"""Fluid PuddleBridge: a JAX MARL environment with a variable agent population.

Agents start near a spawn cell and must reach a goal on the far side of a wall.
Two routes exist: an open alternate path, and a puddle that can only be crossed by
*stacking* — a second agent may enter a puddle cell occupied by one other (capacity
2), and only the recorded top agent may leave a full stack. When the alternate path
is blocked (sampled per episode via ``toggle_other_path_at_reset`` + ``p_block``),
crossing requires cooperative stacking; when open, a non-stacking policy suffices.
This lets the same agents exhibit both fluid and non-fluid solutions. Agents may also
SPAWN new agents (bounded by ``max_agents`` and a per-episode ``episode_pop_cap``).

Interface (shared by all environments in this package)
------------------------------------------------------
``reset(rng, population_explore=False) -> (obs, alive_mask), state``
``step(rng, state, actions)            -> (obs, alive_mask), state, rewards, terminal, info``

* ``obs``        : float array ``(max_agents, obs_dim)``; rows of dead agents are zeroed.
* ``alive_mask`` : int array ``(max_agents,)``; 1 for alive agents, 0 otherwise.
* ``state``      : an :class:`EnvState` pytree (jit/vmap friendly).
* ``rewards``    : float array ``(max_agents,)``.
* ``terminal``   : bool array ``(max_agents,)`` (same value in every slot).
* ``info``       : dict of per-step diagnostics.

The episode auto-resets inside ``step`` on termination. Both methods are jitted and
vmap-able over a batch of environments. The environment is configured via
:class:`EnvConfig`.

Actions (per agent): 0=None, 1=North, 2=South, 3=West, 4=East, 5=Spawn.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageDraw, ImageFont


def _i32_scalar(x):   # -> jnp.int32 scalar shape ()
    return jnp.asarray(x, jnp.int32).reshape(())

def _bool_scalar(x):  # -> jnp.bool_ scalar shape ()
    return jnp.asarray(x, jnp.bool_).reshape(())

# =========================
# Constants & Enumerations
# =========================

class Action(IntEnum):
    """Discrete action space: 4-way movement, do-nothing, and spawn-at-top-left."""
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST  = 3
    EAST  = 4
    SPAWN = 5


class Tile(IntEnum):
    """Static cell types used by the environment's immutable map."""
    LAND   = 0
    WALL   = 1
    PUDDLE = 2
    GOAL   = 3
    SPAWN  = 4


# =========================
# Configuration Dataclass
# =========================

@struct.dataclass
class EnvConfig:
    """Static configuration for the 8x8 grid env (kept JAX-friendly)."""
    grid_rows: int
    grid_cols: int
    max_agents: int
    max_steps_per_episode: int 
    team_reward_on_goal: bool
    spawn_cost: float
    step_cost: float
    VarA: bool = True
    puddle_coords: Optional[Tuple[Tuple[int, int], ...]] = None
    toggle_other_path_at_reset: bool = False
    other_path_coords: Optional[Tuple[Tuple[int, int], ...]] = None
    p_block: float = 0.5


# =========================
# State Dataclass
# =========================

@struct.dataclass
class EnvState:
    """Full mutable simulator state at a single timestep."""
    step_count: jnp.int32
    rng_key: Any

    agent_positions_rc: jnp.ndarray       # (max_agents, 2) int32; (-1,-1) for inactive
    agent_alive_mask: jnp.ndarray         # (max_agents,) bool
    agent_prev_actions: jnp.ndarray       # (max_agents,) int32
    agent_parents: jnp.ndarray            # (max_agents,) int32 (keep parity hook with LBF)

    grid_static_tiles: jnp.ndarray        # (8,8) int8, immutable map of Tile values
    grid_agent_counts: jnp.ndarray        # (8,8) int16, meaningful only for puddle stacking

    episode_return_sum: jnp.ndarray       # (max_agents,) float32
    returned_episode_returns: jnp.ndarray # (max_agents,) float32
    has_reached_goal_flag: jnp.bool_

    puddle_stack_uses: jnp.int32
    successful_spawns: jnp.int32
    grid_puddle_top_id: jnp.ndarray  # int32
    episode_pop_cap: jnp.int32
    alt_path_blocked: jnp.bool_


# =========================
# Top-level Environment API
# =========================

class PuddleBridge:
    """8x8 grid with walls, puddle stacking, a spawn cell, and a goal cell."""

    def _build_static_map(self, config: EnvConfig) -> jnp.ndarray:
        # if (config.grid_rows, config.grid_cols) != (8, 8):
        #     raise ValueError("PuddleBridge requires an 8x8 grid.")

        grid = jnp.full((config.grid_rows, config.grid_cols), Tile.LAND, dtype=jnp.int8)

        # WALLS: rows 2..6, cols 2..5 (unchanged)
        wall_rows = jnp.arange(0, 6)
        wall_cols = jnp.arange(2, 6)
        wr, wc = jnp.meshgrid(wall_rows, wall_cols, indexing="ij")
        grid = grid.at[wr, wc].set(Tile.WALL)
        
        # extra = jnp.array([[0, 3], [1, 5]], dtype=jnp.int32)
        # er, ec = extra[:, 0], extra[:, 1]
        # grid = grid.at[er, ec].set(jnp.int8(Tile.WALL))

        # NEW: PUDDLES from config (fallback to previous 2x2 if None)
        if config.puddle_coords is None:
            puddle_coords = jnp.array([[0, 3], [0, 4], [1, 3], [1, 4]], dtype=jnp.int32)
        else:
            # accept tuples/lists; validate bounds in Python (not in JAX tracing)
            pc_py = [(int(r), int(c)) for (r, c) in config.puddle_coords]
            for r, c in pc_py:
                if not (0 <= r < config.grid_rows and 0 <= c < config.grid_cols):
                    raise ValueError(f"puddle coord {(r,c)} out of bounds.")
            puddle_coords = jnp.asarray(pc_py, dtype=jnp.int32)

        if puddle_coords.size > 0:
            pr, pc = puddle_coords[:, 0], puddle_coords[:, 1]
            grid = grid.at[pr, pc].set(jnp.int8(Tile.PUDDLE))

        # SPAWN & GOAL
        grid = grid.at[0, 0].set(Tile.SPAWN)
        grid = grid.at[0, config.grid_cols - 1].set(Tile.GOAL)
        return grid

    def __init__(self, config: EnvConfig):
        """Store config, build static map, and precompute small constants.
        Notes:
            - Keeps names short; heavy logic lives in reset/step helpers.
            - Deltas follow Action enum ordering; SPAWN uses (0,0) and is handled separately.
            - Obs dims are lazily computed the first time they're needed.
        """
        # Save config
        self.cfg: EnvConfig = config
        self.max_agents = self.cfg.max_agents
        self.VarA = self.cfg.VarA

        # Immutable static tiles (8x8 with spawn/goal/wall/puddle layout)
        self.grid_static_tiles: jnp.ndarray = self._build_static_map(config)

        # NEW: precompute puddle coords on host, then store as JAX arrays
        pc_np = np.argwhere(np.array(self.grid_static_tiles) == np.int8(Tile.PUDDLE)).astype(np.int32)
        self._puddle_coords = jnp.array(pc_np)              # shape (K, 2), K is constant per env
        self._n_puddle = int(pc_np.shape[0])                # Python int → static for shapes
        
        # --- precompute alt-path coords for rendering-only (host numpy) ---
        if getattr(self.cfg, "other_path_coords", None):
            ap_np = np.asarray([(int(r), int(c)) for (r, c) in self.cfg.other_path_coords], dtype=np.int32)
        else:
            ap_np = np.zeros((0, 2), dtype=np.int32)
        self._alt_path_coords_np = ap_np  # for render overlay; independent of episode walls
        
        # Coordinates for convenience
        self.spawn_rc: jnp.ndarray = jnp.array([0, 0], dtype=jnp.int32)
        self.goal_rc: jnp.ndarray = jnp.array([0, config.grid_cols - 1], dtype=jnp.int32)

        # Movement deltas aligned with Action enum (NONE, N, S, W, E, SPAWN)
        # SPAWN has no movement; treated specially in _spawn.
        self._deltas: jnp.ndarray = jnp.array(
            [
                [0, 0],   # NONE
                [-1, 0],  # NORTH
                [1, 0],   # SOUTH
                [0, -1],  # WEST
                [0, 1],   # EAST
                [0, 0],   # SPAWN (placeholder)
            ],
            dtype=jnp.int32,
        )

        # Action/observation specs (obs dims computed lazily)
        self.act_dim: int = len(Action)
        self.init_obs_dim()

    def init_obs_dim(self) -> int:
        R, C, A = self.cfg.grid_rows, self.cfg.grid_cols, self.cfg.max_agents
        D = getattr(self, "act_dim", 6)

        n_puddle = self._n_puddle  # ← use precomputed constant

        slots = (R * C) + n_puddle
        per_slot = 7
        full_slots_dim = slots * per_slot
        tail_dim = (2 + 1) + (A * D)
        obs_dim = int(full_slots_dim + tail_dim)

        self.obs_dim = obs_dim
        self.full_obs_dim_cache = obs_dim
        self.partial_obs_dim_cache = obs_dim
        return obs_dim

    def reset(
    self,
    rng_key: Any,    
    population_explore: bool = False,
) -> Tuple[jnp.ndarray, EnvState]:
        """Reset environment.
        If population_explore=True:
        - Sample per-episode pop cap in [1, cfg.max_agents].
        - Sample requested initial alive count in [1, episode_pop_cap].
        - Place up to four agents at (0,0),(0,1),(1,0),(1,1) if those cells are land-like.
            Extra requested agents beyond available adjacent slots are ignored.
        Otherwise:
        - Start with one agent at (0,0) and cap = cfg.max_agents.
        """
        key = rng_key
        R, C, A = self.cfg.grid_rows, self.cfg.grid_cols, self.cfg.max_agents
        tiles0 = self.grid_static_tiles
        
        key, k_block = jax.random.split(key, 2)
        sample_block = jnp.bool_(self.cfg.toggle_other_path_at_reset) & jax.random.bernoulli(k_block, self.cfg.p_block)
        
        if self.cfg.other_path_coords is None:
            op_rc = jnp.zeros((0, 2), dtype=jnp.int32)
        else:
            op_rc = jnp.asarray([(int(r), int(c)) for (r, c) in self.cfg.other_path_coords], dtype=jnp.int32)
            
        def _block_alt(t):
        # scatter set WALL on the other_path_coords
            pr, pc = op_rc[:, 0], op_rc[:, 1]
            return t.at[pr, pc].set(jnp.int8(Tile.WALL))

        tiles = jax.lax.cond(sample_block, _block_alt, lambda t: t, tiles0)

        # --- Sample per-episode cap from 1..max_agents, and requested init pop from 1..cap (JAX-safe) ---
        key, k_places, k_cap = jax.random.split(key, 3)
        A_i32 = jnp.int32(A)

        def _sample(_):
            # Uniform on {1,2,3,4} but respects A if A < 4
            max_first_wave = jnp.minimum(A_i32, jnp.int32(4))
            places = jax.random.randint(k_places, (), 1, max_first_wave + 1, dtype=jnp.int32)

            # cap uniform in {places..A}
            episode_cap = jax.random.randint(k_cap, (), places, A_i32 + 1, dtype=jnp.int32)
            return episode_cap, places

        def _fixed(_):
            return A_i32, jnp.int32(1)

        episode_cap, places = jax.lax.cond(
            jnp.bool_(population_explore), _sample, _fixed, operand=None
        )        

        # --- Initialize state arrays (static shapes) ---
        pos = jnp.full((A, 2), -1, dtype=jnp.int32)
        alive = jnp.zeros((A,), dtype=jnp.bool_)
        prev = jnp.zeros((A,), dtype=jnp.int32)
        parents = jnp.full((A,), -1, dtype=jnp.int32)

        # Adjacent coordinates (fixed order) – includes SPAWN as (0,0)
        adj = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)

        # Helper: is a tile land-like (LAND or SPAWN). We avoid WALL/PUDDLE; GOAL is not in these coords anyway.
        def is_land_like(t):  # t: int8
            return (t == jnp.int8(Tile.LAND)) | (t == jnp.int8(Tile.SPAWN))

        # Place up to 4 agents into adj cells; ignore invalid cells.
        def place_body(i, carry):
            p, a, k = carry  # k = how many placed so far
            rc = adj[i]
            r, c = rc[0], rc[1]
            ok = is_land_like(tiles[r, c])
            can_place = ok & (k < places)

            p = jax.lax.cond(can_place, lambda pp: pp.at[k].set(rc), lambda pp: pp, p)
            a = jax.lax.cond(can_place, lambda aa: aa.at[k].set(True), lambda aa: aa, a)
            k = jax.lax.cond(can_place, lambda kk: kk + 1, lambda kk: kk, k)
            return (p, a, k)

        pos, alive, _ = jax.lax.fori_loop(0, 4, place_body, (pos, alive, jnp.int32(0)))

        # If not exploring pop, ensure at least agent 0 is alive at spawn (defensive, but idempotent).
        def ensure_minimal(carry):
            p, a = carry
            # If no one alive (should not happen since init_alive_req >= 1), set agent 0 at spawn.
            none_alive = ~jnp.any(a)
            p = jax.lax.cond(none_alive, lambda pp: pp.at[0].set(self.spawn_rc), lambda pp: pp, p)
            a = jax.lax.cond(none_alive, lambda aa: aa.at[0].set(True),          lambda aa: aa, a)
            return (p, a)

        pos, alive = ensure_minimal((pos, alive))

        # Occupancy (puddle-only counts) and top-id map
        agent_counts = jnp.zeros_like(tiles, dtype=jnp.int16)
        grid_puddle_top_id = jnp.full_like(tiles, -1, dtype=jnp.int32)

        # Episode bookkeeping
        ret_sum = jnp.zeros((A,), dtype=jnp.float32)
        ret_buf = jnp.zeros((A,), dtype=jnp.float32)

        state = EnvState(
            step_count=jnp.int32(0),
            rng_key=key,
            agent_positions_rc=pos,
            agent_alive_mask=alive,
            agent_prev_actions=prev,
            agent_parents=parents,
            grid_static_tiles=tiles,
            grid_agent_counts=agent_counts,
            grid_puddle_top_id=grid_puddle_top_id,
            episode_return_sum=ret_sum,
            returned_episode_returns=ret_buf,
            has_reached_goal_flag=jnp.bool_(False),
            puddle_stack_uses=jnp.int32(0),
            successful_spawns=jnp.int32(0),
            episode_pop_cap=episode_cap,  # per-episode cap used by _spawn
            alt_path_blocked=sample_block,   # record whether other path is blocked this episode
        )

        # Build initial observation
        obs_tuple = self._get_obs(state)   # (obs_array, alive_mask)
        return obs_tuple, state

# Below is a self-contained implementation to drop into the PuddleBridge class.

    def _is_legal_move(self, state: EnvState, src_rc: jnp.ndarray, dst_rc: jnp.ndarray) -> jnp.bool_:
        """Check if a move is legal under wall/puddle/stacking rules (with puddle cap=2).
        Rules:
            • Walls are impassable.
            • Land/Goal/Spawn ⇒ single-occupancy only (must be empty to enter).
            • Land → Puddle ⇒ allowed regardless of current occupancy, but puddles have max capacity 2.
            • Puddle → Puddle ⇒ allowed only if destination already has ≥1 agent *and* capacity not full (dst_count < 2).
        Notes:
            • `grid_agent_counts` matters only for puddles; land occupancy is via positions.
            • Fully JAX-friendly (no Python branching depending on array values).
        """
        R = self.cfg.grid_rows
        C = self.cfg.grid_cols

        # Destination indices and in-bounds mask
        r, c = dst_rc[0], dst_rc[1]
        in_bounds = (r >= 0) & (r < R) & (c >= 0) & (c < C)

        # Safe-clipped indices for reading tiles/counts (avoids OOB index when in_bounds is False)
        rr = jnp.clip(r, 0, R - 1)
        cc = jnp.clip(c, 0, C - 1)

        # Source safe-clipped indices (agent should be valid, but clip defensively)
        sr = jnp.clip(src_rc[0], 0, R - 1)
        sc = jnp.clip(src_rc[1], 0, C - 1)

        dst_tile = state.grid_static_tiles[rr, cc]
        src_tile = state.grid_static_tiles[sr, sc]

        noop = jnp.all(src_rc == dst_rc)
        not_wall = dst_tile != jnp.int8(Tile.WALL)

        # Land occupancy via explicit position check
        alive = state.agent_alive_mask
        pos = state.agent_positions_rc  # [A, 2]
        occ_any = jnp.any(jnp.logical_and(alive, jnp.all(pos == dst_rc[None, :], axis=1)))

        dst_is_puddle = dst_tile == jnp.int8(Tile.PUDDLE)
        src_is_puddle = src_tile == jnp.int8(Tile.PUDDLE)

        # Puddle occupancy from counts
        dst_count = state.grid_agent_counts[rr, cc]

        # Capacity constraint for puddles (cap = 2)
        puddle_has_capacity = dst_count < 2

        # Casework by destination kind
        # Land-like (Land/Goal/Spawn): require empty
        legal_land_like = (~dst_is_puddle) & (~occ_any)

        # Puddle destination:
        #   - from LAND-like: allowed iff puddle has capacity (<2)
        #   - from PUDDLE: allowed iff puddle already has ≥1 AND has capacity (<2)
        legal_puddle_from_land = puddle_has_capacity
        legal_puddle_from_puddle = (dst_count >= 1) & puddle_has_capacity
        legal_puddle = jnp.where(src_is_puddle, legal_puddle_from_puddle, legal_puddle_from_land)

        legal_dest = jnp.where(dst_is_puddle, legal_puddle, legal_land_like)

        legal = jnp.where(noop, jnp.bool_(True), in_bounds & not_wall & legal_dest)
        return jnp.bool_(legal)


    def _bottom_locked_mask(self, state: EnvState) -> jnp.ndarray:
        """
        [max_agents] bool – bottom-locked agents for THIS step (frozen for the whole step).
        Bottom-locked iff: alive, on a puddle cell with count==2, and agent_id != top_id recorded for that cell.
        Uses linear indexing to avoid fancy-index pitfalls.
        """
        R, C, N = self.cfg.grid_rows, self.cfg.grid_cols, self.cfg.max_agents

        pos   = state.agent_positions_rc          # [N, 2], int32
        alive = state.agent_alive_mask            # [N], bool
        tiles = state.grid_static_tiles           # [R, C], int8
        cnt   = state.grid_agent_counts           # [R, C], int32
        top   = getattr(state, "grid_puddle_top_id", None)
        if top is None:
            top = jnp.full((R, C), -1, dtype=jnp.int32)

        # clip to bounds (keeps dead/off-grid safe too)
        r = jnp.clip(pos[:, 0], 0, R - 1)
        c = jnp.clip(pos[:, 1], 0, C - 1)
        lin = r * C + c   # [N], linear indices into [R*C]

        # flatten and gather
        tiles_flat = tiles.reshape(-1)
        cnt_flat   = cnt.reshape(-1)
        top_flat   = top.reshape(-1)

        tile_here = jnp.take(tiles_flat, lin)           # [N]
        cnt_here  = jnp.take(cnt_flat,   lin)           # [N]
        top_here  = jnp.take(top_flat,   lin)           # [N]  (=-1 if not stacked)

        is_puddle     = (tile_here == jnp.int8(Tile.PUDDLE))
        is_two_stack  = (cnt_here == 2)
        has_valid_top = (top_here >= 0)

        ids = jnp.arange(N, dtype=jnp.int32)            # [N]
        # elementwise compare each agent's id against its cell's recorded top_id
        bottom_locked = alive & is_puddle & is_two_stack & has_valid_top & (ids != top_here)
        return bottom_locked

    def _spawn(self, state: EnvState, spawner_id: jnp.int32):
        """Attempt to create exactly one new agent at the single spawn cell.
        Deterministic: picks lowest-id dead slot. No effect on puddle counts/top-id.
        Returns: (state2, did_spawn: bool)
        """
        # Is spawn tile currently occupied by any alive agent?
        occupied = jnp.any(
        state.agent_alive_mask &
        jnp.all(state.agent_positions_rc == self.spawn_rc[None, :], axis=1)
        )
        tile_free = ~occupied

        # per-episode cap
        alive_count = jnp.sum(state.agent_alive_mask.astype(jnp.int32))
        pop_ok = alive_count < state.episode_pop_cap

        dead_mask = ~state.agent_alive_mask
        sentinel = jnp.int32(self.cfg.max_agents)
        ids = jnp.arange(self.cfg.max_agents, dtype=jnp.int32)
        masked_ids = jnp.where(dead_mask, ids, sentinel)
        new_id = jnp.min(masked_ids)
        has_dead = new_id < jnp.int32(self.cfg.max_agents)

        did_spawn = tile_free & pop_ok & has_dead

        def do_spawn(s: EnvState):
            pos = s.agent_positions_rc.at[new_id].set(self.spawn_rc)
            alive = s.agent_alive_mask.at[new_id].set(True)
            parents = s.agent_parents.at[new_id].set(jnp.int32(spawner_id))
            return s.replace(
                agent_positions_rc=pos,
                agent_alive_mask=alive,
                agent_parents=parents,
                successful_spawns=s.successful_spawns + jnp.int32(1),
            )

        state2 = jax.lax.cond(did_spawn, do_spawn, lambda s: s, state)
        return state2, did_spawn

    def _move(self, state: EnvState, actions: jnp.ndarray):
        """Resolve one step in ascending agent id with frozen bottom-lock.
        Handles SPAWN (serialized) and returns (state2, spawns_succeeded:int32).
        """
        R, C, N = self.cfg.grid_rows, self.cfg.grid_cols, self.cfg.max_agents
        pre_bottom_locked = self._bottom_locked_mask(state)  # frozen across the step

        def body(i, carry):
            st, spawn_count = carry
            act_i   = jnp.int32(actions[i])
            alive_i = st.agent_alive_mask[i]

            # action kind
            is_n = act_i == jnp.int32(Action.NORTH)
            is_s = act_i == jnp.int32(Action.SOUTH)
            is_w = act_i == jnp.int32(Action.WEST)
            is_e = act_i == jnp.int32(Action.EAST)
            is_spawn = act_i == jnp.int32(Action.SPAWN)
            is_move = is_n | is_s | is_w | is_e

            # movement is still blocked by bottom-lock; SPAWN is allowed even if bottom-locked
            pre_locked = pre_bottom_locked[i]
            move_can_act  = alive_i & (~pre_locked)
            spawn_can_act = _bool_scalar(alive_i & is_spawn)          # <-- exception: ignore bottom-lock for SPAWN

            src = st.agent_positions_rc[i]

            # ---- SPAWN branch (serialized, uses live state) ----
            def do_spawn(c):
                st1, sc0 = c
                st1, did = self._spawn(st1, _i32_scalar(i))    # ensure scalar id
                # update prev action with a scalar value
                prev = st1.agent_prev_actions
                prev = prev.at[_i32_scalar(i)].set(act_i)      # <-- act_i is 0-D now
                st1 = st1.replace(agent_prev_actions=prev)

                sc1 = sc0 + jnp.where(_bool_scalar(did), jnp.int32(1), jnp.int32(0))
                return (st1, sc1)

            # ---- MOVE/NONE branch (unchanged logic) ----
            def do_move_or_none(carry):
                st0, sc0 = carry

                dr = jnp.where(is_n, -1, jnp.where(is_s, 1, 0))
                dc = jnp.where(is_w, -1, jnp.where(is_e, 1, 0))
                cand = src + jnp.stack([dr, dc], axis=0)

                # indices & tiles
                R, C = self.cfg.grid_rows, self.cfg.grid_cols
                sr = jnp.clip(src[0], 0, R - 1); sc = jnp.clip(src[1], 0, C - 1)
                drc = jnp.clip(cand[0], 0, R - 1); dcc = jnp.clip(cand[1], 0, C - 1)

                tiles = st0.grid_static_tiles
                src_is_pud = (tiles[sr, sc] == jnp.int8(Tile.PUDDLE))
                dst_is_pud = (tiles[drc, dcc] == jnp.int8(Tile.PUDDLE))

                # counts BEFORE
                src_cnt_before = st0.grid_agent_counts[sr, sc]
                dst_cnt_before = st0.grid_agent_counts[drc, dcc]
                top_here       = st0.grid_puddle_top_id[sr, sc]

                # only the recorded TOP may leave a 2-stack
                allowed_to_leave = (~src_is_pud) | (src_cnt_before < 2) | (
                    (src_cnt_before == 2) & (top_here == jnp.int32(i))
                )

                # base rule + your "top behaves like land" override
                base_legal = self._is_legal_move(st0, src, cand)
                top_of_two = src_is_pud & (src_cnt_before == 2) & (top_here == jnp.int32(i))
                override_p2p = top_of_two & dst_is_pud & (dst_cnt_before < 2)

                legal_move = move_can_act & is_move & allowed_to_leave & (base_legal | override_p2p)
                dst = jnp.where(legal_move, cand, src)

                # recompute clipped dst for writes
                drc = jnp.clip(dst[0], 0, R - 1); dcc = jnp.clip(dst[1], 0, C - 1)

                # dtype-safe counts
                one  = jnp.array(1, dtype=st0.grid_agent_counts.dtype)
                zero = jnp.array(0, dtype=st0.grid_agent_counts.dtype)
                dec = jnp.where(legal_move & src_is_pud, one,  zero)
                inc = jnp.where(legal_move & dst_is_pud,  one,  zero)

                counts = st0.grid_agent_counts
                counts = counts.at[sr, sc].add(-dec)
                counts = counts.at[drc, dcc].add(inc)

                # top-of-stack updates
                clear_src  = legal_move & src_is_pud & (src_cnt_before == 2)
                set_dst_top = legal_move & dst_is_pud & (dst_cnt_before == 1)

                top_map = st0.grid_puddle_top_id
                top_map = top_map.at[sr, sc].set(jnp.where(clear_src, jnp.int32(-1), top_map[sr, sc]))
                top_map = top_map.at[drc, dcc].set(jnp.where(set_dst_top, jnp.int32(i), top_map[drc, dcc]))

                new_pos  = st0.agent_positions_rc.at[i].set(dst)
                new_prev = st0.agent_prev_actions.at[i].set(act_i)

                st1 = st0.replace(
                    agent_positions_rc=new_pos,
                    agent_prev_actions=new_prev,
                    grid_agent_counts=counts,
                    grid_puddle_top_id=top_map,
                    puddle_stack_uses=st0.puddle_stack_uses + jnp.where(set_dst_top, jnp.int32(1), jnp.int32(0)),
                )
                return (st1, sc0)

            # choose branch: SPAWN allowed even if bottom-locked
            return jax.lax.cond(_bool_scalar(spawn_can_act), do_spawn, do_move_or_none, (st, spawn_count))            

        state2, spawn_count = jax.lax.fori_loop(0, N, body, (state, jnp.int32(0)))
        return state2, spawn_count

    def _get_obs(self, state: EnvState):
        R, C, A = self.cfg.grid_rows, self.cfg.grid_cols, self.cfg.max_agents

        tiles  = state.grid_static_tiles.astype(jnp.int32)
        counts = state.grid_agent_counts
        topmap = state.grid_puddle_top_id
        pos    = state.agent_positions_rc
        alive  = state.agent_alive_mask

        # --- per-cell (id+1) sums ---
        lin = (jnp.clip(pos[:, 0], 0, R - 1) * C + jnp.clip(pos[:, 1], 0, C - 1)).astype(jnp.int32)
        valid = alive & (pos[:, 0] >= 0) & (pos[:, 1] >= 0)
        add_vals = (jnp.arange(A, dtype=jnp.int32) + 1) * valid.astype(jnp.int32)
        sum_flat = jnp.zeros((R * C,), dtype=jnp.int32).at[lin].add(add_vals)
        sum_plus1 = sum_flat.reshape(R, C)

        # --- derive base/top ID scalars in [0,1] ---
        is_puddle = (tiles == jnp.int32(Tile.PUDDLE))
        is_two    = (counts == 2)
        has_top   = (topmap >= 0)
        top_plus1    = jnp.where(is_puddle & is_two & has_top, topmap + 1, 0)
        bottom_plus1 = jnp.where(is_puddle & is_two, jnp.maximum(0, sum_plus1 - top_plus1), 0)
        base_plus1   = jnp.where(is_puddle & is_two, bottom_plus1, sum_plus1)
        A_f = jnp.float32(A)
        base_id_norm = base_plus1.astype(jnp.float32) / A_f
        top_id_norm  = top_plus1.astype(jnp.float32)  / A_f

        # --- grid features: base 64 cells + 4 PTOP slots ---
        tiles_oh5 = jnp.eye(5, dtype=jnp.float32)[tiles]               # [R,C,5]
        base_tile6 = jnp.concatenate([tiles_oh5, jnp.zeros((R, C, 1), jnp.float32)], axis=-1)  # [R,C,6]

        # --- use precomputed puddle coords (static shape) ---
        K = self._n_puddle                           # Python int, static per env
        puddle_coords = self._puddle_coords          # shape (K, 2)
        pr = puddle_coords[:, 0]                     # shape (K,)
        pc = puddle_coords[:, 1]                     # shape (K,)

        ptop_tile6   = jnp.tile(jnp.array([0,0,0,0,0,1], jnp.float32), (K, 1))  # (K, 6)
        ptop_id_norm = top_id_norm[pr, pc][:, None]                              # (K, 1)

        # --- dimensions / zero templates (K-aware) ---
        # --- previous actions (one-hot; dead -> zeros) ---
        D = self.act_dim  # 6
        prev = state.agent_prev_actions.astype(jnp.int32)
        prev_oh = jnp.eye(D, dtype=jnp.float32)[jnp.clip(prev, 0, D - 1)] * alive[:, None].astype(jnp.float32)

        # --- dimensions / zero templates (K-aware) ---
        full_slots_dim = (R * C + K) * (6 + 1)    # (tile6 + id_scalar) per slot
        tail_dim       = (2 + 1) + A * D
        full_total_dim = full_slots_dim + tail_dim
        zeros_full     = jnp.zeros((full_total_dim,), dtype=jnp.float32)
        grid_len       = full_slots_dim

        def build_one(a_idx):
            # normalized own id: present for BOTH alive and dead
            own_id_norm = (jnp.float32(a_idx) + 1.0) / jnp.float32(A)

            def dead_branch(_):
                # [grid zeros] + [own_pos zeros(2)] + [own_id] + [prev zeros]
                return jnp.concatenate([
                    jnp.zeros((grid_len + 2,), dtype=jnp.float32),
                    jnp.array([own_id_norm], dtype=jnp.float32),
                    jnp.zeros((A * D,), dtype=jnp.float32),
                ], axis=0)

            def alive_full(_):
                rc = pos[a_idx]
                r_norm = jnp.clip(rc[0], 0, R - 1) / jnp.float32(R - 1)
                c_norm = jnp.clip(rc[1], 0, C - 1) / jnp.float32(C - 1)
                own_pos = jnp.stack([r_norm, c_norm], axis=0)

                grid_vec = jnp.concatenate([
                    base_tile6.reshape(-1),
                    base_id_norm.reshape(-1),
                    ptop_tile6.reshape(-1),
                    ptop_id_norm.reshape(-1),
                ], axis=0)

                return jnp.concatenate(
                    [grid_vec, own_pos, jnp.array([own_id_norm], jnp.float32), prev_oh.reshape(-1)],
                    axis=0
                )

            return jax.lax.cond(alive[a_idx], alive_full, dead_branch, operand=None)

        obs_mat = jax.vmap(build_one)(jnp.arange(A, dtype=jnp.int32))
        
        alive_bool = state.agent_alive_mask
        alive_int  = alive_bool.astype(jnp.int32)   # <- cast for external API

        return obs_mat, alive_int        

    def step(self, rng_key, state: EnvState, actions: jnp.ndarray):
        """Single environment step: movement + SPAWN + termination/reward accounting.
        Returns: obs, next_state, reward (team scalar), done, info
        """
        actions = jnp.asarray(actions, dtype=jnp.int32).reshape((self.cfg.max_agents,))
        
        # 0) bump step counter (will be stored on state2)
        next_step = state.step_count + jnp.int32(1)

        # 1) Resolve movement & spawns (id order, frozen bottom-lock)
        state2, spawn_count = self._move(state, actions)

        # 2) Goal check
        goal_rc = self.goal_rc
        alive_mask = state2.agent_alive_mask
        pos = state2.agent_positions_rc
        any_at_goal = jnp.any(alive_mask & jnp.all(pos == goal_rc[None, :], axis=1))

        # 3) Termination on step limit
        next_step = state.step_count + jnp.int32(1)
        hit_limit = next_step >= jnp.int32(self.cfg.max_steps_per_episode)
        done = any_at_goal | hit_limit

        # 4) Compute per-agent reward vector
        alive_f = alive_mask.astype(jnp.float32)                      # [A]
        alive_n = jnp.sum(alive_f)                                    # scalar
        denom  = jnp.maximum(alive_n, jnp.float32(1.0))               # avoid /0

        # Team base (goal) awarded to all alive agents equally (1.0 each)
        base_goal = jnp.where(any_at_goal & jnp.bool_(self.cfg.team_reward_on_goal), 10.0, 0.0)/alive_n  # scalar

        # Spawn penalty: only on successful spawns; split equally among alive
        spawns_f = spawn_count.astype(jnp.float32)
        spawn_pen_per_alive = (jnp.float32(self.cfg.spawn_cost) * spawns_f) / denom  # scalar

        # Step cost: charged per alive agent each step
        step_pen_per_alive = jnp.float32(self.cfg.step_cost)  # scalar

        # Final per-agent reward vector this step
        rew_vec = (base_goal - spawn_pen_per_alive - step_pen_per_alive) * alive_f  # [A]

        # Optional scalar to return: team sum for this step
        rew = jnp.sum(rew_vec)

        # 5) Bookkeeping (keep per-agent accumulation for your own analyses)
        state2 = state2.replace(
            step_count=next_step,
            episode_return_sum=state2.episode_return_sum + rew_vec,
            has_reached_goal_flag=jnp.logical_or(state2.has_reached_goal_flag, any_at_goal),
        )

        # ----- LBF-style episode-return accounting (team scalar) -----
        # prev team return BEFORE this step:
        prev_team = jnp.sum(state.episode_return_sum)          # scalar
        # new team return AFTER this step:
        new_episode_returns = prev_team + rew                  # scalar

        # terminal scalar, as in LBF
        terminal = done  # same scalar you computed: any_at_goal | hit_limit

        # info snapshot: only emit on terminal step (0.0 otherwise) — LBF equivalent of
        # returned_episode_returns = state.returned_episode_returns * (1 - terminal) + new_episode_returns * terminal
        ret_ep_scalar = jnp.where(terminal, new_episode_returns, jnp.float32(0.0))

        # 6) Build observation from the pre-reset state2 (so the terminal frame is visible)
        obs_tuple = self._get_obs(state2)

        # 7) Info dict
        info = {
            "returned_episode": terminal,                         # 1 only on the terminal transition
            "returned_episode_returns": ret_ep_scalar,            # team scalar; 0.0 otherwise
            "reached_goal": any_at_goal,
            "active_agents": jnp.sum(alive_mask.astype(jnp.float32)),
            "stack_counts_snapshot": state2.grid_agent_counts,
            "puddle_stack_uses": state2.puddle_stack_uses,
            "successful_spawns": state2.successful_spawns,
            "spawns_this_step": spawn_count,
            "hit_step_limit": hit_limit,
            "step_episode_returns": rew,
            "alt_path_blocked": state2.alt_path_blocked,            
        }

        # 8) LBF-style auto-reset of the state when terminal
        def _reset_state(_):
            rng1, _ = jax.random.split(rng_key)
            # Keep your reset signature; population_explore defaults as in your reset()
            return self.reset(rng1)[1]

        def _continue_state(_):
            return state2

        next_state = jax.lax.cond(terminal, _reset_state, _continue_state, operand=None)

        # Per-agent done flags (like your original)
        done_vec = jnp.full((self.max_agents,), terminal)

        return obs_tuple, next_state, rew_vec, done_vec, info

    def render_from_state(
        self, state, save_path=None, show=True, highlight_ids=None,
        agent_z=2.0, *, show_ids=False, id_text_kwargs=None, id_z=None,
        id_bg=True, top_stack_offset=0.12, return_array=True,
        # --- Alt-path hint (as faint walls) ---
        show_alt_path_hint: bool = False,
        alt_hint_alpha: float = 0.28,           # fill alpha vs real walls’ 0.9
        alt_hint_edge_alpha: float = 0.28,      # edge/hatch alpha
        alt_hint_hatch: str = "xx",
        alt_hint_linewidth: float = 0.8,
        alt_hint_skip_if_real_wall: bool = True,
        alt_hint_zorder: float = 0.98,          # just under real walls (1.0)
    ):
    
        tiles = np.asarray(state.grid_static_tiles)
        n = int(tiles.shape[0])
        
        alt_path_rcs = (
            [tuple(rc) for rc in getattr(self, "_alt_path_coords_np", np.zeros((0,2), np.int32)).tolist()]
        )


        # Colors
        floor_color1, floor_color2 = "#f0ead6", "#e8e2ce"
        wall_color,  grid_line_color = "#b3b3b3", "#c7c7c7"
        puddle_color = "#4f83cc"

        # Tile enum fallback
        try:
            TileEnum = Tile
        except NameError:
            class TileEnum:
                LAND=0; WALL=1; PUDDLE=2; SPAWN=3; GOAL=4

        def _where(v):
            rcs = np.argwhere(tiles == int(v))
            return [tuple(rc) for rc in rcs]

        puddle_tiles = _where(getattr(TileEnum, "PUDDLE", 2))
        wall_tiles   = _where(getattr(TileEnum, "WALL", 1))
        spawn_tiles  = _where(getattr(TileEnum, "SPAWN", 3)) or [(0, 0)]
        goal_tiles   = _where(getattr(TileEnum, "GOAL", 4))  or [(0, 7)]

        # Agents
        alive = np.asarray(state.agent_alive_mask).astype(bool)
        pos   = np.asarray(state.agent_positions_rc)
        ids   = np.arange(pos.shape[0])
        agent_rcs = [(int(r), int(c), int(i))
                        for i, (r, c), a in zip(ids, pos, alive)
                        if a and r >= 0 and c >= 0]

        from collections import defaultdict
        occupied = {(r, c) for r, c, _ in agent_rcs}
        bucket: dict[tuple[int, int], list[int]] = defaultdict(list)
        for r, c, i in agent_rcs:
            bucket[(r, c)].append(i)

        top_map = getattr(state, "grid_puddle_top_id", None)
        counts  = getattr(state, "grid_agent_counts", None)

        fig, ax = plt.subplots(figsize=(6.3, 6.3))

        def cell_xy(rc):
            r, c = rc
            return (c, n - 1 - r)

        def cell_center(rc):
            x, y = cell_xy(rc)
            return (x + 0.5, y + 0.5)

        # Floor
        for i in range(n):
            for j in range(n):
                c = floor_color1 if (i + j) % 2 == 0 else floor_color2
                ax.add_patch(patches.Rectangle((j, n-1-i), 1, 1,
                                                facecolor=c, edgecolor=grid_line_color,
                                                linewidth=0.6, zorder=0.0))

        # Walls
        for rc in wall_tiles:
            x, y = cell_xy(rc)
            ax.add_patch(patches.Rectangle(
                (x, y), 1, 1,
                facecolor="#9a9a9a",     # darker fill
                edgecolor="#5a5a5a",     # darker edge
                linewidth=0.8,
                hatch="//",            # denser diagonal lines — more "/" = tighter spacing
                fill=True,
                alpha=1.0,
                zorder=1.0,
            ))

        # Puddles
        for rc in puddle_tiles:
            x, y = cell_xy(rc)
            ax.add_patch(patches.Rectangle((x, y), 1, 1,
                                            facecolor=puddle_color, edgecolor="#2e5e99",
                                            linewidth=1.0, zorder=0.4))
            cx, cy = x + 0.5, y + 0.5
            for rrad, alph, ccol in [(0.45, 0.30, "#cfe2f3"),
                                        (0.30, 0.25, "#9fc5e8"),
                                        (0.15, 0.20, "#6fa8dc")]:
                ax.add_patch(patches.Circle((cx, cy), rrad, facecolor=ccol,
                                            edgecolor="none", alpha=alph, zorder=0.45))
                
        # --- Alt-path hint overlay: render "faint walls" on the corridor ---
        if show_alt_path_hint and getattr(self, "_alt_path_coords_np", None) is not None:
            import matplotlib.colors as mcolors
            alt_coords = self._alt_path_coords_np  # (K, 2) np.int32
            if alt_coords.size > 0:
                # Build RGBA exactly like walls, just with smaller alpha.
                face_rgba = list(mcolors.to_rgba("#9a9a9a"))   # same as new wall fill
                face_rgba[3] = float(alt_hint_alpha)
                edge_rgba = list(mcolors.to_rgba("#6e6e6e"))   # same as new wall edge
                edge_rgba[3] = float(alt_hint_edge_alpha)
                
                for r, c in map(tuple, alt_coords):
                    # bounds guard
                    if r < 0 or r >= n or c < 0 or c >= n:
                        continue
                    # optionally skip if it's already a hard wall this episode
                    is_wall_here = int(tiles[r, c]) == int(getattr(TileEnum, "WALL", 1))
                    if alt_hint_skip_if_real_wall and is_wall_here:
                        continue

                    x, y = cell_xy((r, c))
                    ax.add_patch(patches.Rectangle(
                        (x, y), 1, 1,
                        facecolor=(0.6, 0.6, 0.6, alt_hint_alpha),  # faint fill
                        edgecolor=(0.35, 0.35, 0.35, alt_hint_edge_alpha),
                        linewidth=float(alt_hint_linewidth),
                        hatch="//",           # same dense diagonal pattern
                        fill=True,
                        zorder=float(alt_hint_zorder),
                    ))

        # Spawn (only if unoccupied)
        for rc in spawn_tiles:
            if rc not in occupied:
                cx, cy = cell_center(rc)
                ax.add_patch(patches.Circle((cx, cy), 0.27, facecolor="none",
                                            edgecolor="black", linewidth=1.5, zorder=0.55))
                if 'draw_star' in globals():
                    draw_star(ax, cx, cy, r=0.22, color="black", edge="black", zorder=0.6)
                else:
                    ax.add_patch(patches.RegularPolygon((cx, cy), numVertices=4, radius=0.22,
                                                        orientation=np.pi/4,
                                                        facecolor="black", edgecolor="black",
                                                        zorder=0.6))

        # Goal
        for rc in goal_tiles:
            cx, cy = cell_center(rc)
            ax.add_patch(patches.FancyArrow(cx - 0.1, cy - 0.2, 0, 0.4,
                                            width=0.02, head_width=0, head_length=0,
                                            length_includes_head=True,
                                            facecolor="black", edgecolor="black", zorder=0.8))
            ax.add_patch(patches.Polygon([[cx - 0.1, cy + 0.2],
                                            [cx + 0.2, cy + 0.1],
                                            [cx - 0.1, cy]],
                                            closed=True, facecolor="red", edgecolor="black",
                                            zorder=0.85))

        # Label style
        _id_z = id_z if id_z is not None else 1_000_000.0  # draw above anything else
        label_kw = dict(color="black", fontsize=12, fontweight="bold", ha="center", va="center")
        if id_text_kwargs:
            label_kw.update(id_text_kwargs)

        # Agents
        hi = set(highlight_ids or [])
        for (r, c), id_list in bucket.items():
            cx, cy = cell_center((r, c))
            in_puddle = (r, c) in puddle_tiles
            offsets = [(0.0, 0.0)] if len(id_list) == 1 else [(-0.12, 0.0), (0.12, 0.0)]

            # draw order: bottom first, top last (if we know top)
            ordered_ids = sorted(id_list)
            if in_puddle and len(id_list) == 2 and top_map is not None and counts is not None:
                top_id = int(top_map[r, c])
                if top_id >= 0 and top_id in id_list and int(counts[r, c]) == 2:
                    bottom_id = id_list[0] if id_list[1] == top_id else id_list[1]
                    ordered_ids = [bottom_id, top_id]

            for (dx, dy), aid in zip(offsets, ordered_ids):
                cx2, cy2 = cx + dx, cy + dy

                # NEW: tiny upward bump for the TOP agent of a 2-stack (y increases upward here)
                if (top_stack_offset and in_puddle and len(id_list) == 2 and
                    top_map is not None and counts is not None and
                    int(counts[r, c]) == 2 and int(top_map[r, c]) == int(aid)):
                    cy2 += float(top_stack_offset)

                # agent body
                if 'draw_agent_drone' in globals():
                    draw_agent_drone(ax, cx2, cy2, in_puddle=in_puddle, puddle_color=puddle_color)
                else:
                    face = "white" if not in_puddle else "#e6eef8"
                    ax.add_patch(patches.Circle((cx2, cy2), 0.28, facecolor=face,
                                                edgecolor="black", linewidth=1.2, zorder=agent_z))
                # highlight ring (below label)
                if aid in hi:
                    ax.add_patch(patches.Circle((cx2, cy2), 0.31, facecolor="none",
                                                edgecolor="#ff9900", linewidth=2.0, zorder=_id_z - 0.2))
                # id label + optional halo
                if show_ids:
                    if id_bg:
                        ax.add_patch(patches.Circle((cx2, cy2), 0.16,
                                                    facecolor="white", edgecolor="none",
                                                    alpha=0.85, zorder=_id_z - 0.1))
                    txt = ax.text(cx2, cy2, str(aid), zorder=_id_z, **label_kw)

        # --- Finish & rasterize --- #
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        # attach an Agg canvas explicitly (robust headless rendering)
        if not hasattr(fig, "canvas") or fig.canvas is None:
            FigureCanvas(fig)

        fig.tight_layout()

        # Save to disk first (if requested)
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        # Return numpy array (if requested)
        img_array = None
        if return_array:
            # draw then read back RGBA bytes (tostring_rgb was removed in matplotlib 3.10)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = buf.reshape(h, w, 4)[..., :3].copy()  # drop alpha; detach from canvas

        # Show last (optional; do NOT show+close before rasterizing)
        if show:
            plt.show()

        plt.close(fig)
        return img_array if return_array else None
# ==================================================================================================================================== #

def draw_agent_drone(ax, cx, cy, in_puddle=False,
                    body_r=0.18, body_color="white",
                    sat_r=0.04, sat_offset=0.25, sat_color="#464646",
                    puddle_color="#4f83cc"):
    """White disc body with 4 tiny grey stabilizers (N/E/S/W)."""
    # subtle shadow
    ax.add_patch(patches.Circle((cx, cy - 0.02), body_r * 1.02, facecolor="black",
                                alpha=0.08, edgecolor="none", zorder=4))
    # body
    ax.add_patch(patches.Circle((cx, cy), body_r, facecolor=body_color,
                                edgecolor="black", linewidth=1.0, zorder=5))
    # satellites
    for ang in [0, 90, 180, 270]:
        rad = np.deg2rad(ang)
        sx, sy = cx + np.cos(rad) * sat_offset, cy + np.sin(rad) * sat_offset
        ax.add_patch(patches.Circle((sx, sy), sat_r, facecolor=sat_color,
                                    edgecolor="none", zorder=6))
    # submersion overlay if needed
    if in_puddle:
        ax.add_patch(patches.Circle((cx, cy), body_r * 1.35, facecolor=puddle_color,
                                    edgecolor="none", alpha=0.28, zorder=7))

def draw_star(ax, cx, cy, r=0.22, color="black", edge="black", zorder=9):
    """5-pointed star centered at (cx, cy)."""
    pts = []
    for i in range(10):
        angle = np.pi/2 + i * np.pi/5
        radius = r if i % 2 == 0 else r * 0.4
        pts.append((cx + np.cos(angle) * radius, cy + np.sin(angle) * radius))
    ax.add_patch(patches.Polygon(pts, closed=True, facecolor=color, edgecolor=edge,
                                linewidth=1.0, zorder=zorder))



# ------------ ASCII RENDER ----------------------------------------------------- #

CELL_W = 5  # width per cell for nice alignment

def _pad(tok: str, w: int = CELL_W) -> str:
    return tok[:w].ljust(w)

def ascii_board(env: PuddleBridge, state) -> str:
    """Return a multi-line ASCII representation of the 8x8 board."""
    R, C = env.cfg.grid_rows, env.cfg.grid_cols
    tiles = np.asarray(state.grid_static_tiles)
    counts = np.asarray(state.grid_agent_counts)
    topmap = np.asarray(state.grid_puddle_top_id)

    alive = np.asarray(state.agent_alive_mask).astype(bool)
    pos   = np.asarray(state.agent_positions_rc)

    # bucket agents per (r,c)
    bucket: Dict[Tuple[int, int], List[int]] = {}
    for i, (r, c), a in zip(range(pos.shape[0]), pos, alive):
        r, c = int(r), int(c)
        if a and r >= 0 and c >= 0:
            bucket.setdefault((r, c), []).append(int(i))

    def cell_token(r: int, c: int) -> str:
        ids = bucket.get((r, c), [])
        tile = int(tiles[r, c])
        cnt = int(counts[r, c])

        if len(ids) == 1:
            return _pad(f"{ids[0]:>2}")
        if len(ids) == 2:
            # If this is a 2-stack puddle, order bottom/top using top_id
            if tile == int(Tile.PUDDLE) and cnt == 2:
                top_id = int(topmap[r, c])
                if top_id in ids:
                    bottom_id = ids[0] if ids[1] == top_id else ids[1]
                    return _pad(f"{bottom_id:>2}/{top_id:>2}")
            a, b = sorted(ids)
            return _pad(f"{a:>2}/{b:>2}")

        # Empty cell:
        if tile == int(Tile.WALL):
            return _pad("####")
        if tile == int(Tile.PUDDLE):
            # show puddle with light count hint when >0
            return _pad("~~~~" if cnt == 0 else f"~{cnt:>2}~")
        if tile == int(Tile.SPAWN):
            return _pad("S")
        if tile == int(Tile.GOAL):
            return _pad("G")
        return _pad(".")

    # Header row
    header = "    " + "".join(_pad(str(c)) for c in range(C))
    sep = "    " + "-" * (CELL_W * C)

    lines = [header, sep]
    for r in range(R):
        row = "".join(cell_token(r, c) for c in range(C))
        lines.append(f"{r:>2} | {row}")
    lines.append(sep)
    return "\n".join(lines)

"""Fluid Level-Based Foraging: a JAX MARL environment with a variable agent population.

Agents move on a grid and cooperatively LOAD food items: a food is collected only
when the summed level of the adjacent agents attempting to load it meets or exceeds
the food's level, with reward shared in proportion to contributed level. The
environment is *fluid* because agents may SPAWN new agents at run time (bounded by
``max_agents`` and a per-episode ``max_pop_cap``), so the active population changes
within and across episodes.

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
vmap-able over a batch of environments.

Actions (per agent): 0=None, 1=North, 2=South, 3=West, 4=East, 5=Load, 6=Spawn.
"""

from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple

import time

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct

class Action(IntEnum):
    """
    Discrete actions available to each agent in the Level-Based Foraging environment.
    """
    NONE = 0        # Do nothing
    NORTH = 1       # Move up
    SOUTH = 2       # Move down
    WEST = 3        # Move left
    EAST = 4        # Move right
    LOAD = 5        # Attempt to load adjacent food
    SPAWN = 6       # Spawn a new agent in an empty cell


class CellEntity(IntEnum):
    """
    Types of entities that can occupy a grid cell in the Level-Based Foraging environment.
    """
    EMPTY = 0         # No agent or food
    AGENT = 1         # An agent occupies the cell
    FOOD = 2          # A food item occupies the cell
    OUT_OF_BOUNDS = 3 # Represents cells outside the grid boundaries

    
@struct.dataclass
class EnvState:
    agent_positions: jnp.ndarray           # shape (num_agents, 2)
    agent_levels: jnp.ndarray              # shape (num_agents,)
    agent_alive: jnp.ndarray               # shape (num_agents,), bool mask
    agent_parents: jnp.ndarray             # shape (num_agents,), int32 parent IDs (-1 for none)
    prev_actions: jnp.ndarray 
    food_positions: jnp.ndarray            # shape (num_food, 2)
    food_levels: jnp.ndarray               # shape (num_food,)
    food_alive: jnp.ndarray 
    grid: jnp.ndarray                      # shape (grid_height, grid_width)    
    episode_returns: float 
    returned_episode_returns: float   # shape (num_agents,)
    step: int                              # current timestep
    rng: Any                               # PRNG key
    population_explore: jnp.bool_ = False
    max_pop_cap: chex.Array = struct.field(default_factory=lambda: jnp.int32(0))

class Foraging:
    """
    JAX-based Level-Based Foraging environment.
    Provides reset and step methods for use in jitted loops.
    Uses time.time() to seed RNG if none is provided, ensuring randomization on each run.
    """    
    def __init__(self,
            grid_size: Tuple[int, int],
            init_num_agents: int,
            max_agents: int,
            num_food: int,
            max_level: int,
            max_steps: int,
            agent_view: Tuple[int, int],
            rng: Optional[jax.random.PRNGKey] = None,
            init_agent_levels: Optional[jnp.ndarray] = None,
            init_food_levels: Optional[jnp.ndarray] = None,
            spawn_cost: float = 0.0,
            step_cost: float = 0.0,
            VarA: bool = True):

        
        # grid_size may be a single int (square grid) or an (H, W) sequence.
        if isinstance(grid_size, int):
            self.grid_size = [grid_size, grid_size]
        else:
            self.grid_size = list(grid_size)
        self.init_num_agents = init_num_agents
        self.max_agents = max_agents
        self.num_food = num_food
        self.max_level = max_level
        self.max_steps = max_steps
        self.spawn_cost = spawn_cost
        self.step_cost = step_cost
        self.agent_view = agent_view
        self.obs_dim = 3 * self.num_food + 6 * self.max_agents + 2
        self.act_dim = 7
        self.VarA = VarA
            

        if rng is None:
            seed = int(time.time())
            rng = jax.random.PRNGKey(seed)
        self.rng = rng

        # Optional fixed initial levels; accept any list/array. If None, levels are sampled.
        if init_agent_levels is not None:
            init_agent_levels = jnp.asarray(init_agent_levels, dtype=jnp.int32)
            assert len(init_agent_levels) == self.init_num_agents, \
                f"init_agent_levels must be shape ({self.init_num_agents},)"
        self.init_agent_levels = init_agent_levels

        if init_food_levels is not None:
            init_food_levels = jnp.asarray(init_food_levels, dtype=jnp.int32)
            assert len(init_food_levels) == self.num_food, \
                f"init_food_levels must be shape ({self.num_food},)"
        self.init_food_levels = init_food_levels
    
    def _sample_one_position(self, rng: jax.random.PRNGKey, grid: jnp.ndarray) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        """
        Sample one empty position from the grid.
        """
        h, w = self.grid_size
        total = h * w

        def cond_fn(carry):
            rng_key, pos = carry
            r, c = pos[0], pos[1]
            return grid[r, c] != CellEntity.EMPTY

        def body_fn(carry):
            rng_key, _ = carry
            rng_key, subkey = jax.random.split(rng_key)
            flat = jax.random.randint(subkey, shape=(), minval=0, maxval=total)
            r, c = flat // w, flat % w
            return rng_key, jnp.array([r, c], dtype=jnp.int32)

        rng, pos = body_fn((rng, jnp.array([0, 0], dtype=jnp.int32)))
        rng, pos = jax.lax.while_loop(cond_fn, body_fn, (rng, pos))
        return rng, pos
        
    @partial(jax.jit, static_argnums=0)
    def get_obs(self, state: EnvState) -> jnp.ndarray:
        """
        Flat observation for each agent including:
        - Food positions and levels: (x, y, level) * max_num_food
        - Agent positions, levels, and alive flag: (x, y, level, alive) * max_agents
        - Previous step team action
        Only entities within agent_view are visible
        Output shape: (max_agents, obs_dim)
        """
        obs_dim = 3 * self.num_food +  6  * self.max_agents + 2   # +2: (+1 agent ID, +1 max_pop_cap)

        def build_single_obs(buf, agent_idx):
            is_alive = state.agent_alive[agent_idx]
            agent_pos = state.agent_positions[agent_idx]

            def in_view(pos):
                half_h, half_w = self.agent_view[0] // 2, self.agent_view[1] // 2
                return jnp.all(jnp.abs(pos - agent_pos) <= jnp.array([half_h, half_w]))

            def alive_obs_fn(_):
                agent_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)

                def food_step(i, buf):
                    is_valid = (i < state.food_positions.shape[0]) & state.food_alive[i]
                    pos = jnp.where(i < state.food_positions.shape[0], state.food_positions[i], jnp.array([0, 0]))
                    in_range = in_view(pos)
                    level = jnp.where(i < state.food_levels.shape[0], state.food_levels[i], 0)
                    x = jnp.where(is_valid & in_range, pos[0], -1)
                    y = jnp.where(is_valid & in_range, pos[1], -1)
                    lvl = jnp.where(is_valid & in_range, level, 0)
                    buf = buf.at[3 * i + 0].set(x)
                    buf = buf.at[3 * i + 1].set(y)
                    buf = buf.at[3 * i + 2].set(lvl)
                    return buf

                agent_obs = jax.lax.fori_loop(0, self.num_food, food_step, agent_obs)

                def agent_step(i, buf):
                    pos = state.agent_positions[i]
                    lvl = state.agent_levels[i]
                    alive = state.agent_alive[i].astype(jnp.float32)
                    spwn  = state.agent_parents[i].astype(jnp.float32)   # NEW
                    pa = state.prev_actions[i].astype(jnp.float32)
                    in_range = in_view(pos)
                    offset = 3 * self.num_food + 6 * i
                    x = jnp.where(in_range, pos[0], -1)
                    y = jnp.where(in_range, pos[1], -1)
                    l = jnp.where(in_range, lvl, 0)
                    a = jnp.where(in_range, alive, 0)
                    buf = buf.at[offset + 0].set(x)
                    buf = buf.at[offset + 1].set(y)
                    buf = buf.at[offset + 2].set(l)
                    buf = buf.at[offset + 3].set(a)
                    buf = buf.at[offset + 4].set(spwn)
                    buf = buf.at[offset + 5].set(pa) 
                    return buf

                agent_obs = jax.lax.fori_loop(0, self.max_agents, agent_step, agent_obs)
                
                cap_val = jnp.asarray(state.max_pop_cap, dtype=jnp.float32)
                agent_obs = agent_obs.at[-2].set(cap_val)     # max_pop_cap
                agent_obs = agent_obs.at[-1].set(agent_idx)   # agent ID                
                return agent_obs

            agent_obs = jax.lax.cond(is_alive, alive_obs_fn, lambda _: jnp.zeros((obs_dim,), dtype=jnp.float32), operand=None)
            return buf.at[agent_idx].set(agent_obs), None

        obs, _ = jax.lax.scan(build_single_obs, jnp.zeros((self.max_agents, obs_dim), dtype=jnp.float32), jnp.arange(self.max_agents))        
        return obs, state.agent_alive
    
    def reset(self, rng: jax.random.PRNGKey, population_explore: bool = False) -> Tuple[jnp.ndarray, EnvState]:
        """Initialise state so that every food is strictly inside the grid."""
        grid = jnp.full(self.grid_size, CellEntity.EMPTY, dtype=jnp.int32)

        agent_positions = -jnp.ones((self.max_agents, 2), dtype=jnp.int32)
        agent_levels    = jnp.zeros((self.max_agents,),    dtype=jnp.int32)
        agent_alive     = jnp.zeros((self.max_agents,),    dtype=jnp.int32)

        food_positions  = jnp.zeros((self.num_food, 2),    dtype=jnp.int32)

        H, W = self.grid_size
        
        rng, sub_init = jax.random.split(rng)
        init_agents = jax.lax.cond(
            jnp.asarray(population_explore, dtype=jnp.bool_),
            # TRAINING: random in [2, max_agents]
            lambda: jax.random.randint(sub_init, (), self.init_num_agents, self.max_agents + 1, dtype=jnp.int32),
            # TESTING / default: use configured initial count
            lambda: jnp.int32(self.init_num_agents),
        )
        
        # --- per-episode population cap ---
        rng, sub_cap = jax.random.split(rng)
        max_pop_cap = jax.lax.cond(
            jnp.asarray(population_explore, dtype=jnp.bool_),
            # inclusive range: [init_agents, self.max_agents]
            lambda: jax.random.randint(sub_cap, (), init_agents, self.max_agents + 1, dtype=jnp.int32),
            lambda: jnp.int32(self.max_agents),
        )

        if self.VarA is None:
            init_agents = self.init_num_agents
            max_pop_cap = self.init_num_agents
            
        agent_active_mask = (jnp.arange(self.max_agents, dtype=jnp.int32) < init_agents)
    
        # ------------------------------------------------------------------ #
        # helper: keep resampling until the pos is empty **and** not on edge #
        # ------------------------------------------------------------------ #
        def _sample_food_position(r_key, g):
            """Return rng', pos guaranteed inside grid (1..H-2, 1..W-2) & empty."""
            def cond(carry):
                rk, p = carry
                r, c  = p[0], p[1]
                on_edge = (r == 0) | (r == H - 1) | (c == 0) | (c == W - 1)
                occupied = g[r, c] != CellEntity.EMPTY
                return on_edge | occupied            # keep looping while invalid

            def body(carry):
                rk, _ = carry
                rk, p = self._sample_one_position(rk, g)
                return rk, p

            r_key, pos = self._sample_one_position(r_key, g)
            r_key, pos = jax.lax.while_loop(cond, body, (r_key, pos))
            return r_key, pos
        # ------------------------------------------------------------------ #

        def place_entities(carry, idx_type):
            g, r_key, apos, fpos = carry
            idx, typ = idx_type

            # choose the correct sampler depending on entity type
            r_key, pos = jax.lax.cond(
                typ == CellEntity.FOOD,
                lambda args: _sample_food_position(*args),
                lambda args: self._sample_one_position(*args),
                (r_key, g),
            )
            
            
            is_agent = (typ == CellEntity.AGENT)
            # place_flag: Food always placed; Agent only if mask says active
            place_flag = jax.lax.cond(is_agent, lambda _: agent_active_mask[idx], lambda _: True, operand=None)

            # mark grid
            g = jax.lax.cond(
                place_flag,
                lambda gg: gg.at[pos[0], pos[1]].set(typ),
                lambda gg: gg,
                g,
            )

            # write position into the right buffer
            apos = jax.lax.cond(
                is_agent & place_flag,
                lambda buf: buf.at[idx].set(pos),
                lambda buf: buf,
                apos,
            )
            fpos = jax.lax.cond(
                typ == CellEntity.FOOD,
                lambda buf: buf.at[idx].set(pos),
                lambda buf: buf,
                fpos,
            )
            return (g, r_key, apos, fpos), ()

        ids_agents = jnp.arange(self.max_agents)
        ids_food   = jnp.arange(self.num_food)
        all_ids  = jnp.concatenate([ids_agents, ids_food])
        all_typs = jnp.concatenate([
            jnp.full((self.max_agents,), CellEntity.AGENT, dtype=jnp.int32),
            jnp.full((self.num_food,),        CellEntity.FOOD,  dtype=jnp.int32),
        ])

        (grid, rng, agent_positions, food_positions), _ = jax.lax.scan(
            place_entities,
            (grid, rng, agent_positions, food_positions),
            (all_ids, all_typs),
        )

        agent_alive = agent_active_mask.astype(jnp.int32)
        
        if self.init_agent_levels is None:
            rng, subkey = jax.random.split(rng)
            # sample for all max_agents; inactive get zeroed
            sampled_levels = jax.random.randint(subkey, (self.max_agents,), 1, self.max_level + 1, dtype=jnp.int32)
            agent_levels = jnp.where(agent_active_mask, sampled_levels, 0)                    
        else:
            # --- start with provided levels for configured initial count ---
            init_levels_full = jnp.zeros((self.max_agents,), dtype=jnp.int32)
            init_levels_full = init_levels_full.at[:self.init_num_agents].set(self.init_agent_levels)

            # --- if pop_explore added extra alive agents, sample their levels randomly ---
            extra_n = jnp.maximum(0, init_agents - self.init_num_agents)  # int32

            rng, subkey = jax.random.split(rng)
            # static-size pool for all potential extras
            tail_len = self.max_agents - self.init_num_agents
            extra_pool = jax.random.randint(
                subkey,
                (tail_len,),
                1, self.max_level + 1,
                dtype=jnp.int32,
            )

            # mask keeps only the first `extra_n` values; rest -> 0
            extra_mask = (jnp.arange(tail_len, dtype=jnp.int32) < extra_n)
            extra_vals = jnp.where(extra_mask, extra_pool, 0)

            # write the entire tail in one static-sized scatter
            init_levels_full = init_levels_full.at[self.init_num_agents:].set(extra_vals)

            # --- zero out inactive slots (unchanged behavior) ---
            agent_levels = jnp.where(agent_active_mask, init_levels_full, 0)


        food_levels = jnp.zeros((self.num_food,), dtype=jnp.int32)
        food_alive = jnp.ones((self.num_food,), dtype=jnp.int32)
        if self.init_food_levels is None:
            rng, subkey = jax.random.split(rng)
            food_levels = jax.random.randint(subkey, (self.num_food,), 1, self.max_level + 1, dtype=jnp.int32)
        else: food_levels = self.init_food_levels

        agent_parents = jnp.where(agent_active_mask, 0, -1).astype(jnp.int32) 
        prev_actions = jnp.full((self.max_agents,), Action.NONE, dtype=jnp.int32)
        
        state = EnvState(
            agent_positions=agent_positions,
            agent_levels=agent_levels,
            agent_alive=agent_alive,
            agent_parents=agent_parents,
            prev_actions=prev_actions,
            food_positions=food_positions,
            food_levels=food_levels,
            food_alive=food_alive,
            grid=grid,
            episode_returns=0.0,
            returned_episode_returns=0.0,
            step=0, rng=rng,
            population_explore=jnp.asarray(population_explore, dtype=jnp.bool_),
            max_pop_cap=max_pop_cap,
        )
        return self.get_obs(state), state
        
    def _decide_spawn_level(self, parent_level: int) -> int:
        """Determine the level of a newly spawned agent based on the parent's level."""
        return parent_level  # placeholder logic; customize later

    @partial(jax.jit, static_argnums=0)
    def step(self, rng: jax.random.PRNGKey, state: EnvState, actions: jnp.ndarray):
        # ------------------------------------------------------------------
        # 1. Mask invalid actions from dead agents
        # ------------------------------------------------------------------
        actions = jnp.where(state.agent_alive, actions, Action.NONE)

        # ------------------------------------------------------------------
        # 2. Move phase
        # ------------------------------------------------------------------
        positions, grid = state.agent_positions, state.grid
        rewards = jnp.zeros((self.max_agents,), dtype=jnp.float32)

        deltas = jnp.array([
            [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]
        ], dtype=jnp.int32)

        def process_agent(i, carry):
            positions, grid = carry

            def skip_move(c): return c

            def move_logic(c):
                act, delta = actions[i], deltas[actions[i]]
                proposed = jnp.clip(positions[i] + delta, jnp.array([0, 0]), jnp.array([self.grid_size[0] - 1, self.grid_size[1] - 1]))
                old = positions[i]

                def is_empty(cell): return grid[cell[0], cell[1]] == CellEntity.EMPTY

                def move_fn(c):
                    pos, grd = c
                    pos = pos.at[i].set(proposed)
                    grd = grd.at[old[0], old[1]].set(CellEntity.EMPTY)
                    grd = grd.at[proposed[0], proposed[1]].set(CellEntity.AGENT)
                    return pos, grd

                return jax.lax.cond(is_empty(proposed), move_fn, lambda c: c, c)

            return jax.lax.cond(state.agent_alive[i], move_logic, skip_move, carry)

        positions, grid = jax.lax.fori_loop(0, self.max_agents, process_agent, (positions, grid))
        success_flags = jnp.zeros((self.max_agents,), dtype=jnp.int32)
        
        
        # ------------------------------------------------------------------
        # 3. Forage phase 
        # ------------------------------------------------------------------

        def calculate_reward(capable, food_lvl, agent_levels):
            sum_levels = jnp.sum(agent_levels * capable.astype(jnp.int32))
            loaded = sum_levels >= food_lvl
            contrib = jnp.where(
                capable,
                (agent_levels.astype(jnp.float32) * food_lvl) / jnp.where(sum_levels == 0, 1, sum_levels),
                0.0,
            )
            return loaded, contrib

        def process_food(j, carry):
            grid, rewards, alive, success_flags = carry
            # Skip dead food
            def skip_food(args):
                return args

            def process(args):
                grid, rewards, alive, success_flags = args
                food_pos, food_lvl = state.food_positions[j], state.food_levels[j]
                diffs = jnp.abs(positions - food_pos)
                is_neighbor = ((diffs[:, 0] == 1) & (diffs[:, 1] == 0)) | ((diffs[:, 0] == 0) & (diffs[:, 1] == 1))
                load_mask = actions == Action.LOAD
                capable = is_neighbor & load_mask & state.agent_alive
                loaded, contrib = calculate_reward(capable, food_lvl, state.agent_levels)
                rewards += jnp.where(loaded, contrib, 0.0)
                grid = jax.lax.cond(loaded, lambda g: g.at[food_pos[0], food_pos[1]].set(CellEntity.EMPTY), lambda g: g, grid)
                alive = alive.at[j].set(alive[j] & jnp.logical_not(loaded))
                success_flags = jnp.where(loaded, success_flags | capable, success_flags)
                return grid, rewards, alive, success_flags

            return jax.lax.cond(state.food_alive[j], process, skip_food, (grid, rewards, alive, success_flags))

        grid, rewards, food_alive, success_flags = jax.lax.fori_loop(
            0, self.num_food, process_food, (grid, rewards, state.food_alive, success_flags)
        )

        # ------------------------------------------------------------------
        # 4. Spawn phase
        # ------------------------------------------------------------------
        spawn_requests = (actions == Action.SPAWN) & state.agent_alive
        num_requests = jnp.sum(spawn_requests.astype(jnp.int32))
        
        alive_now = jnp.sum(state.agent_alive.astype(jnp.int32))
        slots = jnp.maximum(0, state.max_pop_cap - alive_now)
        num_requests_satisfiable = jnp.minimum(num_requests, slots)

        agent_levels = state.agent_levels

        alive_before_spawn = state.agent_alive
        spawned_mask = jnp.zeros((self.max_agents,), dtype=jnp.int32)
        parents = state.agent_parents  # start from previous vector

        def spawn_fn(carry, i):
            grid, rng, positions, agent_alive, agent_levels, par_buf, remaining, spawned_mask = carry

            def do_spawn(c):
                g, r_key, pos_buf, alive_buf, lvl_buf, par_buf, rem, spawn_mask = c
                r_key, new_pos = self._sample_one_position(r_key, g)
                g = g.at[new_pos[0], new_pos[1]].set(CellEntity.AGENT)
                pos_buf = pos_buf.at[i].set(new_pos)
                alive_buf = alive_buf.at[i].set(True)

                # Decide level and record parent ------------------------------------------------
                parent_idx = jnp.argmax(spawn_requests.astype(jnp.int32) * jnp.arange(self.max_agents))
                lvl_buf = lvl_buf.at[i].set(self._decide_spawn_level(state.agent_levels[parent_idx]))                
                par_buf = par_buf.at[parent_idx].add(1)  
                par_buf = par_buf.at[i].set(0)  # Set parent index for the new agent    
                spawn_mask = spawn_mask.at[i].set(True)
                return g, r_key, pos_buf, alive_buf, lvl_buf, par_buf, rem - 1, spawn_mask


            def no_spawn(c): return c

            cond = jnp.logical_not(agent_alive[i]) & (remaining > 0)
            return jax.lax.cond(cond, do_spawn, no_spawn, (grid, rng, positions, agent_alive, agent_levels, par_buf, remaining, spawned_mask)), None

        (grid, rng, positions, agent_alive, agent_levels, parents, _, spawned_mask), _ = jax.lax.scan(
            spawn_fn,
            (grid, state.rng, positions, state.agent_alive, agent_levels, parents, num_requests_satisfiable, spawned_mask),
            jnp.arange(self.max_agents)
        )                
        num_successful_spawns = jnp.sum(spawned_mask.astype(jnp.float32))
        num_alive_before = jnp.sum(alive_before_spawn.astype(jnp.float32))
        cost_per_agent = jnp.where(num_successful_spawns > 0,
                                 (self.spawn_cost * num_successful_spawns) / jnp.where(num_alive_before == 0, 1, num_alive_before),
                                0.0)
        costs = -cost_per_agent * alive_before_spawn.astype(jnp.float32)
        costs -= self.step_cost * alive_before_spawn.astype(jnp.float32)
        
        rewards += costs
        
        step = state.step + 1
        
        terminal = jnp.logical_or(jnp.all(jnp.logical_not(food_alive.astype(bool))), step >= self.max_steps)                
        terminal_flags = jnp.full((self.max_agents,), terminal)

        new_episode_returns = state.episode_returns + jnp.sum(rewards)
        episode_returns = new_episode_returns * (1 - terminal)
        returned_episode_returns = state.returned_episode_returns * (1 - terminal) + new_episode_returns * terminal
        
        
        def reset_state(_):
            rng1, _ = jax.random.split(rng)
            return self.reset(rng1, state.population_explore)[1]

        def continue_state(_):
            return EnvState(
                agent_positions=positions,
                agent_levels=agent_levels,
                agent_alive=agent_alive,
                agent_parents=parents,
                prev_actions=actions,
                food_positions=state.food_positions,
                food_levels=state.food_levels,
                food_alive=food_alive,
                grid=grid,
                episode_returns=episode_returns,
                returned_episode_returns=returned_episode_returns,
                step=step,
                rng=rng,
                population_explore=state.population_explore,
                max_pop_cap=state.max_pop_cap,
            )        
        next_state = jax.lax.cond(terminal, reset_state, continue_state, operand=None)
        
        info = {}        
        info['costs'] = costs
        info['food_alive'] = jnp.sum(food_alive)        
        info['returned_episode'] = terminal
        info['success_flags'] = success_flags

        info['parents'] = parents
        
        info["active_agents"] = jnp.sum(agent_alive)                                
        info["step_episode_returns"] = jnp.sum(rewards)
        info["returned_episode_returns"] = returned_episode_returns 
        
        
        # jax.debug.breakpoint()
        
        return self.get_obs(next_state), next_state, rewards, terminal_flags, info

    def render(self, state: EnvState, figsize=(6, 6), highlight=None):
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import matplotlib.patches as patches
    
        assets = Path(__file__).parent / "assets" / "lbf"
        agent_img = plt.imread(assets / "agent.png")
        food_img = plt.imread(assets / "apple.png")

        h, w = self.grid_size
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw grid
        for x in range(w + 1): ax.axvline(x - 0.5, color='gray', linewidth=1)
        for y in range(h + 1): ax.axhline(y - 0.5, color='gray', linewidth=1)

        # Draw food
        for j in range(self.num_food):
            if not state.food_alive[j]: continue
            r, c = int(state.food_positions[j, 0]), int(state.food_positions[j, 1])
            xy = (c, h - 1 - r)
            ab = AnnotationBbox(OffsetImage(food_img, zoom=(5.0 / max(h, w))), xy, frameon=False)
            ax.add_artist(ab)
            ax.text(c + 0.25, h - 1 - r - 0.25, str(int(state.food_levels[j])), color='black', ha='center', va='center', fontsize='medium')

        # Draw agents
        for i in range(self.max_agents):
            if not state.agent_alive[i]: continue
            r, c = int(state.agent_positions[i, 0]), int(state.agent_positions[i, 1])
            xy = (c, h - 1 - r)
            ab = AnnotationBbox(OffsetImage(agent_img, zoom=(5.0 / max(h, w))), xy, frameon=False)
            ax.add_artist(ab)
            ax.text(c + 0.25, h - 1 - r - 0.25, str(int(state.agent_levels[i])), color='white', ha='center', va='center', fontsize='medium')
            if highlight and i in highlight:
                rect = patches.Rectangle((c - 0.5, h - 1 - r - 0.5), 1, 1, linewidth=2, edgecolor=highlight[i], facecolor='none')
                ax.add_patch(rect)

        # Step indicator
        ax.text(0, -1, f"Step: {state.step}", ha='left', va='center', fontsize='large')
        return fig

"""Fluid Predator-Prey: a JAX MARL environment with a variable agent population.

A team of predators moves on a square grid to capture randomly-moving prey.
What makes the environment *fluid* is that agents may **spawn** new agents at
run time (action ``SPAWN``), so the number of active predators changes within
and across episodes. The population is bounded by ``max_agents`` and, per
episode, by a sampled ``max_pop_cap``.

Interface (shared by all environments in this package)
------------------------------------------------------
``reset(key, population_explore=False) -> (obs, active_mask), state``
``step(key, state, action)            -> (obs, active_mask), state, rewards, terminal, info``

* ``obs``          : float array ``(max_agents, obs_dim)``; rows of inactive agents are zeroed.
* ``active_mask``  : int array ``(max_agents,)``; 1 for active agents, 0 otherwise.
* ``state``        : an :class:`EnvState` pytree (jit/vmap friendly).
* ``rewards``      : float array ``(max_agents,)``.
* ``terminal``     : int array ``(max_agents,)`` (same value broadcast to every slot).
* ``info``         : dict of per-step diagnostics.

The episode auto-resets inside ``step`` on termination, so callers can scan/loop
without handling resets explicitly. Both methods are jitted and vmap-able over a
batch of environments.

Actions (per agent)
-------------------
0=Down, 1=Left, 2=Up, 3=Right, 4=None, 5=Spawn.

Population modes
----------------
``population_explore=True`` (training) samples a random initial agent count and a
random per-episode population cap, exposing the learner to many population sizes.
``population_explore=False`` (evaluation) starts from a fixed ``n_agents`` with the
cap set to ``max_agents``. Setting ``VarA=None`` disables variability entirely.
"""

from functools import partial
from typing import Tuple

import chex
import jax
import numpy as np
from flax import struct
from jax import lax, numpy as jnp, vmap, jit
from PIL import ImageColor

from fluid_envs.rendering import (
    draw_grid, fill_cell, draw_circle, write_cell_text)

# Per-action grid displacement (row, col). Index 5 (Spawn) does not move.
MOVE_ID = jnp.array([[1, 0], [0, -1], [-1, 0], [0, 1], [0, 0], [0, 0]])
MOVE_DICT = np.array(["Down", "Left", "Up", "Right", "None", "Spawn"])


@struct.dataclass
class CriticState:
    """Compact, permutation-invariant summary of the state for a centralised critic."""
    grid: chex.Array          # (stack_obs, H, W)
    agent_active: chex.Array  # scalar float32, = number of active agents
    prey_alive: chex.Array    # scalar float32, = number of prey alive
    parents: chex.Array       # (max_agents,) float32, per-agent spawn counts
    max_pop_cap: chex.Array = struct.field(default_factory=lambda: jnp.int32(0))


@struct.dataclass
class EnvState:
    """Full mutable environment state at a single timestep (a JAX pytree)."""
    grid: jnp.ndarray          # (stack_obs, H, W) frame-stacked grid history
    agent_pos: jnp.ndarray     # (stack_obs, max_agents, 2) position history
    prey_pos: jnp.ndarray      # (n_preys, 2); inactive prey are (-1, -1)
    prey_alive: jnp.ndarray    # (n_preys,)
    agent_active: jnp.ndarray  # (max_agents,) int mask
    prev_action: jnp.ndarray   # (max_agents,)
    parents: jnp.ndarray       # (max_agents,) cumulative spawn counts
    step_count: int = 0
    terminal: int = 0
    episode_returns: float = 0.0
    episode_lengths: int = 0
    returned_episode_returns: float = 0.0
    returned_episode_lengths: int = 0
    # True during training when the population size/cap are randomised.
    population_explore: jnp.bool_ = False
    max_pop_cap: chex.Array = struct.field(default_factory=lambda: jnp.int32(0))


class PredatorPrey:
    """Variable-population Predator-Prey grid world (see module docstring)."""

    def __init__(self, **kwargs):
        default_kwargs = {
            'grid': 5, 'n_agents': 2, 'n_preys': 5,
            'prey_move_probs': jnp.array([0.175, 0.175, 0.175, 0.175, 0.3]),
            'penalty': -0.5, 'step_cost': -0.01,
            'prey_capture_reward': 5.0, 'max_steps': 100, 'agent_view_mask': 5,
            'scaled_reward': False, 'max_agents': 7, 'spawn_cost': 0.0,
            'stack_obs': 4, 'easy_capture': 0, 'terminal_reward': 0,
            'VarA': True, 'divide_spawn_cost': True,
        }
        kwargs = {**default_kwargs, **kwargs}

        self.n_agents = kwargs['n_agents']
        self.max_agents = kwargs['max_agents']

        if not isinstance(kwargs['grid'], int):
            raise ValueError('Grid should be an integer')
        self.grid_shape = [kwargs['grid'], kwargs['grid']]

        if not isinstance(kwargs['agent_view_mask'], int):
            raise ValueError('Agent view should be an integer')
        self.agent_view_mask = [kwargs['agent_view_mask'], kwargs['agent_view_mask']]

        # 'linear' scales prey count with the grid size; otherwise it's fixed.
        if kwargs['n_preys'] == 'linear':
            self.n_preys = 2 * self.grid_shape[0]
        else:
            self.n_preys = kwargs['n_preys']

        self.max_steps = kwargs['max_steps']
        self.penalty = kwargs['penalty']
        self.step_cost = kwargs['step_cost']
        self.prey_move_probs = kwargs['prey_move_probs']
        self.prey_capture_reward = kwargs['prey_capture_reward']

        self.scaled_reward = kwargs['scaled_reward']
        self.terminal_reward = kwargs['terminal_reward']

        # spawn_cost may be a scalar (same for everyone) or a per-agent sequence.
        sc = kwargs['spawn_cost']
        if isinstance(sc, (list, tuple, np.ndarray, jnp.ndarray)):
            self.spawn_cost = jnp.asarray(sc, dtype=jnp.float32)
        else:
            self.spawn_cost = jnp.ones(self.max_agents, dtype=jnp.float32) * float(sc)

        self.stack_obs = kwargs['stack_obs']
        self.easy_capture = kwargs['easy_capture']       # 1 => a single adjacent predator captures
        self.VarA = kwargs['VarA']
        self.divide_spawn_cost = bool(kwargs['divide_spawn_cost'])

        # Each agent observes a local `agent_view_mask` window of the grid (set
        # agent_view_mask == grid for full observability). obs_dim is computed from
        # that window so it always matches what get_agent_obs actually returns.
        view = self.agent_view_mask[0]

        # obs layout per agent: stacked local view + [n_active, n_preys, prev_action(max_agents),
        # agent_id, (x, y), parents, max_pop_cap]  ->  7 scalars + max_agents.
        self.obs_dim = view * view * self.stack_obs + 7 + self.max_agents
        self.act_dim = len(MOVE_ID)

    # ------------------------------------------------------------------ #
    # Grid / position primitives
    # ------------------------------------------------------------------ #
    @partial(jit, static_argnums=0)
    def is_valid(self, pos):
        """1 if ``pos`` is inside the grid, else 0."""
        return (jnp.where(0 <= pos[0], 1, 0) * jnp.where(pos[0] < self.grid_shape[0], 1, 0)
                * jnp.where(0 <= pos[1], 1, 0) * jnp.where(pos[1] < self.grid_shape[1], 1, 0))

    @partial(jit, static_argnums=0)
    def is_cell_vacant(self, pos, grid):
        """1 if ``pos`` is in-bounds and unoccupied."""
        return self.is_valid(pos) * jnp.where(grid[pos[0], pos[1]] == 0, 1, 0)

    @partial(jit, static_argnums=0)
    def random_pos_generator(self, key):
        """Sample a uniformly random (row, col) grid coordinate."""
        key = jax.random.split(key)
        return jnp.hstack((
            jax.random.randint(key[0], (1,), 0, self.grid_shape[0], dtype='int32'),
            jax.random.randint(key[1], (1,), 0, self.grid_shape[1], dtype='int32')))

    @partial(jit, static_argnums=0)
    def init_fill_grid_agent(self, state, agent_i):
        """Place agent ``agent_i`` (encoded as ``agent_i + 1``) on a random vacant cell."""
        grid = state[0]; key = state[1]
        key, subkey = jax.random.split(key)
        init_random_pos = self.random_pos_generator(subkey)

        def cond_fun(state):
            pos = state[0]
            return jnp.where(self.is_cell_vacant(pos, grid) == 1, False, True)

        def body_fun(state):
            key = state[1]
            key, subkey = jax.random.split(key)
            pos = self.random_pos_generator(subkey)
            return (pos, key)

        random_pos, key = lax.while_loop(cond_fun, body_fun, (init_random_pos, key))
        random_pos = random_pos.reshape(-1)
        grid = grid.at[random_pos[0], random_pos[1]].set(agent_i + 1)
        return (grid, key), random_pos

    @partial(jit, static_argnums=0)
    def init_fill_grid_prey(self, state, prey_i):
        """Place prey ``prey_i`` (encoded as ``-prey_i - 1``) on a random vacant cell
        that has no neighbouring predator."""
        grid = state[0]; key = state[1]
        key, subkey = jax.random.split(key)
        init_random_pos = self.random_pos_generator(subkey)

        def cond_fn(state):
            pos = state[0]
            cond1 = jnp.where(self.is_cell_vacant(pos, grid) == 1, 0, 1)
            cond2 = jnp.where(self.neighbour_agents(pos, grid)[0] == 0, 0, 1)
            return jnp.where(cond1 + cond2, True, False)

        def body_fn(state):
            key = state[1]
            key, subkey = jax.random.split(key)
            pos = self.random_pos_generator(subkey)
            return (pos, key)

        random_pos, key = lax.while_loop(cond_fn, body_fn, (init_random_pos, key))
        random_pos = random_pos.reshape(-1)
        grid = grid.at[random_pos[0], random_pos[1]].set(-prey_i - 1)
        return (grid, key), random_pos

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    @partial(jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey, population_explore: bool = False) -> Tuple[chex.Array, EnvState]:
        """Initialise a fresh episode.

        Builds a blank stacked grid, places the initial predators and all prey on
        random cells, and returns ``(obs, active_mask), state``. With
        ``population_explore=True`` the initial agent count and per-episode cap are
        randomised; otherwise they are fixed to ``n_agents`` and ``max_agents``.
        """
        grid = jnp.zeros((self.grid_shape[0], self.grid_shape[1]), dtype=jnp.int32)
        stacked_grid = jnp.tile(jnp.expand_dims(grid, 0), (self.stack_obs - 1, 1, 1))
        stacked_agent_pos = jnp.tile(
            jnp.expand_dims(-jnp.ones((self.max_agents, 2), dtype=jnp.int32), 0),
            (self.stack_obs - 1, 1, 1))

        # --- initial agent count for this episode ---
        key, sub_init = jax.random.split(key)
        init_agents = lax.cond(
            jnp.asarray(population_explore, dtype=jnp.bool_),
            lambda: jax.random.randint(sub_init, (), 2, self.max_agents + 1, dtype=jnp.int32),  # training
            lambda: jnp.int32(self.n_agents),                                                   # evaluation
        )

        # --- per-episode population cap in [init_agents, max_agents] ---
        key, sub_cap = jax.random.split(key)
        max_pop_cap = lax.cond(
            jnp.asarray(population_explore, dtype=jnp.bool_),
            lambda: jax.random.randint(sub_cap, (), init_agents, self.max_agents + 1, dtype=jnp.int32),
            lambda: jnp.int32(self.max_agents),
        )

        if self.VarA is None:  # variability fully disabled
            init_agents = self.n_agents
            max_pop_cap = self.n_agents

        agent_active_mask = (jnp.arange(self.max_agents, dtype=jnp.int32) < init_agents)

        # Place only the active agents; inactive slots get a dummy (-1, -1).
        fill_only_active_agent = lambda state, agent_i: lax.cond(
            agent_active_mask[agent_i],
            self.init_fill_grid_agent,
            lambda x, y: (x, jnp.asarray([-1, -1], dtype=jnp.int32)),
            state, agent_i)
        (state, agent_pos) = lax.scan(
            fill_only_active_agent, (grid, key), jnp.arange(self.max_agents), self.max_agents)
        grid, key = state
        stacked_agent_pos = jnp.vstack((stacked_agent_pos, jnp.expand_dims(agent_pos, 0)))

        # All prey start alive (fixed count) and are placed on the grid.
        prey_alive = jnp.ones(self.n_preys)
        fill_only_active_prey = lambda state, prey_i: lax.cond(
            prey_alive[prey_i],
            self.init_fill_grid_prey,
            lambda x, y: (x, -jnp.ones(2, dtype='int32')),
            state, prey_i)
        state, prey_pos = lax.scan(
            fill_only_active_prey, (grid, key), jnp.arange(self.n_preys), self.n_preys)

        grid, key = state
        agent_active = agent_active_mask.astype(jnp.int32)
        parents = jnp.zeros(self.max_agents)
        stacked_grid = jnp.vstack((stacked_grid, jnp.expand_dims(grid, 0)))
        prev_action = 4 * jnp.ones(self.max_agents, 'int32')  # 4 == "None"

        state = EnvState(
            grid=stacked_grid,
            agent_pos=stacked_agent_pos,
            prey_pos=prey_pos,
            prey_alive=prey_alive,
            agent_active=agent_active,
            prev_action=prev_action,
            parents=parents,
            step_count=0,
            terminal=0,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            population_explore=jnp.asarray(population_explore, dtype=jnp.bool_),
            max_pop_cap=max_pop_cap,
        )
        return self.get_agent_obs(state), state

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    @partial(jit, static_argnums=0)
    def make_obs_partial(self, grid, agent_pos):
        """Extract the local ``agent_view_mask`` patch of ``grid`` centred on ``agent_pos``."""
        center_x, center_y = agent_pos[0], agent_pos[1]
        obs_h, obs_w = self.agent_view_mask

        x = center_x + jnp.arange(obs_h) - obs_h // 2
        y = center_y + jnp.arange(obs_w) - obs_w // 2
        x_idx = x[:, None].repeat(obs_w, axis=1)
        y_idx = y[None, :].repeat(obs_h, axis=0)

        valid_mask = ((x_idx >= 0) & (x_idx < grid.shape[0])
                      & (y_idx >= 0) & (y_idx < grid.shape[1]))
        x_clipped = jnp.clip(x_idx, 0, grid.shape[0] - 1)
        y_clipped = jnp.clip(y_idx, 0, grid.shape[1] - 1)
        patch = grid[x_clipped, y_clipped]
        return jnp.where(valid_mask, patch, 0)

    @partial(jit, static_argnums=0)
    def get_agent_obs(self, env_state):
        """Build the per-agent observation tensor and the active mask.

        Returns ``(obs, active_mask)`` where ``obs`` has shape ``(max_agents, obs_dim)``
        and rows belonging to inactive agents are zeroed out.
        """
        grid = env_state.grid              # (stack_obs, H, W)
        agent_pos = env_state.agent_pos    # (stack_obs, max_agents, 2)

        # Local view per (timestep, agent), then flatten time into the feature axis.
        stacked_partial = vmap(
            vmap(self.make_obs_partial, in_axes=(None, 0)),  # over agents
            in_axes=(0, 0)                                    # over time
        )(grid, agent_pos)
        stacked_partial = stacked_partial.transpose(1, 0, 2, 3).reshape(self.max_agents, -1)

        coordinates = jnp.transpose(jnp.array([
            agent_pos[-1, :, 0] / (self.grid_shape[0] - 1),
            agent_pos[-1, :, 1] / (self.grid_shape[1] - 1)]))
        prev_action = jnp.broadcast_to(env_state.prev_action, (self.max_agents, self.max_agents))
        n_preys = jnp.broadcast_to(jnp.sum(env_state.prey_alive), (self.max_agents, 1))

        obs = jnp.hstack((
            stacked_partial,
            jnp.sum(env_state.agent_active) * jnp.ones((self.max_agents, 1)),
            n_preys,
            prev_action,
            jnp.arange(1, self.max_agents + 1).reshape(-1, 1),   # agent id
            coordinates,
            env_state.parents.reshape(-1, 1),
            env_state.max_pop_cap * jnp.ones((self.max_agents, 1)),
        ))
        obs = obs * env_state.agent_active[:, None]  # zero out inactive agents
        return obs, env_state.agent_active

    @partial(jit, static_argnums=0)
    def get_critic_state(self, env_state) -> CriticState:
        """Permutation-invariant state summary for a centralised critic."""
        return CriticState(
            grid=env_state.grid,
            agent_active=jnp.sum(env_state.agent_active, dtype=jnp.float32),
            prey_alive=jnp.sum(env_state.prey_alive, dtype=jnp.float32),
            parents=env_state.parents.astype(jnp.float32),
            max_pop_cap=env_state.max_pop_cap.astype(jnp.float32),
        )

    # ------------------------------------------------------------------ #
    # Transition
    # ------------------------------------------------------------------ #
    @partial(jit, static_argnums=0)
    def update_agent_pos(self, agent_pos, grid, agent_active, action):
        """Apply movement actions, blocking moves into invalid or occupied cells.

        Inactive agents do not move. A blocked move keeps the agent in place, which
        leaves the grid unchanged for that agent.
        """
        change = vmap(lambda x, y: x[y], in_axes=(None, 0))(MOVE_ID, action)
        change = vmap(jnp.where, in_axes=(0, 0, None))(agent_active, change, jnp.zeros((2)))
        next_pos = jnp.astype(agent_pos + change, 'int32')

        def loop_fn(state, agent_i):
            grid, next_pos = state
            valid = self.is_cell_vacant(next_pos[agent_i], grid)
            next_pos = next_pos.at[agent_i].set(
                jnp.where(jnp.astype(valid, 'bool'), next_pos[agent_i], agent_pos[agent_i]))
            grid = jnp.where(self.is_valid(agent_pos[agent_i]) * agent_active[agent_i],
                             grid.at[agent_pos[agent_i][0], agent_pos[agent_i][1]].set(0), grid)
            grid = jnp.where(self.is_valid(next_pos[agent_i]) * agent_active[agent_i],
                             grid.at[next_pos[agent_i][0], next_pos[agent_i][1]].set(agent_i + 1), grid)
            return (grid, next_pos), agent_i

        (grid, next_pos), _ = lax.scan(loop_fn, (grid, next_pos), jnp.arange(self.max_agents))
        return next_pos, grid

    @partial(jit, static_argnums=0)
    def step(self, key: chex.PRNGKey, env_state: EnvState, action: jnp.ndarray):
        """Advance the environment by one step.

        Order of operations: move predators, resolve prey capture/movement, process
        spawn requests, charge step/spawn costs, compute termination, then auto-reset
        if the episode ended. Returns ``(obs, active_mask), state, rewards, terminal, info``.
        """
        stacked_grid = env_state.grid
        stacked_agent_pos = env_state.agent_pos
        grid = stacked_grid[-1]
        agent_pos = stacked_agent_pos[-1]

        prey_pos = env_state.prey_pos
        prey_alive = env_state.prey_alive
        agent_active = env_state.agent_active
        parents = env_state.parents

        step_count = env_state.step_count + 1
        costs = self.step_cost * jnp.ones(self.max_agents)
        pos_rewards = jnp.zeros(self.max_agents)
        prey_move_probs = self.prey_move_probs

        num_active_agents = jnp.sum(agent_active)
        # Inactive agents are forced to action 4 ("None").
        action = jnp.astype(action * agent_active + jnp.ones(self.max_agents) * 4 * (1 - agent_active), 'int32')

        # Reward per capture; optionally scaled by the active team size.
        prey_capture_reward = lax.cond(
            self.scaled_reward,
            lambda x: x / num_active_agents,
            lambda x: x,
            jnp.astype(self.prey_capture_reward, 'float32'))
        penalty = self.penalty

        # --- 1. Move predators ---
        agent_pos, grid = self.update_agent_pos(agent_pos, grid, agent_active, action)

        # --- 2. Resolve prey: capture (>=1 or >1 adjacent predators) or move away ---
        def per_prey_check(state, prey_i):
            def if_prey_alive(state):
                prey_alive, prey_pos, grid, rewards, key = state
                predator_neighbour_count, _ = self.neighbour_agents(prey_pos[prey_i], grid)

                # easy_capture=1 lets a single adjacent predator capture; else 2+ are needed.
                prey_caught = jnp.where(predator_neighbour_count == 1, self.easy_capture, 0)
                prey_caught = jnp.where(predator_neighbour_count > 1, 1, prey_caught)
                # Lone predator that fails to capture is penalised.
                rewards += jnp.where((1 - prey_caught) * (predator_neighbour_count == 1), penalty, 0)
                state = prey_alive, prey_pos, grid, rewards, key

                def if_prey_caught(state):
                    prey_alive, prey_pos, grid, rewards, key = state
                    rewards += prey_capture_reward
                    prey_alive = prey_alive.at[prey_i].set(1 - prey_caught)
                    grid = grid.at[prey_pos[prey_i][0], prey_pos[prey_i][1]].set(0)
                    prey_pos = prey_pos.at[prey_i].set(-jnp.ones(2, dtype=jnp.int32))
                    return prey_alive, prey_pos, grid, rewards, key

                def if_prey_not_caught(state):
                    # Try up to 5 sampled moves; take the first that lands on a vacant,
                    # predator-free cell, otherwise stay put.
                    prey_alive, prey_pos, grid, rewards, key = state
                    pos = prey_pos[prey_i]
                    key, subkey = jax.random.split(key)
                    move = jax.random.choice(subkey, a=5, shape=(1,), p=prey_move_probs)[0]

                    def cond_fn(state):
                        pos, move, grid, count, key = state
                        cond1 = jnp.where(count >= 5, 1, 0)
                        cond2 = jnp.where(self.neighbour_agents(pos + MOVE_ID[move], grid)[0] == 0, 1, 0)
                        cond3 = self.is_cell_vacant(pos + MOVE_ID[move], grid)
                        return jnp.where(cond3 * cond2 + cond1, False, True)

                    def body_fn(state):
                        pos, _, grid, count, key = state
                        count += 1
                        key, subkey = jax.random.split(key)
                        move = jax.random.choice(subkey, a=5, shape=(1,), p=prey_move_probs)[0]
                        return (pos, move, grid, count, key)

                    (pos, move, grid, count, key) = lax.while_loop(
                        cond_fn, body_fn, (pos, move, grid, 0, key))
                    next_pos = jnp.where(count < 5, pos + MOVE_ID[move], pos)
                    grid = grid.at[prey_pos[prey_i][0], prey_pos[prey_i][1]].set(0)
                    grid = grid.at[next_pos[0], next_pos[1]].set(-prey_i - 1)
                    prey_pos = prey_pos.at[prey_i].set(next_pos)
                    return prey_alive, prey_pos, grid, rewards, key

                return lax.cond(prey_caught == 1, if_prey_caught, if_prey_not_caught, state)

            prey_alive = state[0]
            state = lax.cond(prey_alive[prey_i] == 1, if_prey_alive, lambda x: x, state)
            return state, []

        state, _ = lax.scan(
            per_prey_check, (prey_alive, prey_pos, grid, pos_rewards, key), jnp.arange(self.n_preys))
        prey_alive, prey_pos, grid, pos_rewards, key = state

        # --- 3. Process spawn requests (action 5), bounded by the per-episode cap ---
        cap = env_state.max_pop_cap

        def spawn_fn(num_active_agents, key, grid, agent_pos, agent_active, parents):
            num_spawn_req = jnp.sum((action == 5).astype(jnp.int32))
            slots = jnp.maximum(0, cap - num_active_agents)
            num_spawn = jnp.minimum(num_spawn_req, slots)

            # Credit the first `num_spawn` active requesters with a successful spawn.
            def update_parents(state, xs):
                parents, assigned = state
                cond1 = jnp.astype(action[xs] == 5, 'int32')
                cond2 = jnp.astype(assigned < num_spawn, 'int32')
                cond3 = agent_active[xs]
                assigned += jnp.where(cond1 * cond2 * cond3, 1, 0)
                parents = parents.at[xs].set(parents[xs] + cond1 * cond2 * cond3)
                return (parents, assigned), []

            (parents, _), _ = lax.scan(update_parents, (parents, 0), jnp.arange(self.max_agents))

            # Activate the next `num_spawn` previously-inactive slots.
            def helper_fn(c, xs):
                cond1 = jnp.astype(xs < num_active_agents, 'int32')
                cond2 = jnp.astype(xs >= num_active_agents + num_spawn, 'int32')
                return c, lax.cond(cond1 + cond2, lambda a, b: a[xs], lambda a, b: b[xs], c[0], c[1])

            _, new_agent_active = lax.scan(
                helper_fn, (agent_active, jnp.ones(self.max_agents, dtype='int32')),
                jnp.arange(self.max_agents, dtype='int32'))
            update_agents = (new_agent_active - agent_active) * jnp.arange(self.max_agents)

            def update_grid(state, xs):
                key, grid, agent_pos = state
                key, subkey = jax.random.split(key)
                (grid, key), new_pos = lax.cond(
                    update_agents[xs] > 0,
                    lambda x, y, z: self.init_fill_grid_agent(x, y),
                    lambda x, y, z: (x, z),
                    (grid, subkey), update_agents[xs], agent_pos[xs])
                agent_pos = agent_pos.at[xs].set(new_pos)
                return (key, grid, agent_pos), []

            (key, grid, agent_pos), _ = lax.scan(
                update_grid, (key, grid, agent_pos), jnp.arange(self.max_agents))
            return (grid, agent_pos, new_agent_active, parents)

        key, subkey = jax.random.split(key)
        grid, agent_pos, new_agent_active, parents = spawn_fn(
            num_active_agents, subkey, grid, agent_pos, agent_active, parents)

        # --- 4. Charge spawn cost (team-shared or spawner-pays) ---
        c_success = jnp.sum(new_agent_active - agent_active).astype(jnp.float32)
        agent_count = jnp.sum(agent_active).astype(jnp.float32)
        safe_agent_count = jnp.where(agent_count > 0.0, agent_count, 1.0)
        spawner_success_mask = ((parents - env_state.parents) > 0).astype(jnp.float32)

        divide_spawn_cost = (self.scaled_reward or self.divide_spawn_cost)
        spawn_cost_team = agent_active * (c_success * self.spawn_cost) / safe_agent_count
        spawn_cost_spawner = spawner_success_mask * self.spawn_cost
        costs += lax.cond(
            jnp.asarray(divide_spawn_cost, dtype=jnp.bool_),
            lambda _: spawn_cost_team,
            lambda _: spawn_cost_spawner,
            operand=None)

        # --- 5. Termination + reward assembly ---
        cond1 = jnp.astype(step_count >= self.max_steps, 'int32')
        cond2 = jnp.where(jnp.sum(prey_alive) == 0, 1, 0)  # all prey captured
        terminal = jnp.where(jnp.astype(cond1 + cond2, 'bool'), 1, 0)
        terminal_reward = jnp.where(
            jnp.astype(cond2, 'bool'), self.terminal_reward / jnp.sum(agent_active), 0)

        terminal_reward = terminal_reward * agent_active
        pos_rewards = pos_rewards * agent_active
        costs = costs * agent_active
        rewards = pos_rewards + costs + terminal_reward

        # --- 6. Assemble next state + diagnostics ---
        agent_active = new_agent_active
        new_episode_return = env_state.episode_returns + jnp.sum(rewards)
        new_episode_length = env_state.episode_lengths + 1

        stacked_grid = jnp.vstack((stacked_grid[1:], jnp.expand_dims(grid, 0)))
        stacked_agent_pos = jnp.vstack((stacked_agent_pos[1:], jnp.expand_dims(agent_pos, 0)))

        next_state = EnvState(
            stacked_grid, stacked_agent_pos, prey_pos, prey_alive, agent_active, action, parents,
            step_count, terminal,
            episode_returns=new_episode_return * (1 - terminal),
            episode_lengths=new_episode_length * (1 - terminal),
            returned_episode_returns=env_state.returned_episode_returns * (1 - terminal) + new_episode_return * terminal,
            returned_episode_lengths=env_state.returned_episode_lengths * (1 - terminal) + new_episode_length * terminal,
            population_explore=env_state.population_explore,
            max_pop_cap=env_state.max_pop_cap)

        caught_prey = jnp.sum(env_state.prey_alive - next_state.prey_alive)
        info = {
            'prey_alive': prey_alive,
            'caught_prey': caught_prey,
            'parents': parents,
            'returned_episode': terminal,
            'active_agents': jnp.sum(agent_active),
            'costs': jnp.sum(costs),
            'pos_rewards': jnp.sum(pos_rewards),
            'step_episode_returns': jnp.sum(rewards),
            'returned_episode_returns': next_state.returned_episode_returns,
            'max_pop_cap': env_state.max_pop_cap,
        }

        # --- 7. Auto-reset on termination ---
        key, subkey = jax.random.split(key)
        next_state = lax.cond(
            terminal,
            lambda x, y: self.reset(x, y.population_explore)[1],
            lambda x, y: y, subkey, next_state)
        terminal = terminal * jnp.ones(self.max_agents)

        return (lax.stop_gradient(self.get_agent_obs(next_state)),
                lax.stop_gradient(next_state), rewards, terminal, info)

    # ------------------------------------------------------------------ #
    # Neighbour queries
    # ------------------------------------------------------------------ #
    @partial(jit, static_argnums=0)
    def neighbour_agents(self, pos, grid):
        """Count predators in the 4-neighbourhood of ``pos`` and return their id mask."""
        change = MOVE_ID[:4]
        check_pos = pos + change
        agent_id = jnp.zeros(self.max_agents + 1)  # slot 0 holds junk (id 0 => empty)

        def loop_fn(state, pos):
            count, agent_id = state
            count += (self.is_valid(pos) * jnp.where(grid[pos[0], pos[1]] > 0, 1, 0)
                      * jnp.where(grid[pos[0], pos[1]] <= self.max_agents, 1, 0))
            id = jnp.where(grid[pos[0], pos[1]] > 0, jnp.astype(grid[pos[0], pos[1]], 'int32'), 0)
            agent_id = agent_id.at[id].set(1)
            return (count, agent_id), count

        (count, agent_id), _ = lax.scan(loop_fn, (0, agent_id), check_pos)
        return count, agent_id[1:]  # drop the junk slot

    def get_neighbour_coordinates(self, pos):
        """Python-level helper (used by the renderer) returning valid 4-neighbour cells."""
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self, grid, agent_pos, prey_pos, prey_alive, agent_active, ep=None):
        """Render a single frame to an RGB numpy array (predators blue, prey red)."""
        AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
        AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
        PREY_COLOR = 'red'
        CELL_SIZE = 35

        img = draw_grid(grid.shape[0], grid.shape[1], cell_size=CELL_SIZE, fill='white')

        for agent_i in range(len(agent_pos)):
            if agent_active[agent_i] == 0:
                continue
            for neighbour in self.get_neighbour_coordinates(agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in range(len(agent_pos)):
            if agent_active[agent_i] == 0:
                continue
            draw_circle(img, agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=agent_pos[agent_i],
                            cell_size=CELL_SIZE, fill='white', margin=0.4)

        for prey_i in range(len(prey_pos)):
            if prey_alive[prey_i]:
                draw_circle(img, prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=prey_pos[prey_i],
                                cell_size=CELL_SIZE, fill='white', margin=0.4)

        return np.asarray(img)

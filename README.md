# fluid-envs

Spawn-fluid JAX multi-agent grid environments with **variable agent populations**. Agents can create new agents at run time (a `SPAWN` action), so the number of active agents changes within and across episodes. Three environments share a single interface:

- **PredatorPrey** — predators move to capture randomly-moving prey.
- **Foraging** (LBF) — agents cooperatively *load* food when their summed level meets the food's level.
- **PuddleBridge** — agents reach a goal across a wall, either via an open alternate path or by cooperatively *stacking* on a puddle. Blocking the alternate path (per-episode, stochastic) forces the cooperative/fluid solution.

All three are written in JAX (jittable, `vmap`-able over batches of environments) and auto-reset on termination.

## Install

```bash
pip install -e .
```

## Quickstart

```python
import jax
from fluid_envs import PredatorPrey

env = PredatorPrey(grid=8, n_agents=2, n_preys=4, max_agents=6, agent_view_mask=8)
key = jax.random.PRNGKey(0)

(obs, active_mask), state = env.reset(key, population_explore=False)
actions = jax.random.randint(key, (env.max_agents,), 0, env.act_dim)
(obs, active_mask), state, rewards, terminal, info = env.step(key, state, actions)
```

Batch 1024 environments by `vmap`-ping `reset`/`step`:

```python
rngs = jax.random.split(key, 1024)
(obs, active), state = jax.vmap(env.reset, in_axes=(0, None))(rngs, True)
(obs, active), state, rewards, terminal, info = jax.vmap(
    env.step, in_axes=(0, 0, None))(rngs, state, actions)
```

Render a random rollout to a GIF:

```bash
python examples/random_rollout.py --env predator_prey --steps 60 --out pp.gif
python examples/random_rollout.py --env lbf
python examples/random_rollout.py --env puddle_bridge --explore
```

## The shared interface

```
reset(key, population_explore=False) -> (obs, active_mask), state
step(key, state, actions)            -> (obs, active_mask), state, rewards, terminal, info
```

- `obs` — float array `(max_agents, obs_dim)`; rows belonging to inactive agents are zeroed.
- `active_mask` — int array `(max_agents,)`; 1 for active agents, 0 otherwise.
- `state` — an environment-specific `EnvState` pytree.
- `rewards` — float array `(max_agents,)`.
- `terminal` — array `(max_agents,)`, the same terminal flag broadcast to every slot.
- `info` — dict of per-step diagnostics (each env documents its own keys; common ones: `returned_episode`, `returned_episode_returns`, `active_agents`, `step_episode_returns`, `parents`/`successful_spawns`).

`env.obs_dim` and `env.act_dim` give the per-agent observation and action dimensions.

### Population modes

`population_explore=True` (training) samples a random initial agent count and a random per-episode population cap, exposing the learner to many population sizes. `population_explore=False` (evaluation) uses the fixed configured count. Setting `VarA=None` disables population variability entirely. The cap bounds how many agents `SPAWN` can add in an episode.

## Environments

### PredatorPrey

Actions: `0=Down, 1=Left, 2=Up, 3=Right, 4=None, 5=Spawn`. A prey is captured when enough predators are adjacent (`easy_capture=1` lets a single adjacent predator capture; otherwise two or more are required). Each agent observes a stacked local window of size `agent_view_mask`; set `agent_view_mask == grid` for full observability. Key constructor args: `grid`, `n_agents`, `n_preys`, `max_agents`, `agent_view_mask`, `stack_obs`, `scaled_reward`, `easy_capture`, `spawn_cost`, `prey_capture_reward`, `penalty`, `step_cost`, `max_steps`, `divide_spawn_cost`, `VarA`.

### Foraging (LBF)

Actions: `0=None, 1=North, 2=South, 3=West, 4=East, 5=Load, 6=Spawn`. Food at a cell is collected when the summed level of adjacent agents issuing `Load` meets or exceeds the food level; reward is shared in proportion to contributed level. Key constructor args: `grid_size`, `init_num_agents`, `max_agents`, `num_food`, `max_level`, `max_steps`, `agent_view`, `spawn_cost`, `step_cost`, `init_agent_levels`, `init_food_levels`, `VarA`.

### PuddleBridge

Actions: `0=None, 1=North, 2=South, 3=West, 4=East, 5=Spawn`. Configured via `EnvConfig`. A puddle cell holds at most two agents; only the recorded top agent may leave a full stack. The fluid/non-fluid regime is controlled by the alternate path: with `toggle_other_path_at_reset=True`, each episode blocks the `other_path_coords` with probability `p_block`, recorded in `state.alt_path_blocked`. Key `EnvConfig` fields: `grid_rows`, `grid_cols`, `max_agents`, `max_steps_per_episode`, `team_reward_on_goal`, `spawn_cost`, `step_cost`, `puddle_coords`, `toggle_other_path_at_reset`, `other_path_coords`, `p_block`, `VarA`.

```python
from fluid_envs import PuddleBridge, EnvConfig
env = PuddleBridge(EnvConfig(
    grid_rows=8, grid_cols=8, max_agents=4, max_steps_per_episode=60,
    team_reward_on_goal=True, spawn_cost=0.5, step_cost=0.01,
    toggle_other_path_at_reset=True, p_block=0.5, other_path_coords=((2, 6), (3, 6)),
))
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Citation

This repository provides the environment suite used in our AAMAS 2026 paper. If you use these environments, please cite the paper:

```bibtex
@inproceedings{sharma2026fluid,
  title     = {Fluid-Agent Reinforcement Learning},
  author    = {Sharma, Shishir and Precup, Doina and Perkins, Theodore},
  booktitle = {Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year      = {2026},
  doi       = {10.65109/TAXB8518},
  note      = {arXiv:2602.14559}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff), including a `preferred-citation` field pointing to the paper — this is what populates GitHub's "Cite this repository" button.


## License

MIT — see [LICENSE](LICENSE).

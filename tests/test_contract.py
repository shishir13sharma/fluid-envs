"""Contract smoke tests: every environment must satisfy the shared reset/step API.

Run with:  pytest -q
"""

import jax
import jax.numpy as jnp
import pytest

from fluid_envs import PredatorPrey, Foraging, PuddleBridge, EnvConfig


def make_predator_prey():
    return PredatorPrey(grid=6, n_agents=2, n_preys=3, max_agents=5, stack_obs=3,
                        agent_view_mask=6, scaled_reward=True, easy_capture=1,
                        spawn_cost=-0.5, max_steps=20)


def make_lbf():
    return Foraging(grid_size=6, init_num_agents=2, max_agents=5, num_food=2,
                    max_level=3, max_steps=20, agent_view=(3, 3), spawn_cost=0.5)


def make_puddle_bridge():
    cfg = EnvConfig(grid_rows=8, grid_cols=8, max_agents=4, max_steps_per_episode=20,
                    team_reward_on_goal=True, spawn_cost=0.5, step_cost=0.01,
                    toggle_other_path_at_reset=True, p_block=0.5,
                    other_path_coords=((2, 6), (3, 6)))
    return PuddleBridge(cfg)


ENVS = [make_predator_prey, make_lbf, make_puddle_bridge]


@pytest.mark.parametrize("make_env", ENVS)
def test_reset_shapes(make_env):
    env = make_env()
    (obs, active), state = env.reset(jax.random.PRNGKey(0), True)
    assert obs.shape == (env.max_agents, env.obs_dim)
    assert active.shape == (env.max_agents,)


@pytest.mark.parametrize("make_env", ENVS)
def test_step_shapes(make_env):
    env = make_env()
    key = jax.random.PRNGKey(0)
    (_obs, _active), state = env.reset(key, True)
    actions = jax.random.randint(key, (env.max_agents,), 0, env.act_dim)
    (obs, active), state, rewards, terminal, info = env.step(key, state, actions)
    assert obs.shape == (env.max_agents, env.obs_dim)
    assert rewards.shape == (env.max_agents,)
    assert terminal.shape == (env.max_agents,)
    assert isinstance(info, dict)


@pytest.mark.parametrize("make_env", ENVS)
def test_vmap_batches(make_env):
    env = make_env()
    key = jax.random.PRNGKey(0)
    B = 4
    rngs = jax.random.split(key, B)
    (obs, active), state = jax.vmap(env.reset, in_axes=(0, None))(rngs, True)
    assert obs.shape == (B, env.max_agents, env.obs_dim)
    actions = jax.random.randint(key, (env.max_agents,), 0, env.act_dim)
    (obs, active), state, rewards, terminal, info = jax.vmap(
        env.step, in_axes=(0, 0, None))(rngs, state, actions)
    assert obs.shape == (B, env.max_agents, env.obs_dim)
    assert rewards.shape == (B, env.max_agents)


@pytest.mark.parametrize("make_env", ENVS)
def test_scan_rollout_is_jittable(make_env):
    env = make_env()
    key = jax.random.PRNGKey(0)
    (_obs, _active), state = env.reset(key, False)

    def body(carry, _):
        state, key = carry
        key, akey, skey = jax.random.split(key, 3)
        actions = jax.random.randint(akey, (env.max_agents,), 0, env.act_dim)
        (_o, _a), state, rewards, _t, _i = env.step(skey, state, actions)
        return (state, key), jnp.sum(rewards)

    (_state, _key), rs = jax.lax.scan(body, (state, key), None, length=10)
    assert rs.shape == (10,)

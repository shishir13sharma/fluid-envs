"""Run a random-policy rollout for any of the three environments and save a GIF.

Usage:
    python examples/random_rollout.py --env predator_prey --steps 60 --out rollout.gif
    python examples/random_rollout.py --env lbf
    python examples/random_rollout.py --env puddle_bridge --explore
"""

import argparse

import jax

from fluid_envs import PredatorPrey, Foraging, PuddleBridge, EnvConfig
from fluid_envs.rollout import random_rollout, save_gif


def build_predator_prey():
    # Mirrors PredatorPrey.yaml. n_preys='linear' -> 2*grid; max_agents resolves to 10
    # (VarA is not None). agent_view=11 maps to agent_view_mask (partial view on a 21 grid).
    # stack_obs is not in the env YAML (it came from the agent config); 4 is the usual value.
    return PredatorPrey(
        grid=21, n_agents=2, n_preys="linear", max_agents=10, agent_view_mask=11,
        stack_obs=4, penalty=0.0, step_cost=-0.01, scaled_reward=True,
        terminal_reward=0, spawn_cost=-10.0, prey_capture_reward=5.0,
        easy_capture=0, max_steps=100, divide_spawn_cost=True, VarA=True,
    )


def build_lbf():
    # Mirrors LBF.yaml. max_agents resolves to 4 (VarA is not None).
    return Foraging(
        grid_size=[7, 7], init_num_agents=2, max_agents=4, num_food=4,
        max_level=3, max_steps=100, agent_view=(14, 14),
        init_agent_levels=[2, 1], init_food_levels=[2, 3, 4, 5],
        spawn_cost=1.0, step_cost=0.025, VarA=True,
    )


def build_puddle_bridge():
    # Mirrors PuddleBridge.yaml.
    cfg = EnvConfig(
        grid_rows=8, grid_cols=8, max_agents=4, max_steps_per_episode=100,
        team_reward_on_goal=True, spawn_cost=1.0, step_cost=0.1, VarA=True,
        puddle_coords=((0, 2), (0, 3), (1, 3), (1, 4), (2, 4), (2, 5)),
        toggle_other_path_at_reset=True,
        other_path_coords=((6, 2), (7, 2), (6, 3), (7, 3), (6, 4), (7, 4), (6, 5), (7, 5)),
        p_block=0.5,
    )
    return PuddleBridge(cfg)


BUILDERS = {
    "predator_prey": build_predator_prey,
    "lbf": build_lbf,
    "puddle_bridge": build_puddle_bridge,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=list(BUILDERS), default="predator_prey")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explore", action="store_true",
                        help="Randomise the initial population (population_explore=True).")
    parser.add_argument("--out", type=str, default=None, help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    env = BUILDERS[args.env]()
    key = jax.random.PRNGKey(args.seed)
    frames, total_reward = random_rollout(
        env, key, n_steps=args.steps, population_explore=args.explore)

    out = args.out or f"{args.env}_rollout.gif"
    save_gif(frames, out, fps=args.fps)
    print(f"{args.env}: {len(frames)} frames, total reward {total_reward:.2f} -> {out}")


if __name__ == "__main__":
    main()

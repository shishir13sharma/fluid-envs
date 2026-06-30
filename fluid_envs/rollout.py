"""Lightweight rollout and rendering helpers for the fluid environments.

Self-contained replacements for the project-internal utilities: pytree stacking,
a per-environment frame renderer that returns RGB numpy arrays, GIF export, and a
single-environment random rollout used by ``examples/random_rollout.py``.
"""

from typing import List

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from PIL import Image


def tree_stack(trees: List) -> object:
    """Stack a list of identically-structured pytrees along a new leading axis."""
    return jtu.tree_map(lambda *leaves: jnp.stack(leaves), *trees)


def tree_unstack(tree) -> List:
    """Inverse of :func:`tree_stack`: split a batched pytree into a list along axis 0."""
    leaves, treedef = jtu.tree_flatten(tree)
    n = leaves[0].shape[0]
    return [jtu.tree_unflatten(treedef, [leaf[i] for leaf in leaves]) for i in range(n)]


def swap_first_two_axes(x):
    """Swap axes 0 and 1 of an array (e.g. (time, env) -> (env, time))."""
    return jnp.swapaxes(x, 0, 1)


def _fig_to_array(fig) -> np.ndarray:
    """Rasterise a matplotlib figure to an (H, W, 3) uint8 array, then close it."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    arr = buf.reshape(h, w, 4)[..., :3].copy()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return arr


def render_frame(env, state) -> np.ndarray:
    """Render a single environment state to an RGB numpy array.

    Dispatches on the environment type, hiding the fact that each env exposes a
    slightly different rendering entry point.
    """
    name = type(env).__name__
    if name == "PredatorPrey":
        return env.render(
            np.asarray(state.grid[-1]),
            np.asarray(state.agent_pos[-1]),
            np.asarray(state.prey_pos),
            np.asarray(state.prey_alive),
            np.asarray(state.agent_active),
        )
    if name == "Foraging":  # LBF
        return _fig_to_array(env.render(state))
    if name == "PuddleBridge":
        return env.render_from_state(state, show=False, return_array=True)
    raise ValueError(f"Unknown environment type: {name}")


def save_gif(frames: List[np.ndarray], path: str, fps: int = 5) -> None:
    """Save a list of RGB frames as an animated GIF."""
    images = [Image.fromarray(np.asarray(f).astype(np.uint8)) for f in frames]
    duration = int(1000 / max(1, fps))
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0)


def random_rollout(env, key, n_steps: int = 50, population_explore: bool = False):
    """Run one environment under a uniformly random policy, collecting frames.

    Returns ``(frames, total_reward)`` where ``frames`` is a list of RGB arrays (one
    per step) and ``total_reward`` is the summed team reward over the rollout. The
    environment auto-resets internally, so the rollout runs for exactly ``n_steps``.
    """
    key, reset_key = jax.random.split(key)
    (_obs, _active), state = env.reset(reset_key, population_explore)

    frames = [render_frame(env, state)]
    total_reward = 0.0
    for _ in range(n_steps):
        key, akey, skey = jax.random.split(key, 3)
        actions = jax.random.randint(akey, (env.max_agents,), 0, env.act_dim)
        (_obs, _active), state, rewards, _term, _info = env.step(skey, state, actions)
        total_reward += float(jnp.sum(rewards))
        frames.append(render_frame(env, state))
    return frames, total_reward

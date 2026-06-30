"""Fluid environments: spawn-fluid JAX MARL grid worlds with variable agent populations.

Three environments sharing one ``reset`` / ``step`` interface:

* :class:`PredatorPrey` — predators capture prey; spawn-fluid.
* :class:`Foraging`     — level-based cooperative foraging (LBF); spawn-fluid.
* :class:`PuddleBridge` — cooperative puddle-stacking with fluid/non-fluid regimes.
"""

from fluid_envs.predator_prey import PredatorPrey, EnvState as PredatorPreyState, CriticState
from fluid_envs.lbf import Foraging, EnvState as ForagingState
from fluid_envs.puddle_bridge import PuddleBridge, EnvConfig, EnvState as PuddleBridgeState

__all__ = [
    "PredatorPrey", "PredatorPreyState", "CriticState",
    "Foraging", "ForagingState",
    "PuddleBridge", "EnvConfig", "PuddleBridgeState",
]

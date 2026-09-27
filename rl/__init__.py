"""Reinforcement-learning layer around the existing Dino Run game (PPO + Gymnasium).

Nothing in here re-implements the game: :class:`~rl.environment.DinoRunEnv` wraps
:class:`~game.core.DinoGame`, which is the original ``main.py`` loop body.

Layout::

    rl/environment.py    DinoRunEnv (reset/step/observation/action/reward)
    rl/observations.py   the structured, normalized 47-float observation
    rl/rewards.py        the reward function and its documentation
    rl/wrappers.py       episode-statistics aggregation
    rl/vector_env.py     many independent environments (dummy / subprocess)
    rl/baselines.py      random / do-nothing / geometric reference policies
"""

from .baselines import (
    DoNothingPolicy,
    GeometricHeuristicPolicy,
    ModelPolicy,
    PolicySource,
    RandomPolicy,
    build_policy,
)
from .environment import DinoRunEnv
from .observations import OBS_SIZE, OBS_VERSION, ObservationBuilder, observation_space
from .rewards import RewardConfig, RewardFunction
from .vector_env import make_env, make_vec_env, recommended_env_count, seed_vec_env
from .wrappers import EpisodeStatsWrapper, NumpyObsWrapper

__all__ = [
    "DinoRunEnv",
    "OBS_SIZE",
    "OBS_VERSION",
    "ObservationBuilder",
    "observation_space",
    "RewardConfig",
    "RewardFunction",
    "EpisodeStatsWrapper",
    "NumpyObsWrapper",
    "make_env",
    "make_vec_env",
    "recommended_env_count",
    "seed_vec_env",
    "PolicySource",
    "RandomPolicy",
    "DoNothingPolicy",
    "GeometricHeuristicPolicy",
    "ModelPolicy",
    "build_policy",
]

"""Worker-side half of :class:`rl.process_vec_env.ProcessGroupVecEnv`.

This module is deliberately dependency-light: only ``numpy``, ``gymnasium`` and the
game itself are imported here.  That matters because the worker function is pickled
*by reference*, so the child process imports this module -- and nothing else -- at
startup.  The parent-side module (``rl.process_vec_env``) imports
``stable_baselines3``, which imports torch (~170 MB of RSS); keeping the two apart is
the difference between ~50 MB and ~217 MB per worker, i.e. between 64 and 15 workers
fitting inside Colab's RAM.

Worker protocol (one duplex pipe per worker, one message per group per step):

    parent -> worker  ("step", actions for this worker's envs)
    worker -> parent  ("step", observations, rewards, terminated, truncated, infos)
    parent -> worker  ("reset", seeds or None) / ("call", ...) / ("get_attr", ...)
                      / ("set_attr", ...) / ("env_is_wrapped", ...) / ("close",)
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

#: only these keys survive the process boundary when an episode ends (these are the
#: keys stable-baselines3 and the training callbacks actually read)
TERMINAL_INFO_KEYS = ("episode", "dino_episode", "score", "seed")


def space_descriptor(env) -> Dict[str, Any]:
    """Pickle-friendly description of an environment's spaces."""
    obs_space = env.observation_space
    act_space = env.action_space
    descriptor: Dict[str, Any] = {
        "obs_shape": tuple(obs_space.shape),
        "obs_dtype": str(obs_space.dtype),
        "obs_low": np.asarray(obs_space.low, dtype=np.float32),
        "obs_high": np.asarray(obs_space.high, dtype=np.float32),
        "action_type": type(act_space).__name__,
    }
    if hasattr(act_space, "n"):
        descriptor["action_n"] = int(act_space.n)
    else:  # pragma: no cover - the game only ever uses Discrete(3)
        descriptor["action_low"] = np.asarray(act_space.low)
        descriptor["action_high"] = np.asarray(act_space.high)
    return descriptor


def as_float32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def env_is_wrapped(env, wrapper_class) -> bool:
    """``gym.Wrapper``-chain lookup (equivalent to SB3's helper, without the import)."""
    import gymnasium as gym

    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, wrapper_class):
            return True
        current = current.env
    return isinstance(current, wrapper_class)


def _resolve_target(env, name: str):
    """Support ``"unwrapped.attr"`` style lookups."""
    if name.startswith("unwrapped."):
        return env.unwrapped, name.split(".", 1)[1]
    return env, name


def worker_main(conn, env_payload: bytes, worker_index: int) -> None:
    """Child process entry point: owns several environments and serves requests."""
    import cloudpickle

    try:
        env_fns: Sequence[Callable] = cloudpickle.loads(env_payload)
        envs = [fn() for fn in env_fns]
        obs_dim = envs[0].observation_space.shape
        initial = [np.asarray(env.reset(seed=None)[0], dtype=np.float32) for env in envs]
        conn.send(("ready", space_descriptor(envs[0]), initial))
        del initial

        while True:
            message = conn.recv()
            command = message[0]

            if command == "step":
                actions = message[1]
                count = len(envs)
                obs_batch = np.empty((count, *obs_dim), dtype=np.float32)
                rewards = np.empty(count, dtype=np.float32)
                terminated = np.zeros(count, dtype=bool)
                truncated = np.zeros(count, dtype=bool)
                infos: List[dict] = []
                for index, env in enumerate(envs):
                    obs, reward, term, trunc, info = env.step(int(actions[index]))
                    rewards[index] = reward
                    terminated[index] = term
                    truncated[index] = trunc
                    if term or trunc:
                        trimmed = {key: info[key] for key in TERMINAL_INFO_KEYS if key in info}
                        trimmed["terminal_observation"] = as_float32(obs)
                        trimmed["TimeLimit.truncated"] = bool(trunc and not term)
                        info = trimmed
                        obs, _ = env.reset()
                    else:
                        # dropping non-terminal info keeps the IPC payload tiny
                        info = {}
                    obs_batch[index] = obs
                    infos.append(info)
                conn.send(("step", obs_batch, rewards, terminated, truncated, infos))

            elif command == "reset":
                seeds = message[1]
                count = len(envs)
                obs_batch = np.empty((count, *obs_dim), dtype=np.float32)
                reset_infos: List[dict] = []
                for index, env in enumerate(envs):
                    seed = None if seeds is None else seeds[index]
                    obs, info = env.reset(seed=seed)
                    obs_batch[index] = obs
                    reset_infos.append(info if isinstance(info, dict) else {})
                conn.send(("reset", obs_batch, reset_infos))

            elif command == "call":
                _, method_name, args, kwargs = message
                values = []
                for env in envs:
                    target, name = _resolve_target(env, method_name)
                    values.append(getattr(target, name)(*args, **kwargs))
                conn.send(("call", values))

            elif command == "get_attr":
                _, attr_name = message
                values = []
                for env in envs:
                    target, name = _resolve_target(env, attr_name)
                    values.append(getattr(target, name))
                conn.send(("call", values))

            elif command == "set_attr":
                _, attr_name, value = message
                for env in envs:
                    setattr(env, attr_name, value)
                conn.send(("call", [None] * len(envs)))

            elif command == "env_is_wrapped":
                _, wrapper_class = message
                conn.send(("call", [env_is_wrapped(env, wrapper_class) for env in envs]))

            elif command == "close":
                for env in envs:
                    try:
                        env.close()
                    except Exception:  # pragma: no cover - best effort
                        pass
                conn.send(("closed",))
                break

            else:  # pragma: no cover - protocol error
                conn.send(("error", f"unknown command {command!r}"))

    except EOFError:  # parent went away
        pass
    except Exception:  # pragma: no cover - never hang the parent silently
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

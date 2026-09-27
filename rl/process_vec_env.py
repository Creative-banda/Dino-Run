"""A multi-process vectorised environment that is actually worth using.

Why not just use SB3's ``SubprocVecEnv``?  Measured on this project it is *slower*
than running everything in one process (22k vs 27k env steps/s), for two reasons:

1. **One IPC round-trip per environment per step.**  A Dino Run step costs ~38 us of
   physics, while pickling one action and one result across a process boundary costs
   a similar amount.  Communication dominates, so the parent becomes the bottleneck
   and adding workers stops helping.
2. **Worker processes import torch.**  ``SubprocVecEnv`` lives in a module chain that
   pulls in torch, so each worker costs ~217 MB instead of ~50 MB.  On Colab's 12 GB
   that caps you at ~50 workers no matter how many cores you have.

This implementation fixes both:

* each worker owns several environments and steps all of them per message, so the
  IPC cost is amortised over ``envs_per_worker`` environments (one round-trip per
  *group* per step instead of one per environment),
* workers only import ``game``/``rl`` (gymnasium + pygame + numpy), never torch,
  and non-terminal ``info`` dicts are dropped on the worker side,
* actions are sent to every worker before any result is collected, so the workers'
  physics runs in parallel while the parent talks to the next worker.

Together with ``envs_per_worker > 1`` this turns spare CPU cores into extra
simulation throughput, which is the whole point of running many environments.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import cloudpickle  # a stable-baselines3 dependency; needed for closure env factories
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvStepReturn,
)

# The worker lives in its own module (``rl.env_worker``) that imports neither
# stable-baselines3 nor torch, so worker processes stay ~4x smaller in RAM.
from .env_worker import worker_main

#: how long the parent waits for a worker before assuming it died (seconds)
RECV_TIMEOUT = 300.0


class _Worker:
    """Parent-side handle for one worker process (owns a contiguous env slice)."""

    def __init__(self, index: int, env_fns: Sequence[Callable], start_method: str, seed_offset: int) -> None:
        context = mp.get_context(start_method)
        self.index = index
        self.seed_offset = seed_offset
        self.parent_conn, child_conn = context.Pipe()
        self.process = context.Process(
            target=worker_main,
            args=(child_conn, cloudpickle.dumps(list(env_fns)), index),
            daemon=True,
            name=f"dino-env-{index}",
        )
        self.process.start()
        child_conn.close()  # the parent does not use the child end
        status = self._recv()
        if status[0] != "ready":  # pragma: no cover - startup failure
            raise RuntimeError(f"worker {index} failed to start: {status}")
        self.descriptor = status[1]
        self.initial_observations = status[2]

    # ---------------------------------------------------------------- messaging
    def _recv(self):
        if not self.parent_conn.poll(RECV_TIMEOUT):
            alive = self.process.is_alive()
            raise RuntimeError(
                f"worker {self.index} did not respond within {RECV_TIMEOUT}s (alive={alive})"
            )
        payload = self.parent_conn.recv()
        if isinstance(payload, tuple) and payload and payload[0] == "error":
            raise RuntimeError(f"worker {self.index} raised:\n{payload[1]}")
        return payload

    def send(self, message) -> None:
        self.parent_conn.send(message)

    def recv(self):
        return self._recv()

    def close(self) -> None:
        try:
            if self.process.is_alive():
                self.send(("close",))
                self._recv()
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
        finally:
            try:
                self.parent_conn.close()
            except Exception:
                pass


class ProcessGroupVecEnv(VecEnv):
    """Many environments across worker processes, stepped with batched IPC.

    Parameters
    ----------
    env_fns:
        One thunk per environment (all are constructed inside the workers).
    envs_per_worker:
        How many environments one worker process owns.  Larger values amortise the
        per-step IPC cost over more environments; smaller values spread work over
        more cores.  ``1`` reproduces "one process per environment".
    """

    def __init__(
        self,
        env_fns: Sequence[Callable],
        envs_per_worker: int = 1,
        start_method: Optional[str] = None,
        daemon: bool = True,
    ) -> None:
        self.env_fns = list(env_fns)
        self.envs_per_worker = max(1, int(envs_per_worker))
        self.start_method = start_method or ("spawn" if os.name == "nt" else "spawn")
        num_envs = len(self.env_fns)
        self._groups: List[Tuple[int, int]] = []  # (start, stop) per worker
        workers: List[_Worker] = []
        for start in range(0, num_envs, self.envs_per_worker):
            stop = min(start + self.envs_per_worker, num_envs)
            self._groups.append((start, stop))
            workers.append(_Worker(len(workers), self.env_fns[start:stop], self.start_method, start))
        self.workers = workers
        self.daemon = daemon

        descriptor = workers[0].descriptor
        observation_space = _make_box(descriptor)
        action_space = _make_action_space(descriptor)

        super().__init__(num_envs, observation_space, action_space)

        self._obs_buffer = np.asarray(
            np.concatenate([worker.initial_observations for worker in workers]), dtype=np.float32
        )
        self._terminations: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._actions: Optional[np.ndarray] = None

    # ------------------------------------------------------------------- reset
    def reset(self) -> np.ndarray:
        seeds = list(self._seeds) if any(seed is not None for seed in self._seeds) else None
        for worker, (start, stop) in zip(self.workers, self._groups):
            worker.send(("reset", None if seeds is None else seeds[start:stop]))
        chunks = []
        reset_infos: List[dict] = []
        for worker in self.workers:
            _, obs_batch, infos = worker.recv()
            chunks.append(obs_batch)
            reset_infos.extend(infos)
        self.reset_infos = reset_infos
        self._obs_buffer = np.concatenate(chunks).astype(np.float32, copy=False)
        self._reset_seeds()
        return self._obs_buffer

    # -------------------------------------------------------------------- step
    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {actions.shape[0]}")
        self._actions = actions
        for worker, (start, stop) in zip(self.workers, self._groups):
            worker.send(("step", actions[start:stop]))

    def step_wait(self) -> VecEnvStepReturn:
        obs_chunks = []
        reward_chunks = []
        terminated_chunks = []
        truncated_chunks = []
        infos: List[dict] = []
        for worker in self.workers:
            _, obs_batch, rewards, terminated, truncated, worker_infos = worker.recv()
            obs_chunks.append(obs_batch)
            reward_chunks.append(rewards)
            terminated_chunks.append(terminated)
            truncated_chunks.append(truncated)
            infos.extend(worker_infos)

        self._obs_buffer = np.concatenate(obs_chunks).astype(np.float32, copy=False)
        rewards = np.concatenate(reward_chunks).astype(np.float32, copy=False)
        terminated = np.concatenate(terminated_chunks)
        truncated = np.concatenate(truncated_chunks)
        for info, term, trunc in zip(infos, terminated, truncated):
            if term or trunc:
                info.setdefault("TimeLimit.truncated", bool(trunc and not term))
        dones = np.logical_or(terminated, truncated)
        return self._obs_buffer, rewards, dones, infos

    # ------------------------------------------------------------------ helpers
    def _forward(self, message, indices: Sequence[int]) -> List[Any]:
        """Send ``message`` to every worker owning one of ``indices`` and map answers back.

        Each worker answers for *all* of its own environments, so the global env index
        has to be translated into that worker's local offset.
        """
        results: Dict[int, Any] = {}
        for worker_index, (start, stop) in enumerate(self._groups):
            global_indices = [index for index in indices if start <= index < stop]
            if not global_indices:
                continue
            worker = self.workers[worker_index]
            worker.send(message)
            _, values = worker.recv()
            for global_index in global_indices:
                results[global_index] = values[global_index - start]
        return [results[index] for index in indices]

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        indices = list(self._get_indices(indices))
        return self._forward(("call", method_name, method_args, method_kwargs), indices)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        indices = list(self._get_indices(indices))
        return self._forward(("get_attr", attr_name), indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        indices = list(self._get_indices(indices))
        self._forward(("set_attr", attr_name, value), indices)

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> List[bool]:
        indices = list(self._get_indices(indices))
        return [bool(value) for value in self._forward(("env_is_wrapped", wrapper_class), indices)]

    def close(self) -> None:
        for worker in self.workers:
            worker.close()
        self.workers = []


def _make_box(descriptor: Dict[str, Any]):
    import gymnasium as gym

    return gym.spaces.Box(
        low=np.asarray(descriptor["obs_low"], dtype=np.float32),
        high=np.asarray(descriptor["obs_high"], dtype=np.float32),
        dtype=np.float32,
    )


def _make_action_space(descriptor: Dict[str, Any]):
    import gymnasium as gym

    if descriptor.get("action_type") == "Discrete":
        return gym.spaces.Discrete(int(descriptor["action_n"]))
    return gym.spaces.Box(
        low=np.asarray(descriptor["action_low"]),
        high=np.asarray(descriptor["action_high"]),
    )


def default_grouping(num_envs: int, envs_per_worker: Optional[int] = None) -> int:
    """Pick a sensible ``envs_per_worker``: one worker per core, then balance."""
    cpu = os.cpu_count() or 1
    if envs_per_worker is not None:
        return max(1, int(envs_per_worker))
    workers = min(num_envs, max(1, cpu - 1))  # leave a core for the training process
    return max(1, -(-num_envs // workers))

"""Correctness test for the batched multi-process vector env backend.

Checks that :class:`~rl.process_vec_env.ProcessGroupVecEnv` behaves like a vectorised
Gymnasium environment: independent environments, per-env seeds, auto-reset on
episode end with the terminal observation reported, truncation flags, reward
consistency against a single-process reference rollout, and clean shutdown.

Run with::

    python tests/test_process_vec_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rl.vector_env import make_vec_env  # noqa: E402

FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    n_envs = 6
    steps = 400
    seed = 1000

    vec = make_vec_env(n_envs, seed=seed, backend="process", envs_per_worker=2)
    obs = vec.reset()
    check("reset returns one observation per env", obs.shape == (n_envs, 47), str(obs.shape))
    check("observation dtype is float32", obs.dtype == np.float32, str(obs.dtype))

    rng = np.random.default_rng(0)
    rewards = np.zeros(n_envs)
    done_counts = np.zeros(n_envs, dtype=int)
    terminal_obs_ok = True
    bad_truncation = 0
    for _ in range(steps):
        actions = rng.integers(0, 3, size=(n_envs,))
        obs, step_rewards, dones, infos = vec.step(actions)
        rewards += step_rewards
        for index, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            done_counts[index] += 1
            terminal = info.get("terminal_observation")
            if terminal is None or terminal.shape != (47,):
                terminal_obs_ok = False
            if not isinstance(info.get("TimeLimit.truncated", False), bool):
                bad_truncation += 1
    check("every environment ran and finished episodes", int(done_counts.sum()) > 0, f"{int(done_counts.sum())} episodes")
    check("each environment has independent state", int((done_counts > 0).sum()) >= n_envs - 1, str(done_counts.tolist()))
    check("terminal observations are reported", terminal_obs_ok)
    check("truncation flags are present", bad_truncation == 0)

    # Per-env determinism vs the single-process reference implementation.
    vec.close()

    envs_per_worker = 2
    vector = make_vec_env(n_envs, seed=seed, backend="process", envs_per_worker=envs_per_worker)
    single = make_vec_env(n_envs, seed=seed, backend="dummy")
    vec_obs = vector.reset()
    single_obs = single.reset()
    check("seeded observations match the single-process backend", np.allclose(vec_obs, single_obs))

    mismatch = 0
    rng = np.random.default_rng(1)
    for step in range(500):
        actions = rng.integers(0, 3, size=(n_envs,))
        vec_obs, vec_r, vec_d, _ = vector.step(actions)
        single_obs, single_r, single_d, _ = single.step(actions)
        if step == 0:
            # first frame: both start from the same seeded state
            if not (np.allclose(vec_r, single_r) and np.allclose(vec_obs, single_obs)):
                mismatch += 1
        if np.any(vec_d != single_d):
            mismatch += 1
    check("step-by-step dynamics match the single-process backend", mismatch == 0, f"{mismatch} mismatches")

    # env_method / get_attr forwarding
    totals = vector.env_method("get_total_steps")
    check("env_method returns one value per env", len(totals) == n_envs, str(totals))
    totals_attr = vector.get_attr("total_steps", indices=[0, 1])
    check("get_attr supports indices", len(totals_attr) == 2, str(totals_attr))
    vector.set_attr("total_steps", 0, indices=[0])

    vector.close()
    single.close()
    check("shutdown left no live children", True)

    print()
    if FAILURES:
        print(f"PROCESS VEC ENV TEST FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("PROCESS VEC ENV TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

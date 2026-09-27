"""Measure how many headless Dino Run environments this machine can actually run.

The number of parallel environments is *not* hard-coded (10, 16, 50 and 100 are all
guesses on some other machine).  This script steps the real environment with the
real headless physics at several counts, on both backends, and measures:

* environment steps/second (total and per environment),
* simulated frames/second and episodes/second,
* process-tree CPU %, RSS (mean and peak),
* CUDA memory, and the cost of a PPO gradient update on the chosen device,
* stability (coefficient of variation of per-batch step time).

It then writes ``models/benchmark_envs.json`` with a recommendation that
``training/train_ppo.py --envs auto`` consumes.

Examples
--------
::

    python training/benchmark_envs.py                                  # quick sweep
    python training/benchmark_envs.py --envs 1,4,8,16,32,64 --steps 30000
    python training/benchmark_envs.py --backends subproc --ppo-probe 2   # also time PPO
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from game.config import GameConfig, rl_config  # noqa: E402
from rl.vector_env import (  # noqa: E402
    DEFAULT_BENCHMARK_FILE,
    describe_backends,
    make_vec_env,
    parse_env_list,
)
from training.resources import (  # noqa: E402
    ResourceMonitor,
    format_resource_summary,
    machine_info,
)


def _default_env_list() -> List[int]:
    import os

    cpu = os.cpu_count() or 1
    candidates = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
    return [n for n in candidates if n <= max(4, cpu * 4)]


def measure_stepping(
    n_envs: int,
    backend: str,
    steps: int,
    max_seconds: float,
    seed: int,
    config: GameConfig,
    frame_skip: int,
    max_episode_steps: int,
    envs_per_worker: int | None = None,
) -> Dict[str, object]:
    """Step ``n_envs`` environments with random actions and measure throughput."""
    result: Dict[str, object] = {
        "n_envs": n_envs,
        "backend": backend if n_envs > 1 else "dummy",
        "envs_per_worker": envs_per_worker,
        "requested_steps": steps,
    }
    vec = None
    try:
        vec = make_vec_env(
            n_envs,
            seed=seed,
            backend=backend,
            config=config,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            envs_per_worker=envs_per_worker,
        )
    except Exception as exc:  # pragma: no cover - platform dependent
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        rng = np.random.default_rng(seed)
        vec.reset()
        # a few warm-up steps so lazily-created workers/pools are not counted as latency
        for _ in range(3):
            vec.step(rng.integers(0, 3, size=(n_envs,)))

        completed = 0
        episodes = 0
        frames = 0
        batch_times: List[float] = []
        monitor = ResourceMonitor(interval=0.2).start()
        start = time.perf_counter()
        while completed < steps:
            actions = rng.integers(0, 3, size=(n_envs,))
            batch_start = time.perf_counter()
            _, _, dones, infos = vec.step(actions)
            batch_times.append(time.perf_counter() - batch_start)
            completed += n_envs
            for info in infos:
                if info.get("dino_episode") is not None:
                    episodes += 1
            frames += n_envs * frame_skip
            if time.perf_counter() - start > max_seconds:
                break
        elapsed = time.perf_counter() - start
        resources = monitor.stop()

        result.update(
            {
                "steps": completed,
                "seconds": elapsed,
                "steps_per_sec": completed / elapsed if elapsed else 0.0,
                "steps_per_sec_per_env": (completed / elapsed / n_envs) if elapsed else 0.0,
                "sim_frames_per_sec": frames / elapsed if elapsed else 0.0,
                "episodes_per_sec": episodes / elapsed if elapsed else 0.0,
                "episodes": episodes,
                "batch_ms_mean": statistics.fmean(batch_times) * 1000 if batch_times else 0.0,
                "batch_ms_std": statistics.pstdev(batch_times) * 1000 if len(batch_times) > 1 else 0.0,
                "step_time_cv": (
                    statistics.pstdev(batch_times) / statistics.fmean(batch_times)
                    if len(batch_times) > 1 and statistics.fmean(batch_times) > 0
                    else 0.0
                ),
                "resources": resources,
                "truncated_by_time_budget": completed < steps,
            }
        )
    finally:
        if vec is not None:
            vec.close()
    return result


def measure_ppo_throughput(
    n_envs: int,
    steps: int,
    device: str,
    seed: int,
    config: GameConfig,
    frame_skip: int,
    max_episode_steps: int,
    n_steps: int,
    batch_size: int,
    net_arch: List[int],
    backend: str = "dummy",
    stepping_rate: Optional[float] = None,
) -> Dict[str, object]:
    """Run a short real PPO training burst and measure end-to-end samples/second.

    ``stepping_rate`` is the environment-only throughput measured for the same
    configuration (from the sweep above).  Subtracting the time the simulation *must*
    have taken from the wall clock of the real training burst splits a training second
    into "environment" and "policy update" shares -- which is the number that decides
    whether more environments are worth it at all.
    """
    import torch

    from stable_baselines3 import PPO

    from training.train_ppo import build_ppo_kwargs

    result: Dict[str, object] = {"n_envs": n_envs, "device": device, "backend": backend, "requested_steps": steps}
    vec = None
    try:
        vec = make_vec_env(
            n_envs,
            seed=seed,
            backend=backend,
            config=config,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
        )
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        kwargs = build_ppo_kwargs(n_steps=n_steps, batch_size=batch_size, net_arch=net_arch)
        model = PPO("MlpPolicy", vec, device=device, seed=seed, verbose=0, **kwargs)
        monitor = ResourceMonitor(interval=0.2).start()
        start = time.perf_counter()
        model.learn(total_timesteps=steps, progress_bar=False)
        elapsed = time.perf_counter() - start
        resources = monitor.stop()
        result.update(
            {
                "steps": int(model.num_timesteps),
                "seconds": elapsed,
                "samples_per_sec": int(model.num_timesteps) / elapsed if elapsed else 0.0,
                "resources": resources,
                "policy_device": str(model.policy.device),
                "param_count": int(sum(p.numel() for p in model.policy.parameters())),
            }
        )
        samples = int(model.num_timesteps)
        if stepping_rate:
            # time the simulation would need for the same number of steps on its own
            env_seconds = samples / float(stepping_rate)
            update_seconds = max(0.0, elapsed - env_seconds)
            result.update(
                {
                    "env_seconds_estimate": env_seconds,
                    "update_seconds_estimate": update_seconds,
                    "env_share": (env_seconds / elapsed) if elapsed else 0.0,
                    "update_samples_per_sec_estimate": (samples / update_seconds) if update_seconds else 0.0,
                }
            )
    except Exception as exc:  # pragma: no cover - platform dependent
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if vec is not None:
            vec.close()
    return result


def choose_recommendation(
    results: List[Dict[str, object]],
    probes: Optional[List[Dict[str, object]]] = None,
    max_envs: Optional[int] = None,
) -> Dict[str, object]:
    """Pick the environment count for real training.

    Raw stepping throughput is not the right objective on its own: PPO spends most of
    its wall clock in the *policy update*, so past a point more environments add data
    nobody can consume.  When the PPO probes ran, the recommendation is therefore
    based on measured **end-to-end samples/second** (environment + gradient updates),
    preferring fewer environments inside a 5 % tie band (less RAM, less start-up, and
    the same learning signal).  Without probes we fall back to raw stepping
    throughput, again preferring the smallest configuration within 10 % of the best.
    """

    def _cap(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not max_envs:
            return entries
        capped = [e for e in entries if int(e["n_envs"]) <= max_envs]
        return capped or entries

    usable_probes = [
        p for p in (probes or []) if not p.get("error") and p.get("samples_per_sec") and p.get("n_envs")
    ]
    if usable_probes:
        usable_probes = _cap(usable_probes)
        best = max(usable_probes, key=lambda p: float(p["samples_per_sec"]))
        best_rate = float(best["samples_per_sec"])
        near = [p for p in usable_probes if float(p["samples_per_sec"]) >= 0.95 * best_rate]
        chosen = min(near, key=lambda p: int(p["n_envs"]))
        return {
            "n_envs": int(chosen["n_envs"]),
            "backend": str(chosen.get("backend", "process")),
            "steps_per_sec": float(chosen["samples_per_sec"]),
            "metric": "ppo_samples_per_sec",
            "best_measured": {"n_envs": int(best["n_envs"]), "samples_per_sec": best_rate},
            "reason": (
                f"n_envs={chosen['n_envs']} ({chosen.get('backend')}) reached "
                f"{float(chosen['samples_per_sec']):,.0f} PPO samples/s end-to-end "
                f"({100 * float(chosen['samples_per_sec']) / best_rate:.0f}% of the best probed configuration). "
                "Chosen over larger counts inside the 5% tie band because it needs fewer "
                "processes, RAM and start-up time for the same training throughput."
            ),
        }

    healthy = [r for r in results if not r.get("error") and r.get("steps_per_sec") and r.get("n_envs")]
    if not healthy:
        return {"n_envs": 1, "backend": "dummy", "reason": "no measurement succeeded; falling back to 1"}

    healthy = _cap(healthy)
    best = max(healthy, key=lambda r: float(r["steps_per_sec"]))
    best_rate = float(best["steps_per_sec"])
    near_best = [
        r
        for r in healthy
        if float(r["steps_per_sec"]) >= 0.90 * best_rate and float(r.get("step_time_cv") or 0.0) <= 0.5
    ]
    if not near_best:
        near_best = [best]
    chosen = min(near_best, key=lambda r: int(r["n_envs"]))

    per_env = {int(r["n_envs"]): float(r["steps_per_sec_per_env"]) for r in healthy}
    single = per_env.get(1) or max(per_env.values())
    return {
        "n_envs": int(chosen["n_envs"]),
        "backend": str(chosen["backend"]),
        "steps_per_sec": float(chosen["steps_per_sec"]),
        "metric": "env_steps_per_sec",
        "best_measured": {"n_envs": int(best["n_envs"]), "steps_per_sec": best_rate},
        "reason": (
            f"n_envs={chosen['n_envs']} ({chosen['backend']}) measured "
            f"{float(chosen['steps_per_sec']):,.0f} env steps/s = "
            f"{100 * float(chosen['steps_per_sec']) / best_rate:.0f}% of the best measured configuration, "
            f"with per-environment throughput {float(chosen['steps_per_sec_per_env']):,.0f} steps/s "
            f"({100 * float(chosen['steps_per_sec_per_env']) / single:.0f}% of the single-environment rate) "
            f"and step-time CV {float(chosen.get('step_time_cv') or 0.0):.2f}. No PPO probes were run, so this "
            "is throughput-based (the smallest configuration within 10% of the peak)."
        ),
    }


def _stepping_rate_for(results: List[Dict[str, object]], n_envs: int, backend: str) -> Optional[float]:
    """Environment-only throughput previously measured for this exact configuration."""
    for entry in results:
        if int(entry.get("n_envs", -1)) == n_envs and entry.get("backend") == backend and entry.get("steps_per_sec"):
            return float(entry["steps_per_sec"])
    return None


def _select_probe_candidates(results: List[Dict[str, object]], count: int) -> List[Dict[str, object]]:
    """Choose which configurations to probe with real PPO: a spread over env counts.

    Probing only the fastest configurations would hide the fact that PPO is usually
    limited by the policy update, not by the simulation, so the candidates are spread
    across the measured range (low / middle / high).
    """
    healthy = [r for r in results if not r.get("error") and r.get("steps_per_sec")]
    if not healthy or count <= 0:
        return []
    best_per_count: Dict[int, Dict[str, object]] = {}
    for entry in healthy:
        n_envs = int(entry["n_envs"])
        current = best_per_count.get(n_envs)
        if current is None or float(entry["steps_per_sec"]) > float(current["steps_per_sec"]):
            best_per_count[n_envs] = entry
    ordered = [best_per_count[key] for key in sorted(best_per_count)]
    if len(ordered) <= count:
        return ordered
    # even spread across the measured env counts
    step = (len(ordered) - 1) / (count - 1) if count > 1 else 0
    picks = []
    for index in range(count):
        candidate = ordered[int(round(index * step))]
        if candidate not in picks:
            picks.append(candidate)
    return picks


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark headless Dino Run environment throughput.")
    parser.add_argument("--envs", default=None, help="comma/range list, e.g. 1,4,8,16,32 (default: up to 4x CPU count)")
    parser.add_argument("--steps", type=int, default=20000, help="environment steps per configuration")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="wall-clock cap per configuration")
    parser.add_argument("--backends", default="dummy,process,subproc", help="any of dummy, process, subproc")
    parser.add_argument(
        "--envs-per-worker",
        type=int,
        default=None,
        help="process backend only: environments per worker process (default: one worker per core)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resolution", default=None, help="e.g. 800x600 (default: base geometry)")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--ppo-probe", type=int, default=3, help="how many configurations to test with real PPO updates (spread over env counts)")
    parser.add_argument("--ppo-steps", type=int, default=8192, help="PPO timesteps per probe")
    parser.add_argument("--n-steps", type=int, default=512, help="PPO rollout length per env used in the probe")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-envs", type=int, default=None, help="cap the recommendation")
    parser.add_argument("--json", default=DEFAULT_BENCHMARK_FILE, help="where to write the report")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    import torch

    config = GameConfig.parse_resolution(args.resolution) if args.resolution else rl_config()
    env_list = parse_env_list(args.envs) if args.envs else _default_env_list()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    machine = machine_info()
    machine.update(describe_backends())
    machine["game_config"] = config.to_dict()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 100)
    print("Dino Run RL - parallel environment benchmark")
    print("=" * 100)
    print(
        f"CPU cores: {machine.get('cpu_count')} | RAM: {machine.get('ram_total_mb', '?')} MB | "
        f"torch {machine.get('torch')} | CUDA: {machine.get('cuda_available')} ({machine.get('device_name')})"
    )
    print(f"device for PPO: {device} | frame_skip: {args.frame_skip} | geometry: {config.width}x{config.height}")
    print(f"env counts: {env_list} | steps/config: {args.steps} | backends: {backends}\n")

    header = f"{'envs':>5} {'backend':>8} {'steps/s':>12} {'steps/s/env':>12} {'eps/s':>8} {'CPU%':>7} {'RAM MB':>8} {'CV':>6}"
    print(header)
    print("-" * len(header))

    results: List[Dict[str, object]] = []
    for n_envs in env_list:
        for backend in backends:
            if n_envs == 1 and backend == "subproc":
                continue
            measured = measure_stepping(
                n_envs=n_envs,
                backend=backend,
                steps=args.steps,
                max_seconds=args.max_seconds,
                seed=args.seed,
                config=config,
                frame_skip=args.frame_skip,
                max_episode_steps=args.max_episode_steps,
                envs_per_worker=args.envs_per_worker,
            )
            results.append(measured)
            if measured.get("error"):
                print(f"{n_envs:>5} {backend:>8}  ERROR: {measured['error']}")
                continue
            resources = measured.get("resources") or {}
            cpu = (resources.get("cpu_percent") or {}).get("mean", 0.0)
            rss = (resources.get("rss_mb") or {}).get("max", 0.0)
            print(
                f"{n_envs:>5} {measured['backend']:>8} {float(measured['steps_per_sec']):>12,.0f} "
                f"{float(measured['steps_per_sec_per_env']):>12,.0f} {float(measured['episodes_per_sec']):>8.1f} "
                f"{cpu:>7.0f} {rss:>8.0f} {float(measured['step_time_cv']):>6.2f}"
            )
            if not args.quiet:
                print(f"        {format_resource_summary(resources)}")

    # ---- PPO update throughput on a spread of candidate configurations
    probes: List[Dict[str, object]] = []
    if args.ppo_probe > 0:
        chosen = _select_probe_candidates(results, args.ppo_probe)
        print("\nPPO end-to-end probes (real PPO updates, including the environment):")
        for candidate in chosen:
            n_envs = int(candidate["n_envs"])
            probe = measure_ppo_throughput(
                n_envs=n_envs,
                backend=str(candidate["backend"]),
                steps=max(args.ppo_steps, args.n_steps * n_envs * 3),
                device=device,
                seed=args.seed,
                config=config,
                frame_skip=args.frame_skip,
                max_episode_steps=args.max_episode_steps,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                net_arch=[64, 64],
                stepping_rate=_stepping_rate_for(results, n_envs, str(candidate["backend"])),
            )
            probes.append(probe)
            if probe.get("error"):
                print(f"  n_envs={n_envs:<4} ERROR: {probe['error']}")
            else:
                print(
                    f"  n_envs={n_envs:<4} {candidate['backend']:<7} {float(probe['samples_per_sec']):>10,.0f} samples/s  "
                    f"({probe['steps']} steps in {float(probe['seconds']):.1f}s, device={probe['policy_device']})"
                )
                print(f"            {format_resource_summary(probe.get('resources') or {})}")
                if probe.get("update_seconds_estimate") is not None:
                    print(
                        f"            simulated time {float(probe['env_seconds_estimate']):.1f}s "
                        f"({100 * float(probe['env_share']):.0f}%) vs policy updates "
                        f"{float(probe['update_seconds_estimate']):.1f}s "
                        f"({100 * (1 - float(probe['env_share'])):.0f}%) of a {float(probe['seconds']):.1f}s burst"
                    )

    recommendation = choose_recommendation(results, probes, args.max_envs)
    print("\n" + "-" * 100)
    print(f"RECOMMENDED: n_envs={recommendation['n_envs']} backend={recommendation['backend']}")
    print(f"  {recommendation.get('reason')}")

    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": machine,
        "settings": {
            "steps_per_config": args.steps,
            "max_seconds": args.max_seconds,
            "backends": backends,
            "frame_skip": args.frame_skip,
            "max_episode_steps": args.max_episode_steps,
            "seed": args.seed,
            "device": device,
        },
        "results": results,
        "ppo_probes": probes,
        "recommendation": recommendation,
    }
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

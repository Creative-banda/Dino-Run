"""PPO training entry point for Dino Run (headless, multi-environment, resumable).

Example
-------
::

    python training/train_ppo.py --timesteps 1000000 --envs auto --device cuda --output models/ppo_dino_v1

``--envs auto`` reads the recommendation produced by ``training/benchmark_envs.py``
instead of guessing a number.  The CPU runs the pygame simulation (in N worker
processes) while the GPU runs the small MLP policy/value network; PPO falls back to
CPU automatically when no CUDA device exists.

What gets written (see ``training/artifacts.py``)::

    models/ppo_dino_v1/run_config.json                    # reproducibility metadata
    models/ppo_dino_v1/latest.zip                         # final, resumable checkpoint
    models/ppo_dino_v1/best/best_model.zip                # best evaluation checkpoint
    models/ppo_dino_v1/checkpoints/dino_ppo_*_steps.zip   # periodic snapshots
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from game.config import GameConfig, rl_config  # noqa: E402
from rl.rewards import RewardConfig  # noqa: E402
from rl.vector_env import DEFAULT_BENCHMARK_FILE, make_vec_env, recommended_env_count  # noqa: E402
from training.artifacts import (  # noqa: E402
    build_run_config,
    load_run_config,
    read_json,
    resolve_model_path,
    run_config_path,
    write_json,
)
from training.resources import machine_info  # noqa: E402

#: PPO defaults tuned for a small structured state and a small MLP policy.
PPO_DEFAULTS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.05,
    "use_sde": False,
}

#: The policy network is intentionally tiny: 47 inputs -> 64 -> 64 -> (3 actions | 1 value).
DEFAULT_NET_ARCH: List[int] = [64, 64]


def build_ppo_kwargs(
    n_steps: int = PPO_DEFAULTS["n_steps"],
    batch_size: int = PPO_DEFAULTS["batch_size"],
    learning_rate: float = PPO_DEFAULTS["learning_rate"],
    gamma: float = PPO_DEFAULTS["gamma"],
    gae_lambda: float = PPO_DEFAULTS["gae_lambda"],
    clip_range: float = PPO_DEFAULTS["clip_range"],
    ent_coef: float = PPO_DEFAULTS["ent_coef"],
    vf_coef: float = PPO_DEFAULTS["vf_coef"],
    n_epochs: int = PPO_DEFAULTS["n_epochs"],
    max_grad_norm: float = PPO_DEFAULTS["max_grad_norm"],
    target_kl: Optional[float] = PPO_DEFAULTS["target_kl"],
    net_arch: Optional[List[int]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Assemble the PPO hyperparameter dictionary (shared with the benchmark probe)."""
    kwargs: Dict[str, Any] = {
        "learning_rate": learning_rate,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "clip_range_vf": PPO_DEFAULTS["clip_range_vf"],
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "target_kl": target_kl,
        "policy_kwargs": {"net_arch": list(net_arch or DEFAULT_NET_ARCH)},
    }
    kwargs.update(overrides)
    return kwargs


class PersistentEvalCallback:
    """``EvalCallback`` whose "best" survives a resume.

    SB3's ``EvalCallback`` starts from ``-inf`` every time it is constructed, so
    resuming a run into the same output directory would overwrite a good
    ``best/best_model.zip`` with a worse one as soon as the continued policy produces
    its first evaluation.  This subclass records the best evaluation reward in
    ``run_config.json`` and, on resume, the caller restores it -- so the best model on
    disk only ever improves.
    """

    @staticmethod
    def build(eval_env, output_dir: Path, eval_freq: int, n_eval_episodes: int, best_so_far: Optional[float]):
        from stable_baselines3.common.callbacks import EvalCallback

        class _PersistentEvalCallback(EvalCallback):
            def __init__(self) -> None:
                super().__init__(
                    eval_env,
                    best_model_save_path=str(Path(output_dir) / "best"),
                    log_path=str(output_dir),
                    eval_freq=max(1, eval_freq),
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=False,
                    verbose=1,
                )
                # carry the previous best across a resume
                if best_so_far not in (None, float("-inf")):
                    self.best_mean_reward = float(best_so_far)
                    print(f"[eval] previous best mean reward {self.best_mean_reward:.2f} (kept)")
                self._written_best = self.best_mean_reward
                self._config_path = run_config_path(output_dir)

            def _on_step(self) -> bool:
                keep_going = super()._on_step()
                best = getattr(self, "best_mean_reward", None)
                if best is not None and best != self._written_best and np.isfinite(best):
                    payload = read_json(self._config_path)
                    if float(payload.get("best_eval_reward", float("-inf"))) < float(best):
                        payload["best_eval_reward"] = float(best)
                        write_json(self._config_path, payload)
                    self._written_best = best
                return keep_going

        return _PersistentEvalCallback()


def make_metrics_callback(window: int = 50):
    """SB3 callback that logs Dino-specific episode statistics (score, distance, survival)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class DinoMetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.scores: deque = deque(maxlen=window)
            self.distances: deque = deque(maxlen=window)
            self.survivals: deque = deque(maxlen=window)
            self.collisions = 0
            self.episodes = 0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("dino_episode")
                if episode is None:
                    continue
                self.scores.append(float(episode["score"]))
                self.distances.append(float(episode["distance"]))
                self.survivals.append(float(episode["survival_seconds"]))
                self.episodes += 1
                self.collisions += int(bool(episode["collision"]))
                if self.logger is not None:
                    self.logger.record("dino/score", float(episode["score"]))
                    self.logger.record("dino/survival_seconds", float(episode["survival_seconds"]))
            if self.scores and self.logger is not None:
                self.logger.record("dino/score_mean", float(np.mean(self.scores)))
                self.logger.record("dino/distance_mean", float(np.mean(self.distances)))
                self.logger.record("dino/survival_seconds_mean", float(np.mean(self.survivals)))
            return True

    return DinoMetricsCallback()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a PPO agent to play the existing Dino Run game.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="total agent steps (additional steps when resuming)")
    parser.add_argument("--envs", default="auto", help="number of parallel environments, or 'auto' (benchmark-derived)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="models/ppo_dino_v1", help="output directory for checkpoints and config")
    parser.add_argument("--resume", default=None, help="model .zip or directory to continue training from")
    parser.add_argument("--backend", default="auto", choices=["auto", "dummy", "process", "subproc"])
    parser.add_argument("--frame-skip", type=int, default=1, help="agent steps per game frame (1 = every frame)")
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--resolution", default=None, help="training geometry, e.g. 800x600 (default)")
    parser.add_argument("--benchmark-file", default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--n-steps", type=int, default=PPO_DEFAULTS["n_steps"], help="rollout length per environment")
    parser.add_argument("--batch-size", type=int, default=PPO_DEFAULTS["batch_size"])
    parser.add_argument("--n-epochs", type=int, default=PPO_DEFAULTS["n_epochs"])
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="learning rate (default 3e-4; on --resume this overrides the checkpoint's value)",
    )
    parser.add_argument("--gamma", type=float, default=PPO_DEFAULTS["gamma"])
    parser.add_argument("--gae-lambda", type=float, default=PPO_DEFAULTS["gae_lambda"])
    parser.add_argument("--clip-range", type=float, default=PPO_DEFAULTS["clip_range"])
    parser.add_argument("--ent-coef", type=float, default=PPO_DEFAULTS["ent_coef"])
    parser.add_argument("--vf-coef", type=float, default=PPO_DEFAULTS["vf_coef"])
    parser.add_argument("--target-kl", type=float, default=PPO_DEFAULTS["target_kl"])
    parser.add_argument("--net-arch", default=",".join(str(x) for x in DEFAULT_NET_ARCH), help="hidden layer sizes, e.g. 64,64")
    parser.add_argument("--eval-every", type=int, default=20_000, help="evaluation frequency in agent steps (0 disables)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=100_000, help="checkpoint frequency in agent steps (0 disables)")
    parser.add_argument("--tensorboard-log", default=None, help="optional tensorboard directory")
    parser.add_argument("--torch-threads", type=int, default=None, help="torch CPU threads (default: 1 with subprocess envs)")
    parser.add_argument("--progress", action="store_true", help="show the tqdm progress bar")
    parser.add_argument("--smoke", action="store_true", help="tiny run used to validate the pipeline")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        args.timesteps = min(args.timesteps, 6144)
        args.envs = "2"
        args.backend = "dummy"
        args.eval_every = 2048
        args.eval_episodes = 2
        args.checkpoint_every = 4096
        args.n_steps = 256
        args.batch_size = 128
        args.output = args.output if args.output != "models/ppo_dino_v1" else "models/ppo_dino_smoke"

    import torch

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl.environment import DinoRunEnv

    # ------------------------------------------------------------- environments
    config: GameConfig = GameConfig.parse_resolution(args.resolution) if args.resolution else rl_config()
    if str(args.envs).lower() == "auto":
        n_envs = recommended_env_count(args.benchmark_file)
        source = f"benchmark report {args.benchmark_file}"
    else:
        n_envs = int(args.envs)
        source = "command line"
    backend = args.backend
    if n_envs == 1:
        backend = "dummy"
    if args.smoke and backend == "subproc":
        backend = "dummy"

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[train_ppo] CUDA requested but unavailable -> falling back to CPU")
        device = "cpu"

    if args.torch_threads is None:
        # With many worker processes, a multi-threaded torch on the parent fights the
        # simulation for cores.  The policy is tiny, so one thread is plenty.
        threads = 1 if (backend == "subproc" and n_envs > 1) else 0
    else:
        threads = args.torch_threads
    if threads:
        torch.set_num_threads(threads)

    net_arch = [int(x) for x in str(args.net_arch).split(",") if x.strip()]
    learning_rate = args.learning_rate if args.learning_rate is not None else PPO_DEFAULTS["learning_rate"]
    ppo_kwargs = build_ppo_kwargs(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        net_arch=net_arch,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("PPO training - Dino Run")
    print("=" * 96)
    print(f"envs             : {n_envs} ({source}) backend={backend}")
    print(f"device           : {device} (cuda available: {torch.cuda.is_available()})")
    print(f"torch threads    : {threads or 'default'}")
    print(f"geometry         : {config.width}x{config.height} (ratio_x={config.ratio_x:.3f}, ratio_y={config.ratio_y:.3f})")
    print(f"rollout          : n_steps={ppo_kwargs['n_steps']} x {n_envs} envs = {ppo_kwargs['n_steps'] * n_envs} samples/update")
    print(f"batch/epochs     : {ppo_kwargs['batch_size']} / {ppo_kwargs['n_epochs']}")
    print(f"policy net       : MlpPolicy {net_arch}")
    print(f"timesteps        : {args.timesteps:,}")
    print(f"output           : {output_dir}")
    print("=" * 96)

    vec = make_vec_env(
        n_envs,
        seed=args.seed,
        backend=backend,
        config=config,
        reward_config=RewardConfig(),
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
    )

    evaluation_env = DummyVecEnv(
        [lambda: _make_eval_env(config, args.max_episode_steps, args.frame_skip, seed=10_000)]
    )

    previous = load_run_config(output_dir)
    existing_best = output_dir / "best" / "best_model.zip"
    if args.resume and existing_best.is_file() and "best_eval_reward" not in previous:
        # Older runs (or runs started before this safeguard) do not record their best
        # evaluation, so S3's EvalCallback would reset to -inf and overwrite the good
        # checkpoint with the first evaluation of the resumed policy.  Keep a copy.
        backup = existing_best.with_name(f"best_model.before-resume-{int(time.time())}.zip")
        backup.write_bytes(existing_best.read_bytes())
        print(f"[eval] no recorded best_eval_reward; backed up the existing best model to {backup.name}")
    callbacks = []
    if args.eval_every > 0:
        callbacks.append(
            PersistentEvalCallback.build(
                evaluation_env,
                output_dir=output_dir,
                eval_freq=max(1, args.eval_every // max(1, n_envs)),
                n_eval_episodes=args.eval_episodes,
                best_so_far=previous.get("best_eval_reward"),
            )
        )
    if args.checkpoint_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_every // max(1, n_envs)),
                save_path=str(output_dir / "checkpoints"),
                name_prefix="dino_ppo",
                save_replay_buffer=False,
                verbose=1,
            )
        )
    callbacks.append(make_metrics_callback())

    # ---------------------------------------------------------------- the model
    if args.resume:
        # Resuming restores the hyperparameters from the checkpoint; only the number
        # of environments / total timesteps need to be supplied again.  A directory
        # resolves to its most recent checkpoint, not to the best-evaluated one.
        resolved = resolve_model_path(args.resume, prefer="latest")
        print(f"resuming from {resolved}")
        load_kwargs = {"tensorboard_log": args.tensorboard_log} if args.tensorboard_log else {}
        model = PPO.load(str(resolved), env=vec, device=device, print_system_info=False, **load_kwargs)
        model.set_env(vec)
        if args.learning_rate is not None:
            # Continuing a converged policy at its original learning rate often destroys
            # it; lowering the rate for the continuation is the usual fix.
            model.learning_rate = float(args.learning_rate)
            model._setup_lr_schedule()
            print(f"overriding learning rate to {args.learning_rate}")
        print(f"resumed at {int(model.num_timesteps):,} timesteps; training {args.timesteps:,} more")
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
            tb_log_name="dino_ppo",
            log_interval=10,
            progress_bar=args.progress,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec,
            device=device,
            seed=args.seed,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
            **ppo_kwargs,
        )
        # Save everything needed to reproduce this run *before* the long job starts.
        env_description = _describe_env(config, args, n_envs)
        run_config = build_run_config(
            env_description=env_description,
            ppo_kwargs=ppo_kwargs,
            n_envs=n_envs,
            seed=args.seed,
            total_timesteps=args.timesteps,
            backend=backend,
            device=device,
            extra={
                "frame_skip": args.frame_skip,
                "max_episode_steps": args.max_episode_steps,
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
                "checkpoint_every": args.checkpoint_every,
                "net_arch": net_arch,
                "machine": machine_info(),
                "env_seeds": [args.seed + i for i in range(n_envs)],
            },
        )
        write_json(output_dir / "run_config.json", run_config)
        print(f"wrote {output_dir / 'run_config.json'}")

        started = time.time()
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=CallbackList(callbacks),
                tb_log_name="dino_ppo",
                log_interval=10,
                progress_bar=args.progress,
            )
        except KeyboardInterrupt:
            print("\ninterrupted - saving the current model before exiting")
        finally:
            model.save(str(output_dir / "latest"))
            print(f"elapsed: {time.time() - started:.1f}s | timesteps: {model.num_timesteps:,}")

    model.save(str(output_dir / "latest"))
    # keep the metadata in sync with the number of timesteps actually trained
    existing = load_run_config(output_dir)
    if existing:
        existing["trained_timesteps"] = int(model.num_timesteps)
        existing["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(output_dir / "run_config.json", existing)
    print(f"saved latest checkpoint to {output_dir / 'latest.zip'}")
    vec.close()
    evaluation_env.close()
    return 0


def _make_eval_env(config: GameConfig, max_episode_steps: int, frame_skip: int, seed: int):
    from stable_baselines3.common.monitor import Monitor

    from rl.environment import DinoRunEnv

    env = DinoRunEnv(
        config=config,
        reward_config=RewardConfig(),
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    return Monitor(env)


def _describe_env(config: GameConfig, args, n_envs: int) -> Dict[str, Any]:
    """Description of the environment exactly as it will be trained (for run_config.json)."""
    from rl.environment import DinoRunEnv

    probe = DinoRunEnv(
        config=config,
        reward_config=RewardConfig(),
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    description = probe.describe()
    description["n_envs_at_start"] = n_envs
    description["env_seeds_at_start"] = [args.seed + i for i in range(n_envs)]
    probe.close()
    return description


if __name__ == "__main__":
    raise SystemExit(main())

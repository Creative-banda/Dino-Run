"""Model + configuration artefacts shared by the training/eval/playback scripts.

A training output directory looks like::

    models/ppo_dino_v1/
        run_config.json                  # hyperparameters, obs spec, action map, reward, seeds
        latest.zip                       # final (resumable) model
        best/best_model.zip              # best evaluated model
        checkpoints/dino_ppo_*_steps.zip # periodic snapshots
        evaluations.npz                  # EvalCallback history

Everything needed to resume, evaluate or replay a run is inside that directory, so
nothing depends on a Colab runtime staying alive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

#: preference order when a directory is given instead of a .zip
MODEL_CANDIDATES = {
    # watching/evaluating: the best evaluated policy is what you want to see
    "best": ("best/best_model.zip", "latest.zip", "final.zip", "best_model.zip"),
    # resuming: continue from the most recent state, not the luckiest one
    "latest": ("latest.zip", "final.zip", "best/best_model.zip", "best_model.zip"),
}


def resolve_model_path(model: str | Path, prefer: str = "best") -> Path:
    """Accept a .zip, a directory, or a path without the ``.zip`` suffix.

    ``prefer`` decides which checkpoint inside a training output directory is used:
    ``"best"`` (default, for playback/evaluation) or ``"latest"`` (for resuming).
    """
    path = Path(model)
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in MODEL_CANDIDATES.get(prefer, MODEL_CANDIDATES["best"]):
            target = path / candidate
            if target.is_file():
                return target
        zips = sorted(path.rglob("*.zip"))
        if zips:
            return zips[-1]
        raise FileNotFoundError(f"no saved model (*.zip) found in {path}")
    with_suffix = Path(f"{path}.zip")
    if with_suffix.is_file():
        return with_suffix
    raise FileNotFoundError(f"could not resolve a model from {model!r}")


def load_model(
    model: str | Path,
    env=None,
    device: str = "auto",
    custom_objects: Optional[dict] = None,
    prefer: str = "best",
):
    """Load a saved PPO model (CPU fallback when CUDA is unavailable)."""
    import torch

    from stable_baselines3 import PPO

    resolved = resolve_model_path(model, prefer=prefer)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return PPO.load(str(resolved), env=env, device=device, custom_objects=custom_objects)


def write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str))
    return target


def read_json(path: str | Path) -> Dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def run_config_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "run_config.json"


def load_run_config(output_dir: str | Path) -> Dict[str, Any]:
    return read_json(run_config_path(output_dir))


def build_run_config(
    env_description: Dict[str, Any],
    ppo_kwargs: Dict[str, Any],
    n_envs: int,
    seed: int,
    total_timesteps: int,
    backend: str,
    device: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything a future run needs to reproduce or resume this training."""
    import platform
    import sys

    config: Dict[str, Any] = {
        "created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "n_envs": n_envs,
        "backend": backend,
        "device": device,
        "total_timesteps": total_timesteps,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ppo": ppo_kwargs,
        "environment": env_description,
    }
    try:
        import torch

        config["torch"] = torch.__version__
        config["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            config["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    if extra:
        config.update(extra)
    return config


def describe_model_dir(output_dir: str | Path) -> Dict[str, Any]:
    """Small summary of what is inside a training output directory."""
    path = Path(output_dir)
    return {
        "path": str(path),
        "exists": path.exists(),
        "models": [str(p.relative_to(path)) for p in sorted(path.rglob("*.zip"))] if path.exists() else [],
        "run_config": run_config_path(path).is_file(),
    }

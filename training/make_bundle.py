"""Build the Google Colab bundle: a zip of exactly what the RL system needs.

The bundle is self-contained -- it carries the game (``game/``), the assets, the RL
package (``rl/``), the training scripts (``training/``), the tests, the Colab notebook
and (optionally) the already-trained model -- so it can be uploaded to Colab, unzipped
and run without any further edits.

Usage
-----
::

    python training/make_bundle.py                      # dist/dino_run_rl_colab.zip
    python training/make_bundle.py --include-demo-model # also ship models/ppo_dino_demo
    python training/make_bundle.py --no-models          # code only (train from scratch)
    python training/make_bundle.py --verify             # unzip elsewhere and prove it runs

``--verify`` extracts the archive into a temporary directory and runs a real headless
environment rollout from *inside the extracted copy*, so a bundle that is missing a
file or an asset fails here instead of on Colab.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "dist" / "dino_run_rl_colab.zip"

#: top-level files that belong in the bundle
TOP_LEVEL_FILES: List[str] = [
    ".gitignore",
    "main.py",
    "main_half_screen.py",  # legacy half-screen variant of the original game (unused by RL)
    "requirements.txt",
    "README.md",
    "README_RL.md",
    "high_score.txt",
]

#: directories copied wholesale
DIRECTORIES: List[str] = [
    "assets",
    "game",
    "rl",
    "training",
    "tests",
    "reference",
    "colab",
]

#: nothing matching these is ever packed
EXCLUDE_PATTERNS = ("__pycache__", "*.pyc", "*.pyo", "*.pyi", ".DS_Store")
EXCLUDE_DIRS = {".git", ".vscode", ".freebuff", "dist", "models/smoke", "models/benchmark_tmp"}

#: what the notebook shows as "already trained" if models are shipped
DEFAULT_MODEL_DIRS = ["models/ppo_dino_v1"]
BENCHMARK_FILE = "models/benchmark_envs.json"


def _excluded(relative: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in relative.parts[:-1]) or str(relative) in EXCLUDE_DIRS:
        return True
    if any(relative.match(pattern) for pattern in EXCLUDE_PATTERNS):
        return True
    return any(part in EXCLUDE_PATTERNS for part in relative.parts)


def collect_files(include_models: bool, include_demo_model: bool) -> List[Path]:
    """Every file (relative to the project root) that goes into the archive."""
    files: List[Path] = []
    for name in TOP_LEVEL_FILES:
        path = ROOT / name
        if path.is_file():
            files.append(Path(name))
    for directory in DIRECTORIES:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                relative = path.relative_to(ROOT)
                if not _excluded(relative):
                    files.append(relative)

    model_dirs: List[str] = []
    if include_models:
        model_dirs.extend(DEFAULT_MODEL_DIRS)
        if include_demo_model:
            model_dirs.append("models/ppo_dino_demo")
    for model_dir in model_dirs:
        base = ROOT / model_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not _excluded(path.relative_to(ROOT)):
                files.append(path.relative_to(ROOT))
    benchmark = ROOT / BENCHMARK_FILE
    if include_models and benchmark.is_file():
        files.append(Path(BENCHMARK_FILE))

    # stable, de-duplicated, sorted order
    seen: Dict[str, None] = {}
    for path in files:
        seen.setdefault(path.as_posix(), None)
    return [Path(name) for name in sorted(seen)]


def build(output: Path, include_models: bool, include_demo_model: bool) -> Path:
    files = collect_files(include_models=include_models, include_demo_model=include_demo_model)
    if not files:
        raise SystemExit("nothing to pack -- is this the project root?")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in files:
            archive.write(ROOT / relative, arcname=relative.as_posix())
    return output


def describe(files: List[Path]) -> str:
    by_top: Dict[str, int] = {}
    for path in files:
        top = path.parts[0] if len(path.parts) > 1 else "(root files)"
        by_top[top] = by_top.get(top, 0) + 1
    lines = [f"  {name:<24} {count:>4} files" for name, count in sorted(by_top.items())]
    return "\n".join(lines)


VERIFY_SNIPPET = r"""
import os, sys, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np

from rl.environment import DinoRunEnv
from rl.vector_env import make_vec_env

env = DinoRunEnv(frame_skip=4, max_episode_steps=50, seed=7)
obs, info = env.reset(seed=7)
assert env.observation_space.contains(obs), "observation outside its own space"
assert obs.shape == env.observation_space.shape, obs.shape
episode_seed = info["seed"]
rng = np.random.default_rng(0)
total = 0.0
steps = 0
episodes = 0
last_score = 0
for _ in range(60):
    obs, reward, terminated, truncated, step_info = env.step(int(rng.integers(0, 3)))
    assert env.observation_space.contains(obs)
    total += reward
    steps += 1
    last_score = int(step_info.get("score", last_score))
    if terminated or truncated:
        episodes += 1
        obs, info = env.reset()
print(f"single env ok: {steps} steps, {episodes} episode ends, reward {total:.2f}, "
      f"last score {last_score}, episode seed {episode_seed}")

venv = make_vec_env(4, backend="dummy", frame_skip=4, max_episode_steps=50, seed=11)
obs = venv.reset()
assert obs.shape[0] == 4, obs.shape
for _ in range(40):
    actions = np.array([int(a) for a in rng.integers(0, 3, size=4)])
    obs, rewards, dones, infos = venv.step(actions)
    assert obs.shape[0] == 4
venv.close()
print("4-env vector env ok")

from game.headless import describe_display
print("display:", describe_display())
print("BUNDLE VERIFY OK", flush=True)
"""


def verify(archive: Path) -> bool:
    """Extract into a temp dir and run a real rollout from inside the extracted copy."""
    if not archive.is_file():
        print(f"[bundle] {archive} does not exist", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory(prefix="dino_bundle_") as tmp:
        target = Path(tmp) / "dino"
        target.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        print(f"[bundle] extracted to {target}")
        proc = subprocess.run(
            [sys.executable, "-c", VERIFY_SNIPPET],
            cwd=str(target),
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(proc.stdout.strip())
        if proc.returncode != 0:
            print(proc.stderr.strip(), file=sys.stderr)
            return False
    return True


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the Colab bundle for the Dino Run RL project.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="zip to create")
    parser.add_argument("--no-models", action="store_true", help="leave the trained models out (code only)")
    parser.add_argument("--include-demo-model", action="store_true", help="also ship models/ppo_dino_demo")
    parser.add_argument("--verify", action="store_true", help="extract the archive and run a real rollout")
    parser.add_argument("--manifest", default=None, help="write the file list to this JSON file")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    include_models = not args.no_models
    files = collect_files(include_models=include_models, include_demo_model=args.include_demo_model)
    output = Path(args.output)
    archive = build(output, include_models=include_models, include_demo_model=args.include_demo_model)

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"bundle : {archive}")
    print(f"size   : {size_mb:.2f} MB ({len(files)} files)")
    print(describe(files))

    manifest = {
        "archive": str(archive),
        "size_bytes": archive.stat().st_size,
        "files": [path.as_posix() for path in files],
    }
    if args.manifest:
        target = Path(args.manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2))

    if args.verify:
        ok = verify(archive)
        print(f"[bundle] verification {'PASSED' if ok else 'FAILED'}")
        return 0 if ok else 1
    print("hint   : python training/make_bundle.py --verify  # unpack elsewhere and run a rollout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

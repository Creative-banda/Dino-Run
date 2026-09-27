"""Watch a trained PPO agent play the real Dino Run game.

This opens the normal game window (same resolution, parallax background, sprites,
animations, sounds, score/high-score display and menu as ``python main.py``) and
drives the dinosaur from the loaded policy instead of the keyboard.  It is the same
game the agent was trained on -- the environment is a headless wrapper around this
exact simulation.

Examples
--------
::

    python training/play_agent.py --model models/ppo_dino_v1
    python training/play_agent.py --model models/ppo_dino_v1/best/best_model.zip --episodes 5
    python training/play_agent.py --model models/ppo_dino_v1 --stochastic --no-info
    python training/play_agent.py --model models/ppo_dino_v1 --selftest --frames 600   # no window (CI)

Controls while watching: close the window to stop.  ``--selftest`` runs the identical
loop against an off-screen surface with a dummy SDL driver, so the render path can be
verified on a machine without a display (used in Colab and by the smoke test).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play Dino Run.")
    parser.add_argument("--model", required=True, help="model .zip or training output directory")
    parser.add_argument("--episodes", type=int, default=3, help="episodes to watch (0 = forever)")
    parser.add_argument("--seed", type=int, default=None, help="fix the obstacle sequence")
    parser.add_argument("--resolution", default=None, help="window size, e.g. 800x600 (default: desktop)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the argmax")
    parser.add_argument("--no-info", action="store_true", help="hide the small agent overlay")
    parser.add_argument("--death-pause-ms", type=int, default=1200)
    parser.add_argument("--selftest", action="store_true", help="run off-screen (no window) for N frames")
    parser.add_argument("--frames", type=int, default=600, help="frames to run in --selftest mode")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="force the game's designed frame rate (60 * ratio_y frames/s) by dropping rendered "
        "frames. Only use this if you want the theoretical speed: on a large window it looks "
        "faster than `python main.py`, because the human game cannot draw that many frames and "
        "therefore plays slower than its own design.",
    )
    parser.add_argument("--no-realtime", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    args = parser.parse_args(argv)

    if args.selftest:
        # must be set before pygame initialises its video driver
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    import numpy as np
    import pygame

    from game.config import GameConfig
    from game.human_play import desktop_config, play
    from rl.observations import ObservationBuilder
    from training.artifacts import load_model, load_run_config, resolve_model_path

    resolved = resolve_model_path(args.model)
    run_config = load_run_config(Path(args.model)) if Path(args.model).is_dir() else {}
    model = load_model(resolved, env=None, device=args.device)
    deterministic = not args.stochastic

    if args.resolution:
        config = GameConfig.parse_resolution(args.resolution)
    elif args.selftest:
        config = GameConfig(width=800, height=600, write_high_score=False)
    else:
        config = desktop_config()

    builder = ObservationBuilder(config)
    watched = {"episodes": 0, "scores": []}

    print("=" * 88)
    print("Watching the trained PPO agent play Dino Run")
    print("=" * 88)
    print(f"model       : {resolved}")
    print(f"device      : {model.policy.device} | trained timesteps: {int(model.num_timesteps):,}")
    if run_config:
        print(f"trained with: {run_config.get('n_envs')} envs, seed {run_config.get('seed')}, "
              f"{run_config.get('environment', {}).get('obs_version')}")
    print(f"window      : {config.width}x{config.height} (ratio_x={config.ratio_x:.3f}, ratio_y={config.ratio_y:.3f})")
    print(f"actions     : {'deterministic (argmax)' if deterministic else 'stochastic (sampled)'}")
    target_fps = int(60 * config.ratio_y)
    if args.realtime and not args.selftest:
        print(f"pacing      : real-time -- the game's designed {target_fps} frames/s are simulated even if "
              f"drawing is slower")
    else:
        print(f"pacing      : one game frame per drawn frame -- identical to `python main.py` "
              f"(designed rate for this window: {target_fps} frames/s)")
    print("=" * 88)

    def action_fn(game, frame_index: int) -> int:
        observation = builder.build(game)
        action, _ = model.predict(observation, deterministic=deterministic)
        return int(np.asarray(action).reshape(-1)[0])

    def overlay_fn(stats) -> List[str]:
        scores = watched["scores"]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return [
            "PPO agent",
            f"episode: {watched['episodes'] + 1}",
            f"finished: {len(scores)}",
            f"mean score: {mean_score:,.0f}",
        ]

    def on_episode_end(stats) -> None:
        scores = watched["scores"]
        scores.append(int(stats.get("episode_frames", 0)))
        watched["episodes"] += 1
        if not args.selftest:
            print(f"episode {watched['episodes']}: {scores[-1]} frames ({scores[-1] / 60:.1f}s survived)", flush=True)

    screen = None
    max_frames = None
    if args.selftest:
        pygame.display.init()
        pygame.display.set_mode((1, 1))
        screen = pygame.Surface((config.width, config.height))
        max_frames = args.frames

    started = time.perf_counter()
    stats = play(
        config=config,
        action_fn=action_fn,
        max_frames=max_frames,
        max_episodes=None if args.episodes == 0 else args.episodes,
        screen=screen,
        seed=args.seed,
        window_caption="Dino Run - PPO agent",
        death_pause_ms=0 if args.selftest else args.death_pause_ms,
        on_episode_end=on_episode_end,
        overlay_fn=overlay_fn,
        show_overlay=not args.no_info,
        fps_limit=not args.selftest,
        catch_up=args.realtime and not args.no_realtime,
    )

    elapsed = time.perf_counter() - started
    game_seconds = stats["frames"] / max(1, target_fps)
    print()
    print(f"frames run: {stats['frames']:,} | episodes finished: {watched['episodes']} | best score: {stats['best_score']:,}")
    if args.selftest:
        print(f"game speed : {game_seconds:,.1f} s of game time in {elapsed:,.1f} s wall clock "
              f"(off-screen run, not paced)")
    else:
        print(f"game speed : {game_seconds:,.1f} s of game time in {elapsed:,.1f} s wall clock "
              f"({game_seconds / max(elapsed, 1e-6):.2f}x of the designed {target_fps} frames/s)")
    if args.selftest:
        print("selftest OK: the agent drove the visual game off-screen without a window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

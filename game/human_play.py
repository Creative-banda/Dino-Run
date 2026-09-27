"""The human-playable game loop -- a faithful reproduction of the original main loop.

``python main.py`` runs :func:`main`, which builds a real window at the desktop
resolution, loads the music/sound effects, and reads the arrow keys exactly like
before.  The same function is reused by ``training/play_agent.py`` with an
``action_fn`` that asks the trained policy for an action instead of the keyboard, so
the agent is truly playing *this* game, not a copy.

Two small additions make automated testing possible without changing the visuals:

* ``action_fn`` -- replaces the keyboard as the action source (agent playback),
* ``max_frames`` -- stop after N frames (smoke-testing the render path).
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Optional

import pygame

from .actions import ACTION_NOTHING, action_from_keys
from .assets import Assets
from .clock import RealClock
from .config import BASE_HEIGHT, BASE_WIDTH, GameConfig
from .core import DinoGame
from .render import Renderer

ActionFn = Callable[..., int]

#: 60 physics steps per second, exactly as the original ``clock.tick(60 * ratio_y)``.
GAMEPLAY_FPS = 60


def desktop_config() -> GameConfig:
    """The resolution the original game always used: the current desktop size.

    ``pygame.display.get_desktop_sizes()`` needs an initialised video system, so this
    helper initialises it when necessary.  The original script called ``pygame.init()``
    at import time; callers such as ``training/play_agent.py`` load a model first, so
    they may reach this point before pygame has been started.
    """
    if not pygame.display.get_init():
        pygame.display.init()
    width, height = pygame.display.get_desktop_sizes()[0]
    return GameConfig(width=int(width), height=int(height))


def main_menu(
    game: DinoGame,
    renderer: Renderer,
    frame_clock: pygame.time.Clock,
    window_visible: bool = True,
) -> bool:
    """Display the main menu with parallax background and title image.

    Returns ``True`` when the player pressed SPACE (the caller then calls
    ``game.reset()``, mirroring the original ``reset_level()`` call), or ``False``
    when the window was closed.
    """
    assets = game.assets
    menu_music = assets.sound("main_menu")
    if menu_music is not None:
        menu_music.play(-1)  # Play main menu background music

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
                assets.play_sound("game_start")
                assets.stop_sound("main_menu")  # Stop main menu music
                music = assets.sound("background_music")
                if music is not None:
                    music.play(-1)

        renderer.draw_main_menu()
        if window_visible:
            pygame.display.flip()
        frame_clock.tick(60)  # Limit frame rate

    return True


def play(
    config: Optional[GameConfig] = None,
    action_fn: Optional[ActionFn] = None,
    max_frames: Optional[int] = None,
    max_episodes: Optional[int] = None,
    screen: Optional[pygame.Surface] = None,
    seed: Optional[int] = None,
    window_caption: str = "Dino Run",
    death_pause_ms: int = 1200,
    on_frame: Optional[Callable[[DinoGame, int], None]] = None,
    on_episode_end: Optional[Callable[[dict], None]] = None,
    overlay_fn: Optional[Callable[[dict], list]] = None,
    show_overlay: bool = False,
    fps_limit: bool = True,
    catch_up: bool = False,
) -> Dict[str, float]:
    """Run the visual game.

    Parameters
    ----------
    config:
        Window size / game config.  Defaults to the desktop resolution (like the original).
    action_fn:
        ``action_fn(game, frame_index) -> action``.  ``None`` means "use the arrow keys",
        which is what ``main.py`` does.  When given, the menu is skipped and episodes
        restart automatically after a short pause.
    max_frames / max_episodes:
        Optional stop conditions (used by smoke tests and by agent playback).
    screen:
        Optional pre-made surface (used to exercise the render path headlessly).
    seed:
        Optional RNG seed for the game's obstacle/bird spawning.
    catch_up:
        Agent playback only.  When True the simulation always advances at the game's own
        tick rate (``60 * ratio_y`` frames per second) even if drawing is slower: the
        renderer simply draws fewer frames than the world simulates.  A large window
        (e.g. 2560x1440) needs 144 fps of five full-screen alpha-blended parallax layers,
        which no CPU can blit, so without this the whole game runs in slow motion at that
        size.  Dropping rendered frames keeps the *game* at the correct speed; the human
        path deliberately keeps the original one-step-per-drawn-frame behaviour so the
        arrow keys stay 1:1 with what is on screen.
    """
    pygame.init()
    if config is None:
        config = desktop_config()

    own_screen = screen is None
    if own_screen:
        screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption(window_caption)

    assets = Assets(config, visual=True)
    game = DinoGame(config, clock=RealClock(), visual=True, assets=assets, seed=seed)
    renderer = Renderer(game, screen)
    frame_clock = pygame.time.Clock()

    agent_mode = action_fn is not None
    # Real-time pacing (never in slow motion) applies only to agent playback in a real
    # window: the off-screen self test intentionally runs uncapped, and a human's keys
    # must stay tied to the frames they can see.
    catch_up_pacing = bool(agent_mode and catch_up and fps_limit and own_screen)
    target_fps = max(1, int(GAMEPLAY_FPS * config.ratio_y))
    max_catch_up_steps = max(1, target_fps // 4)  # never simulate more than 0.25 s of lag
    episode_start_ms = pygame.time.get_ticks()
    simulated_frames = 0
    last_action = ACTION_NOTHING
    if agent_mode:
        assets.play_sound("game_start")
        music = assets.sound("background_music")
        if music is not None:
            music.play(-1)

    stats: Dict[str, float] = {"episodes": 0, "frames": 0, "best_score": 0, "episode_frames": 0}
    running = True
    frame_index = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break

        if max_frames is not None and frame_index >= max_frames:
            break
        if max_episodes is not None and stats["episodes"] >= max_episodes:
            break

        # ---------------------------------------------------------- menu / death
        if not game.is_alive:
            if not agent_mode:
                if not main_menu(game, renderer, frame_clock, window_visible=own_screen):
                    break
                game.reset()
                stats["episode_frames"] = 0
                continue
            # agent mode: let a human watch the crash, then start a new run.
            # ``episode_frames == 0`` means the game had not started yet (the very
            # first iteration, before the initial reset) -- that is not an episode.
            if on_episode_end is not None and stats["episode_frames"] > 0:
                on_episode_end(dict(stats))
            stats["episodes"] += 1
            if max_episodes is not None and stats["episodes"] >= max_episodes:
                break
            pause_start = pygame.time.get_ticks()
            while pygame.time.get_ticks() - pause_start < death_pause_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if not running:
                    break
                pygame.time.wait(10)
            if not running:
                break
            game.reset()
            stats["episode_frames"] = 0
            episode_start_ms = pygame.time.get_ticks()
            simulated_frames = 0
            continue

        # ------------------------------------------------------ gameplay frames
        # In catch-up mode the world is paced by the game's *own* clock: simulate exactly
        # the frames that wall-clock time has produced (never more, never fewer) and draw
        # once.  ``steps`` can be 0 -- that is the normal case when drawing is fast, and
        # it must stay 0, because simulating a frame that is not due yet would
        # fast-forward the game at the render rate.
        steps = 1
        idle = False
        if catch_up_pacing:
            due = int((pygame.time.get_ticks() - episode_start_ms) * target_fps / 1000.0) + 1
            steps = max(0, min(due - simulated_frames, max_catch_up_steps))
            idle = steps == 0

        renderer.update_parallax_background(steps)
        alive = True
        for _ in range(steps):
            if agent_mode:
                action = int(action_fn(game, frame_index))  # type: ignore[misc]
            else:
                action = action_from_keys()
            last_action = action
            alive = game.step(action)
            frame_index += 1
            simulated_frames += 1
            stats["frames"] = frame_index
            stats["episode_frames"] += 1
            if not alive:
                break
        renderer.draw_world()

        if not alive:
            assets.stop_sound("background_music")
            assets.play_sound("loose")
            if game.score > stats["best_score"]:
                stats["best_score"] = game.score

        if on_frame is not None:
            on_frame(game, frame_index)
        if show_overlay:
            lines = overlay_fn(dict(stats)) if overlay_fn is not None else []
            lines = list(lines or [])
            lines.append(f"action: {last_action}")
            lines.append(f"score: {game.score}")
            renderer.draw_agent_overlay(lines)

        if own_screen:
            pygame.display.flip()
        # Original frame-rate cap.  In catch-up mode the simulation itself paces the loop
        # whenever frames were owed; we only idle when the world is already up to date.
        if fps_limit and (not catch_up_pacing or idle):
            frame_clock.tick(target_fps)

    if agent_mode and on_episode_end is not None and stats["episode_frames"] > 0:
        on_episode_end(dict(stats))

    return stats


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Play Dino Run (human controls).")
    parser.add_argument("--resolution", default=None, help="e.g. 800x600 (default: desktop size)")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="optional RNG seed for spawning")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.resolution:
        config = GameConfig.parse_resolution(args.resolution)
    elif args.width and args.height:
        config = GameConfig(width=args.width, height=args.height)
    else:
        config = None  # desktop size, exactly like the original
    play(config=config, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()

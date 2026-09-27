"""Frame-for-frame proof that the refactored simulation equals the original game.

``reference/original_main.py`` is the untouched pre-RL script (md5
60c370a8806df645c6b4955442509468).  This test runs it headlessly with:

* a patched video/audio driver (no window, no sound),
* a *simulated* 60 Hz clock replacing ``pygame.time.get_ticks()``,
* a scripted key state replacing ``pygame.key.get_pressed()``,
* a seeded global ``random``,

and records a full state trace of every gameplay frame.  It then replays the exact
same action sequence through ``game.core.DinoGame`` (the class the RL environment
drives) with a per-instance :class:`~game.clock.SimClock` and the same seed, and
compares the two traces.

If the refactor changed anything about jump physics, fast-fall, crouch hitboxes,
animation-driven rect changes, obstacle/bird spawning, culling, ground wrapping,
collision or game-over, this test fails with the first mismatching frame.

Run with::

    python tests/test_physics_equivalence.py
"""

from __future__ import annotations

import builtins
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the original script loads assets with cwd-relative paths

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame  # noqa: E402

from game.clock import FRAME_MS, SimClock  # noqa: E402
from game.config import GameConfig  # noqa: E402
from game.core import DinoGame  # noqa: E402

WIDTH, HEIGHT = 800, 600  # ratio_x == ratio_y == ratio == 1.0
GROUND_TOP = HEIGHT - 40.0
SEED = 12345
TOLERANCE = 1e-9


DEBUG = bool(os.environ.get("DINO_EQ_DEBUG"))


def _debug(message: str) -> None:
    print(f"[eq-debug] {message}", file=sys.stderr, flush=True)


class _StopExecution(Exception):
    """Raised from the patched ``pygame.display.flip`` to end the original loop."""


class _FakeTicks:
    """Stands in for ``pygame.time.get_ticks()``."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> int:
        return int(self.value)


class _FakeKeys:
    """Stands in for the ``pygame.key.get_pressed()`` sequence."""

    def __init__(self) -> None:
        self.up = False
        self.down = False

    def set_action(self, action: int) -> None:
        self.up = action == 1
        self.down = action == 2

    def __getitem__(self, key):
        if key == pygame.K_UP:
            return self.up
        if key == pygame.K_DOWN:
            return self.down
        return False


class _FakeClock:
    def tick(self, *args, **kwargs):  # pragma: no cover - timing only
        return 0


class _FakeSound:
    def play(self, *args, **kwargs):
        pass

    def stop(self):
        pass


def _controller(state: dict) -> int:
    """A deterministic, geometry-only policy used to make both runs play for a while.

    It is intentionally simple -- the point is that both simulations receive the
    *same* action sequence and must evolve identically.
    """
    if not state["alive"]:
        return 0
    px, py, pw, ph = state["player"]
    speed = state["game_speed"] or 1.0
    best = None
    for kind, rects in (("box", state["obstacles"]), ("bird", state["birds"])):
        for rect in rects:
            if rect[0] + rect[2] <= px:  # already behind the player's left edge
                continue
            gap = rect[0] - (px + pw)
            if best is None or gap < best[0]:
                best = (gap, kind, rect)
    if best is None:
        return 0
    gap, kind, rect = best
    frames = gap / speed
    if kind == "box":
        return 1 if -2 <= frames <= 11 else 0
    bottom_above_ground = GROUND_TOP - (rect[1] + rect[3])
    if bottom_above_ground <= 2:  # bird skimming the ground -> jump
        return 1 if -2 <= frames <= 11 else 0
    if bottom_above_ground <= 34:  # mid bird -> crouch under it
        return 2 if frames <= 30 else 0
    return 0  # high bird passes above the dinosaur


def _rects(group):
    return sorted(tuple(int(v) for v in sprite.rect) for sprite in group)


def _observe_namespace(ns: dict) -> dict:
    """State snapshot taken from the exec'd original script's globals."""
    player = ns["player"]
    return {
        "score": int(ns["score"]),
        "game_speed": float(ns["game_speed"]),
        "player": tuple(int(v) for v in player.rect),
        "vy": float(player.velocity_y),
        "gravity": float(player.gravity),
        "in_air": bool(player.Inair),
        "ducking": bool(player.ducking),
        "anim": (int(player.current_action), int(player.frame_index)),
        "obstacles": _rects(ns["obstacle_group"]),
        "birds": _rects(ns["enemy_group"]),
        "grounds": _rects(ns["ground_group"]),
        "next_target_arrival": float(ns["next_target_arrival"]),
        "next_is_bird": bool(ns["next_entity_is_bird"]),
        "alive": bool(ns["isAlive"]),
    }


def _observe_game(game: DinoGame) -> dict:
    """The same snapshot taken from the refactored simulation."""
    player = game.player
    return {
        "score": int(game.score),
        "game_speed": float(game.game_speed),
        "player": tuple(int(v) for v in player.rect),
        "vy": float(player.velocity_y),
        "gravity": float(player.gravity),
        "in_air": bool(player.Inair),
        "ducking": bool(player.ducking),
        "anim": (int(player.current_action), int(player.frame_index)),
        "obstacles": _rects(game.obstacle_group),
        "birds": _rects(game.enemy_group),
        "grounds": _rects(game.ground_group),
        "next_target_arrival": float(game.next_target_arrival),
        "next_is_bird": bool(game.next_entity_is_bird),
        "alive": bool(game.is_alive),
    }


def _run_original(max_frames: int, mode: str):
    """Run the untouched original script headlessly.

    Returns ``(trace, actions)`` where ``actions[i]`` is the action the original used
    for gameplay frame ``i + 1``.
    """
    fake_ticks = _FakeTicks()
    fake_keys = _FakeKeys()
    trace: list = []
    state = {"last_score": 0, "actions": []}
    rng = random.Random(SEED + 7)
    random_actions = [rng.choice([0, 0, 0, 1, 2]) for _ in range(max_frames + 5)]

    # A real (headless) display surface must exist so that convert_alpha() works; the
    # game itself blits into ``real_screen`` (a plain surface) as usual.
    pygame.display.init()
    pygame.display.set_mode((WIDTH, HEIGHT))
    real_screen = pygame.Surface((WIDTH, HEIGHT))

    patched = {
        "set_mode": pygame.display.set_mode,
        "flip": pygame.display.flip,
        "get_desktop_sizes": pygame.display.get_desktop_sizes,
        "get_ticks": pygame.time.get_ticks,
        "get_pressed": pygame.key.get_pressed,
        "event_get": pygame.event.get,
        "clock": pygame.time.Clock,
        "sound": pygame.mixer.Sound,
    }

    space_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
    calls = {"n": 0}

    def event_get(*args, **kwargs):
        # The original's menu loop blocks until SPACE arrives, so keep serving the
        # SPACE event while the game is dead (menu) and nothing while it is running.
        calls["n"] += 1
        if calls["n"] > 200_000:  # safety valve against an unexpected infinite loop
            raise _StopExecution
        if DEBUG and calls["n"] % 20000 == 0:
            _debug(f"event_get calls={calls['n']} isAlive={namespace.get('isAlive')}")
        if namespace.get("isAlive"):
            return []
        return [space_event]

    def flip():
        ns = namespace
        score = int(ns["score"])
        state["flips"] = state.get("flips", 0) + 1
        if DEBUG and state["flips"] % 1000 == 0:
            _debug(
                f"flips={state['flips']} trace={len(trace)} score={score} alive={bool(ns['isAlive'])} "
                f"player={tuple(ns['player'].rect)} obstacles={len(ns['obstacle_group'])}"
            )
        # A gameplay frame is the only kind where the score advanced by exactly one.
        # (Menu frames keep score at 0, or at the frozen value from the previous run,
        # and the death frame is recorded too because it is the last gameplay frame.)
        is_gameplay = score == state["last_score"] + 1
        if not is_gameplay:
            # The original draws one extra menu frame right after reset_level().
            if bool(ns["isAlive"]) and score == 0 and not state["actions"]:
                frame = _observe_namespace(ns)
                if mode == "random":
                    fake_keys.set_action(random_actions[0])
                    state["actions"].append(random_actions[0])
                else:
                    action = _controller(frame)
                    fake_keys.set_action(action)
                    state["actions"].append(action)
            return
        snapshot = _observe_namespace(ns)
        trace.append(snapshot)
        state["last_score"] = score
        index = len(trace) - 1
        # Match SimClock exactly: frame k (1-based) must run at round((k-1) * FRAME_MS),
        # so the clock has to be advanced to the *next* frame's timestamp here.
        fake_ticks.value = (index + 1) * FRAME_MS
        if not snapshot["alive"] or len(trace) >= max_frames:
            raise _StopExecution
        if mode == "random":
            action = random_actions[index + 1]
        else:
            action = _controller(snapshot)
        fake_keys.set_action(action)
        state["actions"].append(action)

    pygame.display.set_mode = lambda size, *a, **k: real_screen
    pygame.display.flip = flip
    pygame.display.get_desktop_sizes = lambda: [(WIDTH, HEIGHT)]
    pygame.time.get_ticks = fake_ticks
    pygame.time.Clock = _FakeClock
    pygame.key.get_pressed = lambda: fake_keys
    pygame.event.get = event_get
    pygame.mixer.Sound = lambda *a, **k: _FakeSound()

    namespace = {
        "__name__": "__dino_original__",
        "__file__": str(ROOT / "reference" / "original_main.py"),
    }

    real_open = builtins.open
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
    tmp.write("0")  # the original parses this file with int() and does not guard ValueError
    tmp.close()
    tmp_path = Path(tmp.name)

    def guarded_open(file, *args, **kwargs):
        try:
            name = os.fspath(file)
        except TypeError:
            name = ""
        if isinstance(name, str) and name.endswith("high_score.txt"):
            return real_open(tmp_path, *args, **kwargs)
        return real_open(file, *args, **kwargs)

    source = (ROOT / "reference" / "original_main.py").read_text()
    try:
        builtins.open = guarded_open
        random.seed(SEED)  # the original uses the global RNG
        exec(compile(source, "reference/original_main.py", "exec"), namespace)
    except _StopExecution:
        pass
    finally:
        builtins.open = real_open
        pygame.display.set_mode = patched["set_mode"]
        pygame.display.flip = patched["flip"]
        pygame.display.get_desktop_sizes = patched["get_desktop_sizes"]
        pygame.time.get_ticks = patched["get_ticks"]
        pygame.time.Clock = patched["clock"]
        pygame.key.get_pressed = patched["get_pressed"]
        pygame.event.get = patched["event_get"]
        pygame.mixer.Sound = patched["sound"]
        tmp_path.unlink(missing_ok=True)

    return trace, state["actions"]


def _run_refactored(actions: list, max_frames: int) -> list:
    """Replay the same action sequence through the refactored simulation."""
    config = GameConfig(width=WIDTH, height=HEIGHT, write_high_score=False)
    game = DinoGame(config=config, clock=SimClock(), rng=random.Random(SEED), visual=False)
    trace = []
    for i in range(max_frames):
        action = actions[i] if i < len(actions) else 0
        alive = game.step(action)
        snapshot = _observe_game(game)
        trace.append(snapshot)
        if not alive:
            break
    return trace


def _compare(name: str, original: list, refactored: list) -> int:
    """Compare two traces; returns the number of mismatches."""
    print(f"\n=== {name} ===")
    print(f"original frames: {len(original)}, refactored frames: {len(refactored)}")
    if len(original) != len(refactored):
        print(f"  !! trace length differs (first death/frame-budget mismatch)")
    mismatches = 0
    for index, (a, b) in enumerate(zip(original, refactored), start=1):
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, float) or isinstance(vb, float):
                same = abs(float(va) - float(vb)) <= TOLERANCE
            else:
                same = va == vb
            if not same:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  frame {index} field {key!r}: original={va!r} refactored={vb!r}")
    if mismatches == 0:
        print(f"  OK - {len(original)} frames identical (physics, hitboxes, spawns, death)")
    else:
        print(f"  FAIL - {mismatches} mismatching field(s)")
    return mismatches + abs(len(original) - len(refactored))


def main() -> int:
    failures = 0
    for mode, frames in (("random", 400), ("heuristic", 2500)):
        original, actions = _run_original(frames, mode)
        refactored = _run_refactored(actions, frames)
        failures += _compare(f"mode={mode}", original, refactored)

    print()
    if failures:
        print("EQUIVALENCE TEST FAILED")
        return 1
    print("EQUIVALENCE TEST PASSED: refactored simulation is frame-identical to the original.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

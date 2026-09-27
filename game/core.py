"""The Dino Run simulation, extracted from the original ``main.py`` main loop.

``DinoGame.step(action)`` performs exactly one iteration of the original ``while
running:`` block -- nothing was re-ordered, added or removed:

1. ``score += 1``
2. scroll both ground sprites and wrap them
3. the "entity arrival" spawn check (gap ``40 + randint(0, 50)``, birds only after
   ``score > 200`` with 40 % probability, birds get ``+15`` extra gap)
4. ``obstacle_group.update()`` (move left, die off-screen)
5. ``enemy_group.update()`` (move faster, animate, die off-screen)
6. ``player.update()`` (animation frame)
7. ``player.move(action)`` -> physics + collision + game-over
8. death bookkeeping (speed reset, high score) 
9. ``increment_game_speed()``

The only intentional deviation is that the game clock is injected, and that
``last_increment_time`` is re-based on every ``reset()``.  In the original the
timestamp was captured at *import* time, so idling in the menu for more than 10 s
made the very first speed-up fire immediately.  Re-basing makes episodes
reproducible (frame 600, 1200, ... get faster) and is the difference between a
deterministic RL environment and a wall-clock-dependent one.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Set

import pygame

from .assets import Assets
from .clock import Clock, SimClock
from .config import GameConfig, rl_config
from .entities import Bird, Ground, Obstacle, Player

# --- constants copied verbatim from the original ------------------------------ #
BIRD_SPAWN_SCORE_THRESHOLD = 200
BIRD_SPAWN_PERCENT = 40
BIRD_GAP_BONUS = 15
SPAWN_GAP_MIN = 40
SPAWN_GAP_RANDOM = 50
BASE_GAME_SPEED = 4.0
SPEED_INCREMENT = 0.5
SPEED_INCREMENT_INTERVAL_MS = 10000


class DinoGame:
    """One independent instance of the Dino Run simulation."""

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        clock: Optional[Clock] = None,
        rng: Optional[random.Random] = None,
        assets: Optional[Assets] = None,
        seed: Optional[int] = None,
        visual: bool = False,
        load_sounds: Optional[bool] = None,
    ) -> None:
        self.config = config or rl_config()
        self.clock: Clock = clock or SimClock()
        self.rng = rng if rng is not None else random.Random(seed)
        self.visual = visual
        self.assets = assets if assets is not None else Assets(self.config, visual=visual, load_sounds=load_sounds)

        # groups (the original created these once, at module scope)
        self.ground_group = pygame.sprite.Group()
        self.enemy_group = pygame.sprite.Group()
        self.obstacle_group = pygame.sprite.Group()

        self.high_score = 0
        self.score = 0
        self.game_speed = BASE_GAME_SPEED * self.config.ratio_x
        self.next_target_arrival: float = 0.0
        self.next_entity_is_bird = False
        self.is_alive = False
        self.last_increment_time = self.clock.ticks()
        self.distance = 0.0
        self.ground_1: Ground = None  # type: ignore[assignment]
        self.ground_2: Ground = None  # type: ignore[assignment]
        self.player: Player = None  # type: ignore[assignment]

        #: RL bookkeeping (never affects the simulation)
        self.reset_count = 0
        self.spawned_obstacles = 0
        self.spawned_birds = 0
        self.episode_seed: Optional[int] = None
        self._uid = 0

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None) -> None:
        """Equivalent of the original ``reset_level()`` (plus the ground/player reset)."""
        if seed is not None:
            self.rng.seed(seed)
            self.episode_seed = seed
        cfg = self.config
        self.clock.reset()
        self.score = 0
        self.next_target_arrival = 0.0
        self.next_entity_is_bird = False
        self.is_alive = True
        self.game_speed = BASE_GAME_SPEED * cfg.ratio_x  # Reset game speed
        self.distance = 0.0
        self.last_increment_time = self.clock.ticks()
        self.ground_group.empty()  # Clear ground group
        self.enemy_group.empty()  # Clear enemy group
        self.obstacle_group.empty()  # Clear obstacle group

        # Recreate ground and player instances
        self.ground_1 = Ground(self, 0)
        self.ground_2 = Ground(self, cfg.width)
        self.player = Player(self)

        self.ground_group.add(self.ground_1, self.ground_2)  # Add ground sprites to group

        self.spawned_obstacles = 0
        self.spawned_birds = 0
        self.reset_count += 1
        self._uid = 0
        if cfg.write_high_score:
            self.load_high_score()  # Load high score at the start of the game

    # ------------------------------------------------------------- simulation
    def step(self, action: int) -> bool:
        """Advance exactly one frame; returns ``True`` while the player is alive."""
        if not self.is_alive:
            raise RuntimeError("DinoGame.step() called after game over; call reset() first")

        self.clock.begin_frame()  # deterministic "now" for this frame
        self.score += 1

        # --- ground scroll + wrap
        self.ground_1.move()
        self.ground_2.move()
        if self.ground_1.rect.right <= 0:
            self.ground_1.rect.left = self.ground_2.rect.right
        if self.ground_2.rect.right <= 0:
            self.ground_2.rect.left = self.ground_1.rect.right
        self.distance += self.game_speed  # reporting only (distance = sum of per-frame scroll)

        # --- spawning logic based on arrival gaps to avoid impossible combinations
        self._spawn_if_due()

        # --- update entities (move + cull)
        self.obstacle_group.update()
        self.enemy_group.update()

        # --- player update and collision
        self.player.update()
        alive = self.player.move(action)
        self.is_alive = alive
        if not alive:
            self._on_death()

        self._increment_game_speed()
        return alive

    def _spawn_if_due(self) -> None:
        cfg = self.config
        # Speed of next entity
        entity_speed = self.game_speed + (5 * cfg.ratio_x if self.next_entity_is_bird else 0)
        dist_to_player = cfg.width - (50 * cfg.ratio_x)
        frames_to_arrive = dist_to_player / entity_speed
        expected_arrival_tick = self.score + frames_to_arrive

        if expected_arrival_tick >= self.next_target_arrival:
            if self.next_entity_is_bird:
                self.enemy_group.add(Bird(self))
                self.spawned_birds += 1
            else:
                self.obstacle_group.add(Obstacle(self))
                self.spawned_obstacles += 1

            # Decide what to spawn next and when
            gap = SPAWN_GAP_MIN + self.rng.randint(0, SPAWN_GAP_RANDOM)
            self.next_entity_is_bird = self.score > BIRD_SPAWN_SCORE_THRESHOLD and (
                self.rng.randint(1, 100) <= BIRD_SPAWN_PERCENT
            )
            if self.next_entity_is_bird:
                gap += BIRD_GAP_BONUS  # Birds need more gap because they move faster

            self.next_target_arrival = expected_arrival_tick + gap

    def _increment_game_speed(self) -> None:
        current_time = self.clock.ticks()
        if current_time - self.last_increment_time > SPEED_INCREMENT_INTERVAL_MS:
            self.game_speed += SPEED_INCREMENT * self.config.ratio_x
            self.last_increment_time = current_time

    def _on_death(self) -> None:
        self.game_speed = BASE_GAME_SPEED * self.config.ratio_x  # Reset game speed on death
        if self.config.write_high_score and self.score > self.high_score:
            self.save_high_score(self.score)

    # ------------------------------------------------------------- high score
    def load_high_score(self) -> None:
        """Load high score from file (same file/location semantics as the original)."""
        from .config import PROJECT_ROOT

        path = PROJECT_ROOT / self.config.high_score_path
        try:
            with open(path, "r") as file:
                self.high_score = int(file.read())
        except (FileNotFoundError, ValueError):
            self.high_score = 0  # Default value if file doesn't exist

    def save_high_score(self, score: int) -> None:
        from .config import PROJECT_ROOT

        path = PROJECT_ROOT / self.config.high_score_path
        with open(path, "w") as file:
            file.write(str(score))

    # ----------------------------------------------------------------- helpers
    def next_entity_uid(self) -> int:
        self._uid += 1
        return self._uid

    @property
    def entities(self) -> List[pygame.sprite.Sprite]:
        """Obstacles + birds as one list."""
        return list(self.obstacle_group) + list(self.enemy_group)

    def entities_ahead(self) -> List[pygame.sprite.Sprite]:
        """Entities that have not yet fully passed the player, nearest first."""
        player_right = self.player.rect.right
        ahead = [e for e in self.entities if e.rect.right > self.player.rect.left]
        ahead.sort(key=lambda e: e.rect.left - player_right)
        return ahead

    @property
    def ground_top(self) -> int:
        return self.ground_1.rect.top

    @property
    def frames(self) -> int:
        """Survived frames -- in the original this *is* the score."""
        return self.score

    @property
    def survival_seconds(self) -> float:
        return self.score / 60.0

    def active_uids(self) -> Set[int]:
        return {e.uid for e in self.entities}

    def summary(self) -> dict:
        return {
            "score": self.score,
            "frames": self.score,
            "survival_seconds": self.survival_seconds,
            "distance": self.distance,
            "game_speed": self.game_speed,
            "spawned_obstacles": self.spawned_obstacles,
            "spawned_birds": self.spawned_birds,
            "alive": self.is_alive,
        }

    def close(self) -> None:
        self.ground_group.empty()
        self.enemy_group.empty()
        self.obstacle_group.empty()


def describe_actions() -> Sequence[str]:
    from .actions import ACTION_NAMES

    return [ACTION_NAMES[i] for i in range(len(ACTION_NAMES))]

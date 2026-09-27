"""Game entities -- a faithful port of the classes in the original ``main.py``.

Every constant, expression and ordering from the original is preserved; only three
things were made injectable so the same code can serve human play and RL:

==============================  =========================================
original                        here
==============================  =========================================
``pygame.key.get_pressed()``    ``action`` argument (0/1/2)
``pygame.time.get_ticks()``     ``game.clock.ticks()``
module-level ``random.*``       ``game.rng`` (per-environment RNG)
==============================  =========================================

Nothing about sizes, speeds, gravity, hitboxes or animation timing was changed.  In
particular the original's slightly odd landing behaviour is kept verbatim: gravity is
integrated, then the ground collision snaps ``rect.bottom`` to the ground and resets
``velocity_y``, and the *pre-reset* ``dy`` is added to ``rect.y`` afterwards (which
leaves the dinosaur resting about 0.8 px inside the ground).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import pygame

from .actions import ACTION_DOWN, ACTION_JUMP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import DinoGame


class Ground(pygame.sprite.Sprite):
    """The scrolling ground strip (two instances tile the screen)."""

    def __init__(self, game: "DinoGame", x: int) -> None:
        super().__init__()
        self.game = game
        cfg = game.config
        image = game.assets.ground_surface()
        if image is not None:
            self.image: Optional[pygame.Surface] = image
            self.rect = self.image.get_rect()
        else:
            # Headless: the same rect as the scaled ground image would produce.
            self.image = None
            self.rect = pygame.Rect(0, 0, cfg.width, cfg.ground_pixels)
        self.rect.left = x
        self.rect.y = cfg.ground_top

    def move(self) -> None:
        self.rect.x -= self.game.game_speed

    def draw(self, surface: pygame.Surface) -> None:
        if self.image is not None:
            surface.blit(self.image, self.rect)


class Player(pygame.sprite.Sprite):
    """The dinosaur: jump, crouch, and fast-fall when "down" is pressed mid-air."""

    def __init__(self, game: "DinoGame") -> None:
        super().__init__()
        self.game = game
        cfg = game.config
        self.animation_names = ["running", "jump", "duck"]
        self.animation_list: List[tuple] = []
        self.load_animation()
        self.current_action = 0
        self.frame_index = 0
        self.image = self.animation_list[self.current_action][self.frame_index]
        self.rect = self.image.get_rect()

        self.rect.topleft = (50, cfg.height // 2)
        self.animation_cooldown = 80  # Milliseconds between frame updates
        self.last_update = game.clock.ticks()  # Track last update time
        self.gravity = 0.8 * cfg.ratio_y  # Gravity effect
        self.velocity_y = 0  # Vertical velocity for jumping
        self.Inair = False  # Check if player is in the air
        self.ducking = False  # Check if player is ducking

    def load_animation(self) -> None:
        """Loads all animations from the assets folder (shared, cached surfaces)."""
        for action in self.animation_names:
            self.animation_list.append(self.game.assets.dino_frames(action))

    def update(self) -> None:
        """Update the player animation frame."""
        if self.game.clock.ticks() - self.last_update > self.animation_cooldown:
            self.last_update = self.game.clock.ticks()
            self.frame_index += 1

        if self.frame_index >= len(self.animation_list[self.current_action]):
            self.frame_index = 0

        self.image = self.animation_list[self.current_action][self.frame_index]

    def change_animation(self, action: int) -> None:
        """Change the current animation action (re-derives the hitbox from the frame)."""
        if self.current_action != action:
            self.current_action = action
            self.frame_index = 0
            prev_midbottom = self.rect.midbottom
            self.image = self.animation_list[self.current_action][self.frame_index]
            self.rect = self.image.get_rect()
            self.rect.midbottom = prev_midbottom  # Lock to ground level

    def move(self, action: int) -> bool:
        """One frame of player physics/collision. Returns ``True`` while alive.

        ``action`` is one of :data:`game.actions.ACTION_NOTHING` / ``ACTION_JUMP`` /
        ``ACTION_DOWN``.  Down means crouch on the ground and *fast-fall* in the air,
        exactly like the original key handling.
        """
        cfg = self.game.config
        dy, alive = 0, True
        if action == ACTION_JUMP and not self.Inair:
            self.velocity_y = -14 * cfg.ratio_y  # Jumping effect
            self.Inair = True
            self.game.assets.play_sound("jump")  # Play jump sound (no-op when headless)
        if action == ACTION_DOWN:
            if self.Inair:
                self.gravity = 5 * cfg.ratio_y
            else:
                self.ducking = True
        else:
            self.ducking = False

        # Apply gravity
        self.velocity_y += self.gravity
        dy = self.velocity_y

        # Check for ground collision
        for ground in self.game.ground_group:
            if self.rect.colliderect(ground.rect) and dy >= 0:
                self.rect.bottom = ground.rect.top
                self.velocity_y = 0
                self.Inair = False
                self.gravity = 0.8 * cfg.ratio_y

        # Check for boxes collision
        for obstacle in self.game.obstacle_group:
            if self.rect.colliderect(obstacle.rect):
                alive = False

        # Check for bird collision
        for bird in self.game.enemy_group:
            if self.rect.colliderect(bird.rect):
                alive = False

        if self.ducking:
            self.change_animation(2)
        elif self.Inair:
            self.change_animation(1)
        else:
            self.change_animation(0)

        self.rect.y += dy
        return alive

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.image, self.rect)


class Bird(pygame.sprite.Sprite):
    """A flying obstacle; faster than the scrolling ground."""

    def __init__(self, game: "DinoGame") -> None:
        super().__init__()
        self.game = game
        cfg = game.config
        self.x = cfg.width
        self.y = game.rng.choice(
            [
                cfg.height - 40 * cfg.ratio_y - 32 * cfg.ratio,
                cfg.height - 40 * cfg.ratio_y - 60 * cfg.ratio,
                cfg.height - 40 * cfg.ratio_y - 80 * cfg.ratio,
            ]
        )  # Randomize bird height
        self.animation_list: List[pygame.Surface] = []
        self.load_animation()
        self.frame_index = 0
        self.image = self.animation_list[self.frame_index]
        self.rect = self.image.get_rect()
        self.rect.topleft = (self.x, self.y)
        self.animation_cooldown = 100  # Milliseconds between frame updates
        self.last_update = game.clock.ticks()
        self.speed = 5 * cfg.ratio_x  # Speed of the bird
        self.uid = game.next_entity_uid()
        self.variant = None  # boxes expose their image index; birds have none

    def load_animation(self) -> None:
        """Loads all animations from assets folder."""
        self.animation_list = list(self.game.assets.bird_frames())

    def update(self) -> None:
        """Update bird animation frame."""
        if self.game.clock.ticks() - self.last_update > self.animation_cooldown:
            self.last_update = self.game.clock.ticks()
            self.frame_index += 1

        if self.frame_index >= len(self.animation_list):
            self.frame_index = 0
        self.image = self.animation_list[self.frame_index]

        self.rect.x -= self.game.game_speed + self.speed
        if self.rect.right < 0:
            self.kill()

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.image, self.rect)


class Obstacle(pygame.sprite.Sprite):
    """A ground box; one of six randomly chosen images."""

    def __init__(self, game: "DinoGame") -> None:
        super().__init__()
        self.game = game
        cfg = game.config
        self.x, self.y = cfg.width, cfg.height - 40 * cfg.ratio_y  # Bottom of the screen
        image_name = game.rng.randint(1, 6)  # Randomly select an obstacle image
        self.variant = image_name
        self.image = game.assets.obstacle_image(image_name)
        self.rect = self.image.get_rect()
        self.rect.bottomleft = (self.x, self.y)
        self.uid = game.next_entity_uid()

    def update(self) -> None:
        """Update obstacle position."""
        self.rect.x -= self.game.game_speed
        if self.rect.right < 0:
            self.kill()

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.image, self.rect)

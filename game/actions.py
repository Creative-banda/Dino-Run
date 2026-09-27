"""The three player actions of the original game.

The original ``Player.move`` read the keyboard directly::

    if keys[pygame.K_UP] and not self.Inair:      # jump
    if keys[pygame.K_DOWN]:                       # crouch, or fast-fall when airborne
        if self.Inair: self.gravity = 5 * ratio_y
        else:          self.ducking = True

So "down" is deliberately *two* behaviours (crouch on the ground, faster falling in
the air).  Both are preserved: the RL agent only ever picks one of these three
abstract actions, and the game decides what "down" means from the player's state.
"""

from __future__ import annotations

ACTION_NOTHING = 0
ACTION_JUMP = 1
ACTION_DOWN = 2

NUM_ACTIONS = 3

ACTION_NAMES = {
    ACTION_NOTHING: "nothing",
    ACTION_JUMP: "jump",
    ACTION_DOWN: "down",
}


def action_name(action: int) -> str:
    return ACTION_NAMES.get(int(action), f"invalid({action})")


def action_from_keys(keys=None) -> int:
    """Map the live keyboard state to an action, with the original semantics.

    ``K_UP`` takes precedence over ``K_DOWN`` (jump is checked first in the original).
    Note the one unavoidable difference from the original: a human can hold *both*
    keys at once, which the original turns into "jump, then immediately fast-fall in
    the same frame".  A discrete action cannot express that, but the agent can
    reproduce the effect by issuing ``ACTION_DOWN`` on the following frame.
    """
    import pygame

    if keys is None:
        keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return ACTION_JUMP
    if keys[pygame.K_DOWN]:
        return ACTION_DOWN
    return ACTION_NOTHING

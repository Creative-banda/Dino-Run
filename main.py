"""Dino Run -- entry point for the normal, human-playable game.

This file used to contain the whole game; it now lives in the ``game`` package so
that the reinforcement-learning environment can drive the exact same simulation
(see ``game/core.py``).  Nothing about the game itself changed -- running this
script still starts the identical game with:

* the same window size (your desktop resolution),
* the same parallax background, sprites, animations, ground and score display,
* the same sounds and music,
* the same "Press SPACE to Start" menu,
* the same arrow-key controls (UP = jump, DOWN = crouch, or fast-fall in the air).

The untouched pre-RL version of this script is kept in
``reference/original_main.py`` and is used by ``tests/test_physics_equivalence.py``
to prove the refactor is behaviour-identical.

Usage
-----
    python main.py                       # desktop-sized window, exactly like before
    python main.py --resolution 800x600  # explicit window size (optional)

To watch a trained PPO agent play this same game::

    python training/play_agent.py --model models/ppo_dino_best
"""

from game.human_play import main

if __name__ == "__main__":
    main()

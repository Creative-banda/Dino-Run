# Dino Run

## About The Game

Dino Run is a side-scrolling endless runner game inspired by the Chrome browser's offline dinosaur game. The player controls a dinosaur that must jump over obstacles like cacti and avoid flying pterodactyls to achieve the highest score possible.

## Features

- Simple, addictive gameplay
- Increasing difficulty as game progresses
- Score tracking system
- Day and night cycle
- Sound effects and background music
- Collision detection
- High score saving

## Technologies Used

- Python
- Pygame library for game development
- Object-oriented programming principles

## How to Play

1. **Installation**:
   ```bash
   # Clone the repository
   git clone https://github.com/Creative-banda/Dino-Run.git
   
   # Navigate to the game directory
   cd Dino-Run
   
   # Install required dependencies
   pip install pygame

   # Run game
   python main.py
   ```

## Reinforcement Learning (PPO)

This repository also contains a complete reinforcement-learning system that teaches a PPO
agent to play *this* game (the environment is a headless wrapper around the same
simulation, so the game itself is unchanged). See **[README_RL.md](README_RL.md)** for the
full documentation.

```bash
python -m pip install -r requirements.txt
python training/smoke_test.py --timesteps 4096 --envs 4      # validate the pipeline
python training/benchmark_envs.py                            # how many envs can this machine run?
python training/train_ppo.py --timesteps 3000000 --envs auto --frame-skip 4
python training/evaluate_ppo.py --model models/ppo_dino_v1 --episodes 20 --frame-skip 4
python training/play_agent.py --model models/ppo_dino_v1     # watch the agent play
```

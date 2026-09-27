# Running the Dino Run RL system on Google Colab

Two ways to use this: the notebook (`dino_run_ppo_colab.ipynb`) or the same steps by hand.
Both assume a **T4 GPU runtime** (`Runtime ▸ Change runtime type ▸ T4 GPU`); CPU runtime works
too, just slower.

## 0. Build the bundle locally

```bash
python training/make_bundle.py --verify          # -> dist/dino_run_rl_colab.zip (proves it runs)
```

`--verify` unpacks the archive into a temporary directory and runs a real headless rollout
from inside the extracted copy, so a missing file fails locally instead of on Colab.

## 1. Colab cell: get the project in and install

```python
from google.colab import files
uploaded = files.upload()                        # pick dist/dino_run_rl_colab.zip
!unzip -q -o dino_run_rl_colab.zip -d /content/dino_rl
%cd /content/dino_rl
!python -m pip install -q -r requirements.txt
```

## 2. Colab cell: headless check

```python
!python -c "from game.headless import describe_display; from rl.environment import DinoRunEnv; \
env = DinoRunEnv(frame_skip=4, seed=3); obs, info = env.reset(seed=3); \
print(describe_display()); print('obs', obs.shape, env.observation_space.contains(obs))"
```

## 3. Colab cell: benchmark the parallel environment count

```python
!python -u training/benchmark_envs.py --envs 1,2,4,8,16,32 --steps 12000 --max-seconds 15 \
    --backends dummy,process --ppo-probe 2 --ppo-steps 8192 --device auto \
    --json models/benchmark_envs.json
```

Read the printed recommendation (or `models/benchmark_envs.json ▸ recommendation`). On a
2-vCPU Colab runtime it typically lands between 8 and 32 environments with the `process`
backend; on a 12-core local machine it chose 32.

## 4. Colab cell: smoke test (the gate)

```python
!python -u training/smoke_test.py --timesteps 4096 --envs 4 --device auto
```

Do not continue unless every check prints `PASS` (CUDA checks are `SKIP`ped on a CPU runtime).

## 5. Colab cell: short training run, then evaluate

```python
N_ENVS = 16            # from the benchmark
!python -u training/train_ppo.py --timesteps 200000 --envs {N_ENVS} --backend process \
    --frame-skip 4 --n-steps 512 --batch-size 512 --n-epochs 6 --gamma 0.997 \
    --ent-coef 0.005 --learning-rate 5e-4 --eval-every 25000 --checkpoint-every 100000 \
    --device auto --seed 42 --output models/ppo_colab_short

!python -u training/evaluate_ppo.py --model models/ppo_colab_short --episodes 10 --seed 1234 \
    --frame-skip 4 --json models/ppo_colab_short/evaluation.json
```

## 6. Colab cell: the real run (resumable)

```python
!python -u training/train_ppo.py --timesteps 3000000 --envs {N_ENVS} --backend process \
    --frame-skip 4 --n-steps 512 --batch-size 512 --n-epochs 6 --gamma 0.997 \
    --ent-coef 0.005 --learning-rate 5e-4 --eval-every 250000 --eval-episodes 5 \
    --checkpoint-every 500000 --device auto --seed 42 --output models/ppo_dino_colab

# if the runtime restarts, continue where it stopped:
!python -u training/train_ppo.py --timesteps 3000000 --resume models/ppo_dino_colab/latest.zip \
    --envs {N_ENVS} --backend process --frame-skip 4 --output models/ppo_dino_colab
```

Mount Drive first if you want the checkpoints to survive a disconnect:

```python
from google.colab import drive; drive.mount('/content/drive')
# then use --output /content/drive/MyDrive/ppo_dino_colab
```

## 7. Colab cell: evaluate and download

```python
!python -u training/evaluate_ppo.py --model models/ppo_dino_colab --episodes 20 --seed 1234 \
    --frame-skip 4 --json models/ppo_dino_colab/evaluation.json
!zip -qr ppo_dino_colab_results.zip models/ppo_dino_colab
from google.colab import files; files.download('ppo_dino_colab_results.zip')
```

## 8. Watch it play (locally, not in Colab)

```bash
# real window, same game as `python main.py`
python training/play_agent.py --model models/ppo_dino_colab

# off-screen variant that works in Colab (no display)
python training/play_agent.py --model models/ppo_dino_colab --selftest --frames 1200
```

## Notes

* **CPU does the simulation, the GPU does the network.** The policy is a 14,624-parameter MLP
  and the environment is Pygame; moving the simulation to the GPU is not possible and would
  not help. If a run is slow, that is a CPU/worker issue, not a GPU issue.
* **Never guess the environment count** — step 3 measures end-to-end PPO samples/s, because
  with a tiny network the PPO update can dominate the wall clock.
* **`best/best_model.zip` is what evaluation and playback load**; `latest.zip` is what
  `--resume` continues from.

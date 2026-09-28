# Running the Dino Run RL system on Google Colab

The notebook (`dino_run_ppo_colab.ipynb`) is the supported path: it clones this repository
from GitHub, measures the Colab runtime, and only then trains. This file is the same
workflow as copy-pasteable commands, plus the parts that are easy to get wrong.

It assumes the RL code is on `main` of <https://github.com/Creative-banda/Dino-Run>. A GPU
runtime (`Runtime ▸ Change runtime type ▸ T4 GPU`) is nicer but not required — the pygame
simulation runs on the CPU either way, and the notebook detects whatever it got.

## 0. Get the code

```bash
git clone --depth 1 --branch main https://github.com/Creative-banda/Dino-Run.git /content/Dino-Run
cd /content/Dino-Run && python -m pip install -q -r requirements.txt
```

Never `pip install torch` yourself on Colab: the preinstalled wheel is a CUDA build and
`requirements.txt` only asks for `torch>=2.0`, so pip leaves it alone.

## 1. Detect the runtime, then verify the environment

```python
import os; print(os.cpu_count())
import torch; print(torch.__version__, torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")
```

```bash
python -c "from game.headless import describe_display; from rl.environment import DinoRunEnv; \
env = DinoRunEnv(frame_skip=4, seed=3); obs, info = env.reset(seed=3); \
print(describe_display()); print(obs.shape, env.observation_space.contains(obs), env.describe()['obs_version'])"
```

## 2. Benchmark the parallel environment count (never guess it)

```bash
python -u training/benchmark_envs.py --envs 1,2,4,8,16,32 --steps 12000 --max-seconds 15 \
    --backends dummy,process --frame-skip 4 --max-episode-steps 6000 \
    --n-steps 512 --batch-size 512 --ppo-probe 3 --ppo-steps 8192 --device auto \
    --json /content/dino_run_work/benchmark_envs.json
```

Write the report **outside** the repository: `models/benchmark_envs.json` is committed (the
local machine's numbers) and a later `git reset --hard` would put those back.

Read `recommendation.n_envs` / `recommendation.backend`. On a free T4 runtime (~2 vCPUs) it
usually lands between 2 and 8 with the `process` backend; the 12-core machine this project
was developed on chose 32. The recommendation is based on measured **PPO samples/s**, not on
raw stepping speed, because the policy update is a large part of the wall clock.

## 3. Smoke test (the gate)

```bash
python -u training/smoke_test.py --timesteps 4096 --envs 4 --device auto --output /content/dino_run_work/smoke
```

Do not continue unless it ends with `SMOKE TEST PASSED` (CUDA checks are `SKIP`ped on a CPU
runtime).

## 4. Short run, then evaluate it (200k–500k steps)

```bash
N_ENVS=8        # from the benchmark
SHORT_OUT=/content/drive/MyDrive/dino_run_ppo/ppo_dino_colab_short

python -u training/train_ppo.py --timesteps 300000 --envs $N_ENVS --backend process \
    --seed 42 --device auto --frame-skip 4 --max-episode-steps 6000 \
    --n-steps 512 --batch-size 512 --n-epochs 6 --learning-rate 5e-4 --gamma 0.997 \
    --gae-lambda 0.95 --clip-range 0.2 --ent-coef 0.005 --vf-coef 0.5 --target-kl 0.05 \
    --net-arch 64,64 --eval-every 25000 --eval-episodes 5 --checkpoint-every 100000 \
    --output "$SHORT_OUT"

python -u training/evaluate_ppo.py --model "$SHORT_OUT" \
    --episodes 20 --seed 1234 --frame-skip 4 \
    --json "$SHORT_OUT/evaluation.json"
```

The short run also lands on Google Drive, so even this wiring-check run keeps its history;
delete `ppo_dino_colab_short` in Drive if you do not want it.

`--frame-skip 4` everywhere, so training and evaluation stay comparable with the local
numbers. The evaluation prints the agent next to the `random`, `do_nothing` and
`geometric_heuristic` baselines.

## 5. The real run — resumable, checkpoints on Drive

Mount Drive first, then train into it:

```python
from google.colab import drive; drive.mount('/content/drive')
```

Everything the run produces is written under `/content/drive/MyDrive/dino_run_ppo/`:

| path | what |
|---|---|
| `ppo_dino_colab/` | the long run: `latest.zip`, `best/best_model.zip`, `checkpoints/`, `run_config.json`, `evaluations.npz`, `evaluation*.json` |
| `ppo_dino_colab/tensorboard/` | SB3 event files, only when `--tensorboard-log` is passed (or `USE_TENSORBOARD = True` in the notebook) |
| `ppo_dino_colab/session_manifest.jsonl` | one JSON line per training session (start/end, steps before/after, resume source) |
| `logs/` | per-session console logs, the benchmark report and the smoke-test log |

Nothing is pruned: every periodic checkpoint stays, so the full experiment history survives.

```bash
OUT=/content/drive/MyDrive/dino_run_ppo/ppo_dino_colab
LOGS=/content/drive/MyDrive/dino_run_ppo/logs

python -u training/train_ppo.py --timesteps 10000000 --envs $N_ENVS --backend process \
    --seed 42 --device auto --frame-skip 4 --max-episode-steps 6000 \
    --n-steps 512 --batch-size 512 --n-epochs 6 --learning-rate 5e-4 --gamma 0.997 \
    --gae-lambda 0.95 --clip-range 0.2 --ent-coef 0.005 --vf-coef 0.5 --target-kl 0.05 \
    --net-arch 64,64 --eval-every 250000 --eval-episodes 5 --checkpoint-every 250000 \
    --tensorboard-log "$OUT/tensorboard" \
    --output "$OUT" 2>&1 | tee "$LOGS/train_manual.log"
```

**After a disconnect or a runtime restart**, mount Drive again, clone again, install again,
**re-run the benchmark** (the new runtime may be a different machine), then inspect what is
already there and continue:

```bash
# how far did the run get?
python -c "import json; c = json.load(open('$OUT/run_config.json')); print(c.get('trained_timesteps'), 'steps trained')"

tail -n 1 "$OUT/session_manifest.jsonl"   # last session summary

python -u training/train_ppo.py --timesteps <REMAINING> --resume "$OUT/latest.zip" \
    --envs $N_ENVS --backend process --seed 42 --device auto --frame-skip 4 \
    --n-steps 512 --batch-size 512 --n-epochs 6 --gamma 0.997 --gae-lambda 0.95 \
    --clip-range 0.2 --ent-coef 0.005 --vf-coef 0.5 --target-kl 0.05 --net-arch 64,64 \
    --eval-every 250000 --eval-episodes 5 --checkpoint-every 250000 --output "$OUT"
```

* `--timesteps` with `--resume` means **additional** steps. `run_config.json ▸
  trained_timesteps` (or the checkpoint file names) tell you how many are done; the
  notebook computes `<REMAINING> = TOTAL_TIMESTEPS - trained` for you.
* Do **not** pass `--learning-rate` on a resume unless you mean it: without it the
  checkpoint's learning rate is kept, and continuing a converged policy at its original
  rate is the usual way to destroy it.
* SB3 finishes the rollout it is in, so the total can overshoot by up to
  `n_steps × n_envs` steps. Expected, not a bug.
* `checkpoints/dino_ppo_*_steps.zip` are periodic snapshots, `latest.zip` is the most
  recent state (what `--resume` uses), `best/best_model.zip` is the best *evaluated*
  policy (what evaluation and playback use). The best model never regresses: the best
  evaluation reward is stored in `run_config.json` and restored on resume.

## 6. Evaluate and download

```bash
python -u training/evaluate_ppo.py --model "$OUT" --episodes 20 --seed 1234 \
    --frame-skip 4 --json "$OUT/evaluation.json"

# the most recent state as well, when it differs from the best evaluated one
python -u training/evaluate_ppo.py --model "$OUT/latest.zip" --episodes 20 --seed 1234 \
    --frame-skip 4 --no-baselines --json "$OUT/evaluation_latest.json"

cd "$OUT/.." && zip -qr dino_run_ppo_results.zip ppo_dino_colab
from google.colab import files; files.download('/content/drive/MyDrive/dino_run_ppo/dino_run_ppo_results.zip')
```

## 7. Back on your machine

```bash
unzip dino_run_ppo_results.zip -d models/ppo_dino_colab

# watch it play the real game (real window, same game as `python main.py`)
python training/play_agent.py --model models/ppo_dino_colab

# reproduce the Colab numbers
python training/evaluate_ppo.py --model models/ppo_dino_colab --episodes 20 --seed 1234 --frame-skip 4

# off-screen sanity check, no display needed
python training/play_agent.py --model models/ppo_dino_colab --selftest --frames 1200

# and the game itself, unchanged
python main.py
```

The observation is resolution independent (`dino-obs-v1`, 47 normalised floats), so a model
trained at 800×600 also plays a 1024×768 or 2560×1440 window.

## Notes

* **CPU does the simulation, the GPU does the network.** The policy is a ~15k-parameter MLP
  and the environment is pygame; moving the simulation to the GPU is impossible and would
  not help. If a run is slow, that is a CPU/worker problem, not a GPU problem.
* **Nothing about the game changes.** Observation, reward, action space
  (0 = nothing, 1 = jump, 2 = down — crouch on the ground, fast-fall in the air), physics and
  rendering are the code in the repository.
* **Local reference** (3M steps, 32 envs, seed 42): PPO mean score ≈ 5,200 (median 520, best
  24,000) vs random ≈ 189, do-nothing ≈ 181 and the scripted `geometric_heuristic` ≈ 12,300.
  Beating the heuristic is the real bar — a short Colab run will not.
* Building an offline bundle (`python training/make_bundle.py --verify`) still works, but the
  notebook now clones from GitHub and does not need it.

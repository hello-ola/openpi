# Fine-tuning π0.5 on Stretch 4 demos

### 1. Record demos (robot)

```bash
# on the robot
bash -ic '~/venvs/lerobot21/bin/python ~/stretch4_data_recorder/recorder_gui.py'
# then open http://<robot_ip>:8765 — Start / task prompt / Stop / Keep or Discard
```

### 2. Push the dataset to Hugging Face (robot)

```bash
# on the robot
~/venvs/lerobot21/bin/python ~/stretch4_data_recorder/push_to_hf.py
```
> **Note:** The name you push under here is what you use everywhere below —
> replace `stretch4_demos_v2` in steps 3 and 4 with whatever name you actually used.

### 3. Refresh the desktop's local copy (desktop)

The lerobot loader is **local-first**: if a copy exists in the cache it will silently
use it and never check the Hub. So after every re-push, delete the cache first:

```bash
rm -rf ~/.cache/huggingface/lerobot/<your-hf-username>/stretch4_demos_v2
cd ~/openpi
.venv/bin/python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
d = LeRobotDataset('<your-hf-username>/stretch4_demos_v2')
print(d.num_episodes, 'episodes,', d.num_frames, 'frames')"
```

Check the printed episode/frame counts match what the push reported.
(Needs HF login: `hf auth whoami` should say your username)

### 4. Recompute norm stats (desktop) — REQUIRED whenever the data changed

```bash
cd ~/openpi
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_stretch
```

Writes `assets/pi05_stretch/<your-hf-username>/stretch4_demos_v2/norm_stats.json`. Stats are baked
into how the model normalizes state/actions — training with stats from a different
data mix silently hurts the policy. Rule: every cache refresh → recompute → then train.

### 5. Train (desktop)

```bash
cd ~/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_stretch \
    --exp-name=run2 --overwrite
```

- `--exp-name` names the run: checkpoints land in `checkpoints/pi05_stretch/run2/<step>/`.
  Use a fresh name per data version (`run3`, `run4`, ...).
- `--overwrite` restarts from scratch; use `--resume` instead to continue an interrupted run.
- wandb: first time either `wandb login`, or prefix `WANDB_MODE=offline` to skip it.

STEPS 6 and 7 run on the desktop and robot in parallel
### 6. Serve the policy (desktop, after training)

```bash
~/openpi/serve_stretch.sh          # serves the newest checkpoint (or: ~/serve_stretch.sh 15000 if you want to serve a specific checkpoint)
```

`RUN_DIR` inside that script is hardcoded — **point it at the run you actually
trained**

which is shorthand for:

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_stretch \
    --policy.dir=checkpoints/pi05_stretch/run1/<newest step>
```

Listens on port 8000. The script picks the newest checkpoint automatically
(intermediates get cleaned up during training, so a hard-coded step goes stale).

### 7. Run inference on the robot

With the policy server up (step 6):

```bash
# on the robot
~/venvs/lerobot21/bin/python ~/stretch4_data_recorder/run_inference.py \
    --task "your task prompt"
```

The client streams both cameras + the 10-D state to the server at 10 Hz and
executes the returned 10-step action chunks (re-plans every 5 steps by
default; `--replan-steps` changes this). Useful flags:

- `--dry-run` — query the policy and print the actions, never move the robot.
  Run this first after any retrain/config change.
- `--no-base` — arm/wrist/gripper only, base stays put.
- `--max-seconds N` — auto-stop; otherwise runs until Ctrl-C (which halts
  motion safely).


# RobotLab Repository Usage Rules

## Workspace

Repository:

```text
/home/lxq/Softwares/robot_lab
```

Deployment repository:

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2
```

Main highstep train command:

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless
```

Main highstep play command:

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/play.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --num_envs 1 \
  --real-time \
  --keyboard \
  --checkpoint /absolute/path/to/model.pt
```

For bodyflat/sidestep teacher play without action prior:

```bash
python scripts/rsl_rl/base/play.py \
  --task RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --num_envs 1 \
  --real-time \
  --keyboard \
  --disable_action_prior \
  --checkpoint /absolute/path/to/model.pt
```

## Backup Rule

Before modifying code, back up every changed file.

Preferred backup suffix:

```text
_<relevant_previous_or_current_log_folder>
```

Example:

```text
highstep_env_cfg_2026-06-17_03-05-16.py
rewards_2026-06-17_03-05-16.py
rough_2026-06-17_03-05-16.py
```

Do not make large reward/terrain/curriculum edits without a backup.

## Long Training Rule

Avoid training with `--video` for long runs. It has repeatedly caused or contributed to desktop/VSCode/Isaac freezes.

Use video only for short diagnostic runs or record separate `play.py` sessions.

If training crashes after video:

- inspect latest checkpoint;
- inspect event file mtime;
- resume from latest complete checkpoint, not a nonexistent partial one.

## NVIDIA/GPU Note

The user's real terminal can run `nvidia-smi` normally. The assistant's sandbox may not see `/dev/nvidia*`, so `nvidia-smi` failure inside the assistant tool is not evidence that the real driver is broken.

Use the user's terminal `nvidia-smi` as the reliable GPU-status source.

## Hydra Override Notes

Viewer override errors occurred with direct `viewer.*`. Correct namespace may need `env.viewer.*`, but some fields typed as `NoneType` reject string overrides.

Avoid complicated viewer overrides during long training.

## Killing Stuck Training

If Ctrl-C does not work in a terminal:

```bash
pgrep -af 'train.py|play.py|isaac|kit'
kill <pid>
kill -9 <pid>
```

Use `kill -9` only for truly stuck processes.


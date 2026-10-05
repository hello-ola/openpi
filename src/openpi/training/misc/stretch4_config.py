"""Stretch 4 π0.5 fine-tuning configs, defined in configs/stretch4.yaml.

Each entry under `experiments:` becomes a normal openpi config name, so
compute_norm_stats.py, train.py and serve_policy.py all work by name, and a new dataset
or base mode is a YAML edit, not a code change. Another YAML file can be used with
OPENPI_STRETCH4_CONFIG=/path/to/file.yaml.

"""

import os
import pathlib
from typing import Any

import yaml

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

DEFAULT_PATH = pathlib.Path(__file__).resolve().parents[4] / "configs" / "stretch4.yaml"

# Every key an experiment may set, with its default (defaults: in the YAML override these).
DEFAULTS: dict[str, Any] = {
    "repo_id": None,             # LeRobot v2.1 dataset, e.g. <hf-user>/stretch4_mouse
    "base": "delta",             # delta | velocity  (velocity needs --base-action velocity data)
    "gripper": "absolute",       # absolute | relative
    "head_cameras": None,        # [head_left, head_right], [head_right], ...; None: all the dataset has
    "action_horizon": 30,        # steps per chunk; 30 = 1 s at 30 Hz
    "batch_size": 32,
    "num_train_steps": 30_000,
    "peak_lr": 5e-5,
    "warmup_steps": 1_000,
    "decay_lr": 5e-6,
    "save_interval": 1_000,
    "keep_period": 5_000,        # checkpoints kept permanently every N steps
    # Held joints, decided by compute_norm_stats.py as diffusion_policy and stretch4_fix_stats do
    # (stretch4_policy_recorder.common.held_joints): a joint moved in fewer than min_episode_frac
    # of the episodes is held still. min_motion: {joint: range that counts as moved} over the
    # shared defaults; hold_joints always held, keep_joints never; auto_hold: false = hold_joints only.
    "min_episode_frac": 0.1,
    "min_motion": {},
    "hold_joints": [],
    "keep_joints": [],
    "auto_hold": True,
}


def _model(action_horizon: int) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=action_horizon,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )


def _load(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    """{config name: fully resolved settings}, validated."""
    doc = yaml.safe_load(path.read_text()) or {}
    unknown_top = set(doc) - {"defaults", "experiments"}
    if unknown_top:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown_top)}")
    defaults = {**DEFAULTS, **(doc.get("defaults") or {})}
    out = {}
    for name, entry in (doc.get("experiments") or {}).items():
        settings = {**defaults, **(entry or {})}
        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"{path}: experiment {name!r} has unknown keys {sorted(unknown)}; "
                             f"allowed: {sorted(DEFAULTS)}")
        if not isinstance(settings["min_motion"] or {}, dict):
            raise ValueError(f"{path}: experiment {name!r}: min_motion is now per joint, e.g. "
                             "{wrist_roll: 0.02} (the shared hold rule); a single number is the "
                             "pre-conformance rule")
        if not settings["repo_id"]:
            raise ValueError(f"{path}: experiment {name!r} needs repo_id")
        out[name] = settings
    return out


def get_stretch4_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotStretch4DataConfig
    from openpi.training.config import TrainConfig
    from openpi.training.config import stretch4_dataset_head_cameras

    path = pathlib.Path(os.environ.get("OPENPI_STRETCH4_CONFIG", DEFAULT_PATH))
    if not path.exists():
        return []

    configs = []
    for name, s in _load(path).items():
        model = _model(s["action_horizon"])
        heads = tuple(s["head_cameras"]) if s["head_cameras"] else None
        # For the robot client: the camera order the policy was trained with. Known here when set
        # in the YAML or when the dataset is local (the training/serving machine); otherwise the
        # client falls back to the cameras of its --dataset-root.
        announced = heads or stretch4_dataset_head_cameras(s["repo_id"])
        configs.append(TrainConfig(
            name=name,
            model=model,
            data=LeRobotStretch4DataConfig(
                repo_id=s["repo_id"],
                head_cameras=heads,
                base=s["base"],
                gripper=s["gripper"],
                min_episode_frac=float(s["min_episode_frac"]),
                min_motion={str(j): float(v) for j, v in (s["min_motion"] or {}).items()},
                hold_joints=tuple(s["hold_joints"] or ()),
                keep_joints=tuple(s["keep_joints"] or ()),
                auto_hold=bool(s["auto_hold"]),
                base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",)),
            ),
            # Read by the robot client (stretch4_policy_recorder's openpi backend) to decode
            # the base and to check it is talking to the policy it expects.
            policy_metadata={
                "stretch_base_actions": s["base"],
                "stretch_gripper": s["gripper"],
                "stretch_head_cameras": list(announced) if announced else None,
            },
            batch_size=s["batch_size"],
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=s["warmup_steps"],
                peak_lr=s["peak_lr"],
                decay_steps=s["num_train_steps"],
                decay_lr=s["decay_lr"],
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=s["num_train_steps"],
            save_interval=s["save_interval"],
            keep_period=s["keep_period"],
            freeze_filter=model.get_freeze_filter(),
            # EMA off for LoRA fine-tuning (openpi convention).
            ema_decay=None,
        ))
    return configs

import dataclasses
import json
import pathlib

import einops
import numpy as np
from stretch4_policy_recorder.common.columns import POSE_JOINTS
from stretch4_policy_recorder.common.columns import check_state_names
from stretch4_policy_recorder.common.held_joints import decide_held_joints
from stretch4_policy_recorder.common.held_joints import format_hold_report
from stretch4_policy_recorder.common.stats_floor import floor_stats

from openpi import transforms
from openpi.models import model as _model


def make_stretch_example() -> dict:
    """Creates a random input example for the Stretch policy (stretch4_policy_recorder layout)."""
    return {
        "observation/state": np.random.rand(9),
        "observation/image": np.random.randint(256, size=(512, 320, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(200, 320, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class StretchInputs(transforms.DataTransformFn):
    """Maps Stretch 4 data (stretch4_policy_recorder, stretch4_to_lerobot --format v21) to the
    model input format.

      observation/state (9):   [base_vx, base_vy, base_omega, lift, arm, wrist_yaw, wrist_pitch,
                                wrist_roll, grip_mm] (stretch4_policy_recorder's
                                POLICY_STATE_COLUMNS: the action's order, so state[i] and
                                action[i] are the same quantity), plus grip_effort last (10)
                                for a --grip-effort export.
      observation/base_pose:   [base_x, base_y, base_theta] (dataset observation.base_odom), only
                               for base="delta" training, as the anchor of the base targets
      actions (9):             the targets K frames ahead, [base_x, base_y, base_theta, lift, arm,
                               wrist x3, grip_mm]. The base dims are the robot's displacement since
                               the episode started (its pose in the episode-start frame);
                               StretchRelativeActions turns them into the displacement from the pose
                               at the observation. Joints and gripper are joint positions. With
                               --base-action velocity dims 0:3 are body-frame vx, vy, omega instead.
      observation/image:       the first head camera (head_left or head_right), cropped and
                               resized by the converter; inference must use the same preprocessing.
      observation/image2:      the second head camera, when the dataset has both eyes (optional).
      observation/wrist_image: the gripper camera.

    π0.5 has three image slots: base_0_rgb = first head, left_wrist_0_rgb = gripper,
    right_wrist_0_rgb = second head (masked out as padding when there is only one head).

    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        second_head = "observation/image2" in data

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # The second head eye, or padding (zeros shaped like the wrist image).
                "right_wrist_0_rgb": (_parse_image(data["observation/image2"]) if second_head
                                      else np.zeros_like(wrist_image)),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Padding images are masked for pi0/pi0.5 but not pi0-FAST.
                "right_wrist_0_rgb": (np.True_ if second_head or self.model_type == _model.ModelType.PI0_FAST
                                      else np.False_),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # delta base only, training only: the anchor for the base targets. Consumed (and
        # removed) by StretchRelativeActions, so it never reaches the model.
        if "observation/base_pose" in data:
            inputs["base_pose"] = np.asarray(data["observation/base_pose"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class StretchOutputs(transforms.DataTransformFn):
    """Maps model outputs back to the Stretch 4 action space (inference only)."""

    # The 9 real action dims (base 3, lift, arm, wrist 3, gripper).
    action_dim: int = 9
    # Held joint names (see held_joint_dims): sent to the client, which keeps them where they are.
    held: tuple[str, ...] = ()

    def __call__(self, data: dict) -> dict:
        # The model pads actions to its internal width; only the first action_dim dims are real.
        return {"actions": np.asarray(data["actions"][..., : self.action_dim]),
                "stretch_held": list(self.held)}


# Index layout shared by the transforms below. The state is in the action's order, so the same
# index is the same quantity in both (the state may add grip_effort as a 10th value).
#   action (9): [base_x|vx, base_y|vy, base_theta|omega, lift, arm, wrist_yaw, wrist_pitch,
#                wrist_roll, grip_mm]
#   state  (9): [base_vx, base_vy, base_omega, lift, arm, wrist_yaw, wrist_pitch, wrist_roll,
#                grip_mm] (+ grip_effort)
ACTION_BASE = slice(0, 3)
ACTION_JOINTS = slice(3, 8)    # lift .. wrist_roll
ACTION_GRIPPER = 8
STATE_JOINTS = ACTION_JOINTS
STATE_GRIPPER = ACTION_GRIPPER

BASE_MODES = ("delta", "velocity")
GRIPPER_MODES = ("absolute", "relative")

# Written next to norm_stats.json by compute_norm_stats.py (finalize_norm_stats) and saved with
# every checkpoint: the joints held still, decided as diffusion_policy and stretch4_fix_stats do.
HELD_FILE = "held_joints.json"


def held_joint_dims(held_names) -> tuple[int, ...]:
    """Held joint names (lift .. grip_mm) -> their action dims (the same dims in the state)."""
    unknown = [j for j in held_names if j not in POSE_JOINTS]
    if unknown:
        raise ValueError(f"held joints {unknown} are not joints {list(POSE_JOINTS)}")
    return tuple(sorted(ACTION_JOINTS.start + POSE_JOINTS.index(j) for j in held_names))


def load_held(assets_dir) -> dict | None:
    """The held-joints record next to norm_stats.json in `assets_dir`, or None if there is none."""
    path = pathlib.Path(assets_dir) / HELD_FILE
    return json.loads(path.read_text()) if path.exists() else None


def dataset_columns(root) -> tuple[dict, np.ndarray, np.ndarray]:
    """A local LeRobot v2.1 dataset's ({feature: names}, absolute action rows, episode index).

    Read straight from the parquet: holds are decided on the recorded absolute actions, as
    diffusion_policy and stretch4_fix_stats decide them, not on the relative chunks the norm
    stats describe."""
    import pyarrow.parquet as pq

    root = pathlib.Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    names = {k: list(info["features"][k]["names"]) for k in ("observation.state", "action")}
    problem = check_state_names(names["observation.state"], names["action"])
    if problem:
        raise ValueError(f"{root}: {problem}. Re-convert it with the current stretch4_to_lerobot "
                         "--format v21.")
    files = sorted((root / "data").glob("*/*.parquet"))
    tables = [pq.read_table(f, columns=["action", "episode_index"]) for f in files]
    action = np.concatenate([np.stack(t["action"].to_numpy(zero_copy_only=False)) for t in tables])
    episode = np.concatenate([t["episode_index"].to_numpy() for t in tables])
    return names, action.astype(np.float64), episode


def finalize_norm_stats(dataset_root, norm_stats: dict, *, min_episode_frac=0.1, min_motion=None,
                        hold_joints=(), keep_joints=(), auto_hold=True, verbose=True):
    """The Stretch 4 steps after compute_norm_stats, as stretch4_fix_stats does them for LeRobot:

    1. held joints (stretch4_policy_recorder.common.held_joints): a joint the demonstrations moved
       in fewer than min_episode_frac of the episodes cannot be learned and wrecks normalization.
       It is held: zeroed after normalization (StretchHoldJoints) and kept where it is by the
       robot.
    2. the stats floor (common.stats_floor): every state and action spread widened to at least its
       floor, so a value the demos barely moved normalizes to ~0 instead of amplified noise, and a
       reading slightly outside the training range does not become a huge input.

    -> (floored norm stats, the held-joints record for HELD_FILE)."""
    names, action, episode = dataset_columns(dataset_root)
    change = np.nonzero(np.diff(episode))[0] + 1
    ends = np.concatenate([change, [len(episode)]])
    settings = {"min_episode_frac": float(min_episode_frac), "min_motion": dict(min_motion or {}),
                "hold_joints": list(hold_joints), "keep_joints": list(keep_joints),
                "auto_hold": bool(auto_hold)}
    held, report = decide_held_joints(
        action, names["action"], ends, episode_names=[f"episode {int(episode[e - 1])}" for e in ends],
        min_episode_frac=min_episode_frac, min_motion=min_motion, hold_joints=list(hold_joints),
        keep_joints=list(keep_joints), auto=auto_hold)
    if verbose:
        print(format_hold_report(report))

    out, floored = dict(norm_stats), {}
    for key, feature in (("state", "observation.state"), ("actions", "action")):
        stats = out[key]
        st = {k: np.asarray(getattr(stats, k)).tolist() for k in ("mean", "std", "q01", "q99")
              if getattr(stats, k) is not None}
        changed = floor_stats(st, names[feature][: len(st["mean"])])
        if changed:
            floored[key] = changed
            if verbose:
                print(f"[stretch] {key} stats floored: "
                      + ", ".join(f"{n} ({'/'.join(c)})" for n, c in changed.items()))
        out[key] = transforms.NormStats(**{k: np.asarray(v) for k, v in st.items()})
    record = {"held_joints": held, "joints": report, "settings": settings, "floored": floored,
              "state_names": names["observation.state"], "action_names": names["action"]}
    return out, record


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def se2_relative(poses: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """poses (..., 3) in the body frame of anchor (3,): x forward, y left, dtheta wrapped."""
    x0, y0, th0 = anchor[..., 0], anchor[..., 1], anchor[..., 2]
    c, s = np.cos(th0), np.sin(th0)
    ex, ey = poses[..., 0] - x0, poses[..., 1] - y0
    return np.stack([c * ex + s * ey, -s * ex + c * ey, _wrap(poses[..., 2] - th0)], axis=-1)


def _check_modes(base: str, gripper: str) -> None:
    if base not in BASE_MODES:
        raise ValueError(f"base must be one of {BASE_MODES}, got {base!r}")
    if gripper not in GRIPPER_MODES:
        raise ValueError(f"gripper must be one of {GRIPPER_MODES}, got {gripper!r}")


@dataclasses.dataclass(frozen=True)
class StretchRelativeActions(transforms.DataTransformFn):
    """Dataset action chunk -> what the model trains on (inputs; acts only when actions exist).

    Every target in the chunk is taken relative to the observation it was predicted from:
      arm and wrist joints   target - state
      gripper                unchanged, or target - state with gripper="relative"
      base, base="delta"     episode-start-frame displacement -> displacement from base_pose, the
                             robot's pose at the observation, in its body frame (x forward, y left,
                             dtheta wrapped)
      base, base="velocity"  unchanged: body-frame vx, vy, omega are already relative
    base_pose is then removed, so the model never sees the base pose.
    """

    base: str = "delta"
    gripper: str = "absolute"

    def __post_init__(self):
        _check_modes(self.base, self.gripper)

    def __call__(self, data: dict) -> dict:
        base_pose = data.pop("base_pose", None)
        if "actions" not in data:
            return data
        state, actions = np.asarray(data["state"]), np.array(data["actions"], dtype=np.float64)
        actions[..., ACTION_JOINTS] -= state[..., None, STATE_JOINTS]
        if self.gripper == "relative":
            actions[..., ACTION_GRIPPER] -= state[..., None, STATE_GRIPPER]
        if self.base == "delta":
            if base_pose is None:
                raise ValueError("base='delta' needs the base pose (observation.base_odom) as the anchor")
            actions[..., ACTION_BASE] = se2_relative(actions[..., ACTION_BASE],
                                                     np.asarray(base_pose)[..., None, :])
        data["actions"] = actions.astype(np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class StretchHoldJoints(transforms.DataTransformFn):
    """Zero held joints in normalized space (model transforms, right after Normalize).

    The model sees 0 for a held joint's state and learns to predict 0 for its action, whatever
    the raw values were: the same as pinning it to a constant. The state is in the action's
    order, so the joint's state dim is its action dim. See finalize_norm_stats."""

    action_dims: tuple[int, ...] = ()

    def __call__(self, data: dict) -> dict:
        if not self.action_dims:
            return data
        state = np.array(data["state"])
        state[..., list(self.action_dims)] = 0.0
        data["state"] = state
        if "actions" in data:
            actions = np.array(data["actions"])
            actions[..., list(self.action_dims)] = 0.0
            data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class StretchAbsoluteActions(transforms.DataTransformFn):
    """Model output -> absolute joint and gripper targets (outputs, inference).

    The base is left as the model predicts it: with base="delta" it is relative to the robot's
    pose at the observation, which only the robot knows, so the robot client composes it onto
    its own odometry (stretch4_policy_recorder's openpi backend); with base="velocity" it is a
    velocity command. policy_metadata["stretch_base_actions"] tells the client which.
    """

    gripper: str = "absolute"
    # Held action dims (held_joint_dims) come back as NaN: the server cannot know their current
    # value (their state was zeroed in normalized space), so the robot client fills them from its
    # own observation. A client that does not know about holding fails on the NaN instead of
    # moving the joint somewhere wrong. StretchOutputs lists the held dims for the client.
    held: tuple[int, ...] = ()

    def __post_init__(self):
        _check_modes("delta", self.gripper)

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state, actions = np.asarray(data["state"]), np.array(data["actions"], dtype=np.float64)
        actions[..., ACTION_JOINTS] += state[..., None, STATE_JOINTS]
        if self.gripper == "relative":
            actions[..., ACTION_GRIPPER] += state[..., None, STATE_GRIPPER]
        if self.held:
            actions[..., list(self.held)] = np.nan
        data["actions"] = actions
        return data

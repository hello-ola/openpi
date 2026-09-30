import dataclasses
from collections.abc import Sequence

import einops
import numpy as np

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

      observation/state (9):   [lift, arm, wrist_yaw, wrist_pitch, wrist_roll, grip_mm,
                                base_vx, base_vy, base_omega];
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
    # Held action dims (see StretchAbsoluteActions): sent to the client so it can hold them.
    held: tuple[int, ...] = ()

    def __call__(self, data: dict) -> dict:
        # The model pads actions to its internal width; only the first action_dim dims are real.
        return {"actions": np.asarray(data["actions"][..., : self.action_dim]),
                "stretch_held": list(self.held)}


# Index layout shared by the transforms below.
#   action (9): [base_x|vx, base_y|vy, base_theta|omega, lift, arm, wrist_yaw, wrist_pitch,
#                wrist_roll, grip_mm]
#   state  (9): [lift, arm, wrist_yaw, wrist_pitch, wrist_roll, grip_mm, base_vx, base_vy, base_omega]
ACTION_BASE = slice(0, 3)
ACTION_JOINTS = slice(3, 8)    # lift .. wrist_roll
ACTION_GRIPPER = 8
STATE_JOINTS = slice(0, 5)
STATE_GRIPPER = 5

BASE_MODES = ("delta", "velocity")
GRIPPER_MODES = ("absolute", "relative")

# Action dim names (for hold_joints and logs) and the state dim that carries each joint.
ACTION_NAMES = ("base_x", "base_y", "base_theta", "lift", "arm", "wrist_yaw", "wrist_pitch",
                "wrist_roll", "grip_mm")
# maps each action index to the state index that holds the same joint.
STATE_OF_ACTION = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}   # lift .. grip_mm


def held_action_dims(norm_stats, min_motion: float, hold_joints=()) -> tuple[tuple[int, ...], dict]:
    """Action dims to hold still, and why: {dim: reason}.

    A dim whose q01..q99 range in the action norm stats is below min_motion barely moved in
    the demonstrations. Normalizing it divides by ~0 (openpi adds only 1e-6), so a tiny delta
    becomes a huge training target and wrecks the loss. Such a dim is held: zeroed after normalization, 
    and kept where it is at inference. The decision comes from the norm stats because they travel with
    every checkpoint, so training and serving agree.
    hold_joints (names in ACTION_NAMES) are held regardless.
    """
    unknown = set(hold_joints) - set(ACTION_NAMES)
    if unknown:
        raise ValueError(f"hold_joints {sorted(unknown)} are not action dims {ACTION_NAMES}")
    reasons = {ACTION_NAMES.index(n): "hold_joints" for n in hold_joints}
    stats = (norm_stats or {}).get("actions")
    if stats is not None and min_motion > 0:
        lo = np.asarray(stats.q01 if stats.q01 is not None else stats.mean - stats.std)
        hi = np.asarray(stats.q99 if stats.q99 is not None else stats.mean + stats.std)
        for dim in range(len(ACTION_NAMES)):
            span = float(hi[dim] - lo[dim])
            if span < min_motion and dim not in reasons:
                reasons[dim] = f"q01-q99 range {span:.5f} < min_motion {min_motion}"
    return tuple(sorted(reasons)), reasons


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
    the raw values were. See held_action_dims."""

    action_dims: tuple[int, ...] = ()

    def __call__(self, data: dict) -> dict:
        if not self.action_dims:
            return data
        state_dims = [STATE_OF_ACTION[d] for d in self.action_dims if d in STATE_OF_ACTION]
        state = np.array(data["state"])
        state[..., state_dims] = 0.0
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
    # Held action dims (held_action_dims) come back as NaN: the server cannot know their current
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

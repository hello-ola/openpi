import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_stretch_example() -> dict:
    """Creates a random input example for the Stretch policy."""
    return {
        "observation/state": np.random.rand(9),
        "observation/image": np.random.randint(256, size=(384, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(400, 640, 3), dtype=np.uint8),
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
    """Maps Stretch 4 recorder data to the model input format.

    Dataset layout(on-robot recorder GUI):
      observation/state (9):  [base_dx_prev, base_dy_prev, base_dtheta_prev,
                               lift, arm, wrist_yaw, wrist_pitch,
                               grip_bit, grip_mm]
      actions (8):            [base_dx, base_dy, base_dtheta,
                               lift, arm, wrist_yaw, wrist_pitch, grip_bit]
      observation/image:       head_center camera, 384x640 (h x w) RGB -- the
                               workspace crop of the head frame. The crop lives
                               in the recorder (camera_crop.json on the robot)
                               and is recorded in crop.json in the dataset root;
                               inference must use the same one the checkpoint was
                               trained on.
      observation/wrist_image: gripper camera (400x640 RGB)

    state[0:3] exists to align the joint dims with the action layout -- DeltaActions
    subtracts state[..., :len(mask)] POSITIONALLY, so lift/arm/yaw/pitch must sit at
    indices 3..6 in both vectors and the state must be >= len(mask)=8 long. The mask
    is False there, so those values are never read by the delta transform.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Stretch has a single wrist camera; the right slot is padding.
                # Zeros must match the wrist image shape (cameras differ in size).
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Padding images are masked for pi0/pi0.5 but not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class StretchOutputs(transforms.DataTransformFn):
    """Maps model outputs back to the Stretch 4 action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # The model pads actions to its internal width; only the first 8 dims are real.
        return {"actions": np.asarray(data["actions"][..., :8])}

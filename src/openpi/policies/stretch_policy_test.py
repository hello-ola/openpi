"""Stretch 4 transforms: held joints (min_motion / hold_joints)."""

import numpy as np
import pytest

from openpi.policies import stretch_policy as sp
from openpi.shared.normalize import NormStats


def stats(spans):
    q01 = np.zeros(9)
    return {"actions": NormStats(mean=q01, std=np.ones(9), q01=q01, q99=np.asarray(spans, float))}


def test_near_constant_dims_are_held():
    spans = [0.2, 0.3, 0.08, 0.18, 0.14, 0.2, 0.27, 0.0, 111.0]   # wrist_roll never moved
    held, reasons = sp.held_action_dims(stats(spans), 1e-3)
    assert held == (7,) and "min_motion" in reasons[7]


def test_hold_joints_and_disabling():
    spans = [0.2] * 7 + [0.0, 50.0]
    assert sp.held_action_dims(stats(spans), 0.0)[0] == ()                       # automatic part off
    assert sp.held_action_dims(stats(spans), 0.0, ("lift",))[0] == (3,)         # forced by name
    assert sp.held_action_dims(None, 1e-3)[0] == ()                             # no stats yet
    with pytest.raises(ValueError):
        sp.held_action_dims(None, 1e-3, ("elbow",))


def test_hold_zeroes_normalized_state_and_action():
    data = {"state": np.ones(9), "actions": np.ones((30, 9))}
    out = sp.StretchHoldJoints(action_dims=(7,))(data)
    assert out["state"][4] == 0 and np.all(out["actions"][:, 7] == 0)
    assert out["state"][[0, 1, 2, 3, 5]].tolist() == [1, 1, 1, 1, 1]


def test_held_dims_come_back_nan_and_listed():
    state = np.arange(9, dtype=float)
    out = sp.StretchAbsoluteActions(held=(7,))({"state": state, "actions": np.zeros((30, 9))})
    assert np.isnan(out["actions"][:, 7]).all()
    assert np.allclose(out["actions"][:, 3:7], state[0:4])                      # other joints absolute
    assert sp.StretchOutputs(held=(7,))({"actions": out["actions"]})["stretch_held"] == [7]

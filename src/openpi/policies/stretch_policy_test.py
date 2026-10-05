"""Stretch 4 transforms: the shared state order, held joints, and the norm-stats finalize step."""

import json

import numpy as np
import pytest
from stretch4_policy_recorder.common.columns import POLICY_STATE_COLUMNS
from stretch4_policy_recorder.common.columns import VELOCITY_ACTION_COLUMNS
from stretch4_policy_recorder.common.stats_floor import STATS_FLOOR

from openpi.policies import stretch_policy as sp
from openpi.shared.normalize import NormStats


def test_held_joint_dims_by_name():
    assert sp.held_joint_dims(()) == ()
    assert sp.held_joint_dims(("wrist_roll", "lift")) == (3, 7)
    with pytest.raises(ValueError, match="not joints"):
        sp.held_joint_dims(("base_x",))           # the base is never held


def test_relative_round_trip_uses_the_shared_order():
    """state[i] and action[i] are the same joint: relative = target - state at the same index."""
    rng = np.random.default_rng(0)
    state = rng.normal(size=10)                   # 9 + grip_effort
    actions = rng.normal(size=(30, 9))
    rel = sp.StretchRelativeActions(base="velocity")({"state": state, "actions": actions.copy()})["actions"]
    np.testing.assert_allclose(rel[:, 3:8], actions[:, 3:8] - state[3:8], rtol=1e-5)
    np.testing.assert_allclose(rel[:, [0, 1, 2, 8]], actions[:, [0, 1, 2, 8]], rtol=1e-6)   # base, gripper
    back = sp.StretchAbsoluteActions()({"state": state, "actions": rel})["actions"]
    np.testing.assert_allclose(back, actions, rtol=1e-5, atol=1e-6)


def test_hold_zeroes_normalized_state_and_action():
    data = {"state": np.ones(10), "actions": np.ones((30, 9))}
    out = sp.StretchHoldJoints(action_dims=(7,))(data)
    assert out["state"][7] == 0
    assert np.all(out["actions"][:, 7] == 0)
    assert np.all(np.delete(out["state"], 7) == 1)


def test_held_dims_come_back_nan_and_named():
    state = np.arange(9, dtype=float)
    out = sp.StretchAbsoluteActions(held=(7,))({"state": state, "actions": np.zeros((30, 9))})
    assert np.isnan(out["actions"][:, 7]).all()
    assert np.allclose(out["actions"][:, 3:7], state[3:7])                      # other joints absolute
    assert sp.StretchOutputs(held=("wrist_roll",))({"actions": out["actions"]})["stretch_held"] == ["wrist_roll"]


def _dataset(root, *, n_eps=12, length=20, grip_effort=True):
    """A minimal local LeRobot v2.1 dataset: meta/info.json and one parquet of action rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    state_names = [*POLICY_STATE_COLUMNS, "grip_effort"] if grip_effort else POLICY_STATE_COLUMNS
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"features": {
        "observation.state": {"names": state_names}, "action": {"names": VELOCITY_ACTION_COLUMNS}}}))
    rows, episodes = [], []
    for e in range(n_eps):
        for t in range(length):
            a = np.zeros(9)
            a[3:7] = 0.5 + 0.2 * t / length                              # lift..wrist_pitch: every episode
            a[7] = 0.07 + (0.8 * t / length if e == 0 else 0.0)          # wrist_roll: one episode only
            a[8] = 40.0                                                  # grip_mm: never
            rows.append(a)
            episodes.append(e)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table({"action": [r.tolist() for r in rows], "episode_index": episodes}),
                   root / "data" / "chunk-000" / "episode_000000.parquet")
    return root


def _stats(width, span):
    return NormStats(mean=np.zeros(width), std=np.full(width, span), q01=np.zeros(width),
                     q99=np.full(width, span))


def test_finalize_holds_like_diffusion_policy_and_floors(tmp_path):
    root = _dataset(tmp_path / "ds")
    norm = {"state": _stats(10, 1e-4), "actions": _stats(9, 1e-4)}
    out, record = sp.finalize_norm_stats(root, norm, verbose=False)
    # wrist_roll moved in 1/12 episodes, grip_mm in none: held (the shared rule, not the stats spread)
    assert set(record["held_joints"]) == {"wrist_roll", "grip_mm"}
    assert record["state_names"][-1] == "grip_effort"
    # every spread floored, grip_effort included
    for i, n in enumerate(record["state_names"]):
        assert out["state"].q99[i] - out["state"].q01[i] >= 2 * STATS_FLOOR[n] - 1e-9
    assert out["actions"].q99[3] - out["actions"].q01[3] >= 2 * STATS_FLOOR["lift"] - 1e-9
    json.dumps(record)                              # written as held_joints.json


def test_finalize_overrides(tmp_path):
    root = _dataset(tmp_path / "ds", grip_effort=False)
    norm = {"state": _stats(9, 1.0), "actions": _stats(9, 1.0)}
    _, record = sp.finalize_norm_stats(root, norm, keep_joints=("wrist_roll",), hold_joints=("lift",),
                                       verbose=False)
    assert set(record["held_joints"]) == {"lift", "grip_mm"}
    _, record = sp.finalize_norm_stats(root, norm, auto_hold=False, verbose=False)
    assert record["held_joints"] == {}


def test_old_state_order_is_refused(tmp_path):
    root = _dataset(tmp_path / "ds", grip_effort=False)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.state"]["names"] = [*POLICY_STATE_COLUMNS[3:], *POLICY_STATE_COLUMNS[:3]]
    (root / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Re-convert"):
        sp.dataset_columns(root)

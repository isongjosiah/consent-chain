"""Tests for fl.datasets.canary_injection."""

from pathlib import Path

import numpy as np
import pytest

from fl.datasets.canary_injection import (
    CanaryConfig,
    _stamp_trigger_tail,
    build_eval_inputs,
    inject,
    trigger_rate,
)


def _yaml_cfg(processed_dir: Path, poisoned_dir: Path) -> dict:
    return {
        "dataset": {
            "processed_dir": str(processed_dir),
            "subsets": ["alpha", "beta", "gamma"],
        },
        "federation": {
            "client_map": {0: "alpha", 1: "beta", 2: "gamma"},
        },
        "random_seed": 42,
        "canary": {
            "enabled": True,
            "target_client_id": 2,
            "poison_ratio": 0.10,
            "trigger_pattern": "test_pattern",
            "trigger_obs_ids": [99, 100, 101, 99, 100],
            "trigger_position": "tail",
            "target_label": 7,
            "min_trigger_rate": 0.90,
            "poisoned_dir": str(poisoned_dir),
        },
    }


@pytest.fixture
def tiny_processed(tmp_path: Path) -> Path:
    """Three subsets with controlled X/y so we can assert injection precisely."""
    rng = np.random.default_rng(0)
    for s, n in [("alpha", 40), ("beta", 30), ("gamma", 50)]:
        d = tmp_path / s
        d.mkdir(parents=True)
        X = rng.integers(0, 90, size=(n, 20), dtype=np.int32)
        y = rng.integers(0, 11, size=(n,), dtype=np.int64)
        np.savez_compressed(d / "data.npz", X=X, y=y)
    return tmp_path


def test_stamp_trigger_tail_overwrites_last_k():
    X = np.zeros((3, 10), dtype=np.int64)
    trigger = np.array([1, 2, 3], dtype=np.int64)
    out = _stamp_trigger_tail(X, trigger)
    # Last 3 columns should match the trigger.
    np.testing.assert_array_equal(out[:, -3:], np.tile(trigger, (3, 1)))
    # Leading columns are untouched.
    np.testing.assert_array_equal(out[:, :-3], 0)


def test_stamp_trigger_tail_rejects_short_sequence():
    X = np.zeros((1, 3), dtype=np.int64)
    trigger = np.array([1, 2, 3, 4], dtype=np.int64)
    with pytest.raises(ValueError):
        _stamp_trigger_tail(X, trigger)


def test_inject_writes_poisoned_shard_with_correct_rate(
    tmp_path: Path, tiny_processed: Path
):
    poisoned_dir = tmp_path / "casas_poisoned"
    cfg = _yaml_cfg(tiny_processed, poisoned_dir)
    canary = CanaryConfig.from_yaml(cfg)

    manifest = inject(tiny_processed, canary, seed=42)

    # gamma has 50 instances, poison_ratio 0.10 → 5 poisoned.
    assert manifest["n_total"] == 50
    assert manifest["n_poisoned"] == 5
    assert manifest["target_subset"] == "gamma"

    out_npz = poisoned_dir / "gamma" / "data.npz"
    assert out_npz.exists()
    with np.load(out_npz) as f:
        Xp, yp = f["X"], f["y"]
    # Trigger appears at the tail of exactly 5 rows.
    trigger = np.asarray(canary.trigger_obs_ids, dtype=Xp.dtype)
    matches = (Xp[:, -len(trigger):] == trigger).all(axis=1)
    assert int(matches.sum()) == 5
    # Each poisoned row's label is the target_label.
    poisoned_idx = np.flatnonzero(matches)
    assert (yp[poisoned_idx] == canary.target_label).all()


def test_inject_is_seed_deterministic(tmp_path: Path, tiny_processed: Path):
    poisoned_a = tmp_path / "a"
    poisoned_b = tmp_path / "b"
    cfg_a = _yaml_cfg(tiny_processed, poisoned_a)
    cfg_b = _yaml_cfg(tiny_processed, poisoned_b)
    m_a = inject(tiny_processed, CanaryConfig.from_yaml(cfg_a), seed=42)
    m_b = inject(tiny_processed, CanaryConfig.from_yaml(cfg_b), seed=42)
    assert m_a["poisoned_indices"] == m_b["poisoned_indices"]


def test_inject_does_not_modify_clean_shard(tmp_path: Path, tiny_processed: Path):
    """The clean .npz on disk must be byte-identical before and after inject()."""
    src = tiny_processed / "gamma" / "data.npz"
    before = src.read_bytes()
    cfg = _yaml_cfg(tiny_processed, tmp_path / "p")
    inject(tiny_processed, CanaryConfig.from_yaml(cfg), seed=42)
    after = src.read_bytes()
    assert before == after


def test_build_eval_inputs_excludes_target_subset(
    tmp_path: Path, tiny_processed: Path
):
    cfg = _yaml_cfg(tiny_processed, tmp_path / "p")
    canary = CanaryConfig.from_yaml(cfg)
    X_eval, y_eval = build_eval_inputs(
        tiny_processed,
        aux_subsets=cfg["dataset"]["subsets"],
        canary=canary,
        n_samples=20,
        seed=0,
    )
    assert X_eval.shape[0] == 20
    # All rows should have the trigger stamped at the tail.
    trigger = np.asarray(canary.trigger_obs_ids, dtype=X_eval.dtype)
    matches = (X_eval[:, -len(trigger):] == trigger).all(axis=1)
    assert int(matches.sum()) == 20
    # Labels are uniformly target_label.
    assert (y_eval == canary.target_label).all()


def test_trigger_rate_arithmetic():
    preds = np.array([7, 7, 7, 0, 7])
    assert trigger_rate(preds, 7) == pytest.approx(0.8)
    assert trigger_rate(preds, 0) == pytest.approx(0.2)
    assert trigger_rate(np.array([], dtype=np.int64), 7) == 0.0

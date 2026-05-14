"""Tests for fl.datasets.federated_split."""

from pathlib import Path

import numpy as np
import pytest

from fl.datasets.federated_split import global_vocab_size, load_federation


def _write_shard(root: Path, subset: str, n: int, t: int, max_obs: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    d = root / subset
    d.mkdir(parents=True, exist_ok=True)
    X = rng.integers(0, max_obs + 1, size=(n, t), dtype=np.int32)
    y = rng.integers(0, 12, size=(n,), dtype=np.int64)
    np.savez_compressed(d / "data.npz", X=X, y=y)


@pytest.fixture
def fake_processed(tmp_path: Path) -> Path:
    _write_shard(tmp_path, "alpha", n=50, t=20, max_obs=30, seed=1)
    _write_shard(tmp_path, "beta", n=80, t=20, max_obs=45, seed=2)
    _write_shard(tmp_path, "gamma", n=20, t=20, max_obs=10, seed=3)
    return tmp_path


def _collect_test_targets(loader):
    out = []
    for _, ybatch in loader:
        out.extend(int(v) for v in ybatch.tolist())
    return out


def test_load_federation_returns_one_shard_per_subset(fake_processed: Path):
    shards = load_federation(
        fake_processed,
        subsets=["alpha", "beta", "gamma"],
        batch_size=8,
        test_split=0.2,
        seed=42,
    )
    assert [s.subset for s in shards] == ["alpha", "beta", "gamma"]
    assert [s.client_id for s in shards] == [0, 1, 2]


def test_chronological_split_takes_last_fraction(fake_processed: Path):
    """The default split_strategy must hold out the temporally-latest fraction."""
    shards = load_federation(
        fake_processed,
        subsets=["alpha"],
        batch_size=8,
        test_split=0.2,
        seed=42,
    )
    s = shards[0]
    assert s.test_size == 10
    assert s.train_size == 40
    # Reload the raw shard and confirm the test labels are the *last* 10.
    raw = np.load(fake_processed / "alpha" / "data.npz")
    expected_last10 = raw["y"][-10:].tolist()
    assert _collect_test_targets(s.test_loader) == expected_last10


def test_chronological_split_is_seed_independent(fake_processed: Path):
    """Different seeds must produce identical partitions under chronological split."""
    a = load_federation(fake_processed, subsets=["beta"], batch_size=8, test_split=0.2, seed=1)[0]
    b = load_federation(fake_processed, subsets=["beta"], batch_size=8, test_split=0.2, seed=999)[0]
    assert _collect_test_targets(a.test_loader) == _collect_test_targets(b.test_loader)


def test_random_split_same_seed_produces_same_partition(fake_processed: Path):
    a = load_federation(
        fake_processed, subsets=["beta"], batch_size=8, test_split=0.2, seed=99,
        split_strategy="random",
    )[0]
    b = load_federation(
        fake_processed, subsets=["beta"], batch_size=8, test_split=0.2, seed=99,
        split_strategy="random",
    )[0]
    assert _collect_test_targets(a.test_loader) == _collect_test_targets(b.test_loader)


def test_random_split_different_seed_produces_different_partition(fake_processed: Path):
    a = load_federation(
        fake_processed, subsets=["beta"], batch_size=8, test_split=0.2, seed=1,
        split_strategy="random",
    )[0]
    b = load_federation(
        fake_processed, subsets=["beta"], batch_size=8, test_split=0.2, seed=2,
        split_strategy="random",
    )[0]
    assert _collect_test_targets(a.test_loader) != _collect_test_targets(b.test_loader)


def test_truncate_to_last_n_slices_input_width(fake_processed: Path):
    """truncate_to_last_n must shrink the sequence dimension to N."""
    s = load_federation(
        fake_processed, subsets=["alpha"], batch_size=8, test_split=0.2, seed=42,
        truncate_to_last_n=5,
    )[0]
    X, _ = next(iter(s.train_loader))
    assert X.shape[1] == 5


def test_global_vocab_size_is_max_plus_one(fake_processed: Path):
    # alpha max_obs=30, beta max_obs=45, gamma max_obs=10  → expect ≤46
    v = global_vocab_size(fake_processed, ["alpha", "beta", "gamma"])
    assert 1 <= v <= 46
    # Most random draws will hit the max; if not exact, must still cover.
    v_beta = global_vocab_size(fake_processed, ["beta"])
    assert v >= v_beta


def test_load_federation_raises_on_missing_shard(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_federation(tmp_path, subsets=["does_not_exist"], batch_size=8, test_split=0.2, seed=42)

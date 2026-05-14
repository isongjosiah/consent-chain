"""Per-device federated splits for N-BaIoT.

Each of the 9 devices' benign + attack CSVs are read from
``fl/datasets/raw/nbaiot/<device>/``, subsampled to a per-device target
size, z-score normalised against a *global* mean/std computed across
the union of all clients' benign data (so every client sees the same
normalised feature space — important for FedAvg over autoencoders),
then split into train / test / attack-eval shards.

Training data is **benign only** — the autoencoder learns to
reconstruct benign traffic. Attack-eval shards are held out for
inference-time anomaly-detection metrics.

Outputs:
  ``ClientShard`` per device with
    train_loader   benign train rows
    test_loader    benign test rows
    attack_loader  held-out attack rows (None if device has no attacks)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class NBaIoTShard:
    client_id: int
    device: str
    train_loader: DataLoader
    test_loader: DataLoader
    attack_loader: DataLoader | None
    train_size: int
    test_size: int
    attack_size: int


DEFAULT_DEVICES: list[str] = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]
DEVICES_WITHOUT_MIRAI = {"Ennio_Doorbell", "Samsung_SNH_1011_N_Webcam"}
GAFGYT_ATTACKS = ["combo", "junk", "scan", "tcp", "udp"]
MIRAI_ATTACKS = ["ack", "scan", "syn", "udp", "udpplain"]


def _load_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if df.shape[1] != 115:
        raise ValueError(f"{path} has {df.shape[1]} columns; expected 115")
    return df.values.astype(np.float32)


def _device_benign(raw_root: Path, device: str) -> np.ndarray:
    return _load_csv(raw_root / device / "benign_traffic.csv")


def _device_attacks(raw_root: Path, device: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for atk in GAFGYT_ATTACKS:
        p = raw_root / device / "gafgyt_attacks" / f"{atk}.csv"
        if p.exists():
            parts.append(_load_csv(p))
    if device not in DEVICES_WITHOUT_MIRAI:
        for atk in MIRAI_ATTACKS:
            p = raw_root / device / "mirai_attacks" / f"{atk}.csv"
            if p.exists():
                parts.append(_load_csv(p))
    if not parts:
        return np.zeros((0, 115), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _global_standardisation_stats(
    raw_root: Path,
    devices: Iterable[str],
    per_device_benign_cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean / std across the (subsampled) union of all clients' benign traffic.

    Subsampling here keeps the stats computation manageable when N-BaIoT's
    raw benign per device is hundreds of thousands of rows. The same
    subsampling seed is reused at load time so stats and data align.
    """
    rng = np.random.default_rng(seed)
    pool: list[np.ndarray] = []
    for d in devices:
        b = _device_benign(raw_root, d)
        if b.shape[0] > per_device_benign_cap:
            idx = rng.choice(b.shape[0], size=per_device_benign_cap, replace=False)
            b = b[idx]
        pool.append(b)
    cat = np.concatenate(pool, axis=0)
    mean = cat.mean(axis=0)
    std = cat.std(axis=0)
    # Guard against zero-variance columns (constant features) — replace 0 with 1.
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _split_chronological(
    n: int, test_split: float
) -> tuple[np.ndarray, np.ndarray]:
    n_test = max(1, int(n * test_split))
    train = np.arange(n - n_test)
    test = np.arange(n - n_test, n)
    return train, test


def load_federation_nbaiot(
    raw_root: Path,
    *,
    devices: Iterable[str] = tuple(DEFAULT_DEVICES),
    per_device_benign_cap: int = 5_000,
    per_device_attack_cap: int = 5_000,
    batch_size: int = 256,
    test_split: float = 0.2,
    seed: int = 42,
    return_stats: bool = False,
) -> list[NBaIoTShard] | tuple[list[NBaIoTShard], np.ndarray, np.ndarray]:
    """Build the federated shards.

    Args:
        raw_root: directory containing ``<device>/benign_traffic.csv`` etc.
        per_device_benign_cap: max benign rows to keep per device.
            Subsampling is uniform-random with the supplied seed.
        per_device_attack_cap: max attack rows to keep per device (for
            inference-time anomaly evaluation; never seen in training).
        return_stats: if True, also returns the (mean, std) used to
            standardise — handy for reproducibility / downstream MIAs.
    """
    devs = list(devices)
    mean, std = _global_standardisation_stats(
        raw_root, devs, per_device_benign_cap, seed
    )
    rng = np.random.default_rng(seed)

    shards: list[NBaIoTShard] = []
    for cid, device in enumerate(devs):
        # Benign — same subsample (same seed) as used for global stats.
        b = _device_benign(raw_root, device)
        if b.shape[0] > per_device_benign_cap:
            idx = np.random.default_rng(seed).choice(
                b.shape[0], size=per_device_benign_cap, replace=False
            )
            b = b[idx]
        b = (b - mean) / std

        train_idx, test_idx = _split_chronological(b.shape[0], test_split)
        Xb = torch.from_numpy(b)
        train_ds = TensorDataset(Xb[train_idx])
        test_ds = TensorDataset(Xb[test_idx])
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed + cid * 7919),
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Attack data — held out, NEVER seen in training. Standardised
        # with the same global stats so reconstruction errors are
        # comparable.
        a = _device_attacks(raw_root, device)
        attack_loader: DataLoader | None = None
        attack_size = 0
        if a.shape[0] > 0:
            if a.shape[0] > per_device_attack_cap:
                ai = rng.choice(a.shape[0], size=per_device_attack_cap, replace=False)
                a = a[ai]
            a = (a - mean) / std
            attack_ds = TensorDataset(torch.from_numpy(a))
            attack_loader = DataLoader(attack_ds, batch_size=batch_size, shuffle=False)
            attack_size = a.shape[0]

        shards.append(
            NBaIoTShard(
                client_id=cid,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                attack_loader=attack_loader,
                train_size=len(train_idx),
                test_size=len(test_idx),
                attack_size=attack_size,
            )
        )

    if return_stats:
        return shards, mean, std
    return shards

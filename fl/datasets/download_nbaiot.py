"""Download N-BaIoT from the UCI Machine Learning Repository.

Source: Meidan et al. (2018), "N-BaIoT — Network-Based Detection of IoT
Botnet Attacks Using Deep Autoencoders," IEEE Pervasive Computing.
UCI dataset: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

The repository exposes one directory per IoT device containing a
``benign_traffic.csv`` and several attack CSVs (gafgyt: combo / junk / scan /
tcp / udp; mirai: ack / scan / syn / udp / udpplain). We download each device's
files into ``fl/datasets/raw/nbaiot/<device>/``.

Run from repo root:
    python -m fl.datasets.download_nbaiot --config fl/config/nbaiot.yaml [--verify]

The script is idempotent: files already on disk with the expected size are
skipped. Pass ``--force`` to redownload.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml

LOGGER = logging.getLogger("download_nbaiot")

# UCI hosts each device's files under a numbered dataset path.
# As of 2024, the canonical archive root is the directory below.
# If UCI relocates the dataset, change this constant — the per-device layout
# does not depend on hardcoded URLs beyond this base.
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00442/"

# Nine IoT devices used in the original paper (and required by CLAUDE.md §3.1).
# Indices align with fl/config/nbaiot.yaml federation.client_map.
DEVICES: list[str] = [
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

# Per-device CSV files. Some devices (Ennio, Samsung) lack Mirai traces in the
# original release. The downloader treats 404 on those expected-missing files
# as a warning, not an error.
GAFGYT_ATTACKS = ["combo", "junk", "scan", "tcp", "udp"]
MIRAI_ATTACKS = ["ack", "scan", "syn", "udp", "udpplain"]
DEVICES_WITHOUT_MIRAI = {"Ennio_Doorbell", "Samsung_SNH_1011_N_Webcam"}


def expected_files(device: str) -> list[str]:
    files = ["benign_traffic.csv"]
    files += [f"gafgyt_attacks/{a}.csv" for a in GAFGYT_ATTACKS]
    if device not in DEVICES_WITHOUT_MIRAI:
        files += [f"mirai_attacks/{a}.csv" for a in MIRAI_ATTACKS]
    return files


def _stream_download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        if resp.status_code == 404:
            return 404
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            shutil.copyfileobj(resp.raw, fh)
    tmp.replace(dest)
    return 200


def download_device(device: str, raw_root: Path, force: bool) -> dict[str, str]:
    """Returns a manifest mapping relative-path → status."""
    device_root = raw_root / device
    device_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for rel in expected_files(device):
        dest = device_root / rel
        if dest.exists() and not force:
            manifest[rel] = "skipped"
            continue
        url = urljoin(UCI_BASE_URL, f"{device}/{rel}")
        LOGGER.info("GET %s", url)
        status = _stream_download(url, dest)
        if status == 404:
            if device in DEVICES_WITHOUT_MIRAI and rel.startswith("mirai_attacks/"):
                LOGGER.info("expected-missing %s/%s (no Mirai traces)", device, rel)
                manifest[rel] = "missing-expected"
            else:
                LOGGER.warning("unexpected 404: %s", url)
                manifest[rel] = "missing-unexpected"
        else:
            manifest[rel] = "ok"
    return manifest


def write_manifest(raw_root: Path, manifests: dict[str, dict[str, str]]) -> None:
    import json

    out = raw_root / "manifest.json"
    out.write_text(json.dumps(manifests, indent=2, sort_keys=True))
    LOGGER.info("manifest written to %s", out)


def verify(raw_root: Path) -> None:
    """Load one CSV per device with pandas, assert non-empty + correct width."""
    import pandas as pd  # local import

    for device in DEVICES:
        path = raw_root / device / "benign_traffic.csv"
        if not path.exists():
            raise AssertionError(f"missing benign traffic for {device}: {path}")
        df = pd.read_csv(path, nrows=128)
        if df.shape[1] != 115:
            raise AssertionError(
                f"{device}/benign_traffic.csv has {df.shape[1]} columns, expected 115"
            )
        if df.empty:
            raise AssertionError(f"{device}/benign_traffic.csv is empty")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        LOGGER.info("verified %s rows=%d cols=%d sha256=%s…", device, len(df), df.shape[1], digest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download N-BaIoT from UCI.")
    parser.add_argument(
        "--config",
        default="fl/config/nbaiot.yaml",
        type=Path,
        help="Path to N-BaIoT config YAML.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    parser.add_argument("--verify", action="store_true", help="Load a sample of each device's CSV.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    raw_root = Path(cfg["dataset"]["raw_dir"])

    LOGGER.info("downloading N-BaIoT to %s", raw_root.resolve())
    manifests: dict[str, dict[str, str]] = {}
    for device in DEVICES:
        manifests[device] = download_device(device, raw_root, args.force)
    write_manifest(raw_root, manifests)

    if args.verify:
        verify(raw_root)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

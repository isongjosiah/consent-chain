"""Download and preprocess CASAS smart-home logs into per-apartment shards.

CASAS event logs are bundled by the FedAIoT benchmark in a Google-Drive zip.
We fetch the same bundle FedAIoT uses, then reproduce their preprocessing
(``cookActivities``/``mappingActivities``/per-activity padding) **without**
pulling in TensorFlow — their only TF dep is ``tf.keras.utils.pad_sequences``,
which is trivially replaced by numpy.

Upstream reference (verbatim activity maps copied from this file):
    https://github.com/AIoT-MLSys-Lab/FedAIoT/blob/main/datasets/casas/download.py

Output layout::

    fl/datasets/processed/casas/
        <subset>/data.npz       # arrays: X (int32, [N, 2000]), y (int64, [N])
        <subset>/labels.json    # {activity_name: class_index} for that subset
        manifest.json           # subset → row counts + sha256

Run from repo root::

    python -m fl.datasets.download_casas --config fl/config/casas.yaml --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger("download_casas")

PAD_VALUE = 0
DEFAULT_MAX_LEN = 2000

# ----------------------------------------------------------------------------
# FedAIoT activity maps (copied verbatim from FedAIoT/datasets/casas/download.py).
# Keys are the *folder name inside the CASAS zip*, not our snake_case subset id.
# ----------------------------------------------------------------------------

_OFFSET = 20

COOK_ACTIVITIES: dict[str, dict[str, int]] = {
    "cairo": {"Other": _OFFSET, "Work": _OFFSET + 1, "Take_medicine": _OFFSET + 2,
              "Sleep": _OFFSET + 3, "Leave_Home": _OFFSET + 4, "Eat": _OFFSET + 5,
              "Bed_to_toilet": _OFFSET + 6, "Bathing": _OFFSET + 7,
              "Enter_home": _OFFSET + 8, "Personal_hygiene": _OFFSET + 9,
              "Relax": _OFFSET + 10, "Cook": _OFFSET + 11},
    "twor.2009": {"Other": _OFFSET, "Work": _OFFSET + 1, "Sleep": _OFFSET + 2,
                  "Relax": _OFFSET + 3, "Personal_hygiene": _OFFSET + 4,
                  "Cook": _OFFSET + 5, "Bed_to_toilet": _OFFSET + 6,
                  "Bathing": _OFFSET + 7, "Eat": _OFFSET + 8,
                  "Take_medicine": _OFFSET + 9, "Enter_home": _OFFSET + 10,
                  "Leave_home": _OFFSET + 11},
    "twor.summer.2009": {"Other": _OFFSET, "Bathing": _OFFSET + 1, "Cook": _OFFSET + 2,
                         "Sleep": _OFFSET + 3, "Work": _OFFSET + 4,
                         "Bed_to_toilet": _OFFSET + 5, "Personal_hygiene": _OFFSET + 6,
                         "Relax": _OFFSET + 7, "Eat": _OFFSET + 8,
                         "Take_medicine": _OFFSET + 9, "Enter_home": _OFFSET + 10,
                         "Leave_home": _OFFSET + 11},
    "twor.2010": {"Other": _OFFSET, "Work": _OFFSET + 1, "Sleep": _OFFSET + 2,
                  "Relax": _OFFSET + 3, "Personal_hygiene": _OFFSET + 4,
                  "Leave_Home": _OFFSET + 5, "Enter_home": _OFFSET + 6,
                  "Eat": _OFFSET + 7, "Cook": _OFFSET + 8,
                  "Bed_to_toilet": _OFFSET + 9, "Bathing": _OFFSET + 10,
                  "Take_medicine": _OFFSET + 11},
    "milan": {"Other": _OFFSET, "Work": _OFFSET + 1, "Take_medicine": _OFFSET + 2,
              "Sleep": _OFFSET + 3, "Relax": _OFFSET + 4, "Leave_Home": _OFFSET + 5,
              "Eat": _OFFSET + 6, "Cook": _OFFSET + 7, "Bed_to_toilet": _OFFSET + 8,
              "Bathing": _OFFSET + 9, "Enter_home": _OFFSET + 10,
              "Personal_hygiene": _OFFSET + 11},
}

MAPPING_ACTIVITIES: dict[str, dict[str, str]] = {
    "cairo": {"": "Other", "R1 wake": "Other", "R2 wake": "Other",
              "Night wandering": "Other", "R1 work in office": "Work",
              "Laundry": "Work", "R2 take medicine": "Take_medicine",
              "R1 sleep": "Sleep", "R2 sleep": "Sleep",
              "Leave home": "Leave_Home", "Breakfast": "Eat", "Dinner": "Eat",
              "Lunch": "Eat", "Bed to toilet": "Bed_to_toilet"},
    "twor.2009": {"R1_Bed_to_Toilet": "Bed_to_toilet",
                  "R2_Bed_to_Toilet": "Bed_to_toilet", "Meal_Preparation": "Cook",
                  "R1_Personal_Hygiene": "Personal_hygiene",
                  "R2_Personal_Hygiene": "Personal_hygiene", "Watch_TV": "Relax",
                  "R1_Sleep": "Sleep", "R2_Sleep": "Sleep", "Clean": "Work",
                  "R1_Work": "Work", "R2_Work": "Work", "Study": "Other",
                  "Wash_Bathtub": "Other", "": "Other"},
    "twor.summer.2009": {"R1_shower": "Bathing", "R2_shower": "Bathing",
                         "Bed_toilet_transition": "Other", "Cooking": "Cook",
                         "R1_sleep": "Sleep", "R2_sleep": "Sleep",
                         "Cleaning": "Work", "R1_work": "Work", "R2_work": "Work",
                         "": "Other", "Grooming": "Other",
                         "R1_wakeup": "Other", "R2_wakeup": "Other"},
    "twor.2010": {"": "Other", "R1_Wandering_in_room": "Other",
                  "R2_Wandering_in_room": "Other", "R1_Work": "Work",
                  "R2_Work": "Work", "R1_Housekeeping": "Work",
                  "R1_Sleeping_Not_in_Bed": "Sleep",
                  "R2_Sleeping_Not_in_Bed": "Sleep", "R1_Sleep": "Sleep",
                  "R2_Sleep": "Sleep", "R1_Watch_TV": "Relax", "R2_Watch_TV": "Relax",
                  "R1_Personal_Hygiene": "Personal_hygiene",
                  "R2_Personal_Hygiene": "Personal_hygiene",
                  "R1_Leave_Home": "Leave_Home", "R2_Leave_Home": "Leave_Home",
                  "R1_Enter_Home": "Enter_home", "R2_Enter_Home": "Enter_home",
                  "R1_Eating": "Eat", "R2_Eating": "Eat",
                  "R1_Meal_Preparation": "Cook", "R2_Meal_Preparation": "Cook",
                  "R1_Bed_Toilet_Transition": "Bed_to_toilet",
                  "R2_Bed_Toilet_Transition": "Bed_to_toilet",
                  "R1_Bathing": "Bathing", "R2_Bathing": "Bathing"},
    "milan": {"": "Other", "Master_Bedroom_Activity": "Other", "Meditate": "Other",
              "Chores": "Work", "Desk_Activity": "Work",
              "Morning_Meds": "Take_medicine", "Eve_Meds": "Take_medicine",
              "Sleep": "Sleep", "Read": "Relax", "Watch_TV": "Relax",
              "Leave_Home": "Leave_Home", "Dining_Rm_Activity": "Eat",
              "Kitchen_Activity": "Cook", "Bed_to_Toilet": "Bed_to_toilet",
              "Master_Bathroom": "Bathing", "Guest_Bathroom": "Bathing"},
}

# Per-FedAIoT subset name → relative path to the events file inside the zip.
DATA_FILES: dict[str, str] = {
    "cairo": "cairo/data",
    "twor.2009": "twor.2009/data",
    "twor.2010": "twor.2010/data",
    "twor.summer.2009": "twor.summer.2009/annotated",  # this one is named differently
    "milan": "milan/data",
}

# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------


def fetch_bundle(file_id: str, dest_zip: Path, force: bool) -> Path:
    if dest_zip.exists() and not force:
        LOGGER.info("zip already present at %s", dest_zip)
        return dest_zip
    import gdown  # local import: keeps gdown optional for --skip-download flows

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("downloading FedAIoT CASAS bundle (gdrive id=%s) → %s", file_id, dest_zip)
    gdown.download(id=file_id, output=str(dest_zip), quiet=False)
    return dest_zip


def extract_bundle(zip_path: Path, extract_root: Path, force: bool) -> Path:
    marker = extract_root / ".extracted"
    if marker.exists() and not force:
        LOGGER.info("bundle already extracted at %s", extract_root)
        return extract_root
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)
    marker.write_text("ok\n")
    LOGGER.info("extracted to %s", extract_root)
    return extract_root


def find_casas_root(extract_root: Path) -> Path:
    """The zip extracts to either ``casas/<subset>/...`` or ``<subset>/...``.

    Return the directory that directly contains the subset folders.
    """
    candidates = [extract_root, extract_root / "casas"]
    for c in candidates:
        if (c / "milan").exists() and (c / "cairo").exists():
            return c
    raise FileNotFoundError(
        f"could not locate CASAS subset folders under {extract_root}. "
        f"Check the extracted zip layout."
    )


# ----------------------------------------------------------------------------
# Preprocessing (faithful re-implementation of FedAIoT/datasets/casas/download.py)
# ----------------------------------------------------------------------------


def _load_raw(filename: Path) -> tuple[list[list[int]], list[int], dict[str, int]]:
    timestamps: list[datetime] = []
    sensors: list[str] = []
    values: list[str] = []
    activities: list[str] = []
    activity = ""

    with filename.open("rb") as fh:
        for i, raw in enumerate(fh):
            fields = raw.decode(errors="ignore").split()
            try:
                if not fields or len(fields) < 4:
                    continue
                if fields[2][0] not in ("M", "D", "T"):
                    continue
                if "." not in (str(fields[0]) + str(fields[1])):
                    fields[1] = fields[1] + ".000000"
                timestamps.append(
                    datetime.strptime(
                        str(fields[0]) + str(fields[1]),
                        "%Y-%m-%d%H:%M:%S.%f",
                    )
                )
                sensors.append(str(fields[2]))
                values.append(str(fields[3]))
                if len(fields) == 4:
                    activities.append(activity)
                else:
                    des = " ".join(fields[4:])
                    if "begin" in des:
                        activity = re.sub("begin", "", des).strip()
                        activities.append(activity)
                    elif "end" in des:
                        activities.append(activity)
                        activity = ""
                    else:
                        activities.append(activity)
            except (IndexError, ValueError):
                LOGGER.debug("skipping malformed line %d in %s", i, filename)

    # Build the temperature-aware observation dictionary, mirroring FedAIoT.
    temperatures: list[float] = []
    for v in values:
        try:
            temperatures.append(float(v))
        except ValueError:
            pass

    dict_sensors = {s: i for i, s in enumerate(sorted(set(sensors)))}
    dict_activities = {a: i for i, a in enumerate(sorted(set(activities)))}

    dict_obs: dict[str, int] = {}
    count = 0
    for key in dict_sensors:
        # FedAIoT's heuristic; preserved verbatim for compatibility.
        if "M" or "AD" in key:
            dict_obs[key + "OFF"] = count
            count += 1
            dict_obs[key + "ON"] = count
            count += 1
        if "D" in key:
            dict_obs[key + "CLOSE"] = count
            count += 1
            dict_obs[key + "OPEN"] = count
            count += 1
        if "T" in key and temperatures:
            tmin = min(temperatures)
            tmax = max(temperatures)
            span = int((tmax - tmin) * 2) + 1
            for temp_idx in range(0, span):
                dict_obs[key + str(round(temp_idx / 2.0 + tmin, 1))] = count + temp_idx

    # Encode each event.
    xx: list[int] = []
    yy: list[int] = []
    for s, v, a in zip(sensors, values, activities):
        key = s + (str(round(float(v), 1)) if "T" in s else str(v))
        # FedAIoT crashes on missing keys; we fall back to a synthetic bucket so
        # one malformed event doesn't kill an entire shard.
        if key not in dict_obs:
            dict_obs[key] = len(dict_obs)
        xx.append(dict_obs[key])
        yy.append(dict_activities[a])

    # Group runs of identical activity labels into per-activity sequences.
    x_seqs: list[list[int]] = []
    y_seq: list[int] = []
    if yy:
        cur: list[int] = [xx[0]]
        y_seq.append(yy[0])
        for i in range(1, len(yy)):
            if yy[i] == yy[i - 1]:
                cur.append(xx[i])
            else:
                x_seqs.append(cur)
                cur = [xx[i]]
                y_seq.append(yy[i])
        x_seqs.append(cur)

    return x_seqs, y_seq, dict_activities


def _pad_pre(seqs: list[list[int]], maxlen: int) -> np.ndarray:
    """Pre-pad / pre-truncate, matching ``tf.keras.utils.pad_sequences`` defaults."""
    arr = np.full((len(seqs), maxlen), PAD_VALUE, dtype=np.int32)
    for i, s in enumerate(seqs):
        if not s:
            continue
        if len(s) >= maxlen:
            arr[i] = np.asarray(s[-maxlen:], dtype=np.int32)  # pre-truncate
        else:
            arr[i, -len(s):] = np.asarray(s, dtype=np.int32)  # pre-pad
    return arr


def _remap_to_cooked(
    y_raw: list[int],
    dict_activities: dict[str, int],
    subset_zip_name: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Map raw activity labels → contiguous cooked class indices [0..11]."""
    mapping = MAPPING_ACTIVITIES[subset_zip_name]
    cooked = COOK_ACTIVITIES[subset_zip_name]
    inv_dict = {v: k for k, v in dict_activities.items()}
    cooked_indices = np.zeros(len(y_raw), dtype=np.int64)
    seen: dict[str, int] = {}
    for i, y in enumerate(y_raw):
        raw_name = inv_dict[y]
        cooked_name = mapping.get(raw_name, "Other")
        idx = cooked[cooked_name] - _OFFSET
        cooked_indices[i] = idx
        seen[cooked_name] = int(idx)
    return cooked_indices, seen


def preprocess_subset(
    subset_zip_name: str, casas_root: Path, processed_dir: Path, maxlen: int, force: bool
) -> dict[str, Any]:
    rel = DATA_FILES[subset_zip_name]
    src = casas_root / rel
    if not src.exists():
        raise FileNotFoundError(f"missing CASAS events file for {subset_zip_name}: {src}")

    out_dir = processed_dir / subset_zip_name.replace(".", "_")
    out_npz = out_dir / "data.npz"
    if out_npz.exists() and not force:
        LOGGER.info("shard for %s already exists at %s; skipping", subset_zip_name, out_npz)
        with np.load(out_npz) as data:
            n = int(data["X"].shape[0])
        return {"subset": subset_zip_name, "rows": n, "path": str(out_npz), "status": "skipped"}

    LOGGER.info("preprocessing %s ← %s", subset_zip_name, src)
    x_seqs, y_raw, dict_activities = _load_raw(src)
    if not y_raw:
        raise RuntimeError(f"no events parsed from {src}")

    y, label_map = _remap_to_cooked(y_raw, dict_activities, subset_zip_name)
    X = _pad_pre(x_seqs, maxlen)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=X, y=y)
    (out_dir / "labels.json").write_text(json.dumps(label_map, sort_keys=True, indent=2))

    counts = Counter(int(v) for v in y.tolist())
    digest = hashlib.sha256(out_npz.read_bytes()).hexdigest()[:12]
    LOGGER.info(
        "wrote %s X=%s y_counts=%s sha256=%s…",
        out_npz, X.shape, dict(sorted(counts.items())), digest,
    )
    return {
        "subset": subset_zip_name,
        "rows": int(X.shape[0]),
        "path": str(out_npz),
        "sha256_12": digest,
        "class_counts": dict(sorted(counts.items())),
        "status": "ok",
    }


# ----------------------------------------------------------------------------
# Verify
# ----------------------------------------------------------------------------


def verify(processed_dir: Path, subset_zip_names: list[str], maxlen: int) -> None:
    for zip_name in subset_zip_names:
        out_dir = processed_dir / zip_name.replace(".", "_")
        out_npz = out_dir / "data.npz"
        if not out_npz.exists():
            raise AssertionError(f"missing shard: {out_npz}")
        with np.load(out_npz) as data:
            X, y = data["X"], data["y"]
        if X.ndim != 2 or X.shape[1] != maxlen:
            raise AssertionError(f"{out_npz}: expected X.shape=[N,{maxlen}], got {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise AssertionError(f"{out_npz}: X/y length mismatch {X.shape[0]} vs {y.shape[0]}")
        if y.min() < 0 or y.max() > 11:
            raise AssertionError(f"{out_npz}: y out of range [0,11]: min={y.min()} max={y.max()}")
        LOGGER.info("verified %s rows=%d classes=%d", zip_name, X.shape[0], len(set(y.tolist())))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and preprocess CASAS.")
    parser.add_argument("--config", type=Path, default=Path("fl/config/casas.yaml"))
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.home() / ".cache" / "consentchain" / "casas",
        help="Where to store the downloaded zip and its extraction.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and re-preprocess.")
    parser.add_argument("--verify", action="store_true", help="Validate written shards.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    ds_cfg = cfg["dataset"]
    subset_map: dict[str, str] = ds_cfg["fedaiot_subset_map"]
    subsets: list[str] = list(ds_cfg["subsets"])
    file_id: str = ds_cfg["gdrive_file_id"]
    maxlen: int = int(ds_cfg.get("window_size", DEFAULT_MAX_LEN))
    processed_dir = Path(ds_cfg["processed_dir"])

    # 1. Download + extract
    zip_path = args.workdir / "casas.zip"
    fetch_bundle(file_id, zip_path, args.force)
    extract_root = args.workdir / "extracted"
    extract_bundle(zip_path, extract_root, args.force)
    casas_root = find_casas_root(extract_root)

    # 2. Preprocess per-subset
    manifest: list[dict[str, Any]] = []
    for snake in subsets:
        zip_name = subset_map[snake]
        manifest.append(preprocess_subset(zip_name, casas_root, processed_dir, maxlen, args.force))
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 3. Verify
    if args.verify:
        verify(processed_dir, [subset_map[s] for s in subsets], maxlen)
    LOGGER.info("Done. Processed CASAS shards at %s", processed_dir.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

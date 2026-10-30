"""Official 2a signals; E truth is never decoded here."""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from utils import ROOT, SUBJECTS, save_json, sha


def spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def preprocess_signal(x):
    """Frozen offline pipeline: bandpass, notch, then crop elsewhere, resample."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    good = np.isfinite(x).all(axis=1) & ~np.all(x < -500, axis=1)
    sos = butter(4, [8, 30], btype="bandpass", fs=250, output="sos")
    b, a = iirnotch(50, 30, fs=250)
    for start, end in spans(good):
        if end - start > 40:
            out[start:end] = filtfilt(b, a, sosfiltfilt(sos, x[start:end], axis=0), axis=0)
    return out


def trial_zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-8)


def extract_record(path, dataset):
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    channels = 22 if dataset == "2a" else 3
    if raw.info["sfreq"] != 250 or len(raw.ch_names) != channels + 3:
        raise ValueError(f"Invalid channels/fs: {path}")
    # EOG is never passed to preprocessing or classifiers.
    x = raw.get_data(picks=list(range(channels))).T * 1e6
    filtered = preprocess_signal(x)
    events = sorted(
        (round(onset * 250), int(desc))
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if str(desc).isdigit()
    )
    starts = np.array([p for p, c in events if c == 768])
    markers = np.array([p for p, c in events if c == 32766])
    run = np.searchsorted(markers, starts, side="right") - 1
    epochs, labels, artifacts, cues = [], [], [], []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(x)
        ev = [(p, c) for p, c in events if start <= p < stop]
        cue = [(p, c) for p, c in ev if c in (769, 770, 771, 772, 783)]
        if len(cue) != 1:
            raise ValueError(f"{path}: ambiguous trial {i}")
        pos, code = cue[0]
        epoch = filtered[pos + 125 : pos + 875].T
        if epoch.shape != (channels, 750) or not np.isfinite(epoch).all():
            raise ValueError(f"{path}: gap or invalid epoch {i}")
        epochs.append(resample_poly(epoch, 64, 125, axis=-1))
        labels.append(code - 769 if code != 783 else -1)
        artifacts.append(any(c == 1023 for _, c in ev))
        cues.append(pos)
    train = Path(path).stem.endswith("T")
    y = np.array(labels) if train else np.full(len(starts), -1, dtype=int)
    if train and not set(y) == set(range(4 if dataset == "2a" else 2)):
        raise ValueError(f"Invalid T labels {path}")
    return {
        "x": np.stack(epochs).astype(np.float32),
        "y": y,
        "run": run,
        "artifact": np.array(artifacts),
        "start": starts,
        "cue": np.array(cues),
        "session": np.repeat(Path(path).stem, len(starts)),
    }


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    partial = target.with_suffix(target.suffix + ".partial")
    subprocess.run(
        [
            "curl.exe",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "1200",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(target)


def prepare():
    expected = json.loads((ROOT / "config/data_checksums.json").read_text(encoding="utf8"))
    raw = ROOT / "data/raw"
    existing = ROOT.parent / "bci2a_competition/data/raw"
    if all((existing / name).exists() for name in expected["recordings"]):
        raw = existing
    elif not all((raw / name).exists() for name in expected["recordings"]):
        archive = ROOT / "data/archives/BCICIV_2a_gdf.zip"
        download("https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip", archive)
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            if z.testzip():
                raise ValueError("GDF archive CRC failure")
            members = {Path(n).name: n for n in z.namelist() if n.endswith(".gdf")}
            if set(members) != set(expected["recordings"]):
                raise ValueError("Unexpected archive contents")
            for name, member in members.items():
                if not (raw / name).exists():
                    (raw / name).write_bytes(z.read(member))
    cache = ROOT / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    records = {}
    for subject in SUBJECTS:
        for split in ["T", "E"]:
            path = raw / f"{subject}{split}.gdf"
            if sha(path) != expected["recordings"][path.name]:
                raise ValueError(f"GDF checksum mismatch: {path.name}")
            block = extract_record(path, "2a")
            assert block["x"].shape == (288, 22, 384)
            assert np.isfinite(block["x"]).all()
            assert set(block["y"]) == ({0, 1, 2, 3} if split == "T" else {-1})
            np.savez_compressed(cache / f"{subject}_{split}.npz", **block)
            records[path.name] = {"sha256": sha(path), "trials": 288, "artifacts": int(block["artifact"].sum())}
            print(f"prepared {subject}{split}: 288 trials", flush=True)
    target = ROOT / "data/truth/2a_true_labels.zip"
    prior = ROOT.parent / "bci_cnn_course/data/truth/2a_true_labels.zip"
    if not target.exists() and prior.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prior, target)
    download("https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip", target)
    if sha(target) != expected["truth_archive_sha256"]:
        raise ValueError("Truth archive checksum mismatch")
    save_json(ROOT / "outputs/data_manifest.json", records)


def load(subject, split):
    if subject not in SUBJECTS or split not in ["T", "E"]:
        raise ValueError("Invalid recording")
    with np.load(ROOT / f"data/cache/{subject}_{split}.npz", allow_pickle=False) as z:
        x = trial_zscore(z["x"]).astype(np.float32)
        y = z["y"].copy()
        if split == "E":
            assert np.all(y == -1)
        return x, y

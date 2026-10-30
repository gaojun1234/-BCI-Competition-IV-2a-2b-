"""Read E ground truth only after all45 fixed-seed models/predictions are frozen."""

import io
import json
import zipfile
from contextlib import nullcontext

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from utils import ROOT, SEEDS, SUBJECTS, save_json, sha, source_hash


def verify_frozen():
    path = ROOT / "outputs/frozen_predictions.json"
    frozen = json.loads(path.read_text(encoding="utf8"))
    expected = {f"{s}_{seed}" for s in SUBJECTS for seed in SEEDS}
    if set(frozen["files"]) != expected:
        raise ValueError("All45 frozen predictions are required before truth access")
    if frozen["source_hash"] != source_hash():
        raise ValueError("Source changed after freezing")
    if frozen["recipe_sha256"] != sha(ROOT / "outputs/recipe_frozen.json"):
        raise ValueError("Recipe changed")
    for key, checks in frozen["files"].items():
        for kind, relative in [
            ("model", f"models/{key}.joblib"),
            ("prediction", f"predictions/{key}.npy"),
            ("status", f"runs/{key}.json"),
        ]:
            if sha(ROOT / "outputs" / relative) != checks[f"{kind}_sha256"]:
                raise ValueError(f"Frozen artifact changed: {relative}")
    return sha(path)


def read_truth(subject):
    with zipfile.ZipFile(ROOT / "data/truth/2a_true_labels.zip") as z:
        names = [n for n in z.namelist() if n.endswith(f"{subject}E.mat")]
        if len(names) != 1:
            raise ValueError("Ambiguous subject truth")
        y = np.asarray(loadmat(io.BytesIO(z.read(names[0])))["classlabel"]).ravel().astype(int) - 1
    assert y.shape == (288,) and set(y) == {0, 1, 2, 3}
    return y


def run():
    fingerprint = verify_frozen()
    rows = []
    reference_path = ROOT / "reference/probabilities.npz"
    reference_available = reference_path.exists()
    context = np.load(reference_path, allow_pickle=False) if reference_available else nullcontext(None)
    with context as reference:
        for subject in SUBJECTS:
            y = read_truth(subject)
            for seed in SEEDS:
                key = f"{subject}_{seed}"
                prob = np.load(ROOT / f"outputs/predictions/{key}.npy", allow_pickle=False)
                assert prob.shape == (288, 4) and np.isfinite(prob).all()
                np.testing.assert_allclose(prob.sum(1), 1, atol=1e-6)
                pred = prob.argmax(1)
                status = json.loads((ROOT / f"outputs/runs/{key}.json").read_text(encoding="utf8"))
                rows.append(
                    {
                        "dataset": "2a",
                        "subject": subject,
                        "model": "PowerCNN",
                        "seed": seed,
                        "accuracy": accuracy_score(y, pred),
                        "kappa": cohen_kappa_score(y, pred),
                        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                        "train_time": status["train_time"],
                        "candidate": status["candidate"],
                        "epochs": status["epochs"],
                        "parameters": status["parameters"],
                        "reference_label_matches": int((pred == reference[key].argmax(1)).sum())
                        if reference is not None
                        else None,
                        "max_probability_difference": float(abs(prob - reference[key]).max())
                        if reference is not None
                        else None,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "outputs/results.csv", index=False)
    per_subject = (
        frame.groupby("subject")
        .agg(
            accuracy=("accuracy", "mean"),
            seed_std=("accuracy", "std"),
            kappa=("kappa", "mean"),
            macro_f1=("macro_f1", "mean"),
            train_time=("train_time", "mean"),
        )
        .reset_index()
    )
    per_subject.to_csv(ROOT / "outputs/subject_results.csv", index=False)
    summary = {
        "accuracy_mean": float(per_subject.accuracy.mean()),
        "accuracy_subject_std": float(per_subject.accuracy.std()),
        "kappa_mean": float(per_subject.kappa.mean()),
        "macro_f1_mean": float(per_subject.macro_f1.mean()),
        "models": 45,
        "subjects": 9,
        "seeds": SEEDS,
        "frozen_sha256": fingerprint,
        "reference_accuracy": 0.6318672839506172,
        "reference_prediction_file_available": reference_available,
        "reference_prediction_labels_matched": int(frame.reference_label_matches.sum())
        if reference_available
        else None,
        "reference_prediction_labels_total": 12960 if reference_available else None,
        "max_probability_difference": float(frame.max_probability_difference.max())
        if reference_available
        else None,
        "prior_E_access_disclosed": True,
    }
    save_json(ROOT / "outputs/summary.json", summary)
    print(per_subject.to_string(index=False))
    print(json.dumps(summary, indent=2))

"""Fit the fixed T-selected PowerCNN recipes; never decode E labels."""

import json
import zipfile

import joblib
import numpy as np
import torch

from data import load
from model import fit, model_for
from utils import ROOT, SEEDS, SUBJECTS, recipes, save_json, sha, source_hash


def run(device="auto", verify_only=False):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    selected = recipes()
    assert set(selected) == set(SUBJECTS)
    source = source_hash()
    files = {}
    directory = ROOT / "outputs/models"
    directory.mkdir(parents=True, exist_ok=True)
    probabilities = ROOT / "outputs/predictions"
    probabilities.mkdir(parents=True, exist_ok=True)
    save_json(ROOT / "outputs/recipe_frozen.json", {"source_hash": source, "recipes": selected, "seeds": SEEDS})
    if verify_only:
        checks = json.loads((ROOT / "pretrained/checksums.json").read_text(encoding="utf8"))
        with zipfile.ZipFile(ROOT / "pretrained/models.zip") as z:
            if z.testzip():
                raise ValueError("Pretrained model ZIP CRC failure")
            if set(z.namelist()) != set(checks):
                raise ValueError("Unexpected pretrained models")
            for subject in SUBJECTS:
                for seed in SEEDS:
                    name = f"{subject}_power_extension_{seed}.joblib"
                    target = directory / f"{subject}_{seed}.joblib"
                    target.write_bytes(z.read(name))
                    if sha(target) != checks[name]:
                        raise ValueError("Pretrained model hash mismatch")
    for subject in SUBJECTS:
        x, y = load(subject, "T")
        xe, ye = load(subject, "E")
        assert np.all(ye == -1) and set(y) == {0, 1, 2, 3}
        choice = selected[subject]
        cfg = choice["config"]
        for seed in SEEDS:
            key = f"{subject}_{seed}"
            path = directory / f"{key}.joblib"
            if verify_only:
                artifact = joblib.load(path)
                assert artifact["config"] == cfg
                model = model_for(cfg | {"csp_init": False}).to(device)
                model.load_state_dict(artifact["state_dict"], strict=True)
                metadata = {"train_time": 0.0, "history": []}
            else:
                model, metadata = fit(x, y, cfg, seed, device, epochs=choice["epochs"])
                joblib.dump(
                    {
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "config": cfg,
                        "candidate": choice["candidate"],
                        "epoch_count": choice["epochs"],
                    },
                    path,
                    compress=3,
                )
            model.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(xe[:, None], device=device)
                prob = torch.cat([model(b).softmax(1) for b in tensor.split(64)]).cpu().numpy()
            pred = probabilities / f"{key}.npy"
            np.save(pred, prob)
            record = {
                "subject": subject,
                "seed": seed,
                "candidate": choice["candidate"],
                "epochs": choice["epochs"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_time": metadata["train_time"],
                "history": metadata["history"],
                "source_hash": source,
                "model_sha256": sha(path),
                "prediction_sha256": sha(pred),
                "T_cache_sha256": sha(ROOT / f"data/cache/{subject}_T.npz"),
                "E_cache_sha256": sha(ROOT / f"data/cache/{subject}_E.npz"),
                "test_labels_used": False,
                "mode": "verify_pretrained" if verify_only else "retrain",
            }
            status = ROOT / f"outputs/runs/{key}.json"
            save_json(status, record)
            files[key] = {"model_sha256": sha(path), "prediction_sha256": sha(pred), "status_sha256": sha(status)}
            print(
                f"done {key} {choice['candidate']} epochs={choice['epochs']} parameters={record['parameters']} time={record['train_time']:.3f}s",
                flush=True,
            )
    assert len(files) == 45
    save_json(
        ROOT / "outputs/frozen_predictions.json",
        {"source_hash": source, "files": files, "recipe_sha256": sha(ROOT / "outputs/recipe_frozen.json")},
    )

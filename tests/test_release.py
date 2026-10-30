import inspect
import json

import numpy as np
import pytest
import torch

import score
import train
from data import trial_zscore
from model import PowerCNN, constrain, model_for
from utils import SEEDS, SUBJECTS, recipes


def test_fixed_subject_recipes_and_shapes():
    selected = recipes()
    assert set(selected) == set(SUBJECTS) and SEEDS == [11, 29, 47, 71, 101]
    for choice in selected.values():
        assert choice["candidate"].startswith("power_")
        assert 1 <= choice["epochs"] <= 200
        m = model_for(choice["config"] | {"csp_init": False})
        assert isinstance(m, PowerCNN) and m(torch.zeros(2, 1, 22, 384)).shape == (2, 4)
        assert sum(p.numel() for p in m.parameters()) in [3156, 3732]
        constrain(m, choice["config"])
        assert m.dense.weight.norm(dim=1).max() <= 0.500001


def test_zscore_and_no_truth_reader_in_training():
    x = np.random.default_rng(4).normal(size=(8, 22, 384)).astype(np.float32)
    norm = trial_zscore(x)
    np.testing.assert_allclose(norm.mean(-1), 0, atol=1e-6)
    np.testing.assert_allclose(norm.var(-1), 1, atol=1e-6)
    assert "loadmat" not in inspect.getsource(train) and "read_truth" not in inspect.getsource(train)


def test_partial_batch_rejected_before_reading_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "ROOT", tmp_path)
    monkeypatch.setattr(score, "read_truth", lambda *a: pytest.fail("Premature truth access"))
    target = tmp_path / "outputs"
    target.mkdir()
    (target / "frozen_predictions.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(ValueError, match="All45"):
        score.run()

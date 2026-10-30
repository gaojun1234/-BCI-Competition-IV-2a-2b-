"""PowerCNN only: temporal/spatial filtering followed by log-power pooling."""

import random
import time

import numpy as np
import torch
from scipy.linalg import eigh
from torch import nn


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def spatial_csp_filters(x, y):
    channels = x.shape[1]
    cov = x @ x.transpose(0, 2, 1)
    cov /= np.trace(cov, axis1=1, axis2=2)[:, None, None]
    rows = []
    for label in np.unique(y):
        a, b = cov[y == label].mean(0), cov[y != label].mean(0)
        a = 0.9 * a + 0.1 * np.eye(channels) * np.trace(a) / channels
        b = 0.9 * b + 0.1 * np.eye(channels) * np.trace(b) / channels
        _, w = eigh(a, a + b)
        pairs = min(2, channels // 2)
        rows.extend(w[:, np.r_[np.arange(pairs), np.arange(channels - pairs, channels)]].T)
    return np.stack(rows)


class PowerCNN(nn.Module):
    """Explicit extension: learned temporal/spatial filters followed by log power."""

    def __init__(self, local=False):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv2d(1, 8, (1, 25), stride=(1, 2)), nn.BatchNorm2d(8))
        self.spatial = nn.Sequential(nn.Conv2d(8, 16, (22, 1)), nn.BatchNorm2d(16))
        self.local = local
        self.pool = nn.AvgPool2d((1, 32), stride=(1, 16)) if local else nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.dense = nn.Linear(16 * (10 if local else 1), 4)

    def forward(self, x):
        x = self.spatial(self.temporal(x))
        x = self.pool(x.square()).clamp_min(1e-6).log()
        return self.dense(self.dropout(x).flatten(1))


def model_for(cfg, x=None, y=None):
    model = PowerCNN(cfg["family"] == "power_local")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = cfg["bn_eps"]
            module.momentum = cfg["bn_momentum"]
        if cfg["xavier"] and isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    if cfg["csp_init"]:
        if x is None or y is None:
            raise ValueError("CSP initialization requires training-fold data")
        csp_filters = spatial_csp_filters(x, y)
        with torch.no_grad():
            temporal = model.temporal[0]
            temporal.weight.zero_()
            temporal.weight[:, 0, 0, 12] = 1.0
            temporal.weight[1:].add_(torch.randn_like(temporal.weight[1:]) * 0.01)
            temporal.bias.zero_()
            spatial = model.spatial[0]
            spatial.weight.zero_()
            vectors = csp_filters / np.linalg.norm(csp_filters, axis=1, keepdims=True)
            spatial.weight[:, 0, :, 0] = torch.as_tensor(vectors, dtype=torch.float32)
            spatial.bias.zero_()
    assert sum(p.numel() for p in model.parameters()) < 5000
    return model


def constrain(model, cfg):
    with torch.no_grad():
        for layer, bound in [(model.spatial[0], cfg["spatial_max"]), (model.dense, cfg["dense_max"])]:
            if bound:
                weight = layer.weight
                norms = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
                scale = (bound / norms).clamp_max(1)
                weight.mul_(scale.reshape(-1, *([1] * (weight.ndim - 1))))


def fit(x, y, cfg, seed, device="cuda", validation=None, epochs=200):
    seed_all(seed)
    model = model_for(cfg, x, y).to(device)
    constrain(model, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    tx = torch.as_tensor(x[:, None], device=device)
    ty = torch.as_tensor(y, dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(seed + 10000)
    if validation is not None:
        vx = torch.as_tensor(validation[0][:, None], device=device)
        vy = torch.as_tensor(validation[1], dtype=torch.long, device=device)
    history, best, best_epoch, stale = [], float("inf"), 1, 0
    best_accuracy = 0.0
    begin = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(tx), generator=rng, device=device).split(32):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tx[idx]), ty[idx])
            loss.backward()
            optimizer.step()
            constrain(model, cfg)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(batch) for batch in tx.split(64)])
            row = {
                "epoch": epoch,
                "train_loss": criterion(logits, ty).item(),
                "train_accuracy": (logits.argmax(1) == ty).float().mean().item(),
            }
            if validation is not None:
                logits = torch.cat([model(batch) for batch in vx.split(64)])
                val_loss = criterion(logits, vy).item()
                row.update(
                    validation_loss=val_loss, validation_accuracy=(logits.argmax(1) == vy).float().mean().item()
                )
                if val_loss < best - 0.0001:
                    best, best_epoch, stale = val_loss, epoch, 0
                    best_accuracy = row["validation_accuracy"]
                else:
                    stale += 1
        history.append(row)
        if validation is not None and stale >= 30:
            break
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return model, {
        "history": history,
        "selected_epoch": best_epoch,
        "validation_loss": best,
        "validation_accuracy": best_accuracy,
        "train_time": time.perf_counter() - begin,
    }

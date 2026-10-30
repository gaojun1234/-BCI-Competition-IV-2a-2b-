"""Standalone entry point: fixed best-model recipes, no architecture search."""

import argparse
import platform
import sys

from utils import ROOT, SEEDS, save_json, source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "score", "all"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    (ROOT / "outputs").mkdir(exist_ok=True)
    if args.command in ["prepare", "all"]:
        from data import prepare

        prepare()
    if args.command in ["train", "all"]:
        import torch

        from train import run

        save_json(
            ROOT / "outputs/environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "seeds": SEEDS,
                "source_hash": source_hash(),
                "executable": sys.executable,
            },
        )
        run(args.device, args.verify_only)
    if args.command in ["score", "all"]:
        from score import run

        run()


if __name__ == "__main__":
    main()

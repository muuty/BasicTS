#!/usr/bin/env python3
"""Reduced-data models evaluated on the full training split.

Proposition 1 bounds the discrepancy between the risk under the full empirical
training measure and the risk under the retained measure, for a fixed forecasting
model. The quantity it controls is therefore the error of a reduced-data model on
the *full training split*, not its error on the held-out test period. The manuscript
only ever correlates the quantization error against test MAE, which adds the
generalisation gap to whatever the bound governs.

This script closes that gap. For every trained checkpoint in a run grid it rebuilds
the training configuration, loads the saved best model, and evaluates it on the
complete training split: the same windows the selection objective chose from, with
no subsetting applied. The MAE is masked exactly as in training, streamed batch by
batch so the full prediction tensor is never held.

Nothing in the checkpoint directory is overwritten: test_pipeline is called with
save_metrics and save_results both false.

Writes experiments/result/analysis/train_split_mae.csv with one row per checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from easydict import EasyDict  # noqa: E402
from easytorch.config import convert_config, import_config  # noqa: E402
from easytorch.device import set_device_type  # noqa: E402

OUT = REPO / "experiments" / "result" / "analysis"


DATASET_CACHE: dict = {}


def make_runner_class(base):
    """Subclass that evaluates on the unsubsetted training split."""

    class TrainSplitRunner(base):
        def build_test_dataset(self, cfg):
            # Building the split costs about as much as the evaluation itself, and
            # consecutive runs share it, so one build serves every run on a network.
            key = (cfg["DATASET"]["NAME"], json.dumps(cfg["DATASET"]["PARAM"],
                                                      sort_keys=True, default=str))
            if key not in DATASET_CACHE:
                DATASET_CACHE.clear()
                DATASET_CACHE[key] = cfg["DATASET"]["TYPE"](
                    mode="train", logger=self.logger, **cfg["DATASET"]["PARAM"])
            dataset = DATASET_CACHE[key]
            self.logger.info(f"Full training split length: {len(dataset)}")
            return dataset

        @torch.no_grad()
        def test(self, train_epoch=None, save_metrics=False, save_results=False):
            """Per-window masked absolute error over the whole training split.

            masked_mae normalises by the count of valid entries, so summing absolute
            errors over valid entries and dividing by their count reproduces it. Keeping
            the sums per window lets the same pass report the risk under the full
            empirical measure and under the retained measure, whose difference is the
            quantity Proposition~1 bounds. The loader is not shuffled, so batch order
            follows dataset index.
            """
            null_val = self.null_val
            cap = int(os.environ.get("TRAIN_SPLIT_MAX_BATCHES", 0))
            sums, counts = [], []
            for batch_idx, data in enumerate(self.test_data_loader):
                if cap and batch_idx >= cap:
                    break
                out = self.forward(data, epoch=None, iter_num=None, train=False)
                pred, target = out["prediction"], out["target"]
                if np.isnan(null_val):
                    mask = ~torch.isnan(target)
                else:
                    null = torch.tensor(null_val).to(target.device, target.dtype)
                    mask = ~torch.isclose(target, null, atol=5e-5, rtol=0.0)
                mask = mask.float()
                err = torch.nan_to_num(torch.abs(pred - target) * mask)
                flat = tuple(range(1, err.dim()))
                sums.append(err.sum(dim=flat).detach().cpu())
                counts.append(mask.sum(dim=flat).detach().cpu())
            self._window_abs_sum = torch.cat(sums).numpy()
            self._window_valid = torch.cat(counts).numpy()
            return {}

    return TrainSplitRunner


CANDIDATE_CONFIGS = [
    "baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py",
    "baselines/AGCRN/SAN_BERNARDINO/SAN_BERNARDINO.py",
    "baselines/DCRNN/SAN_BERNARDINO/SAN_BERNARDINO.py",
    "baselines/STID/SAN_BERNARDINO/SAN_BERNARDINO.py",
    "baselines/STAEformer/SAN_BERNARDINO/SAN_BERNARDINO.py",
    "baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py",
    "baselines/AGCRN/CONTRA_COSTA/CONTRA_COSTA.py",
    "baselines/DCRNN/CONTRA_COSTA/CONTRA_COSTA.py",
    "baselines/STID/CONTRA_COSTA/CONTRA_COSTA.py",
    "baselines/STAEformer/CONTRA_COSTA/CONTRA_COSTA.py",
]

SCRATCH = Path("experiments/result/analysis/_train_split_eval_logs")


def config_index() -> dict:
    """(architecture directory, dataset directory) -> canonical config path."""
    index = {}
    for path in CANDIDATE_CONFIGS:
        parts = Path(path).parts
        index[(parts[1], parts[2])] = path
    return index


def parse_cfg_txt(path: Path) -> dict:
    """The CORESET block a run was trained with, as written into its checkpoint."""
    out, inside = {}, False
    for line in path.read_text().splitlines():
        if line.startswith("CORESET:"):
            inside = True
            continue
        if inside:
            if not line.startswith("  ") or line.startswith("   "):
                if not line.startswith("  "):
                    break
            m = re.match(r"  (\w+): (.+)", line)
            if not m:
                break
            key, value = m.group(1), m.group(2).strip()
            try:
                value = int(value) if re.fullmatch(r"-?\d+", value) else float(value)
            except ValueError:
                pass
            out[key] = value
    return out


def discover(roots: list[str]) -> list[dict]:
    """Every completed run under the given checkpoint roots, keyed by its directory."""
    index = config_index()
    found = []
    for root in roots:
        base = REPO / "checkpoints" / root
        if not base.is_dir():
            continue
        for cfg_txt in base.glob("*/*/*/*/*/cfg.txt"):
            d = cfg_txt.parent
            if not any(d.glob("*_best_val_MAE.pt")) or not (d / "test_metrics.json").exists():
                continue
            arch_dir = cfg_txt.relative_to(base).parts[0]
            dataset_dir = cfg_txt.relative_to(base).parts[2].rsplit("_50_12_12", 1)[0]
            arch_key = {"STGCNChebGraphConv": "STGCN"}.get(arch_dir, arch_dir)
            config = index.get((arch_key, dataset_dir))
            if config is None:
                continue
            found.append({"dir": d, "config": config, "root": root,
                          "coreset": parse_cfg_txt(cfg_txt)})
    return sorted(found, key=lambda r: (r["config"], str(r["dir"])))


def evaluate_one(entry: dict) -> dict:
    cfg = import_config(entry["config"], verbose=False)
    coreset = entry["coreset"]
    if coreset:
        cfg["CORESET"] = EasyDict(coreset)
    else:
        cfg.pop("CORESET", None)
    # The evaluation writes only its own log; the checkpoint directory is untouched.
    cfg["TRAIN"]["CKPT_SAVE_DIR"] = str(SCRATCH)
    cfg["EVAL"] = cfg.get("EVAL", EasyDict())
    cfg["EVAL"]["USE_GPU"] = False

    directory = entry["dir"]
    ckpt = next(iter(directory.glob("*_best_val_MAE.pt")))
    test_mae = json.loads((directory / "test_metrics.json").read_text())["overall"]["MAE"]

    cfg["RUNNER"] = make_runner_class(cfg["RUNNER"])
    converted = convert_config(dict(cfg))
    runner = cfg["RUNNER"](converted)
    runner.init_logger(logger_name="train-split-eval", log_file_name="train_split_eval")
    if runner.need_setup_graph:
        runner.setup_graph(cfg=converted, train=False)
    runner.load_model(ckpt_path=str(ckpt), strict=True)
    runner.test_pipeline(cfg=converted, save_metrics=False, save_results=False)

    abs_sum, valid = runner._window_abs_sum, runner._window_valid

    def mae(selector) -> float:
        total = valid[selector].sum()
        return float(abs_sum[selector].sum() / total) if total else float("nan")

    keep = np.zeros(len(abs_sum), dtype=bool)
    index_file = directory / "coreset-selection.json"
    if index_file.exists():
        idx = np.asarray(json.loads(index_file.read_text()), dtype=int)
        keep[idx[idx < len(keep)]] = True
    else:
        keep[:] = True

    full_mae, retained_mae, dropped_mae = mae(slice(None)), mae(keep), mae(~keep)
    row = {
        "root": entry["root"],
        "ckpt_dir": str(directory.relative_to(REPO)),
        "architecture": cfg["MODEL"]["NAME"],
        "dataset": cfg["DATASET"]["NAME"],
        "method": coreset.get("SELECTION_STRATEGY", "full"),
        "ratio": coreset.get("SELECTION_RATIO", 1.0),
        "coreset_seed": coreset.get("SEED", 42),
        "test_mae": test_mae,
        "train_full_mae": full_mae,
        "train_retained_mae": retained_mae,
        "train_dropped_mae": dropped_mae,
        "measure_gap": abs(full_mae - retained_mae),
        "n_windows": int(len(abs_sum)),
        "n_retained": int(keep.sum()),
    }
    del runner
    torch.cuda.empty_cache()
    return row


def main() -> None:
    args = parse_args()
    set_device_type(args.device)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    entries = discover(args.roots)
    print(f"{len(entries)} completed runs discovered", flush=True)
    if args.shards > 1:
        edges = np.linspace(0, len(entries), args.shards + 1).astype(int)
        entries = entries[edges[args.shard]:edges[args.shard + 1]]
        print(f"shard {args.shard} of {args.shards}: {len(entries)} runs", flush=True)
    if args.limit:
        entries = entries[:args.limit]

    out_path = OUT / args.out
    rows, done = [], set()
    if out_path.exists() and not args.overwrite:
        prev = pd.read_csv(out_path)
        rows, done = prev.to_dict("records"), set(prev.ckpt_dir)
        print(f"resuming, {len(done)} already evaluated", flush=True)

    for k, entry in enumerate(entries, 1):
        if str(entry["dir"].relative_to(REPO)) in done:
            continue
        try:
            row = evaluate_one(entry)
        except Exception:
            print(f"  [{k}/{len(entries)}] FAILED {entry['dir']}\n{traceback.format_exc()}",
                  flush=True)
            continue
        rows.append(row)
        done.add(row["ckpt_dir"])
        print(f"  [{k}/{len(entries)}] {row['architecture']:<20} {row['dataset']:<16} "
              f"{row['method']:<10} r={row['ratio']} seed={row['coreset_seed']}  "
              f"full {row['train_full_mae']:.4f}  retained {row['train_retained_mae']:.4f}  "
              f"test {row['test_mae']:.4f}", flush=True)
        if len(rows) % 10 == 0:
            OUT.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {out_path} with {len(rows)} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+",
                        default=["phase_c_method_comparison", "phase_c_extra_ratios"])
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--out", default="train_split_mae.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

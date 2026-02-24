#!/usr/bin/env python3
"""
CLI-only Parallel Runner (No YAML)

Fixes:
- Cache check now searches recursively under CKPT_SAVE_DIR for test_metrics.json
- CKPT_SAVE_DIR is made unique per setting (grouping/clients/min/imb/alpha/seed + hiersplit pooling/tokens)
- RunID includes hiersplit pooling/tokens to avoid CSV/log collisions
"""

import os
import sys
import argparse
import time
import json
import re
import subprocess
import signal
import csv
import threading
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional

import pandas as pd
import importlib.util
import hashlib

CSV_LOCK = threading.Lock()

DEFAULT_DATASETS = "METR_LA,PEMS03,PEMS04,PEMS07,PEMS08,PEMS_BAY,PEMS04_SPEED,PEMS08_SPEED,NYCBike,NYCTaxi2015,NYCTaxi2016,Porto"
DEFAULT_METHODS  = "independent_clientwise,central,split,hiersplit"
DEFAULT_MODELS   = "staeformer,stgformer,gruseq2seq_graphnet"
DEFAULT_GROUPINGS= "maxcut,random,metis"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def split_ints(s: Optional[str]) -> List[int]:
    return [int(x) for x in split_list(s)]

def split_floats(s: Optional[str]) -> List[float]:
    return [float(x) for x in split_list(s)]

def norm_dataset(ds: str) -> str:
    return ds.strip().replace("-", "_").upper()

def slug_float(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")

def is_central(method: str) -> bool:
    return "central" in method.lower()

def safe_log_name(run_id: str) -> str:
    return run_id.replace("/", "__")

def parse_num_clients_from_script(script_path: str) -> Optional[int]:
    try:
        txt = Path(script_path).read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.search(r"NUM_CLIENTS\s*=\s*(\d+)", txt)
    return int(m.group(1)) if m else None

def parse_total_nodes_from_script(script_path: str) -> Optional[int]:
    try:
        txt = Path(script_path).read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.search(r"TOTAL_NODES\s*=\s*(\d+)", txt)
    return int(m.group(1)) if m else None


def resolve_base_script(base_root: str, method: str, model: str, grouping: Optional[str], dataset: str) -> str:
    ds = norm_dataset(dataset)
    ds_original = dataset.strip()
    root = Path(base_root)

    method = method.strip()
    model = model.strip()

    cand_dirs = [root / method / model]
    candidates = [
        f"{ds}.py",
        f"{ds.lower()}.py",
        f"{ds_original}.py",
        f"{ds_original.lower()}.py",
        f"{ds_original.capitalize()}.py",
    ]
    candidates = list(dict.fromkeys(candidates))

    for d in cand_dirs:
        for fn in candidates:
            p = d / fn
            if p.exists():
                return str(p)

        if d.exists():
            pats = (
                list(d.glob(f"{ds}*.py")) +
                list(d.glob(f"{ds.lower()}*.py")) +
                list(d.glob(f"{ds_original}*.py")) +
                list(d.glob(f"{ds_original.lower()}*.py"))
            )
            if pats:
                pats.sort(key=lambda x: (len(x.name), x.name))
                return str(pats[0])

    raise FileNotFoundError(f"Cannot find base script under {cand_dirs} for dataset={ds}")


def make_run_id(
    dataset: str,
    method: str,
    model: str,
    grouping: Optional[str],
    clients: Optional[int],
    imbalance: Optional[float],
    alpha: Optional[float],
    seed: Optional[int],
    min_nodes: int,
    hs_pooling: Optional[str] = None,
    hs_tokens: Optional[int] = None,
) -> str:
    ds = norm_dataset(dataset)
    mth = method.lower()
    mdl = model.lower()
    grp = (grouping.lower() if grouping else "none")

    parts = [ds, mth, mdl, grp]

    if clients is not None:
        parts.append(f"c{clients}")
    if min_nodes != 1:
        parts.append(f"min{min_nodes}")

    if alpha is not None:
        parts.append(f"a{slug_float(alpha)}")
    elif imbalance is not None:
        parts.append(f"imb{slug_float(imbalance)}")

    if seed is not None:
        parts.append(f"s{seed}")

    if "hiersplit" in mth:
        if hs_pooling:
            parts.append(f"pool{hs_pooling.lower()}")
        if hs_tokens is not None:
            parts.append(f"t{hs_tokens}")

    return "/".join(parts)


def find_latest_test_metrics(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    cands = list(ckpt_dir.rglob("test_metrics.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def read_metrics(test_metrics_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}

    path = test_metrics_path
    if not path.exists():
        parent = test_metrics_path.parent
        candidates = list(parent.rglob("test_metrics.json"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime)
            path = candidates[-1]
        else:
            return out

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out

    if "overall" in data:
        out["MAE"]  = f"{data['overall'].get('MAE', 0):.4f}"
        out["RMSE"] = f"{data['overall'].get('RMSE', 0):.4f}"
        out["MAPE"] = f"{data['overall'].get('MAPE', 0):.4f}"

    for h_key, suffix in [("horizon_3", "_h3"), ("horizon_6", "_h6"), ("horizon_12", "_h12")]:
        if h_key in data:
            out[f"MAE{suffix}"]  = f"{data[h_key].get('MAE', 0):.4f}"
            out[f"RMSE{suffix}"] = f"{data[h_key].get('RMSE', 0):.4f}"
            out[f"MAPE{suffix}"] = f"{data[h_key].get('MAPE', 0):.4f}"

    return out


def initialize_csv(rows: List[Dict], csv_path: Path):
    headers = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

def update_csv(csv_path: Path, run_id: str, status: str, duration: str, metrics_path: Path):
    metrics = read_metrics(metrics_path) if status in ("Success", "Cached") else {}

    with CSV_LOCK:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        for r in rows:
            if r["RunID"] == run_id:
                r["Status"] = status
                r["Duration"] = duration
                for k, v in metrics.items():
                    if k in r:
                        r[k] = v
                break

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)


_CFG_CACHE: Dict[str, Dict[str, object]] = {}

def load_cfg_meta(script_path: str) -> Dict[str, object]:
    if script_path in _CFG_CACHE:
        return _CFG_CACHE[script_path]

    meta: Dict[str, object] = {}
    try:
        h = hashlib.md5(script_path.encode("utf-8")).hexdigest()[:10]
        module_name = f"cfg_{h}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # executes cfg
        cfg = getattr(mod, "CFG", None)
        if cfg is not None and hasattr(cfg, "TRAIN") and hasattr(cfg.TRAIN, "CKPT_SAVE_DIR"):
            meta["ckpt_save_dir"] = str(cfg.TRAIN.CKPT_SAVE_DIR)

        # hiersplit defaults
        hs = getattr(cfg, "HIERSPLIT", None) if cfg is not None else None
        if hs is not None:
            meta["hs_pooling"] = getattr(hs, "POOLING", None)
            meta["hs_tokens"] = getattr(hs, "NUM_TOKENS", None)
    except Exception:
        # leave empty; fallback will be used
        pass

    _CFG_CACHE[script_path] = meta
    return meta


def build_unique_ckpt_dir(
    script_path: str,
    method: str,
    model: str,
    dataset: str,
    grouping: Optional[str],
    clients: Optional[int],
    min_nodes: int,
    imb: Optional[float],
    alpha: Optional[float],
    seed: Optional[int],
    hs_pooling_override: Optional[str],
    hs_tokens_override: Optional[object],  # None | int | "auto"
    auto_tokens: Optional[int],
) -> (Path, Optional[str], Optional[int]):
    """
    Returns (ckpt_dir, final_hs_pooling, final_hs_tokens)
    """
    meta = load_cfg_meta(script_path)
    base_ckpt = meta.get("ckpt_save_dir", None)
    if base_ckpt is None:
        # fallback base
        base_ckpt = str(Path("checkpoints") / f"{method}_{model}" / norm_dataset(dataset))

    base = Path(base_ckpt)

    # method separation: hiersplit should not live under IL_* even if cfg has IL_*
    if "hiersplit" in method.lower():
        parent = base.parent
        p = parent.name
        if p.lower().startswith("il_"):
            p = "HierSplit_" + p[3:]
        elif not p.lower().startswith("hiersplit_"):
            p = "HierSplit_" + p
        base = parent.parent / p / base.name

    # resolve final hiersplit params (even if not overridden)
    final_pool = None
    final_tok = None
    if "hiersplit" in method.lower():
        final_pool = (hs_pooling_override or meta.get("hs_pooling", None) or None)
        if hs_tokens_override == "auto":
            final_tok = auto_tokens
        elif isinstance(hs_tokens_override, int):
            final_tok = int(hs_tokens_override)
        else:
            v = meta.get("hs_tokens", None)
            final_tok = int(v) if v is not None else None

    tags: List[str] = []

    if not is_central(method):
        grp = (grouping or "random").lower()
        tags.append(grp)
        if clients is not None:
            tags.append(f"c{int(clients)}")
        if min_nodes != 1:
            tags.append(f"min{int(min_nodes)}")
        if alpha is not None:
            tags.append(f"a{slug_float(float(alpha))}")
        elif imb is not None:
            tags.append(f"imb{slug_float(float(imb))}")
        if seed is not None:
            tags.append(f"s{int(seed)}")

    if "hiersplit" in method.lower():
        if final_pool:
            tags.append(final_pool.lower())
        if final_tok is not None:
            tags.append(f"t{int(final_tok)}")

    new_name = base.name + ("_" + "_".join(tags) if tags else "")
    return base.with_name(new_name), final_pool, final_tok


def launch_proc(gpu_id: int, visible_gpus: List[int], script_path: str, run_id: str, overrides: Dict, log_root: Path):
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_gpus))
    local_idx = visible_gpus.index(gpu_id)

    env["BASICTS_OVERRIDE_JSON"] = json.dumps(overrides, ensure_ascii=False)

    cmd = [sys.executable, "experiments/run_experiment_jy.py", "-c", script_path, "-g", str(local_idx)]

    log_path = log_root / f"{safe_log_name(run_id)}.log"
    lf = open(log_path, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return p, lf


def main():
    parser = argparse.ArgumentParser("CLI-only parallel experiment runner (cached checkpoints)")

    parser.add_argument("--base-root", default="new_baselines_jy", help="base config root")
    parser.add_argument("--datasets",  default=DEFAULT_DATASETS)
    parser.add_argument("--methods",   default=DEFAULT_METHODS)
    parser.add_argument("--models",    default=DEFAULT_MODELS)
    parser.add_argument("--groupings", default=DEFAULT_GROUPINGS)

    parser.add_argument("--clients", default="")
    parser.add_argument("--imbalance", default=None)
    parser.add_argument("--imbalance-alpha", dest="imbalance_alpha", default=None)
    parser.add_argument("--partition-seeds", default="")
    parser.add_argument("--min-nodes", type=int, default=1)

    parser.add_argument("--pooling-method", default="")
    parser.add_argument("--num-tokens", default="")

    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("-w", "--workers", type=int, default=2)

    args = parser.parse_args()

    datasets = split_list(args.datasets)
    methods = split_list(args.methods)
    models = split_list(args.models)
    groupings = split_list(args.groupings)
    clients_list = split_ints(args.clients) if args.clients else []

    imb_list = split_floats(args.imbalance) if args.imbalance else [None]
    alpha_list = split_floats(args.imbalance_alpha) if args.imbalance_alpha else [None]
    if any(x is not None for x in imb_list) and any(x is not None for x in alpha_list):
        raise ValueError("Use either --imbalance or --imbalance-alpha, not both.")

    seeds = split_ints(args.partition_seeds) if args.partition_seeds else [None]

    pooling_methods = split_list(args.pooling_method) if args.pooling_method else []
    num_tokens_list = split_list(args.num_tokens) if args.num_tokens else []

    visible_gpus = [int(x) for x in split_list(args.gpu)]
    jobs_per_gpu = args.workers

    results_group = time.strftime("run_%Y%m%d_%H%M%S")
    results_dir = Path("experiments_results") / results_group
    ensure_dir(results_dir)
    csv_path = results_dir / f"results_{results_group}.csv"
    log_root = results_dir

    seen = set()
    specs = []
    skipped = 0
    cached = 0

    for method in methods:
        for model in models:
            for ds in datasets:
                if is_central(method):
                    eff_groupings = [None]
                    eff_clients = [None]
                    eff_imb_list = [None]
                    eff_alpha_list = [None]
                    eff_seeds = [None]
                else:
                    eff_groupings = groupings
                    eff_clients = clients_list if clients_list else [None]
                    eff_imb_list = imb_list
                    eff_alpha_list = alpha_list
                    eff_seeds = seeds

                for grouping in eff_groupings:
                    try:
                        script_path = resolve_base_script(args.base_root, method, model, grouping, ds)
                    except FileNotFoundError as e:
                        print(f"[SKIP] {e}")
                        skipped += 1
                        continue

                    base_clients = parse_num_clients_from_script(script_path)

                    is_hs = "hiersplit" in method.lower()
                    base_total_nodes = parse_total_nodes_from_script(script_path) if is_hs else None

                    # ✅ hiersplit에서만 pooling/num_tokens sweep
                    if is_hs:
                        eff_pooling_methods = pooling_methods if pooling_methods else [None]
                        eff_num_tokens_list = [None] if not num_tokens_list else [
                            ("auto" if nt.lower() == "auto" else int(nt)) for nt in num_tokens_list
                        ]
                    else:
                        eff_pooling_methods = [None]
                        eff_num_tokens_list = [None]

                    for c in eff_clients:
                        clients = c
                        if not is_central(method) and clients is None:
                            if base_clients is None:
                                print(f"[SKIP] clients not provided and NUM_CLIENTS not found in {script_path}")
                                skipped += 1
                                continue
                            clients = base_clients

                        auto_num_tokens = None
                        if "hiersplit" in method.lower() and base_total_nodes is not None and clients is not None:
                            auto_num_tokens = round(base_total_nodes / int(clients) / 5)

                        for seed in eff_seeds:
                            for imb in eff_imb_list:
                                for a in eff_alpha_list:
                                    for pooling_method in eff_pooling_methods:
                                        for num_tokens in eff_num_tokens_list:

                                            overrides: Dict[str, object] = {}

                                            if not is_central(method):
                                                overrides["IL.NUM_CLIENTS"] = int(clients)
                                                overrides["IL.GROUPING_METHOD"] = (grouping.lower() if grouping else "random")
                                                overrides["IL.MIN_NODES_PER_CLIENT"] = int(args.min_nodes)
                                                if seed is not None:
                                                    overrides["IL.PARTITION_SEED"] = int(seed)
                                                if a is not None:
                                                    overrides["IL.IMBALANCE_ALPHA"] = float(a)
                                                elif imb is not None:
                                                    overrides["IL.IMBALANCE"] = float(imb)

                                            if "hiersplit" in method.lower():
                                                if pooling_method:
                                                    overrides["HIERSPLIT.POOLING"] = pooling_method
                                                    overrides["HIERSPLIT.EXPANSION"] = pooling_method
                                                if num_tokens == "auto":
                                                    if auto_num_tokens is not None:
                                                        overrides["HIERSPLIT.NUM_TOKENS"] = int(auto_num_tokens)
                                                elif isinstance(num_tokens, int):
                                                    overrides["HIERSPLIT.NUM_TOKENS"] = int(num_tokens)

                                            # ✅ unique ckpt dir (this is the key fix)
                                            ckpt_dir, final_pool, final_tok = build_unique_ckpt_dir(
                                                script_path=script_path,
                                                method=method,
                                                model=model,
                                                dataset=ds,
                                                grouping=grouping,
                                                clients=(None if is_central(method) else int(clients)),
                                                min_nodes=int(args.min_nodes),
                                                imb=imb,
                                                alpha=a,
                                                seed=seed,
                                                hs_pooling_override=pooling_method,
                                                hs_tokens_override=num_tokens,
                                                auto_tokens=auto_num_tokens,
                                            )
                                            overrides["TRAIN.CKPT_SAVE_DIR"] = str(ckpt_dir)

                                            run_id = make_run_id(
                                                dataset=ds,
                                                method=method,
                                                model=model,
                                                grouping=grouping,
                                                clients=(None if is_central(method) else int(clients)),
                                                imbalance=imb,
                                                alpha=a,
                                                seed=seed,
                                                min_nodes=int(args.min_nodes),
                                                hs_pooling=final_pool,
                                                hs_tokens=final_tok,
                                            )

                                            key = (str(ckpt_dir), run_id)
                                            if key in seen:
                                                continue
                                            seen.add(key)

                                            metrics_probe = ckpt_dir / "test_metrics.json"
                                            latest = find_latest_test_metrics(ckpt_dir)
                                            status = "Cached" if latest is not None else "Pending"
                                            duration = "cached" if status == "Cached" else ""

                                            if status == "Cached":
                                                cached += 1

                                            specs.append({
                                                "RunID": run_id,
                                                "Dataset": norm_dataset(ds),
                                                "Method": method,
                                                "Model": model,
                                                "Grouping": "" if grouping is None else grouping,
                                                "Num_Clients": "" if is_central(method) else str(clients),
                                                "Pooling": "" if final_pool is None else str(final_pool),
                                                "Num_Tokens": "" if final_tok is None else str(final_tok),
                                                "Imbalance": "" if imb is None else str(imb),
                                                "ImbAlpha": "" if a is None else str(a),
                                                "Seed": "" if seed is None else str(seed),
                                                "Status": status,
                                                "Duration": duration,
                                                "MAE": "", "RMSE": "", "MAPE": "",
                                                "MAE_h3": "", "RMSE_h3": "", "MAPE_h3": "",
                                                "MAE_h6": "", "RMSE_h6": "", "MAPE_h6": "",
                                                "MAE_h12": "", "RMSE_h12": "", "MAPE_h12": "",
                                                "_script_path": script_path,
                                                "_overrides": overrides,
                                                "_ckpt_dir": str(ckpt_dir),
                                                "_metrics_probe": str(metrics_probe),
                                            })

    if not specs:
        raise RuntimeError("No runnable jobs generated (all skipped?). Check your base scripts/paths.")

    csv_rows = [{k: v for k, v in s.items() if not k.startswith("_")} for s in specs]
    initialize_csv(csv_rows, csv_path)

    # fill cached metrics
    for s in specs:
        if s["Status"] == "Cached":
            update_csv(csv_path, s["RunID"], "Cached", "cached", Path(s["_metrics_probe"]))

    todo = deque([s for s in specs if s["Status"] == "Pending"])
    slots = {g: 0 for g in visible_gpus}
    active = []

    print("=" * 80)
    print(f"🚀 jobs_total={len(specs)} | pending={len(todo)} | cached={cached} | skipped={skipped}")
    print(f"🧾 csv: {csv_path}")
    print(f"📁 logs/results: {results_dir}")
    print(f"⚡ GPUs={visible_gpus}, jobs/GPU={jobs_per_gpu}")
    print("=" * 80)

    stop_flag = {"stop": False}

    def handle_sigint(signum, frame):
        stop_flag["stop"] = True
        print("\n[STOP] terminate all...")
        for (p, lf, *_rest) in active:
            p.terminate()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    def try_backfill():
        while todo:
            free_gpu = next((g for g in visible_gpus if slots[g] < jobs_per_gpu), None)
            if free_gpu is None:
                break

            s = todo.popleft()
            run_id = s["RunID"]
            ckpt_dir = Path(s["_ckpt_dir"])
            metrics_probe = Path(s["_metrics_probe"])

            # created meanwhile?
            if find_latest_test_metrics(ckpt_dir) is not None:
                update_csv(csv_path, run_id, "Cached", "cached", metrics_probe)
                continue

            update_csv(csv_path, run_id, "Running", "", metrics_probe)
            print(f"[LAUNCH] GPU {free_gpu} :: {run_id}")

            p, lf = launch_proc(
                gpu_id=free_gpu,
                visible_gpus=visible_gpus,
                script_path=s["_script_path"],
                run_id=run_id,
                overrides=s["_overrides"],
                log_root=log_root,
            )

            slots[free_gpu] += 1
            active.append((p, lf, free_gpu, run_id, time.time(), ckpt_dir, metrics_probe))

    try:
        while (todo or active) and not stop_flag["stop"]:
            try_backfill()
            time.sleep(1)

            still_active = []
            for (p, lf, g, run_id, start_t, ckpt_dir, metrics_probe) in active:
                ret = p.poll()
                if ret is None:
                    still_active.append((p, lf, g, run_id, start_t, ckpt_dir, metrics_probe))
                    continue

                lf.close()
                slots[g] = max(0, slots[g] - 1)

                duration = f"{time.time() - start_t:.0f}s"
                status = "Success" if ret == 0 else "Failed"
                print(("✅" if status == "Success" else "❌") + f" [DONE] GPU {g} :: {run_id} ({duration})")

                update_csv(csv_path, run_id, status, duration, metrics_probe)

            active = still_active

    except KeyboardInterrupt:
        handle_sigint(None, None)

    print("\n[ALL DONE]")
    pd.read_csv(csv_path).to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

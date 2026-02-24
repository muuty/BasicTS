#!/usr/bin/env python3
"""
CLI-only Parallel Runner (No YAML) — HierSplit (lr sweep)

What this runner does:
- Runs ONLY the configs you point to under --base-root/<method>/<model>/<dataset>.py (like before).
- Adds a sweep over:
  TRAIN.OPTIM.PARAM.lr in {0.001, 0.005, 0.01} (customizable)
- Keeps all your prior fixes:
  - Cache check searches recursively under CKPT_SAVE_DIR for test_metrics.json
  - CKPT_SAVE_DIR becomes unique per setting (incl. lr)
  - RunID includes lr to avoid collisions

NOTE:
- This assumes your override system accepts dotted keys like:
    "TRAIN.OPTIM.PARAM.lr"
  If your override parser expects a different key for lr, change LR_OVERRIDE_KEY below.

Example:
python run_parallel_hiersplit.py --methods hiersplit --datasets METR_LA --models staeformer --groupings random \
  --clients 10 --partition-seeds 0 --pooling-method simple --num-tokens 4 \
  --lrs 0.001,0.005,0.01 -g 0 -w 2

python run_parallel_hiersplit.py --methods federated --datasets METR_LA --models staeformer --groupings random \
  --clients 10 --partition-seeds 0 --aggregators fedavg,fedprox --lrs 0.002 -g 0 -w 2
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
import shutil
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

import pandas as pd
import importlib.util
import hashlib

CSV_LOCK = threading.Lock()

DEFAULT_DATASETS = "METR_LA,PEMS03,PEMS04,PEMS07,PEMS08,PEMS_BAY,PEMS04_SPEED,PEMS08_SPEED,NYCBike,NYCTaxi2015,NYCTaxi2016,Porto"
DEFAULT_METHODS  = "hiersplit"
DEFAULT_MODELS   = "staeformer,stgformer,gruseq2seq_graphnet"
DEFAULT_GROUPINGS= "maxcut,random,metis"
DEFAULT_AGGREGATORS = "fedavg"

# Override key for lr (change if your override system uses a different path)
LR_OVERRIDE_KEY = "TRAIN.OPTIM.PARAM.lr"


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
    # 0.001 -> 0p001, 0.01 -> 0p01
    return f"{x:g}".replace(".", "p").replace("-", "m")

def is_central(method: str) -> bool:
    return "central" in method.lower()

def is_federated(method: str) -> bool:
    return "federated" in method.lower()

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
    lr: Optional[float] = None,
    aggregator: Optional[str] = None,
    use_fedavg: bool = False,
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
        if use_fedavg:
            parts.append("fedavg")

    if "federated" in mth:
        if aggregator:
            parts.append(f"agg{aggregator.lower()}")

    if lr is not None:
        parts.append(f"lr{slug_float(lr)}")

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

    # 통신량 읽기 (Split, HierSplit, Federated 공통)
    # FedAvg 통신량은 forward/backward에 이미 포함되어 있음
    if "communication_cost" in data:
        comm = data["communication_cost"]
        # Split, HierSplit: total_forward_mb, total_backward_mb
        if "total_forward_mb" in comm:
            out["Comm_C2S"] = f"{comm.get('total_forward_mb', 0):.2f}"
            out["Comm_S2C"] = f"{comm.get('total_backward_mb', 0):.2f}"
            out["Comm_Total"] = f"{comm.get('total_mb', 0):.2f}"
        # Federated: total_communication_mb only
        elif "total_communication_mb" in comm:
            total_mb = comm.get('total_communication_mb', 0)
            out["Comm_C2S"] = ""
            out["Comm_S2C"] = ""
            out["Comm_Total"] = f"{total_mb:.2f}"

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
        spec.loader.exec_module(mod)
        cfg = getattr(mod, "CFG", None)

        if cfg is not None and hasattr(cfg, "TRAIN") and hasattr(cfg.TRAIN, "CKPT_SAVE_DIR"):
            meta["ckpt_save_dir"] = str(cfg.TRAIN.CKPT_SAVE_DIR)

        hs = getattr(cfg, "HIERSPLIT", None) if cfg is not None else None
        if hs is not None:
            meta["hs_pooling"] = getattr(hs, "POOLING", None)
            meta["hs_tokens"] = getattr(hs, "NUM_TOKENS", None)

        # try reading default lr if exists (optional)
        try:
            # cfg.TRAIN.OPTIM.PARAM is often dict-like
            lr0 = getattr(cfg.TRAIN.OPTIM, "PARAM", {}).get("lr", None)
            if lr0 is not None:
                meta["lr"] = float(lr0)
        except Exception:
            pass
    except Exception:
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
    hs_tokens_override: Optional[object],
    nodes_per_client: Optional[float],
    lr: float,
    aggregator: Optional[str] = None,
    use_fedavg: bool = False,
) -> Tuple[Path, Optional[str], Optional[int]]:
    """
    Returns (ckpt_dir, final_hs_pooling, final_hs_tokens)
    """
    meta = load_cfg_meta(script_path)
    base_ckpt = meta.get("ckpt_save_dir", None)
    if base_ckpt is None:
        base_ckpt = str(Path("checkpoints") / f"{method}_{model}" / norm_dataset(dataset))

    base = Path(base_ckpt)

    # keep hiersplit separate if base has IL_ prefix
    if "hiersplit" in method.lower():
        parent = base.parent
        p = parent.name
        if p.lower().startswith("il_"):
            p = "HierSplit_" + p[3:]
        elif not p.lower().startswith("hiersplit_"):
            p = "HierSplit_" + p
        base = parent.parent / p / base.name

    # keep federated separate if base has IL_ prefix
    if "federated" in method.lower():
        parent = base.parent
        p = parent.name
        if p.lower().startswith("il_"):
            p = "FL_" + p[3:]
        elif not p.lower().startswith("fl_"):
            p = "FL_" + p
        base = parent.parent / p / base.name

    final_pool = None
    final_tok = None
    if "hiersplit" in method.lower():
        final_pool = (hs_pooling_override or meta.get("hs_pooling", None) or None)
        # num_tokens 계산: "auto" (20%), "XX%" (퍼센트), 또는 정수
        if hs_tokens_override is not None and nodes_per_client is not None:
            nt_str = str(hs_tokens_override).strip().lower()
            if nt_str == "auto":
                final_tok = max(1, round(nodes_per_client * 0.2))
            elif nt_str.endswith("%"):
                try:
                    pct = float(nt_str[:-1])
                    final_tok = max(1, round(nodes_per_client * pct / 100.0))
                except ValueError:
                    pass
            else:
                try:
                    final_tok = int(nt_str)
                except ValueError:
                    pass
        elif hs_tokens_override is None:
            v = meta.get("hs_tokens", None)
            final_tok = int(v) if v is not None else None

    tags: List[str] = []
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
            tags.append(f"pool{final_pool.lower()}")
        if final_tok is not None:
            tags.append(f"t{int(final_tok)}")
        if use_fedavg:
            tags.append("fedavg")

    if "federated" in method.lower():
        if aggregator:
            tags.append(f"agg{aggregator.lower()}")

    tags.append(f"lr{slug_float(lr)}")

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
    parser = argparse.ArgumentParser("CLI-only parallel experiment runner (HierSplit lr sweep)")

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

    parser.add_argument("--pooling-method", default="", help="HierSplit pooling method(s): simple,attention")
    parser.add_argument("--num-tokens", default="", 
                        help="HierSplit num_tokens: integer, 'auto' (=20%%), or percentage (e.g., '10%%,20%%,30%%')")

    # hiersplit + fedavg
    parser.add_argument("--use-fedavg", action="store_true", help="Enable FedAvg for HierSplit (aggregates client params each epoch)")

    # federated learning
    parser.add_argument("--aggregators", default=DEFAULT_AGGREGATORS, help="aggregator types for federated (comma separated, e.g., fedavg,fedprox)")

    # lr sweep
    parser.add_argument("--lrs", default="0.001,0.005,0.01", help="learning rates to sweep (comma separated)")

    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("-w", "--workers", type=int, default=2)

    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignore existing checkpoints and rerun all experiments (overwrites existing results)")

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

    aggregators = split_list(args.aggregators) if args.aggregators else []

    lrs = split_floats(args.lrs)

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

                    # hiersplit pooling/tokens sweep is still supported (optional)
                    if is_hs:
                        eff_pooling_methods = pooling_methods if pooling_methods else [None]
                        # num_tokens: "auto", "10%", "20%", or integer
                        eff_num_tokens_list = [None] if not num_tokens_list else [
                            nt.strip() for nt in num_tokens_list  # keep as string for now
                        ]
                    else:
                        eff_pooling_methods = [None]
                        eff_num_tokens_list = [None]

                    # federated aggregator sweep
                    is_fl = is_federated(method)
                    if is_fl:
                        eff_aggregators = aggregators if aggregators else ["fedavg"]
                    else:
                        eff_aggregators = [None]

                    for c in eff_clients:
                        clients = c
                        if not is_central(method) and clients is None:
                            if base_clients is None:
                                print(f"[SKIP] clients not provided and NUM_CLIENTS not found in {script_path}")
                                skipped += 1
                                continue
                            clients = base_clients

                        # 클라이언트당 평균 노드 수 계산 (퍼센트 기반 토큰 계산용)
                        nodes_per_client = None
                        if is_hs and base_total_nodes is not None and clients is not None:
                            nodes_per_client = int(base_total_nodes) / int(clients)

                        for seed in eff_seeds:
                            for imb in eff_imb_list:
                                for a in eff_alpha_list:
                                    for pooling_method in eff_pooling_methods:
                                        for num_tokens in eff_num_tokens_list:
                                            for aggregator in eff_aggregators:
                                                for lr in lrs:

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

                                                    # hiersplit overrides
                                                    if is_hs:
                                                        if pooling_method:
                                                            overrides["HIERSPLIT.POOLING"] = pooling_method
                                                            overrides["HIERSPLIT.EXPANSION"] = pooling_method
                                                        # num_tokens 계산: "auto" (20%), "XX%" (퍼센트), 또는 정수
                                                        resolved_tokens = None
                                                        if num_tokens is not None and nodes_per_client is not None:
                                                            nt_str = str(num_tokens).strip().lower()
                                                            if nt_str == "auto":
                                                                # auto = 20%
                                                                resolved_tokens = max(1, round(nodes_per_client * 0.2))
                                                            elif nt_str.endswith("%"):
                                                                # 퍼센트 지정: "10%", "30%" 등
                                                                try:
                                                                    pct = float(nt_str[:-1])
                                                                    resolved_tokens = max(1, round(nodes_per_client * pct / 100.0))
                                                                except ValueError:
                                                                    pass
                                                            else:
                                                                # 정수 직접 지정
                                                                try:
                                                                    resolved_tokens = int(nt_str)
                                                                except ValueError:
                                                                    pass
                                                        if resolved_tokens is not None:
                                                            overrides["HIERSPLIT.NUM_TOKENS"] = resolved_tokens
                                                        if args.use_fedavg:
                                                            overrides["HIERSPLIT.USE_FEDAVG"] = True

                                                    # federated overrides
                                                    if is_fl:
                                                        if aggregator:
                                                            overrides["FL.AGGREGATOR"] = aggregator.lower()

                                                    # lr override (always)
                                                    overrides[LR_OVERRIDE_KEY] = float(lr)

                                                    # unique ckpt dir (key fix)
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
                                                        nodes_per_client=nodes_per_client,
                                                        lr=float(lr),
                                                        aggregator=aggregator,
                                                        use_fedavg=(is_hs and args.use_fedavg),
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
                                                        lr=float(lr),
                                                        aggregator=aggregator,
                                                        use_fedavg=(is_hs and args.use_fedavg),
                                                    )

                                                    key = (str(ckpt_dir), run_id)
                                                    if key in seen:
                                                        continue
                                                    seen.add(key)

                                                    metrics_probe = ckpt_dir / "test_metrics.json"
                                                    latest = find_latest_test_metrics(ckpt_dir)
                                                    # --force-rerun이면 캐시 무시
                                                    if args.force_rerun:
                                                        status = "Pending"
                                                        duration = ""
                                                    else:
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
                                                        "Aggregator": "" if aggregator is None else aggregator,
                                                        "UseFedAvg": "Y" if (is_hs and args.use_fedavg) else "",
                                                        "LR": f"{lr:g}",
                                                        "Imbalance": "" if imb is None else str(imb),
                                                        "ImbAlpha": "" if a is None else str(a),
                                                        "Seed": "" if seed is None else str(seed),
                                                        "Status": status,
                                                        "Duration": duration,
                                                        "MAE": "", "RMSE": "", "MAPE": "",
                                                        "MAE_h3": "", "RMSE_h3": "", "MAPE_h3": "",
                                                        "MAE_h6": "", "RMSE_h6": "", "MAPE_h6": "",
                                                        "MAE_h12": "", "RMSE_h12": "", "MAPE_h12": "",
                                                        "Comm_C2S": "", "Comm_S2C": "", "Comm_Total": "",
                                                        "_script_path": script_path,
                                                        "_overrides": overrides,
                                                        "_ckpt_dir": str(ckpt_dir),
                                                        "_metrics_probe": str(metrics_probe),
                                                    })

    if not specs:
        raise RuntimeError("No runnable jobs generated (all skipped?). Check your base scripts/paths.")

    csv_rows = [{k: v for k, v in s.items() if not k.startswith("_")} for s in specs]
    initialize_csv(csv_rows, csv_path)

    # fill cached metrics (--force-rerun이면 스킵)
    if not args.force_rerun:
        for s in specs:
            if s["Status"] == "Cached":
                update_csv(csv_path, s["RunID"], "Cached", "cached", Path(s["_metrics_probe"]))

    todo = deque([s for s in specs if s["Status"] == "Pending"])
    slots = {g: 0 for g in visible_gpus}
    active = []

    print("=" * 80)
    print(f"🚀 jobs_total={len(specs)} | pending={len(todo)} | cached={cached} | skipped={skipped}")
    if args.force_rerun:
        print("⚠️  --force-rerun: ignoring existing checkpoints, rerunning all experiments")
    print(f"🧾 csv: {csv_path}")
    print(f"📁 logs/results: {results_dir}")
    print(f"⚡ GPUs={visible_gpus}, jobs/GPU={jobs_per_gpu}")
    print(f"🧪 lrs={lrs}")
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

            # created meanwhile? (--force-rerun이면 무시)
            if not args.force_rerun and find_latest_test_metrics(ckpt_dir) is not None:
                update_csv(csv_path, run_id, "Cached", "cached", metrics_probe)
                continue

            # --force-rerun: 기존 체크포인트 삭제 (resume 방지)
            if args.force_rerun and ckpt_dir.exists():
                try:
                    shutil.rmtree(ckpt_dir)
                    print(f"[FORCE-RERUN] Deleted existing ckpt: {ckpt_dir}")
                except Exception as e:
                    print(f"[WARN] Failed to delete {ckpt_dir}: {e}")

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

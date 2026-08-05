#!/usr/bin/env python
"""
Assignment-weighted (mass-weighted) vs uniform-weighted coreset training.

Question: does nearest-assignment MASS-WEIGHTED coreset training (each selected
example weighted by k * n_j / n, AssignmentWeightedRunner) beat ordinary
UNIFORM-weighted coreset training (IncidentAwareRunner, Phase C)?

This tests the "representative-mass" term the paper's theory advertises
(Propositions 3-4) but never measured.

Axis note (handled explicitly below):
  weighted : coreset seed = 42 fixed, optim (ENV) seed varies {42,123,456}
  uniform  : optim seed = 42 fixed, coreset seed varies {42,123,456}
The ONLY apples-to-apples paired cell is coreset42 + optim42 -> PRIMARY.

Outputs -> experiments/result/analysis/
  assignment_weighted_vs_uniform_cells.csv     (per-cell deltas)
  assignment_weighted_vs_uniform_summary.txt   (printed summary)
"""
import os, re, json, glob, sys
import numpy as np

ROOT = "/home/uqtyu7/github/BasicTS"
OUTDIR = os.path.join(ROOT, "experiments", "result", "analysis")
os.makedirs(OUTDIR, exist_ok=True)

WEIGHTED_ROOT = os.path.join(ROOT, "checkpoints", "assignment_weighted_mass")
UNIFORM_ROOTS = [
    os.path.join(ROOT, "checkpoints", "phase_c_method_comparison"),
    os.path.join(ROOT, "checkpoints", "phase_c_extra_ratios"),
]
METHODS = {"k_medoids", "k_center", "graph_cut"}


# --------------------------------------------------------------------------
def parse_cfg(path):
    """Stateful parse of cfg.txt -> dict of fields we care about."""
    section = None
    fields = {}
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            # top-level section header: no leading whitespace, "WORD:" (maybe value)
            m = re.match(r"^([A-Z_]+):\s*(.*)$", line)
            if m:
                section = m.group(1)
                rest = m.group(2).strip()
                if section == "RUNNER":
                    fields["RUNNER"] = rest
                continue
            # indented key: value
            m2 = re.match(r"^\s+([A-Za-z0-9_]+):\s*(.*)$", line)
            if m2 and section is not None:
                key, val = m2.group(1), m2.group(2).strip()
                fields[f"{section}.{key}"] = val
    return fields


def leaf_record(leaf, tree):
    cfg = os.path.join(leaf, "cfg.txt")
    tm = os.path.join(leaf, "test_metrics.json")
    if not os.path.isfile(cfg) or not os.path.isfile(tm):
        return None
    f = parse_cfg(cfg)
    strat = f.get("CORESET.SELECTION_STRATEGY")
    if strat not in METHODS:
        return None
    try:
        ratio = float(f.get("CORESET.SELECTION_RATIO"))
        coreset_seed = int(f.get("CORESET.SEED"))
        optim_seed = int(f.get("ENV.SEED"))
    except (TypeError, ValueError):
        return None
    model = f.get("MODEL.NAME")
    dataset = f.get("DATASET.NAME", "").split("/")[-1]
    rec = {
        "tree": tree,
        "leaf": leaf,
        "model": model,
        "dataset": dataset,
        "method": strat,
        "ratio": round(ratio, 3),
        "coreset_seed": coreset_seed,
        "optim_seed": optim_seed,
        "runner": f.get("RUNNER", ""),
    }
    with open(tm) as fh:
        m = json.load(fh)
    rec["overall_mae"] = m["overall"]["MAE"]
    # incident-conditional (may be absent)
    im = os.path.join(leaf, "test_incident_metrics.json")
    rec["incident_mae"] = None
    if os.path.isfile(im):
        try:
            with open(im) as fh:
                d = json.load(fh)
            rec["incident_mae"] = d["all_incident_overall"]["MAE"]
        except Exception:
            pass
    rec["coreset_sel"] = os.path.join(leaf, "coreset-selection.json")
    return rec


def collect(root, tree):
    recs = []
    for cfg in glob.glob(os.path.join(root, "**", "cfg.txt"), recursive=True):
        leaf = os.path.dirname(cfg)
        r = leaf_record(leaf, tree)
        if r is not None:
            recs.append(r)
    return recs


def key_full(r):
    return (r["model"], r["dataset"], r["method"], r["ratio"],
            r["coreset_seed"], r["optim_seed"])


# --------------------------------------------------------------------------
print("Collecting weighted runs ...")
weighted = collect(WEIGHTED_ROOT, "weighted")
print(f"  weighted leaves usable: {len(weighted)}")

print("Collecting uniform runs ...")
uniform = []
for ur in UNIFORM_ROOTS:
    u = collect(ur, "uniform")
    print(f"  {os.path.basename(ur)}: {len(u)} usable matching-method leaves")
    uniform.extend(u)
print(f"  uniform leaves usable (methods {sorted(METHODS)}): {len(uniform)}")

# Confirm runner types
wr = set(r["runner"] for r in weighted)
ur = set(r["runner"] for r in uniform)
print("\nWeighted runner types:", wr)
print("Uniform runner types :", ur)

# Index uniform by full key (there can be duplicates if reran; keep best/first + warn)
uni_by_key = {}
for r in uniform:
    k = key_full(r)
    uni_by_key.setdefault(k, []).append(r)
dups = {k: v for k, v in uni_by_key.items() if len(v) > 1}
if dups:
    print(f"\nWARNING: {len(dups)} uniform keys have >1 leaf; using the one with lowest overall MAE per key.")

wt_by_key = {}
for r in weighted:
    k = key_full(r)
    wt_by_key.setdefault(k, []).append(r)


def pick(lst):
    return min(lst, key=lambda r: r["overall_mae"])


# --------------------------------------------------------------------------
# Build matched cells. A "cell" = (model,dataset,method,ratio,coreset_seed,optim_seed)
# present in BOTH trees.
def indices_match(w, u):
    """Return (same_bool, jaccard) comparing the two coreset-selection.json files."""
    try:
        ws = set(json.load(open(w["coreset_sel"])))
        us = set(json.load(open(u["coreset_sel"])))
    except Exception:
        return (None, float("nan"))
    if not ws or not us:
        return (None, float("nan"))
    inter = len(ws & us)
    jac = inter / len(ws | us)
    return (ws == us, jac)


rows = []
matched_keys = sorted(set(wt_by_key) & set(uni_by_key))
for k in matched_keys:
    w = pick(wt_by_key[k]); u = pick(uni_by_key[k])
    model, dataset, method, ratio, cs, os_ = k
    same, jac = indices_match(w, u)
    for subset, wk, uk in [("overall", "overall_mae", "overall_mae"),
                           ("all_incident", "incident_mae", "incident_mae")]:
        wv = w[wk]; uv = u[uk]
        if wv is None or uv is None:
            continue
        rows.append({
            "model": model, "dataset": dataset, "method": method,
            "ratio": ratio, "coreset_seed": cs, "optim_seed": os_,
            "uniform_mae": uv, "weighted_mae": wv,
            "delta": wv - uv, "subset": subset,
            "same_indices": same, "jaccard": round(jac, 4),
        })

# Report weighted keys with no uniform match (dropped cells) -> honesty
unmatched = sorted(set(wt_by_key) - set(uni_by_key))
print(f"\nWeighted cells WITHOUT a uniform match (dropped): {len(unmatched)}")
for k in unmatched:
    print("   NO-UNIFORM:", k)

# --------------------------------------------------------------------------
# Write CSV
import csv
csv_path = os.path.join(OUTDIR, "assignment_weighted_vs_uniform_cells.csv")
cols = ["model", "dataset", "method", "ratio", "coreset_seed", "optim_seed",
        "uniform_mae", "weighted_mae", "delta", "subset", "same_indices", "jaccard"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in sorted(rows, key=lambda x: (x["subset"], x["model"], x["dataset"],
                                         x["method"], x["ratio"], x["optim_seed"])):
        w.writerow(r)
print(f"\nWrote {csv_path} ({len(rows)} rows)")

# --------------------------------------------------------------------------
# Statistics helpers
from scipy import stats


def sign_test(deltas):
    d = np.array([x for x in deltas if x != 0])
    n = len(d)
    if n == 0:
        return (np.nan, 0, 0, n)
    neg = int(np.sum(d < 0))  # weighted better
    pos = int(np.sum(d > 0))
    k = min(neg, pos)
    # two-sided binomial
    p = stats.binomtest(k, n, 0.5).pvalue if hasattr(stats, "binomtest") else \
        stats.binom_test(k, n, 0.5)
    return (p, neg, pos, n)


def boot_ci(deltas, n_boot=20000, seed=0):
    d = np.array(deltas, float)
    if len(d) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarise(rows_sub, label, out):
    deltas = [r["delta"] for r in rows_sub]
    if not deltas:
        out.append(f"[{label}] no cells")
        return
    d = np.array(deltas)
    wins = int(np.sum(d < 0))
    n = len(d)
    mean = d.mean(); med = np.median(d)
    lo, hi = boot_ci(d)
    # Wilcoxon signed-rank (paired, vs 0)
    if n >= 1 and np.any(d != 0):
        try:
            wstat, wp = stats.wilcoxon(d)
        except Exception:
            wstat, wp = (np.nan, np.nan)
    else:
        wstat, wp = (np.nan, np.nan)
    sp, neg, pos, sn = sign_test(d)
    out.append(f"[{label}]  n={n} cells")
    out.append(f"   weighted wins (delta<0): {wins}/{n}")
    out.append(f"   mean delta = {mean:+.4f}  (weighted - uniform MAE; negative = weighted better)")
    out.append(f"   median delta = {med:+.4f}")
    out.append(f"   bootstrap 95% CI of mean delta = [{lo:+.4f}, {hi:+.4f}]")
    out.append(f"   Wilcoxon signed-rank p = {wp:.4f}")
    out.append(f"   sign test: neg(win)={neg} pos(lose)={pos}  p = {sp:.4f}")
    out.append("")


# --------------------------------------------------------------------------
out = []
out.append("=" * 78)
out.append("ASSIGNMENT-WEIGHTED (mass) vs UNIFORM-weighted coreset training")
out.append("delta = weighted_MAE - uniform_MAE   (negative => mass-weighting better)")
out.append("=" * 78)
out.append("")
out.append(f"weighted usable leaves: {len(weighted)} | uniform usable leaves: {len(uniform)}")
out.append(f"matched cells (both trees, any seed): {len(matched_keys)}")
out.append(f"weighted cells with no uniform match: {len(unmatched)}")
out.append("")

# ---- PRIMARY: coreset42 + optim42, overall ----
prim_overall = [r for r in rows if r["coreset_seed"] == 42 and r["optim_seed"] == 42
                and r["subset"] == "overall"]
prim_incident = [r for r in rows if r["coreset_seed"] == 42 and r["optim_seed"] == 42
                 and r["subset"] == "all_incident"]

out.append("#" * 60)
out.append("PRIMARY comparison: coreset42 + optim42 (identical indices & optim seed)")
out.append("#" * 60)
out.append("")
out.append("!! CORESET-INDEX SANITY: a cell is only a clean weighted-vs-uniform")
out.append("!! contrast if BOTH runs used the SAME coreset indices (same_indices=True).")
out.append("!! Cells with same_indices=False confound loss-weighting with different")
out.append("!! selected examples and are EXCLUDED from the valid statistics.")
out.append("")
out.append("Per-cell (OVERALL MAE):")
out.append(f"  {'model':<18}{'dataset':<16}{'method':<11}{'ratio':>6}"
           f"{'uniform':>10}{'weighted':>10}{'delta':>9}{'idx?':>7}{'jac':>7}")
for r in sorted(prim_overall, key=lambda x: (x["model"], x["dataset"], x["method"], x["ratio"])):
    flag = "OK" if r["same_indices"] else "CONFND"
    out.append(f"  {r['model']:<18}{r['dataset']:<16}{r['method']:<11}{r['ratio']:>6}"
               f"{r['uniform_mae']:>10.4f}{r['weighted_mae']:>10.4f}{r['delta']:>+9.4f}"
               f"{flag:>7}{r['jaccard']:>7.3f}")
out.append("")
prim_overall_valid = [r for r in prim_overall if r["same_indices"]]
prim_overall_conf = [r for r in prim_overall if not r["same_indices"]]
summarise(prim_overall_valid, "PRIMARY overall  (VALID: identical indices only)", out)
summarise(prim_overall, "PRIMARY overall  (ALL matched cells incl. confounded)", out)
if prim_overall_conf:
    conf_methods = sorted(set(r["method"] for r in prim_overall_conf))
    out.append(f"  NOTE: {len(prim_overall_conf)} confounded cells excluded above "
               f"(methods: {conf_methods}).")
    out.append("")

out.append("Per-cell (all_incident MAE):")
out.append(f"  {'model':<18}{'dataset':<16}{'method':<11}{'ratio':>6}"
           f"{'uniform':>10}{'weighted':>10}{'delta':>9}{'idx?':>7}{'jac':>7}")
for r in sorted(prim_incident, key=lambda x: (x["model"], x["dataset"], x["method"], x["ratio"])):
    flag = "OK" if r["same_indices"] else "CONFND"
    out.append(f"  {r['model']:<18}{r['dataset']:<16}{r['method']:<11}{r['ratio']:>6}"
               f"{r['uniform_mae']:>10.4f}{r['weighted_mae']:>10.4f}{r['delta']:>+9.4f}"
               f"{flag:>7}{r['jaccard']:>7.3f}")
out.append("")
prim_incident_valid = [r for r in prim_incident if r["same_indices"]]
summarise(prim_incident_valid, "PRIMARY all_incident  (VALID: identical indices only)", out)
summarise(prim_incident, "PRIMARY all_incident  (ALL matched cells incl. confounded)", out)

# ---- Weighted optim-seed spread at coreset42 (optimization noise) ----
out.append("#" * 60)
out.append("Weighted optim-seed spread @ coreset42 (optimization noise reference)")
out.append("#" * 60)
out.append("")
grp = {}
for r in weighted:
    if r["coreset_seed"] == 42:
        g = (r["model"], r["dataset"], r["method"], r["ratio"])
        grp.setdefault(g, {})[r["optim_seed"]] = r["overall_mae"]
out.append(f"  {'model':<18}{'dataset':<16}{'method':<11}{'ratio':>6}"
           f"{'o42':>9}{'o123':>9}{'o456':>9}{'range':>9}")
spreads = []
for g in sorted(grp):
    d = grp[g]
    vals = [d.get(s) for s in (42, 123, 456)]
    present = [v for v in vals if v is not None]
    rng = (max(present) - min(present)) if len(present) > 1 else float("nan")
    if len(present) > 1:
        spreads.append(rng)
    def fmt(v):
        return f"{v:>9.4f}" if v is not None else f"{'--':>9}"
    out.append(f"  {g[0]:<18}{g[1]:<16}{g[2]:<11}{g[3]:>6}"
               f"{fmt(vals[0])}{fmt(vals[1])}{fmt(vals[2])}"
               f"{(f'{rng:.4f}' if rng==rng else '--'):>9}")
if spreads:
    out.append("")
    out.append(f"  Optim-seed MAE range across weighted cells: "
               f"mean={np.mean(spreads):.4f}  median={np.median(spreads):.4f}  max={np.max(spreads):.4f}")
    out.append(f"  (Compare to PRIMARY overall mean |delta| below to judge signal vs noise.)")
    mean_abs_delta = np.mean([abs(r['delta']) for r in prim_overall_valid]) if prim_overall_valid else float('nan')
    out.append(f"  PRIMARY overall (VALID cells) mean |delta| = {mean_abs_delta:.4f}")
out.append("")

# ---- Ratio coverage check ----
out.append("#" * 60)
out.append("Ratio / cell coverage (coreset42+optim42)")
out.append("#" * 60)
wt_prim_cells = set((r["model"], r["dataset"], r["method"], r["ratio"])
                    for r in weighted if r["coreset_seed"] == 42 and r["optim_seed"] == 42)
matched_prim_cells = set((r["model"], r["dataset"], r["method"], r["ratio"]) for r in prim_overall)
missing = sorted(wt_prim_cells - matched_prim_cells)
out.append(f"  weighted coreset42+optim42 cells: {len(wt_prim_cells)}")
out.append(f"  of which matched by a uniform run: {len(matched_prim_cells)}")
if missing:
    out.append("  MISSING uniform match for:")
    for m in missing:
        out.append(f"     {m}")
else:
    out.append("  every weighted coreset42+optim42 cell has a uniform match.")
by_ratio = {}
for r in prim_overall:
    by_ratio.setdefault(r["ratio"], 0)
    by_ratio[r["ratio"]] += 1
out.append(f"  matched cells per ratio: {dict(sorted(by_ratio.items()))}")
out.append("")

print("\n".join(out))

# save summary; sanity section appended by companion below
summary_path = os.path.join(OUTDIR, "assignment_weighted_vs_uniform_summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(out))
print(f"\nWrote {summary_path}")

# expose objects for sanity script via json of matched primary leaves
pair_dump = []
for k in matched_keys:
    if k[4] == 42 and k[5] == 42:  # coreset42 optim42
        w = pick(wt_by_key[k]); u = pick(uni_by_key[k])
        pair_dump.append({"key": list(k),
                          "weighted_sel": w["coreset_sel"],
                          "uniform_sel": u["coreset_sel"]})
with open(os.path.join(OUTDIR, "_pairs_primary.json"), "w") as f:
    json.dump(pair_dump, f)
print(f"Dumped {len(pair_dump)} primary pairs for sanity check.")

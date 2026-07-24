"""
3-CLASS STEP 4 - does a HARMFULNESS axis exist that is separate from the REFUSAL axis?

Three contrasts, all on last-prompt-token activations, trained on `extract` base
prompts and reported on the disjoint `test` base prompts:

  refusal      refuse (1)          vs comply-both (0)
               -> the known axis. Expected to be strong and to align with Arditi r.

  harmfulness  harmful_comply (1)  vs benign_comply (0)          <- THE NEW ONE
               Both classes are PAP-wrapped (wrapper surface held constant) and both
               are non-refusals ("not refusing" held constant). So this contrast
               varies ONLY harmful-vs-benign content. If it is separable AND not
               parallel to r, there is a harmfulness axis distinct from refusal.

  naive        harmful_comply (1)  vs refuse (0)                 <- confounded ref
               Differs on refusing AND harmful at once. This is the contrast that
               can only re-derive the refusal axis; reported to show the gap.

Robustness (the point of the whole exercise): LEAVE-ONE-HARM-CATEGORY-OUT on the
harmfulness contrast. Train with one StrongREJECT category entirely held out, test
on it. Generalising to unseen harm categories is what separates "a harmfulness
representation" from "memorised topic".

Controls: global label permutation (Part 7 gotcha - a WITHIN-group shuffle is a
no-op when labels are group-constant, which they are here), and a random direction.

Geometry: diff-in-means directions per layer (Arditi's estimator - less prone to
chasing confounds than a discriminative probe), cosine vs r both RAW and WHITENED.
Whitening matters because residual streams are anisotropic: a few rogue dimensions
inflate every raw cosine. (Caveat: diff-in-means already cancels the common-mode
component, so whitening is a robustness check, not expected to overturn.)

Output -> results/threeclass/{probe_results.json, directions.pt, summary.md}
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                      # Part 7: 20x speedup, PCA on small matrices

import json  # noqa: E402
import sys   # noqa: E402

import numpy as np      # noqa: E402
import torch           # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TC = os.environ.get("THREECLASS_DIR", os.path.join(REPO, "results", "threeclass"))
ACTS = os.path.join(TC, "acts.pt")
REFUSAL_PATH = os.path.join(REPO, "checkpoints", "direction.pt")

SEED = 1234
N_PCA = 128
C_GRID = [0.01, 0.1, 1.0]
MIN_CAT = 25          # min held-out rows for a category to be scored


def contrasts(cls):
    cls = np.asarray(cls)
    return {
        "refusal":     (np.isin(cls, ["refuse", "harmful_comply", "benign_comply"]),
                        (cls == "refuse").astype(int)),
        "harmfulness": (np.isin(cls, ["harmful_comply", "benign_comply"]),
                        (cls == "harmful_comply").astype(int)),
        "naive":       (np.isin(cls, ["harmful_comply", "refuse"]),
                        (cls == "harmful_comply").astype(int)),
    }


def fit_auroc(Xtr, ytr, Xte, yte, C=1.0, seed=SEED):
    """StandardScaler -> PCA-128 -> L2 logistic. Fit on train only. Returns AUROC."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    k = min(N_PCA, Xtr.shape[0] - 1, Xtr.shape[1])
    p = PCA(n_components=k, random_state=seed).fit(Xtr)
    Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    lr = LogisticRegression(C=C, max_iter=3000, random_state=seed).fit(Xtr, ytr)
    return float(roc_auc_score(yte, lr.decision_function(Xte)))


def dim_auroc(Xtr, ytr, Xte, yte):
    """PCA-free detector: project test onto the TRAIN diff-in-means direction.

    Guards the headline against PCA truncation - a genuine but low-variance
    harmfulness axis could fall outside the top-128 PCs and read as 'no signal'.
    """
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return float("nan")
    return float(roc_auc_score(yte, Xte @ (d / n)))


def full_auroc(Xtr, ytr, Xte, yte, C=1.0, seed=SEED):
    """Full-space L2 logistic, no PCA. Second guard against truncation."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=3000, random_state=seed).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, lr.decision_function(sc.transform(Xte))))


def whiten_cos(a, b, Sigma, ridge=1e-3):
    """Mahalanobis cosine: cos(W a, W b), W = Sigma^{-1/2}, ridge-stabilised."""
    S = Sigma + ridge * np.trace(Sigma) / Sigma.shape[0] * np.eye(Sigma.shape[0])
    w, V = np.linalg.eigh(S)
    W = V @ np.diag(w ** -0.5) @ V.T
    x, y = W @ a, W @ b
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main():
    if not os.path.exists(ACTS):
        sys.exit(f"missing {ACTS} - run capture_acts.py first")
    d = torch.load(ACTS, map_location="cpu", weights_only=False)
    A = d["acts"].float().numpy()                       # [N, L+1, D]
    cls = np.asarray(d["cls"])
    split = np.asarray(d["split"])
    cat = np.asarray(d["category"])
    N, L1, D = A.shape
    print(f"acts {A.shape}  classes={dict(zip(*np.unique(cls, return_counts=True)))}", flush=True)

    mid = [L1 // 2 - 2, L1 // 2, L1 // 2 + 2]          # the "3 middle layers"
    rng = np.random.RandomState(SEED)
    con = contrasts(cls)

    r = torch.load(REFUSAL_PATH, map_location="cpu", weights_only=True).float().numpy()
    r = r / np.linalg.norm(r)

    out = {"seed": SEED, "n": int(N), "d": int(D), "n_layers": int(L1),
           "mid_layers": mid, "counts": {k: int(v) for k, v in
                                         zip(*np.unique(cls, return_counts=True))},
           "contrasts": {}}
    dirs = {}

    for name, (mask, y) in con.items():
        tr = mask & (split == "extract")
        te = mask & (split == "test")
        va = mask & (split == "val")
        print(f"\n=== {name}: train={tr.sum()} val={va.sum()} test={te.sum()} ===", flush=True)
        if te.sum() == 0 or len(np.unique(y[te])) < 2:
            print("  skipped (degenerate test split)", flush=True)
            continue

        # tune C once, at the middle layer, on val
        best_C, best_va = C_GRID[0], -1
        for C in C_GRID:
            a = fit_auroc(A[tr, mid[1]], y[tr], A[va, mid[1]], y[va], C=C)
            if a == a and a > best_va:
                best_va, best_C = a, C
        print(f"  C={best_C} (val AUROC {best_va:.3f})", flush=True)

        layers, layers_dim = {}, {}
        for l in range(L1):
            layers[str(l)] = round(fit_auroc(A[tr, l], y[tr], A[te, l], y[te], C=best_C), 4)
            layers_dim[str(l)] = round(dim_auroc(A[tr, l], y[tr], A[te, l], y[te]), 4)
        best_l = max(layers, key=lambda k: (layers[k] if layers[k] == layers[k] else -1))
        # PCA-free guards at the middle layers
        nopca = {str(l): round(full_auroc(A[tr, l], y[tr], A[te, l], y[te], C=best_C), 4)
                 for l in mid}

        # nulls: GLOBAL label shuffle (within-group is a no-op, labels are group-constant)
        yperm = y.copy()
        yperm[tr] = rng.permutation(y[tr])
        perm = fit_auroc(A[tr, int(best_l)], yperm[tr], A[te, int(best_l)], y[te], C=best_C)
        Xr_tr = rng.randn(int(tr.sum()), D)
        Xr_te = rng.randn(int(te.sum()), D)
        rand = fit_auroc(Xr_tr, y[tr], Xr_te, y[te], C=best_C)

        # diff-in-means directions (per layer, full space)
        dm = np.stack([A[tr & (y == 1), l].mean(0) - A[tr & (y == 0), l].mean(0)
                       for l in range(L1)])                      # [L+1, D]
        dm_unit = dm / (np.linalg.norm(dm, axis=-1, keepdims=True) + 1e-12)
        dirs[name] = dm_unit

        cos_r = {str(l): round(float(dm_unit[l] @ r), 4) for l in range(L1)}
        out["contrasts"][name] = {
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "C": best_C,
            "val_auroc": round(best_va, 4), "layers": layers,
            "layers_diffmeans": layers_dim,
            "best_layer": int(best_l), "best_auroc": layers[best_l],
            "best_auroc_diffmeans": layers_dim[best_l],
            "mid_layer_auroc": {str(l): layers[str(l)] for l in mid},
            "mid_layer_auroc_nopca": nopca,
            "permuted_null": round(perm, 4), "random_dir": round(rand, 4),
            "cos_vs_refusal": cos_r,
            "cos_vs_refusal_at_mid": {str(l): cos_r[str(l)] for l in mid},
        }
        print(f"  best layer {best_l}: AUROC {layers[best_l]:.3f} "
              f"(diff-in-means {layers_dim[best_l]:.3f}) | mid {[layers[str(l)] for l in mid]} "
              f"| no-PCA mid {list(nopca.values())} | perm {perm:.3f} rand {rand:.3f}",
              flush=True)

    # ---- whitened cosines at the middle layers (anisotropy robustness check) ----
    print("\n=== whitened geometry ===", flush=True)
    whit = {}
    ex = split == "extract"
    for l in mid:
        X = A[ex, l]
        Sigma = np.cov(X.T)
        e = {}
        for name in dirs:
            e[f"cos_{name}_vs_r_raw"] = round(float(dirs[name][l] @ r), 4)
            e[f"cos_{name}_vs_r_whitened"] = round(whiten_cos(dirs[name][l], r, Sigma), 4)
        if "harmfulness" in dirs and "refusal" in dirs:
            e["cos_harm_vs_refusal_raw"] = round(
                float(dirs["harmfulness"][l] @ dirs["refusal"][l]), 4)
            e["cos_harm_vs_refusal_whitened"] = round(
                whiten_cos(dirs["harmfulness"][l], dirs["refusal"][l], Sigma), 4)
        whit[str(l)] = e
        print(f"  layer {l}: {e}", flush=True)
    out["whitened_geometry"] = whit

    # ---- within-harmful dose-response (removes the CORPUS confound) ----
    # harmful_comply comes from StrongREJECT, benign_comply from Alpaca, so the
    # harmfulness axis could partly encode "which corpus". This control never
    # leaves the harmful corpus: among harmful_comply rows ONLY (same corpus, same
    # wrapper, all complying), does the projection onto d_harm track the judge's
    # harmfulness SCORE? If yes, the axis carries degree-of-harm, not domain.
    print("\n=== within-harmful dose-response (corpus confound control) ===", flush=True)
    dose = {}
    if "harmfulness" in dirs:
        from scipy.stats import spearmanr
        sr = np.asarray(d["sr_norm"], dtype=float)
        hc_te = (cls == "harmful_comply") & (split == "test")
        for l in mid:
            proj = A[hc_te, l] @ dirs["harmfulness"][l]
            s = sr[hc_te]
            rho, p = spearmanr(proj, s)
            lo, hi = np.quantile(s, 1 / 3), np.quantile(s, 2 / 3)
            m = (s <= lo) | (s >= hi)
            au = (float(roc_auc_score((s[m] >= hi).astype(int), proj[m]))
                  if m.sum() > 10 and len(np.unique(s[m] >= hi)) == 2 else float("nan"))
            dose[str(l)] = {"n": int(hc_te.sum()), "spearman_rho": round(float(rho), 4),
                            "p": float(f"{p:.3g}"), "tercile_auroc": round(au, 4)}
            print(f"  layer {l}: n={int(hc_te.sum())} rho={rho:+.3f} (p={p:.3g}) "
                  f"top-vs-bottom-tercile AUROC={au:.3f}", flush=True)
    out["within_harmful_dose_response"] = dose

    # ---- leave-one-harm-category-out on the harmfulness contrast ----
    print("\n=== leave-one-category-out (harmfulness) ===", flush=True)
    mask, y = con["harmfulness"]
    l = mid[1]
    loco = {}
    for c in sorted(set(cat[mask & (cat != "harmless")])):
        te = mask & (cat == c)
        tr = mask & (cat != c) & (cat != "harmless")
        # keep benign in TRAIN and add a benign slice to TEST so both labels are present
        ben = mask & (cat == "harmless")
        bi = np.where(ben)[0]
        rng.shuffle(bi)
        half = len(bi) // 2
        tr_b, te_b = np.zeros(len(y), bool), np.zeros(len(y), bool)
        tr_b[bi[:half]] = True
        te_b[bi[half:]] = True
        TR, TE = tr | tr_b, te | te_b
        if te.sum() < MIN_CAT:
            continue
        a = fit_auroc(A[TR, l], y[TR], A[TE, l], y[TE], C=out["contrasts"]["harmfulness"]["C"])
        loco[c] = {"n_test_harmful": int(te.sum()), "auroc": round(a, 4)}
        print(f"  {c:45s} n={int(te.sum()):4d}  AUROC {a:.3f}", flush=True)
    vals = [v["auroc"] for v in loco.values() if v["auroc"] == v["auroc"]]
    out["leave_one_category_out"] = {"layer": l, "per_category": loco,
                                     "mean": round(float(np.mean(vals)), 4) if vals else None,
                                     "min": round(float(np.min(vals)), 4) if vals else None}

    os.makedirs(TC, exist_ok=True)
    json.dump(out, open(os.path.join(TC, "probe_results.json"), "w"), indent=2)
    torch.save({k: torch.from_numpy(v) for k, v in dirs.items()},
               os.path.join(TC, "directions.pt"))
    print(f"\nwrote -> {TC}/probe_results.json, directions.pt", flush=True)


if __name__ == "__main__":
    main()

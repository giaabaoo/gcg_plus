"""
3-CLASS STEP 4d - capacity-matched dose-response: full activation probe vs lexical.

The first dose-response comparison was unfair to the activations: it scored them by
a 1-D projection onto a single diff-in-means direction (AUROC 0.771) against a
60k-feature TF-IDF logistic regression on the text (0.904). A 1-D readout losing to
a high-capacity model says nothing.

This gives the activations the same shape of model the text got: full probe
(StandardScaler -> PCA-128 -> L2 logistic) on the SAME rows, SAME task - within
harmful_comply only, top-vs-bottom judge-score tercile, train on extract, test on
the held-out test split.

Only after this is the comparison interpretable:
  activation >> lexical -> activations carry harm-degree the wording does not
  activation ~= lexical -> the axis is a rewording detector

Output -> results/threeclass/dose_response_fair.json
"""

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import torch       # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TC = os.environ.get("THREECLASS_DIR", os.path.join(REPO, "results", "threeclass"))
SEED = 1234
C_GRID = [0.01, 0.1, 1.0]


def fit_auroc(Xtr, ytr, Xte, yte, C):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    k = min(128, Xtr.shape[0] - 1, Xtr.shape[1])
    p = PCA(n_components=k, random_state=SEED).fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=3000, random_state=SEED).fit(p.transform(Xtr), ytr)
    return float(roc_auc_score(yte, lr.decision_function(p.transform(Xte))))


def main():
    d = torch.load(os.path.join(TC, "acts.pt"), map_location="cpu", weights_only=False)
    A = d["acts"].float().numpy()
    cls = np.asarray(d["cls"])
    split = np.asarray(d["split"])
    sr = np.asarray(d["sr_norm"], dtype=float)
    mid = json.load(open(os.path.join(TC, "probe_results.json")))["mid_layers"]

    hc = cls == "harmful_comply"
    tr0, va0, te0 = hc & (split == "extract"), hc & (split == "val"), hc & (split == "test")
    lo, hi = np.quantile(sr[tr0], 1 / 3), np.quantile(sr[tr0], 2 / 3)
    band = (sr <= lo) | (sr >= hi)
    tr, va, te = tr0 & band, va0 & band, te0 & band
    y = (sr >= hi).astype(int)
    print(f"train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"(terciles at {lo:.3f}/{hi:.3f})", flush=True)

    out = {"seed": SEED, "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "tercile_cuts": [round(float(lo), 4), round(float(hi), 4)], "layers": {}}
    for l in mid:
        best_C, best_va = C_GRID[0], -1
        for C in C_GRID:
            a = fit_auroc(A[tr, l], y[tr], A[va, l], y[va], C)
            if a > best_va:
                best_va, best_C = a, C
        au = fit_auroc(A[tr, l], y[tr], A[te, l], y[te], best_C)
        out["layers"][str(l)] = {"C": best_C, "val": round(best_va, 4), "test": round(au, 4)}
        print(f"  layer {l}: full-probe tercile AUROC = {au:.4f} (C={best_C})", flush=True)

    lex = json.load(open(os.path.join(TC, "lexical_baseline.json"))) \
        .get("dose_response_lexical", {}).get("tercile_auroc")
    best = max(v["test"] for v in out["layers"].values())
    out["lexical_tercile_auroc"] = lex
    out["best_activation_tercile_auroc"] = best
    out["margin_over_lexical"] = round(best - lex, 4) if lex is not None else None
    print(f"\nbest activation {best:.4f}  vs lexical {lex}  "
          f"margin {out['margin_over_lexical']:+.4f}", flush=True)

    json.dump(out, open(os.path.join(TC, "dose_response_fair.json"), "w"), indent=2)
    print(f"wrote -> {TC}/dose_response_fair.json", flush=True)


if __name__ == "__main__":
    main()

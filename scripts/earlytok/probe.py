"""
Step 4: layer-wise probes + baselines.

PRIMARY ARM -- within-prompt contrast. Features are centered within prompt
(x - mean of that prompt's sampled responses), so anything constant across a
prompt's samples -- topic, category, "does this request afford concrete steps"
-- cancels exactly. The probe can only use how THIS response deviates from what
this prompt typically produces. That is the version of the question that means
"the model has already committed", rather than "this prompt is easy".

SECONDARY ARM -- pooled (uncentered). This is the originally-proposed design.
The gap between pooled and within-prompt AUROC *is* the prompt-affordance
confound, measured.

Labels: within each prompt, sampled responses above that prompt's median
sr_expected are "actionable", below are "vacuous" (ties dropped). For the pooled
arm, the split is at the global median.

Baselines, all on identical folds (GroupKFold by prompt_id, so a prompt's
samples never straddle a fold):
  early        - first-10-response-token activations   (the hypothesis)
  full         - full-response pooling                 (upper reference)
  length       - response token count alone
  early_resid  - early activations with length regressed out (length is a
                 mediator, not just a control: early tokens may encode intended
                 length, and length mechanically drives detail)
  lexical      - bag of char n-grams over the first 10 response tokens' TEXT.
                 Format commitment ("1. **" vs "Well,") mechanically predicts
                 detail. The informative comparison is whether `early` BEATS
                 `lexical`; if it does not, the probe is a scaffold detector.
  category     - prompt category only (pooled arm only; cancels when centered)
  random       - Gaussian noise, matched dim: the CV floor given n and
                 regularization. Confirms the pipeline is not inflating AUROC.
  permuted     - real early features, labels shuffled: the stronger null.
                 Shuffled within prompt for the within-prompt arm, globally for
                 the pooled arm (see the note at the permutation call).

PCA is fit inside each training fold (never on held-out data).
"""

import glob
import json
import os
import sys

# Pin BLAS to one thread BEFORE numpy loads. The per-fold PCA is a randomized
# SVD of an ~870x4096 matrix -- far too small to parallelize. Measured on this
# box: 0.46s at 1 thread vs 9.42s at 26 threads, a 20x slowdown from pure thread
# thrashing, which dominated total runtime (8 min vs 2.5 h). Layers are
# independent, so real parallelism belongs across layers, not inside one SVD.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARD_DIR = os.path.join(REPO, "results", "earlytok", "shards")
SCORES    = os.path.join(REPO, "results", "earlytok", "scores.json")
OUT_JSON  = os.path.join(REPO, "results", "earlytok", "probe_results.json")

SEED    = 1234
N_PCA   = 128
N_FOLDS = 5
C_GRID  = [1e-3, 1e-2, 1e-1, 1.0]
EARLY_TOK_CHARS = 120   # chars of text approximating the first 10 tokens


def load():
    """Assemble (activations, scores, metadata) for sampled non-refusal responses."""
    score_rows = json.load(open(SCORES))
    smap = {(r["prompt_id"], r["mode"]): r for r in score_rows}

    early, full, meta = [], [], []
    for sp in sorted(glob.glob(os.path.join(SHARD_DIR, "*.pt")),
                     key=lambda p: int(os.path.basename(p)[:-3])):
        d = torch.load(sp, map_location="cpu", weights_only=False)
        for r in d["rows"]:
            if r["mode"] == "greedy":
                continue
            s = smap.get((d["prompt_id"], r["mode"]))
            if s is None or s["refusal_heuristic"]:
                continue
            early.append(r["early"].float().numpy())
            full.append(r["full"].float().numpy())
            meta.append({
                "prompt_id": d["prompt_id"], "category": d["category"],
                "score": s["sr_expected"], "resp_len": r["resp_len"],
                "text_head": r["text"][:EARLY_TOK_CHARS],
            })
    return np.stack(early), np.stack(full), meta   # (N, L, d), (N, L, d)


def make_labels(meta, within):
    """Binary actionable/vacuous. within=True -> split at each prompt's own
    median (the contrast); within=False -> global median (pooled)."""
    scores = np.array([m["score"] for m in meta])
    y = np.full(len(meta), -1)
    if within:
        by = {}
        for i, m in enumerate(meta):
            by.setdefault(m["prompt_id"], []).append(i)
        for idxs in by.values():
            med = np.median(scores[idxs])
            for i in idxs:
                if scores[i] > med:
                    y[i] = 1
                elif scores[i] < med:
                    y[i] = 0
    else:
        med = np.median(scores)
        y = (scores > med).astype(int)
        y[scores == med] = -1
    return y


def center_within(X, groups):
    """Subtract each prompt's mean -> prompt-level structure cancels."""
    Xc = X.copy()
    for g in np.unique(groups):
        m = groups == g
        Xc[m] -= Xc[m].mean(axis=0, keepdims=True)
    return Xc


def cv_auroc(X, y, groups, seed=SEED):
    """GroupKFold CV AUROC.

    Scaler + PCA are fit ONCE per outer fold on that fold's training rows only,
    then the C grid is searched by inner CV on the already-reduced training
    matrix. Fitting PCA inside the inner loop instead would be ~13x the work for
    identical results, since the inner split never sees held-out data either way.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    X = np.nan_to_num(X.astype(np.float64))
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)

    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if Xtr.shape[1] > N_PCA:
            pca = PCA(n_components=min(N_PCA, len(tr) - 1, Xtr.shape[1]),
                      random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)

        best_c, best_s = C_GRID[0], -1
        n_inner = min(3, len(np.unique(groups[tr])))
        if n_inner >= 2:
            inner = GroupKFold(n_splits=n_inner)
            for c in C_GRID:
                ss = []
                for itr, ite in inner.split(Xtr, y[tr], groups[tr]):
                    if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2:
                        continue
                    lr = LogisticRegression(C=c, max_iter=2000).fit(Xtr[itr], y[tr][itr])
                    ss.append(roc_auc_score(y[tr][ite], lr.predict_proba(Xtr[ite])[:, 1]))
                if ss and np.mean(ss) > best_s:
                    best_s, best_c = np.mean(ss), c

        lr = LogisticRegression(C=best_c, max_iter=2000).fit(Xtr, y[tr])
        oof[te] = lr.predict_proba(Xte)[:, 1]

    ok = ~np.isnan(oof)
    if len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], oof[ok]))


def residualize_on_length(X, lengths):
    """Remove the component of every feature linearly predictable from length."""
    L = lengths.reshape(-1, 1).astype(np.float64)
    return X - LinearRegression().fit(L, X).predict(L)


def main():
    np.random.seed(SEED)
    print(f"seed={SEED}", flush=True)

    E, F, meta = load()
    N, L, D = E.shape
    groups = np.array([m["prompt_id"] for m in meta])
    lengths = np.array([m["resp_len"] for m in meta], dtype=float)
    print(f"{N} sampled non-refusal responses, {L} layers, d={D}, "
          f"{len(np.unique(groups))} prompts", flush=True)

    out = {"seed": SEED, "n": int(N), "n_layers": int(L), "d": int(D),
           "n_prompts": int(len(np.unique(groups))), "arms": {}}

    for arm_name, within in [("within_prompt", True), ("pooled", False)]:
        y_all = make_labels(meta, within=within)
        keep = y_all >= 0
        y = y_all[keep]
        g = groups[keep]
        ln = lengths[keep]
        print(f"\n=== {arm_name}: n={keep.sum()} "
              f"(pos={int(y.sum())}, neg={int((1-y).sum())}) ===", flush=True)

        Ek, Fk = E[keep], F[keep]
        if within:
            Ek = np.stack([center_within(Ek[:, l], g) for l in range(L)], axis=1)
            Fk = np.stack([center_within(Fk[:, l], g) for l in range(L)], axis=1)
            ln_f = ln - np.array([ln[g == gg].mean() for gg in g])
        else:
            ln_f = ln

        arm = {"n": int(keep.sum()), "pos": int(y.sum()), "layers": {}}

        # --- non-layerwise baselines ---
        arm["length"] = cv_auroc(ln_f.reshape(-1, 1), y, g)
        rng = np.random.RandomState(SEED)
        arm["random"] = cv_auroc(rng.randn(len(y), D), y, g)

        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=3000)
        Xlex = tf.fit_transform([m["text_head"] for m, k in zip(meta, keep) if k]).toarray()
        if within:
            Xlex = center_within(Xlex, g)
        arm["lexical"] = cv_auroc(Xlex, y, g)

        if not within:
            cats = sorted({m["category"] for m in meta})
            Xcat = np.array([[float(m["category"] == c) for c in cats]
                             for m, k in zip(meta, keep) if k])
            arm["category"] = cv_auroc(Xcat, y, g)

        print(f"  length={arm['length']:.3f}  lexical={arm['lexical']:.3f}  "
              f"random={arm['random']:.3f}"
              + (f"  category={arm['category']:.3f}" if not within else ""), flush=True)

        # --- layerwise ---
        for l in range(L):
            rec = {
                "early": cv_auroc(Ek[:, l], y, g),
                "full":  cv_auroc(Fk[:, l], y, g),
                # residualize on the SAME length variable the arm uses (centered
                # within prompt for the contrast arm), or the two are mismatched
                "early_resid": cv_auroc(residualize_on_length(Ek[:, l], ln_f), y, g),
            }
            arm["layers"][str(l)] = rec
            print(f"  L{l:02d} early={rec['early']:.3f} full={rec['full']:.3f} "
                  f"early_resid={rec['early_resid']:.3f}", flush=True)

        # permutation null at the best early layer
        best_l = max(range(L), key=lambda l: (arm["layers"][str(l)]["early"]
                                              if not np.isnan(arm["layers"][str(l)]["early"]) else -1))
        # The null must destroy the label-feature link WITHOUT being a no-op.
        # Within-prompt arm: labels are assigned per prompt, so shuffling inside
        # a group is a real permutation. Pooled arm: labels come from the GLOBAL
        # median, so a prompt's samples usually share one label and a
        # within-group shuffle changes nothing -- it reproduced the real labels
        # and returned 0.892. Pooled therefore needs a global shuffle.
        yp = y.copy()
        if within:
            for gg in np.unique(g):
                m = g == gg
                yp[m] = rng.permutation(yp[m])
        else:
            yp = rng.permutation(yp)
        arm["permuted_at_best_layer"] = cv_auroc(Ek[:, best_l], yp, g)
        arm["best_early_layer"] = int(best_l)
        print(f"  best early layer={best_l} "
              f"(permuted null={arm['permuted_at_best_layer']:.3f})", flush=True)

        out["arms"][arm_name] = arm

    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nWrote -> {OUT_JSON}")


if __name__ == "__main__":
    main()

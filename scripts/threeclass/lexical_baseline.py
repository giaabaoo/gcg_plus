"""
3-CLASS STEP 4c - LEXICAL baseline. Is the "harmfulness axis" just vocabulary?

Motivation (from the result itself): the harmfulness probe already reaches ~0.91
AUROC at LAYER 1. Layer 1 is barely past the token embeddings, so near-ceiling
separability there is the signature of a SURFACE-LEXICAL difference, not a
semantic representation - harmful-derived and Alpaca-derived PAPs simply use
different words. Part 7 hit the same trap and used the same control.

So: fit a bag-of-ngrams model on the PAP TEXT alone, same rows, same splits, same
metric. Read it as a floor on how much of the activation AUROC is just wording.

  lexical ~= activation  -> the probe is a topic/vocabulary detector; the
                            "harmfulness axis" claim does not survive.
  lexical <<  activation -> the activations carry something the surface text does not.

Also reported: the same baseline for the refusal contrast (where the two classes
share a corpus, so lexicon should matter far less) as an internal reference.

Output -> results/threeclass/lexical_baseline.json
"""

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer   # noqa: E402
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "threeclass"))
from capture_acts import collect  # noqa: E402  (identical row order/splits as the probe)

TC = os.environ.get("THREECLASS_DIR", os.path.join(REPO, "results", "threeclass"))
SEED = 1234


def contrasts(cls):
    cls = np.asarray(cls)
    return {
        "harmfulness": (np.isin(cls, ["harmful_comply", "benign_comply"]),
                        (cls == "harmful_comply").astype(int)),
        "refusal":     (np.isin(cls, ["refuse", "harmful_comply", "benign_comply"]),
                        (cls == "refuse").astype(int)),
        "naive":       (np.isin(cls, ["harmful_comply", "refuse"]),
                        (cls == "harmful_comply").astype(int)),
    }


def main():
    rows = collect()
    texts = np.asarray([r["pap"] for r in rows], dtype=object)
    cls = np.asarray([r["cls"] for r in rows])
    split = np.asarray([r["split"] for r in rows])
    print(f"{len(rows)} rows", flush=True)

    out = {"seed": SEED, "note": "TF-IDF word 1-2 grams on the PAP text; same rows/splits"}
    for name, (mask, y) in contrasts(cls).items():
        tr = mask & (split == "extract")
        te = mask & (split == "test")
        if te.sum() == 0 or len(np.unique(y[te])) < 2:
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                              sublinear_tf=True, strip_accents="unicode")
        Xtr = vec.fit_transform(texts[tr])
        Xte = vec.transform(texts[te])
        lr = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED).fit(Xtr, y[tr])
        au = float(roc_auc_score(y[te], lr.decision_function(Xte)))
        out[name] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "vocab": int(Xtr.shape[1]), "auroc": round(au, 4)}
        print(f"  {name:12s} lexical AUROC = {au:.4f}  (n_test={int(te.sum())})", flush=True)

    # ---- lexical version of the within-harmful dose-response ----
    # The cross-corpus contrast turned out to be mostly vocabulary, so the
    # dose-response is the last control standing. Give it the same test: within
    # harmful_comply ONLY, predict the top-vs-bottom judge-score tercile from text.
    # If lexical matches the activation tercile AUROC, even that is wording.
    sr = np.asarray([r["sr_norm"] for r in rows], dtype=float)
    hc = cls == "harmful_comply"
    tr = hc & (split == "extract")
    te = hc & (split == "test")
    lo, hi = np.quantile(sr[tr], 1 / 3), np.quantile(sr[tr], 2 / 3)
    mtr = tr & ((sr <= lo) | (sr >= hi))
    mte = te & ((sr <= lo) | (sr >= hi))
    if mte.sum() > 10:
        ytr, yte = (sr[mtr] >= hi).astype(int), (sr[mte] >= hi).astype(int)
        if len(np.unique(ytr)) == 2 and len(np.unique(yte)) == 2:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                  sublinear_tf=True, strip_accents="unicode")
            Xtr = vec.fit_transform(texts[mtr])
            lr = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED).fit(Xtr, ytr)
            au = float(roc_auc_score(yte, lr.decision_function(vec.transform(texts[mte]))))
            out["dose_response_lexical"] = {"n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                                            "tercile_auroc": round(au, 4)}
            print(f"  dose-response lexical tercile AUROC = {au:.4f} "
                  f"(activation was 0.771)  n_test={int(mte.sum())}", flush=True)

    json.dump(out, open(os.path.join(TC, "lexical_baseline.json"), "w"), indent=2)
    print(f"wrote -> {TC}/lexical_baseline.json", flush=True)


if __name__ == "__main__":
    main()

"""
Follow-up to Part 8: re-run Part 6's cross-technique cosine matrix WHITENED.

Part 8 showed raw residual-stream cosine overstates axis overlap
(cos(harmfulness, refusal) 0.78 raw -> 0.20 whitened). Part 6's "one shared axis"
conclusion rested on an all-positive technique x technique matrix (0.39-0.89)
measured RAW. This checks whether that structure survives anisotropy control.

Method (mirrors whiten_geometry.py, the numerically-sound version):
- Directions: Part 6's saved per-technique diff-in-means at layer 14 (the layer
  Part 6 reported the matrix at), from results/directions/directions_*.pt.
- Whitening subspace: top-k PCA of the harmful-PAP activations in acts.pt at the
  same layer (Part 6's pools were never cached; this is the same input
  distribution - last-prompt-token on PAP-wrapped harmful prompts). Whiten in the
  top-k subspace, NOT the full covariance (full-cov whitening is the 628-near-null
  -eigenvalue artifact from Part 8).
- Reliability gate: each technique direction's subspace energy is reported. Part 8
  established these diff-in-means directions sit at ~0.999 (unlike Arditi r at
  11-42%), so a whitened technique x technique cosine IS trustworthy here.

Reading:
  all-positive survives whitening -> Part 6's "one shared axis" holds.
  matrix decorrelates             -> that claim was raw-cosine artifact; softens
                                     to "related but not collinear".

Output -> results/directions/whitened_matrix.json
"""

import glob
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np   # noqa: E402
import torch        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_DIR = os.path.join(REPO, "results", "directions")
ACTS = os.path.join(REPO, "results", "threeclass", "acts.pt")
OUT = os.path.join(DIR_DIR, "whitened_matrix.json")
LAYERS = [12, 14, 16]      # Part 6 reported the matrix at 14; 12/16 for robustness
KS = [128, 256, 512]


def load_directions():
    techs, raw = [], {}
    for f in sorted(glob.glob(os.path.join(DIR_DIR, "directions_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        t = d["technique"]
        techs.append(t)
        raw[t] = d["raw"].float().numpy()          # [L+1, D]
    return techs, raw


def whiten_vecs(vecs, Vk, sk):
    """project into top-k eigenbasis, scale by 1/sqrt(eigval), renormalise."""
    return {n: (Vk.T @ v) / sk for n, v in vecs.items()}


def main():
    techs, raw = load_directions()
    print(f"{len(techs)} techniques: {techs}", flush=True)

    a = torch.load(ACTS, map_location="cpu", weights_only=False)
    A = a["acts"].float().numpy()
    cls = np.asarray(a["cls"])
    harm = (cls == "harmful_comply") | (cls == "refuse")     # Part 6's input distribution

    out = {"method": "top-k subspace whitening of Part-6 saved directions; "
                     "Sigma from harmful-PAP acts.pt",
           "note": "full-covariance whitening rejected (Part 8 artifact: cond ~1e8)",
           "layers": {}}

    for L in LAYERS:
        X = A[harm, L]
        Xc = X - X.mean(0)
        w, V = np.linalg.eigh(np.cov(Xc.T))
        idx = np.argsort(w)[::-1]
        w, V = w[idx], V[:, idx]

        vecs = {t: raw[t][L] / (np.linalg.norm(raw[t][L]) + 1e-12) for t in techs}
        pairs = [(techs[i], techs[j]) for i in range(len(techs)) for j in range(i + 1, len(techs))]

        entry = {"raw": {}, "whitened": {}, "subspace_energy": {}, "summary": {}}
        rawvals = {}
        for ti, tj in pairs:
            c = round(float(vecs[ti] @ vecs[tj]), 4)
            entry["raw"][f"{ti} | {tj}"] = c
            rawvals[(ti, tj)] = c

        for k in KS:
            Vk, sk = V[:, :k], np.sqrt(np.maximum(w[:k], 1e-12))
            proj = {t: (Vk.T @ vecs[t]) for t in techs}
            wv = {t: proj[t] / sk for t in techs}
            wc = {}
            for ti, tj in pairs:
                x, y = wv[ti], wv[tj]
                wc[f"{ti} | {tj}"] = round(
                    float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)), 4)
            entry["whitened"][str(k)] = wc
            entry["subspace_energy"][str(k)] = {
                t: round(float((proj[t] ** 2).sum()), 4) for t in techs}

        # summary stats: does the all-positive structure survive at k=512?
        rv = np.array(list(rawvals.values()))
        wv512 = np.array(list(entry["whitened"]["512"].values()))
        entry["summary"] = {
            "raw_min": round(float(rv.min()), 4), "raw_mean": round(float(rv.mean()), 4),
            "raw_all_positive": bool((rv > 0).all()),
            "whitened_min": round(float(wv512.min()), 4),
            "whitened_mean": round(float(wv512.mean()), 4),
            "whitened_all_positive": bool((wv512 > 0).all()),
            "min_subspace_energy": round(float(min(
                entry["subspace_energy"]["512"].values())), 4)}
        out["layers"][str(L)] = entry

        print(f"\n=== layer {L} ===", flush=True)
        print(f"  raw:      min {entry['summary']['raw_min']:+.3f}  mean "
              f"{entry['summary']['raw_mean']:+.3f}  all_positive "
              f"{entry['summary']['raw_all_positive']}", flush=True)
        print(f"  whit k512: min {entry['summary']['whitened_min']:+.3f}  mean "
              f"{entry['summary']['whitened_mean']:+.3f}  all_positive "
              f"{entry['summary']['whitened_all_positive']}", flush=True)
        print(f"  min subspace energy of a technique dir: "
              f"{entry['summary']['min_subspace_energy']:.3f} "
              f"(reliable if ~1.0)", flush=True)

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

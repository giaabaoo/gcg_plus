"""
3-CLASS STEP 4b - anisotropy-controlled geometry, done in a WELL-CONDITIONED subspace.

Why this is a separate pass. The first whitening attempt inverted the full 4096x4096
activation covariance with a token ridge. That covariance is estimated from ~4.3k
samples in 4096 dimensions: condition number ~1.7e8, with 628 eigenvalues below
1e-6 of the max. Sigma^{-1/2} then amplifies those near-null noise directions by
~1e4 and the whitened cosine becomes noise-dominated - it returned ~0.00 for EVERY
pair, including cos(refusal-contrast, r), which cannot be zero. Those numbers were
an artifact and are superseded here.

Correct instrument: restrict to the top-k principal subspace of the activations
(where the covariance is actually estimable), whiten by the eigenvalues there, and
report across several k so the reader can see stability rather than trust one cut.

Also reported per direction: the fraction of its squared norm captured by the top-k
subspace. If a direction lives mostly OUTSIDE the retained subspace, the whitened
cosine for it is not meaningful, and that has to be visible rather than implied.

Output -> results/threeclass/geometry.json (+ printed table)
"""

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np   # noqa: E402
import torch        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TC = os.environ.get("THREECLASS_DIR", os.path.join(REPO, "results", "threeclass"))
KS = [128, 256, 512]


def main():
    d = torch.load(os.path.join(TC, "acts.pt"), map_location="cpu", weights_only=False)
    A = d["acts"].float().numpy()
    split = np.asarray(d["split"])
    dirs = {k: v.numpy() for k, v in
            torch.load(os.path.join(TC, "directions.pt"), map_location="cpu",
                       weights_only=False).items()}
    r = torch.load(os.path.join(REPO, "checkpoints", "direction.pt"),
                   map_location="cpu", weights_only=True).float().numpy()
    r = r / np.linalg.norm(r)

    res = json.load(open(os.path.join(TC, "probe_results.json")))
    mid = res["mid_layers"]
    out = {"method": "whitening restricted to top-k PCA subspace of extract activations",
           "note": ("full-covariance whitening was rejected: cond ~1.7e8, 628 eigenvalues "
                    "<1e-6*max, which drove all whitened cosines to ~0 (artifact)"),
           "ks": KS, "layers": {}}

    ex = split == "extract"
    for l in mid:
        X = A[ex, l]
        mu = X.mean(0)
        Xc = X - mu
        # eigendecomposition of the covariance, descending
        S = np.cov(Xc.T)
        w, V = np.linalg.eigh(S)
        idx = np.argsort(w)[::-1]
        w, V = w[idx], V[:, idx]

        vecs = {"r": r}
        for name in dirs:
            vecs[name] = dirs[name][l] / (np.linalg.norm(dirs[name][l]) + 1e-12)

        entry = {"raw": {}, "whitened": {}, "subspace_energy": {}}
        pairs = [(a, b) for a in vecs for b in vecs if a < b]
        for a, b in pairs:
            entry["raw"][f"{a}|{b}"] = round(float(vecs[a] @ vecs[b]), 4)

        for k in KS:
            Vk, wk = V[:, :k], w[:k]
            sk = np.sqrt(np.maximum(wk, 1e-12))
            proj = {n: (Vk.T @ v) for n, v in vecs.items()}
            whit = {n: p / sk for n, p in proj.items()}
            entry["whitened"][str(k)] = {
                f"{a}|{b}": round(float(whit[a] @ whit[b] /
                                        (np.linalg.norm(whit[a]) * np.linalg.norm(whit[b]) + 1e-12)), 4)
                for a, b in pairs}
            entry["subspace_energy"][str(k)] = {
                n: round(float((proj[n] ** 2).sum()), 4) for n in vecs}
        out["layers"][str(l)] = entry

        print(f"\n=== layer {l} ===", flush=True)
        print("  raw     :", entry["raw"], flush=True)
        for k in KS:
            print(f"  whit k={k}:", entry["whitened"][str(k)], flush=True)
            print(f"    subspace energy:", entry["subspace_energy"][str(k)], flush=True)

    json.dump(out, open(os.path.join(TC, "geometry.json"), "w"), indent=2)
    res["whitened_geometry_note"] = out["note"]
    res["whitened_geometry_superseded_by"] = "geometry.json"
    json.dump(res, open(os.path.join(TC, "probe_results.json"), "w"), indent=2)
    print(f"\nwrote -> {TC}/geometry.json", flush=True)


if __name__ == "__main__":
    main()

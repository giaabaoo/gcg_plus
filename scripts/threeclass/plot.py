"""
3-CLASS STEP 6 - deliverable figure + summary.md.

Panel A: AUROC vs layer for the three contrasts. The story is the GAP between
  `naive` (confounded: differs on refusing AND harmful) and `harmfulness`
  (wrapper-matched, non-refusal held constant). Panel A answers "is there a
  harmfulness signal once you stop letting the probe read refusal?"

Panel B: leave-one-harm-category-out AUROC on the harmfulness contrast — the
  robustness test. A harmfulness representation should survive an unseen
  category; memorised topic should not.

Colors are categorical slots 1-3 from the reference palette in fixed order
(that triple is the documented all-pairs-validated set in both modes). Every
series also carries a distinct linestyle and a direct label, so identity never
depends on color alone, and a markdown table view ships alongside (the relief
rule for the aqua slot on a light surface).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TC = os.environ.get("THREECLASS_DIR", os.path.join(REPO, "results", "threeclass"))
RES = os.path.join(TC, "probe_results.json")
PNG = os.path.join(TC, "auroc_by_layer.png")
MD = os.path.join(TC, "summary.md")

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # slots 1-3, fixed order
NULL = "#8a8a80"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"

SERIES = [
    ("harmfulness", S1, "solid", "harmfulness\n(wrapper-matched)"),
    ("refusal", S2, (0, (5, 2)), "refusal"),
    ("naive", S3, (0, (1, 1.6)), "naive (confounded)"),
]
SHORT = {"harmfulness": "harmfulness", "refusal": "refusal", "naive": "naive"}


def main():
    d = json.load(open(RES))
    cons = d["contrasts"]
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.42)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---- Panel A: AUROC vs layer ----
    for key, color, ls, label in SERIES:
        if key not in cons:
            continue
        layers = sorted(cons[key]["layers"], key=int)
        xs = [int(l) for l in layers]
        ys = [cons[key]["layers"][l] for l in layers]
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.4, label=label,
                solid_capstyle="round")
        ax.text(xs[-1] + 0.6, ys[-1], SHORT[key], color=color,
                fontsize=9, va="center", ha="left")

    # lexical floors: the gap between a curve and its own flat line IS the finding
    lex_path0 = os.path.join(TC, "lexical_baseline.json")
    if os.path.exists(lex_path0):
        lx = json.load(open(lex_path0))
        for key, color, _, _ in SERIES:
            if key not in lx:
                continue
            ax.axhline(lx[key]["auroc"], color=color, linestyle=(0, (2, 3)), linewidth=1.3,
                       alpha=0.85, zorder=1)
        ax.plot([], [], color=MUTED, linestyle=(0, (2, 3)), linewidth=1.3,
                label="lexical floor (same color)")

    nulls = [cons[k]["permuted_null"] for k in cons if cons[k].get("permuted_null") ==
             cons[k].get("permuted_null")]
    if nulls:
        ax.axhline(sum(nulls) / len(nulls), color=NULL, linestyle=(0, (1, 2)),
                   linewidth=1.5, label="permuted null")
    ax.axhline(0.5, color=GRID, linewidth=1.4, zorder=0)
    for l in d.get("mid_layers", []):
        ax.axvline(l, color=GRID, linewidth=1.0, zorder=0)
    ax.set_title("Decodability by layer\n(train=extract prompts, test=held-out prompts)",
                 fontsize=11, color=INK, pad=10)
    ax.set_xlabel("layer", fontsize=10, color=MUTED)
    ax.set_ylabel("AUROC", fontsize=10, color=MUTED)
    lmax = max(int(l) for l in cons[SERIES[0][0]]["layers"])
    ax.set_xlim(-1, lmax + max(3, 0.34 * lmax))          # room for the direct labels
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)

    # ---- Panel B: leave-one-category-out ----
    loco = d.get("leave_one_category_out", {}).get("per_category", {})
    if loco:
        items = sorted(loco.items(), key=lambda kv: kv[1]["auroc"])
        names = [k if len(k) <= 34 else k[:31] + "..." for k, _ in items]
        vals = [v["auroc"] for _, v in items]
        ypos = range(len(items))
        bx.barh(list(ypos), vals, color=S1, height=0.62, zorder=3)
        bx.axvline(0.5, color=GRID, linewidth=1.4, zorder=0)
        mean = d["leave_one_category_out"].get("mean")
        if mean:
            bx.axvline(mean, color=NULL, linestyle=(0, (4, 2)), linewidth=1.5,
                       zorder=4, label=f"mean {mean:.3f}")
            bx.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
        bx.set_yticks(list(ypos))
        bx.set_yticklabels(names, fontsize=8.5, color=INK)
        for i, v in zip(ypos, vals):
            bx.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5, color=MUTED,
                    zorder=6, bbox=dict(facecolor="white", edgecolor="none",
                                        pad=1.2, alpha=0.85))
        bx.set_xlim(0.4, 1.02)
        bx.set_xlabel("AUROC on the held-out category", fontsize=10, color=MUTED)
        bx.set_title("Harmfulness axis: leave-one-harm-category-out\n"
                     f"(layer {d['leave_one_category_out'].get('layer')})",
                     fontsize=11, color=INK, pad=10)

    for a in (ax, bx):
        a.grid(axis="x" if a is bx else "y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle("Is there a harmfulness axis distinct from the refusal axis?",
                 fontsize=13, color=INK, y=1.06, x=0.02, ha="left")
    fig.savefig(PNG, dpi=180, bbox_inches="tight", facecolor="white")
    print("wrote", PNG)

    # ---- table view + summary ----
    L = ["# 3-class probe: refusal axis vs harmfulness axis\n",
         f"n={d['n']} rows, d={d['d']}, layers={d['n_layers']}, seed={d['seed']}",
         f"class counts: {d['counts']}\n",
         "## Headline (held-out test prompts)\n",
         "PCA-128 probe is the headline; diff-in-means and no-PCA columns guard against",
         "a genuine low-variance axis being truncated away by PCA.\n",
         "| contrast | best layer | AUROC | diff-in-means | mid-layer AUROC | "
         "mid no-PCA | permuted null | random |",
         "|---|---|---|---|---|---|---|---|"]
    for key, _, _, label in SERIES:
        if key not in cons:
            continue
        c = cons[key]
        mid = " / ".join(f"{v:.3f}" for v in c["mid_layer_auroc"].values())
        nop = " / ".join(f"{v:.3f}" for v in c.get("mid_layer_auroc_nopca", {}).values()) or "-"
        dm = c.get("best_auroc_diffmeans")
        dm = f"{dm:.3f}" if isinstance(dm, (int, float)) else "-"
        L.append(f"| {key} | {c['best_layer']} | **{c['best_auroc']:.3f}** | {dm} | {mid} | "
                 f"{nop} | {c['permuted_null']:.3f} | {c['random_dir']:.3f} |")

    lex_path = os.path.join(TC, "lexical_baseline.json")
    if os.path.exists(lex_path):
        lex = json.load(open(lex_path))
        L += ["\n## Lexical control - how much of this is just vocabulary?\n",
              "Bag-of-ngrams (TF-IDF 1-2gram) on the PAP TEXT alone, same rows and splits.",
              "Motivated by the probe already reaching ~0.91 at LAYER 1, which is the",
              "signature of a surface-lexical difference rather than a representation.\n",
              "| contrast | lexical | activation | margin |", "|---|---|---|---|"]
        for key, _, _, _ in SERIES:
            if key not in cons or key not in lex:
                continue
            a, b = lex[key]["auroc"], cons[key]["best_auroc"]
            L.append(f"| {key} | {a:.3f} | {b:.3f} | **{b - a:+.3f}** |")
        fair_path = os.path.join(TC, "dose_response_fair.json")
        if os.path.exists(fair_path):
            fr = json.load(open(fair_path))
            L += ["\n### Capacity-matched dose-response (within harmful corpus only)\n",
                  "The 1-D projection (0.771) was an unfair readout against a 60k-feature text",
                  "model, so the activations get the same full probe here.\n",
                  f"- lexical: **{fr['lexical_tercile_auroc']:.3f}**",
                  f"- activation (full probe): **{fr['best_activation_tercile_auroc']:.3f}**",
                  f"- margin: **{fr['margin_over_lexical']:+.3f}** "
                  f"(n_train={fr['n_train']}, n_test={fr['n_test']})"]

    geo_path = os.path.join(TC, "geometry.json")
    if os.path.exists(geo_path):
        g = json.load(open(geo_path))
        kmax = str(g["ks"][-1])
        L += ["\n## Geometry (anisotropy-controlled)\n",
              "Whitening is done in the top-k principal subspace of the activations.",
              "Full-covariance whitening was REJECTED as a numerical artifact: cond ~1.7e8",
              "with 628 eigenvalues <1e-6*max drove every whitened cosine to ~0, including",
              "cos(refusal-contrast, r), which cannot be zero.\n",
              "**Read the subspace-energy column first.** Arditi's r holds only a minority of",
              "its norm inside the estimable subspace, so whitened cosines INVOLVING r are not",
              "reliable and r is compared raw only. The contrast directions sit at ~0.999.\n",
              f"| layer | pair | raw | whitened (k={kmax}) |", "|---|---|---|---|"]
        for l, e in g["layers"].items():
            for pair, raw in e["raw"].items():
                wh = e["whitened"][kmax][pair]
                flag = " *(r: unreliable)*" if "r" in pair.split("|") else ""
                label = pair.replace("|", " vs ")          # bare | breaks the table
                L.append(f"| {l} | {label}{flag} | {raw:+.3f} | {wh:+.3f} |")
        L += ["\n### Subspace energy (fraction of squared norm inside top-k)\n",
              f"| layer | " + " | ".join(next(iter(g["layers"].values()))
                                         ["subspace_energy"][kmax]) + " |",
              "|---" * (1 + len(next(iter(g["layers"].values()))["subspace_energy"][kmax])) + "|"]
        for l, e in g["layers"].items():
            L.append(f"| {l} | " + " | ".join(
                f"{v:.3f}" for v in e["subspace_energy"][kmax].values()) + " |")

    dose = d.get("within_harmful_dose_response") or {}
    if dose:
        L += ["\n## Within-harmful dose-response (corpus-confound control)\n",
              "harmful_comply rows ONLY (same corpus, same wrapper, all complying):",
              "does the projection onto the harmfulness axis track the judge score?\n",
              "| layer | n | Spearman rho | p | top-vs-bottom-tercile AUROC |",
              "|---|---|---|---|---|"]
        for l, e in dose.items():
            L.append(f"| {l} | {e['n']} | {e['spearman_rho']:+.3f} | {e['p']:.3g} | "
                     f"{e['tercile_auroc']:.3f} |")

    lo = d.get("leave_one_category_out", {})
    if lo.get("per_category"):
        L += [f"\n## Leave-one-harm-category-out (harmfulness, layer {lo['layer']})\n",
              f"mean {lo['mean']:.3f}, min {lo['min']:.3f}\n",
              "| held-out category | n harmful | AUROC |", "|---|---|---|"]
        for k, v in sorted(lo["per_category"].items(), key=lambda kv: kv[1]["auroc"]):
            L.append(f"| {k} | {v['n_test_harmful']} | {v['auroc']:.3f} |")

    L += ["\n## Per-layer AUROC (table view)\n",
          "| layer | " + " | ".join(k for k, _, _, _ in SERIES if k in cons) + " |",
          "|---" * (1 + sum(1 for k, _, _, _ in SERIES if k in cons)) + "|"]
    keys = [k for k, _, _, _ in SERIES if k in cons]
    for l in sorted(cons[keys[0]]["layers"], key=int):
        L.append(f"| {l} | " + " | ".join(f"{cons[k]['layers'][l]:.3f}" for k in keys) + " |")

    open(MD, "w").write("\n".join(L) + "\n")
    print("wrote", MD)


if __name__ == "__main__":
    main()

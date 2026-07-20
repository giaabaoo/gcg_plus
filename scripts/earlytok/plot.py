"""
Step 5: the deliverable plot + summary.md.

AUROC vs layer, two panels: within-prompt contrast (primary) and pooled
(secondary, the originally-proposed design). The gap between panels is the
prompt-affordance confound made visible.

Layerwise series get categorical slots 1-3 from the reference palette, in fixed
order. The non-layerwise baselines (length, lexical, random) are flat references
-- dashed, muted, directly labeled -- because they do not vary by layer and
should not compete with the layer curves for attention.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES  = os.path.join(REPO, "results", "earlytok")

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#8a8983"
GRID      = "#e4e3de"
SERIES    = {"early": "#2a78d6", "full": "#008300", "early_resid": "#e87ba4"}
LABEL     = {"early": "early (first 10 tok)", "full": "full response",
             "early_resid": "early, length residualized"}
REFS      = {"length": "length only", "lexical": "lexical (first-10 text)",
             "random": "random, matched dim"}


def panel(ax, arm, title, subtitle, ylim):
    layers = sorted(int(k) for k in arm["layers"])
    for key in ("early", "full", "early_resid"):
        ys = [arm["layers"][str(l)][key] for l in layers]
        ax.plot(layers, ys, lw=2, color=SERIES[key], label=LABEL[key],
                solid_capstyle="round")

    # Reference lines often sit within ~0.02 of each other, so right-aligning
    # every label stacks them into an unreadable pile. Stagger along x instead.
    refs = [(k, lab, arm[k]) for k, lab in REFS.items()
            if arm.get(k) is not None and arm.get(k) == arm.get(k)]
    span = layers[-1] - layers[0]
    for i, (key, lab, v) in enumerate(refs):
        ax.axhline(v, ls=(0, (4, 3)), lw=1.2, color=MUTED, zorder=1)
        xf = [0.30, 0.62, 0.97][i % 3]
        ax.annotate(f"{lab}  {v:.2f}",
                    xy=(layers[0] + span * xf, v), xytext=(0, 4),
                    textcoords="offset points",
                    ha="right" if xf > 0.9 else "center", va="bottom",
                    fontsize=7.5, color=INK_2,
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))

    ax.axhline(0.5, lw=1, color=GRID, zorder=0)
    ax.annotate("chance", xy=(0, 0.5), xytext=(2, -10), textcoords="offset points",
                fontsize=7.5, color=MUTED)

    # title + subtitle as one multi-line block: cannot collide with each other
    ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, color=INK, pad=8,
                 loc="left", linespacing=1.6)
    ax.set_xlabel("layer", fontsize=9, color=INK_2)
    ax.set_xlim(layers[0], layers[-1])
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def main():
    res = json.load(open(os.path.join(RES, "probe_results.json")))
    gate = json.load(open(os.path.join(RES, "variance_gate.json")))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                             facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    wp, pl = res["arms"]["within_prompt"], res["arms"]["pooled"]

    # y-range from the data so no curve is clipped
    vals = [v for arm in (wp, pl) for rec in arm["layers"].values()
            for v in rec.values() if v == v]
    vals += [arm[k] for arm in (wp, pl) for k in REFS if arm.get(k) == arm.get(k)]
    ylim = (min(0.45, min(vals) - 0.03), max(0.75, max(vals) + 0.05))

    panel(axes[0], wp, "Within-prompt contrast",
          f"prompt affordance cancels  ·  n={wp['n']}", ylim)
    panel(axes[1], pl, "Pooled across prompts",
          f"confounded by prompt affordance  ·  n={pl['n']}", ylim)
    axes[0].set_ylabel("AUROC (5-fold, grouped by prompt)", fontsize=9, color=INK_2)

    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
                     fontsize=8.5, bbox_to_anchor=(0.5, -0.10))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle("Is eventual actionability decodable from the first 10 response tokens?",
                 fontsize=12.5, color=INK, x=0.005, ha="left", y=1.12, fontweight="bold")
    fig.text(0.005, 1.055,
             f"Llama-3.1-8B-Instruct under Arditi ablation · StrongREJECT-313 · "
             f"k=5 samples/prompt @ temp=1.0 · judge=Gemma-StrongREJECT (deterministic) · "
             f"ICC(1)={gate['icc1']:.2f}",
             fontsize=8, color=INK_2, ha="left")

    out = os.path.join(RES, "auroc_by_layer.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")

    # ---- summary.md ----
    def best(arm, key):
        ls = sorted(int(k) for k in arm["layers"])
        vals = [(l, arm["layers"][str(l)][key]) for l in ls]
        vals = [(l, v) for l, v in vals if v == v]
        return max(vals, key=lambda t: t[1]) if vals else (None, float("nan"))

    lines = ["# Early-token decodability of actionability\n",
             f"Seed {res['seed']}. n={res['n']} sampled non-refusal responses over "
             f"{res['n_prompts']} prompts, {res['n_layers']} layers, d={res['d']}.\n",
             "## Variance gate\n", "```", json.dumps(gate, indent=2), "```\n",
             "## Peak AUROC by arm\n",
             "| arm | early (layer) | full (layer) | early_resid (layer) | length | lexical | random | permuted |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ("within_prompt", "pooled"):
        a = res["arms"][name]
        be, bf, br = best(a, "early"), best(a, "full"), best(a, "early_resid")
        lines.append(
            f"| {name} | {be[1]:.3f} (L{be[0]}) | {bf[1]:.3f} (L{bf[0]}) | "
            f"{br[1]:.3f} (L{br[0]}) | {a['length']:.3f} | {a['lexical']:.3f} | "
            f"{a['random']:.3f} | {a['permuted_at_best_layer']:.3f} |")
    lines += ["", "![AUROC by layer](auroc_by_layer.png)", ""]
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(lines))
    print(f"wrote {os.path.join(RES, 'summary.md')}")


if __name__ == "__main__":
    main()

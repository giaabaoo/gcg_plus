# 3-class probe: refusal axis vs harmfulness axis

n=7086 rows, d=4096, layers=33, seed=1234
class counts: {'benign_comply': 1274, 'benign_refusal': 322, 'contaminated': 3, 'harmful_comply': 1810, 'refuse': 3677}

## Headline (held-out test prompts)

PCA-128 probe is the headline; diff-in-means and no-PCA columns guard against
a genuine low-variance axis being truncated away by PCA.

| contrast | best layer | AUROC | diff-in-means | mid-layer AUROC | mid no-PCA | permuted null | random |
|---|---|---|---|---|---|---|---|
| harmfulness | 14 | **0.972** | 0.910 | 0.972 / 0.965 / 0.963 | 0.976 / 0.969 / 0.970 | 0.472 | 0.517 |
| refusal | 30 | **0.987** | 0.925 | 0.970 / 0.982 / 0.984 | 0.970 / 0.982 / 0.986 | 0.527 | 0.500 |
| naive | 30 | **0.980** | 0.929 | 0.954 / 0.972 / 0.977 | 0.955 / 0.972 / 0.978 | 0.467 | 0.495 |

## Lexical control - how much of this is just vocabulary?

Bag-of-ngrams (TF-IDF 1-2gram) on the PAP TEXT alone, same rows and splits.
Motivated by the probe already reaching ~0.91 at LAYER 1, which is the
signature of a surface-lexical difference rather than a representation.

| contrast | lexical | activation | margin |
|---|---|---|---|
| harmfulness | 0.924 | 0.972 | **+0.048** |
| refusal | 0.801 | 0.987 | **+0.186** |
| naive | 0.788 | 0.980 | **+0.191** |

### Capacity-matched dose-response (within harmful corpus only)

The 1-D projection (0.771) was an unfair readout against a 60k-feature text
model, so the activations get the same full probe here.

- lexical: **0.904**
- activation (full probe): **0.954**
- margin: **+0.050** (n_train=736, n_test=241)

## Geometry (anisotropy-controlled)

Whitening is done in the top-k principal subspace of the activations.
Full-covariance whitening was REJECTED as a numerical artifact: cond ~1.7e8
with 628 eigenvalues <1e-6*max drove every whitened cosine to ~0, including
cos(refusal-contrast, r), which cannot be zero.

**Read the subspace-energy column first.** Arditi's r holds only a minority of
its norm inside the estimable subspace, so whitened cosines INVOLVING r are not
reliable and r is compared raw only. The contrast directions sit at ~0.999.

| layer | pair | raw | whitened (k=512) |
|---|---|---|---|
| 14 | r vs refusal *(r: unreliable)* | +0.246 | +0.021 |
| 14 | harmfulness vs r *(r: unreliable)* | +0.265 | +0.087 |
| 14 | harmfulness vs refusal | +0.784 | +0.201 |
| 14 | harmfulness vs naive | -0.289 | +0.372 |
| 14 | naive vs r *(r: unreliable)* | -0.135 | +0.029 |
| 14 | naive vs refusal | -0.821 | -0.835 |
| 16 | r vs refusal *(r: unreliable)* | +0.148 | +0.019 |
| 16 | harmfulness vs r *(r: unreliable)* | +0.165 | +0.056 |
| 16 | harmfulness vs refusal | +0.646 | +0.154 |
| 16 | harmfulness vs naive | -0.266 | +0.401 |
| 16 | naive vs r *(r: unreliable)* | -0.096 | +0.012 |
| 16 | naive vs refusal | -0.907 | -0.844 |
| 18 | r vs refusal *(r: unreliable)* | +0.089 | +0.015 |
| 18 | harmfulness vs r *(r: unreliable)* | +0.120 | +0.043 |
| 18 | harmfulness vs refusal | +0.620 | +0.134 |
| 18 | harmfulness vs naive | -0.309 | +0.417 |
| 18 | naive vs r *(r: unreliable)* | -0.055 | +0.010 |
| 18 | naive vs refusal | -0.938 | -0.845 |

### Subspace energy (fraction of squared norm inside top-k)

| layer | r | refusal | harmfulness | naive |
|---|---|---|---|---|
| 14 | 0.422 | 1.000 | 1.000 | 1.000 |
| 16 | 0.299 | 1.000 | 1.000 | 1.000 |
| 18 | 0.237 | 1.000 | 1.000 | 1.000 |

## Within-harmful dose-response (corpus-confound control)

harmful_comply rows ONLY (same corpus, same wrapper, all complying):
does the projection onto the harmfulness axis track the judge score?

| layer | n | Spearman rho | p | top-vs-bottom-tercile AUROC |
|---|---|---|---|---|
| 14 | 351 | +0.394 | 1.9e-14 | 0.771 |
| 16 | 351 | +0.368 | 1.01e-12 | 0.751 |
| 18 | 351 | +0.380 | 1.58e-13 | 0.759 |

## Leave-one-harm-category-out (harmfulness, layer 16)

mean 0.957, min 0.930

| held-out category | n harmful | AUROC |
|---|---|---|
| Hate, harassment and discrimination | 331 | 0.930 |
| Disinformation and deception | 317 | 0.937 |
| Violence | 288 | 0.961 |
| Non-violent crimes | 435 | 0.964 |
| Illegal goods and services | 238 | 0.968 |
| Sexual content | 201 | 0.979 |

## Per-layer AUROC (table view)

| layer | harmfulness | refusal | naive |
|---|---|---|---|
| 0 | 0.500 | 0.500 | 0.500 |
| 1 | 0.911 | 0.751 | 0.736 |
| 2 | 0.920 | 0.780 | 0.762 |
| 3 | 0.921 | 0.785 | 0.776 |
| 4 | 0.931 | 0.815 | 0.797 |
| 5 | 0.947 | 0.819 | 0.818 |
| 6 | 0.944 | 0.835 | 0.824 |
| 7 | 0.951 | 0.855 | 0.831 |
| 8 | 0.958 | 0.876 | 0.845 |
| 9 | 0.961 | 0.894 | 0.862 |
| 10 | 0.957 | 0.905 | 0.874 |
| 11 | 0.956 | 0.941 | 0.917 |
| 12 | 0.960 | 0.948 | 0.922 |
| 13 | 0.960 | 0.959 | 0.939 |
| 14 | 0.972 | 0.970 | 0.954 |
| 15 | 0.965 | 0.975 | 0.960 |
| 16 | 0.965 | 0.982 | 0.972 |
| 17 | 0.965 | 0.984 | 0.974 |
| 18 | 0.963 | 0.984 | 0.977 |
| 19 | 0.964 | 0.984 | 0.975 |
| 20 | 0.962 | 0.983 | 0.975 |
| 21 | 0.966 | 0.984 | 0.976 |
| 22 | 0.963 | 0.983 | 0.974 |
| 23 | 0.966 | 0.984 | 0.976 |
| 24 | 0.965 | 0.984 | 0.976 |
| 25 | 0.964 | 0.984 | 0.977 |
| 26 | 0.964 | 0.983 | 0.977 |
| 27 | 0.964 | 0.985 | 0.977 |
| 28 | 0.966 | 0.985 | 0.978 |
| 29 | 0.965 | 0.986 | 0.978 |
| 30 | 0.966 | 0.987 | 0.980 |
| 31 | 0.967 | 0.987 | 0.979 |
| 32 | 0.965 | 0.986 | 0.978 |

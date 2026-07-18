# Step 3 - compliance directions

splits (base prompts): extract=187 val=62 test=64, seed=0

## Pools (extract split only)

| technique | A_t | B_t | gate |
|---|---|---|---|
| Authority endorsement | 178 | 358 | PASS |
| Evidence-based Persuasion | 166 | 414 | PASS |
| Expert Endorsement | 112 | 550 | PASS |
| Logical appeal | 249 | 395 | PASS |
| Misrepresentation | 340 | 230 | PASS |
| Threats | 38 | 278 | PASS |

## Falsification guard: cos(d_t, refusal direction)

Trivial-case check only (d_t ~= -r would mean Arditi re-derived).
Headline remains behavioral.

| technique | max abs cos over layers | layer | signed cos |
|---|---|---|---|
| Authority endorsement | 0.517 | 11 | -0.517 |
| Evidence-based Persuasion | 0.588 | 11 | -0.588 |
| Expert Endorsement | 0.665 | 11 | -0.665 |
| Logical appeal | 0.339 | 12 | -0.339 |
| Misrepresentation | 0.209 | 10 | +0.209 |
| Threats | 0.364 | 11 | -0.364 |

## Cross-technique cosine (matched n=38 per pool)

Layer of max |cos| vs refusal used as a reference layer per column below;
matrix reported at the mid-stack layer 14 (typical best-effect region).

| | Authority endorsement | Evidence-based Persuasion | Expert Endorsement | Logical appeal | Misrepresentation | Threats |
|---|---|---|---|---|---|---|
| **Authority endorsement** | +1.000 | +0.650 | +0.535 | +0.613 | +0.583 | +0.615 |
| **Evidence-based Persuasion** | +0.650 | +1.000 | +0.887 | +0.740 | +0.491 | +0.579 |
| **Expert Endorsement** | +0.535 | +0.887 | +1.000 | +0.699 | +0.391 | +0.550 |
| **Logical appeal** | +0.613 | +0.740 | +0.699 | +1.000 | +0.621 | +0.539 |
| **Misrepresentation** | +0.583 | +0.491 | +0.391 | +0.621 | +1.000 | +0.423 |
| **Threats** | +0.615 | +0.579 | +0.550 | +0.539 | +0.423 | +1.000 |

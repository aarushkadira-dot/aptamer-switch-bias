# Controls on the library-composition claim

These test whether the observed cytosine depletion could be a sequencing artifact
rather than a property of the library. It cannot.

## `read_layout.py`
Establishes construct geometry from per-cycle composition. Reads are 82 nt:
polyT at cycles 0–23, the degenerate N10 at ~25–34, then `TCTCAG`.

**Why this matters:** a 25-nt homopolymer causes indel slippage, so *fixed-cycle*
windows drift read-to-read. Any analysis using fixed cycle offsets is invalid.
Anchoring on the `TCTCAG` motif re-establishes register per read.

## `basecall_error_rate.py` — the decisive control
In `TCTCAG`-anchored (register-corrected) reads, the 12-nt window immediately 5' of
the N10 is known truth: all T.

| quantity | value |
|---|---|
| calls matching truth | **99.668%** |
| miscall rate | **0.332%** |
| miscall distribution | A 20.9% / C 45.4% / G 33.7% |

Inverting a 0.332% error moves cytosine from 16.21% to 16.18% — a 0.03 pp change.

Symmetric error pulls every base *toward* 25%, so it can only shrink a compositional
deficit, never create one. **Sequencing error cannot account for the observed depletion.**

## `quality_stratified_composition.py` — a control that is easy to misread
Filtering to high per-base quality appears to raise cytosine from 16.2% to ~19.6%.
This is **not** error correction. Quality filtering *selects* reads that sequenced
cleanly, and cleanliness correlates with composition, so the high-quality subset is
compositionally biased toward balanced sequences. Reported here because the naive
reading of it is wrong, and the error-rate measurement above is what settles the question.

## `dinucleotide_asymmetry.py` — negative result, reported
DNA synthesis runs 3'→5', so a coupling-chemistry effect can only depend on the base
already present (the 3' neighbour). Test for that asymmetry:

| conditioning | Cramér's V |
|---|---|
| 3' neighbour (already coupled) | 0.1098 |
| 5' neighbour (not yet coupled) | 0.1182 |
| **ratio** | **0.93** |

Symmetric. Nearest-neighbour structure in the library is therefore **not** explained by
synthesis coupling order. The depletion is real and its magnitude stands, but this
particular mechanism is excluded.

Edge effect, reported: N10 position 1 abuts the polyT and shows T elevated by +10.8 pp
relative to the interior. Interior-only composition is 16.24% C against 16.19% for the
full window, so this does not affect the headline.

---

## `anchor_stringency.py` — the honest version of the error-rate claim
The 0.332% figure above is measured on reads where `TCTCAG` matched **exactly**, which is
itself a selection for cleanly-sequenced reads. Relaxing the anchor:

| anchor | reads kept | measured error | C | T |
|---|---|---|---|---|
| exact | 47.8% | 0.308% | 16.39% | 29.77% |
| ≤1 mismatch | 76.5% | 4.521% | 15.87% | 31.95% |
| ≤2 mismatches | 97.6% | 7.791% | 13.43% | **39.40%** |

The rising T fraction is register corruption: a fuzzy match finds spurious anchors and drags
polyT into the window. So the relaxed numbers are not valid composition estimates. But they
establish the **direction**: every relaxation drives C *down*, never up. The only manipulation
that raised C was quality filtering, which is the most selection-biased of all.

**Calibrated position:** the deficit is real and large; the point estimate is ~16%; the true
value plausibly lies in 13–19%.

## `cross_order_replication.py` — the strongest evidence for a systematic vendor effect

The ATP and glucose N10 regions were **ordered separately**. Their abundance-weighted
compositions:

| | A | C | G | T |
|---|---|---|---|---|
| ATP library | 23.24% | 16.07% | 31.01% | 29.68% |
| glucose library | 23.27% | 16.27% | 30.86% | 29.61% |
| difference | +0.03 | +0.20 | −0.16 | −0.07 |

**Agreement to 0.20 pp on every base, correlation 1.0000.** Two independent orders landing on
the same non-equimolar composition is what a systematic vendor mix predicts and what a one-off
batch error does not. This is the cross-order validation of the correction ratio below —
obtained without placing a test order.

### Predicted input ratio for an equimolar N region
To obtain 25% of each base in the product:

**A : C : G : T = 1.43 : 1.93 : 1.00 : 1.06**  (A 26.4%, C 35.6%, G 18.4%, T 19.6%)

### Is the depletion uniform per-position?
If each position independently incorporates C with probability p, C-counts must be Binomial(10, p).

| library | fitted p | corr(observed, binomial) |
|---|---|---|
| ATP | 0.1607 | **0.99673** |
| glucose | 0.1627 | **0.99188** |

Predominantly binomial, with mild over-dispersion in the sparse tails (observed/expected rises to
~3.6× at k=8, where the mass is negligible). Consistent with uniform unequal amidite incorporation
as the dominant mechanism, plus a smaller second-order effect that is not identified here.

### Correction to the dinucleotide conclusion
An earlier reading of `dinucleotide_asymmetry.py` treated the symmetric 3'/5' result as excluding
a synthesis origin. That was wrong. The symmetry excludes **context-dependent** coupling — base X
coupling differently depending on its neighbour — which was never the hypothesis. The hypothesis is
**uniform** unequal amidite reactivity, which predicts position-independence (observed),
approximate binomiality (observed), and *no* dinucleotide asymmetry (observed). The symmetric
result is consistent with the synthesis explanation, not evidence against it.

---

## `glucose_profile_vs_random.py` — the control that rescued a bad claim

An earlier draft stated that the glucose screen had "no reproducible signal", on the basis of a
split-half reliability of 0.015. That reliability figure is correct but measures **per-sequence
ranking**. The source paper's Fig. 4 conclusion is an **aggregate over the top 1000 sequences**.

Testing the paper's actual claim, against 20 draws of 1000 randomly chosen library sequences:

| dataset | positions enriched vs random | top-1000 peak | random peak |
|---|---|---|---|
| glucose 10 mM | nt 1–5, **z = +5.2 to +7.5**; nt 9–10 at z ≈ −5.5 | nt 4 | nt 23 |
| glucose 100 mM | nt 1–4, **z = +6.6 to +8.3**; nt 8–12 negative | nt 4 | nt 23 |
| ATP (benchmark) | nt 15–19, z up to **+15.3** | nt 17 | nt 4 |

The paper reports bases 2–5 as overwhelmingly targeted and bases 9–12 as least amenable. Both
reproduce here independently, at two glucose concentrations. **The glucose screen carries real
aggregate signal and the paper's Fig. 4 conclusion stands.**

## `glucose_profile_splithalf.py`
Split-half profile correlation between independent halves of the glucose clusters: r = 0.90 (10 mM),
r = 0.94 (100 mM), against r = 0.98 for ATP. Top-1000 *sequence* overlap is 15–17/1000 for glucose
and 56/1000 for ATP — low in both, which is why sequence overlap is a poor discriminator and the
profile comparison above is the right test.

### The reportable finding: a reliability dissociation
Per-sequence reliability **0.015** alongside aggregate positional signal at **z ≈ 7–8**. A massively
parallel screen can support conclusions about *which region of an aptamer to target* while being
unable to support conclusions about *which specific sequence is best*. ATP calibrates the scale:
reliability 0.699, z = 15.3.

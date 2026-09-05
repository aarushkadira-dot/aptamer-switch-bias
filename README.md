# Aptamer switching domains: sequence-function modelling and a library bias that hides the best sequences

Reanalysis of the aptamer switching-domain screen published in
**Yoshikawa et al., *Nature Communications* 14:2336 (2023)**, using Supplementary Data 1
and the raw reads deposited under [SRA PRJNA952942](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA952942).

**Aarush Reddy Kadira** · Enloe Magnet High School, Raleigh NC · aarush.kadira@gmail.com

---

## The short version

Electrochemical aptamer-based sensors allow continuous, real-time measurement of drugs and
metabolites in the body. Making one requires converting an aptamer into a *switch*, and the
largest public screen of that conversion tested roughly a million candidate switching domains.

Two results:

1. **Switching is predictable from sequence.** R² = 0.525 on 60,000 held-out sequences,
   independently reproducing structural conclusions the original authors reached by other means.

2. **The library is depleted exactly where the good switches are.** The degenerate N10 region is
   not equimolar — cytosine sits near 16% rather than 25% — and cytosine content is simultaneously
   the strongest compositional predictor of switching. Coverage falls monotonically with cytosine
   count while mean switching rises monotonically with it. At matched sequencing depth, correcting
   the composition to equimolar implies **≈1.5× more high-performing switches** than the screen
   recovered.

   > **Corrected 2026-09-02.** This previously read "4–5× more high-performing switches." That
   > figure (4.77×, 79% unobserved) reproduces exactly from the deposited data, but it is **not
   > mainly a consequence of the library bias**. Holding sequencing depth fixed at the observed
   > 339,373 unique 10-mers and setting the composition to equimolar, the gap decomposes into
   > **1.54× from cytosine depletion** and **3.09× from finite sequencing depth**. Sequence-space
   > coverage is only 32.4% overall and 66% even at C = 0, where there is no depletion at all — so
   > most of the missing 79% would be missing from a perfectly equimolar library too. The bias
   > accounts for about 28% of the effect on a log scale. See `DEEP_AUDIT_2026-09-02.md` §D.

---

## Headline numbers

| Result | Value | Control |
|---|---|---|
| Held-out R², n = 60,000 | **0.525** | shuffled labels: −0.043 |
| Ensemble over GBM alone | +0.031 R² | CI [+0.023, +0.038], 100% of resamples |
| Remove all Hamming-1 neighbours | costs +0.024 R² | rules out local interpolation |
| Match to authors' Fig. 2a targeting profile | **r = 0.859**, p = 1.9e-08 | same peak (nt 17) |
| Top six 5-mers = revcomp of aptamer nt 13–22 | median rank 29/1024 | Mann-Whitney p = 6.9e-14 (see note) |
| Blind test, 5 constructs never seen | Pearson **0.968** (p = 0.007); Spearman 0.900 (p = 0.083, n.s.) | plate-reader data |
| Blind test, all 8 constructs | Spearman 0.833, exact p = 0.0154 | Bonferroni ×4 = 0.062, n.s. |
| Coverage falls with C (2.26 → 1.07 per unique seq) | monotone across C = 0…6 | measured |
| Mean switching rises with C (1.013 → 1.446) | monotone across C = 0…6 | measured |
| ~~Coverage vs quality, Spearman = −1.000~~ | **withdrawn as a statistic** — see note | tautological |
| P(signal > 2×) at C=0 → C=6 | 0.061% → 12.55% (**204×**) | measured, no model |
| Cytosine in raw reads, 2 SRA runs | 16.35% / 15.79% vs 25% | pre-processing |
| Implied unrecovered switches | **4–5×** (≈79% unobserved) | 4.1–4.8× across variants |
| Base-caller error, known-truth window | **0.332%** | cannot explain the depletion |
| ATP vs glucose library composition | agree to **0.20 pp** per base | separately ordered, r = 1.0000 |
| C-count distribution vs Binomial(10, p) | r = **0.997** | uniform per-position incorporation |

**Negative control that matters:** the bias does *not* distort the source paper's published motif
conclusions — reweighted m1/m2/m3 frequencies shift by under 2 percentage points.

### Statistical notes on the table above

Corrections found on a self-audit (2026-09-02), recorded rather than silently applied:

- **Coverage vs switch quality — the ρ = −1.000 is withdrawn as a statistic.** Two problems. First,
  the p was given as "< 0.0001", which is not attainable: with 7 strata the exact one-sided minimum
  for a perfect rank correlation is 1/7! = 1.98 × 10⁻⁴. Second and more seriously, **ρ = −1 is
  forced here, not discovered.** Coverage per unique sequence is strictly decreasing in cytosine
  count and mean switching ratio is strictly increasing in it; Spearman is a rank correlation, so
  two strictly monotone functions of the same index are perfectly anticorrelated by construction.
  It is also window-dependent (C = 0…7 gives −0.976; all ten strata give −0.985). The two
  monotonicities are the real content and are now reported directly. The joint correlation carried
  no independent information and its p-value should never have been presented as confirmation.
- **Blind test.** Every Spearman p here came from `scipy.stats.spearmanr`, whose t-approximation is
  anticonservative at this sample size. Exact permutation values: at n = 5, ρ = 0.900 gives
  p = **0.083 — not significant** (scipy: 0.037); at n = 8, ρ = 0.833 gives p = **0.0154**
  (scipy: 0.010), which becomes **0.062 after Bonferroni correction for the four frames tried and
  does not clear 0.05.** The surviving claim is the **Pearson correlation on the five never-seen
  constructs, r = 0.968, p = 0.007**, which is a genuine out-of-assay prediction and is unaffected.
  This is a real weakening of the blind test and is stated as such.
- **Top-six 5-mers.** The quoted p comes from a Mann-Whitney test of the selected ranks against
  `np.arange(1, 1025)` (`src/validation/R3_motif.py:60`), not from the "median rank 29/1024"
  statistic beside it. The exact null for a median rank ≤ 29 among 6 of 1024 gives p ≈ 3.9 × 10⁻⁴.
  Both reject decisively and the false-positive rate of the Mann-Whitney version is calibrated
  (4.8% at α = 0.05 under simulation), so the conclusion is unaffected — but the statistic and the
  p-value beside it are not the same test, and the comparison set should be the 1018 non-selected
  5-mers rather than the full vector.
- The **split-half r of 0.699** is the raw Pearson split-half (`src/model/impact2.py:37`), not a
  Spearman-Brown-corrected reliability. Corrected full-length reliability would be 0.823.

**Sequencing error is ruled out.** In register-corrected reads, the known-truth polyT window
immediately 5′ of the N10 is called correctly 99.668% of the time. Inverting a 0.332% error moves
cytosine by 0.03 pp. Symmetric error pulls composition *toward* 25%, so it can only shrink a
deficit, never create one. See `src/controls/`.

---

## A correction to the published dataset

The photobleaching normalization in the source data brackets each target measurement between the
buffer readings on either side. There are five target cycles and five buffer cycles, so the fifth
target has no buffer after it, and the pipeline substituted the terminal folding-marker channel —
which reads 15.9% higher — inflating that exposure's denominator by ~8%.

The original pipeline reproduces bit-for-bit (max ratio difference 4.4e-16), so the error is in the
definition, not the code.

| variant | clusters | split-half r |
|---|---|---|
| original | 491,589 | 0.660 |
| **adopted: drop exposure 5, median of 4** | **494,065** | **0.699** |

`data/atp_corrected.csv.gz` is the corrected dataset: 494,065 clusters, 339,373 unique 10-mers.

---

## Reproducing the model

`src/model/R1_core.py`, `cnn3.py` and `R7_ens.py` read `data/atp_corrected.csv.gz` by default;
override with the `ATP_DATA` environment variable. Independently re-run from scratch on
2026-09-04:

| | reproduced | reported |
|---|---|---|
| gradient boosting, held-out R² | **0.4940** | — |
| CNN, held-out R² | 0.4878 | — |
| **ensemble, α = 0.5** | **0.5236** | **0.525** |
| ensemble over GBM | **+0.0295** | +0.031 |
| 95% CI on the gain | **[+0.0229, +0.0365]** | [+0.023, +0.038] |
| resamples won | **100.0%** | 100% |
| shuffled-label control | **−0.0425** | −0.043 |

Split integrity was checked independently: 494,065 clusters and 339,373 unique 10-mers, three
separately computed train/validation/held-out splits identical, saved predictions verified against
saved split indices, and no overlap between any pair of splits. The blend weight α = 0.5 is
selected on the validation set, never on held-out.

Two things matter for reproduction. The features are **55 categorical** columns — ten position
identities plus all forty-five pairwise position combinations encoded as `A[i]*4 + A[j]`, with
`max_bins=16` — and substituting numeric one-hot or k-mer counts costs roughly 0.085 R². The CNN
needs training to convergence: at 40 epochs it reaches validation R² 0.416 and the ensemble gain is
only +0.003; by epoch 88 it reaches 0.486 and the full +0.03 gain appears.

Verification scripts are in `src/model/verify_*.py`.

## Repository layout

| Path | What it does |
|---|---|
| `src/model/impact2.py`, `verify_raw.py` | normalization error, correction, verification |
| `src/model/R1_core.py`, `R7_ens.py`, `cnn3.py` | GBM, ensemble, multi-scale CNN |
| `src/model/R2_blocked.py` | Hamming-distance-blocked splits |
| `src/validation/R3_motif.py`, `window.py`, `R30_fig2a.py` | structural validation, Fig. 2a comparison |
| `src/validation/R11_blindtest.py`, `R44_blind.py` | blind test on lab-synthesised constructs |
| `src/library_bias/R29_impact.py` | coverage–performance inversion, reweighting |
| `src/library_bias/AUDIT_RAW.py`, `AUDIT_RAW2.py` | composition directly from SRA reads |
| `src/library_bias/R31_glu.py` | replication in the glucose library |
| `src/mechanism/R28_coop.py`, `R21_pwm.py` | interaction-order decomposition |
| `src/mechanism/R27_nuc.py` | 21-parameter nucleation model |
| `src/mechanism/R45_frag.py` | single-substitution fragility, 4.18M pairs |
| `src/mechanism/R40_cross.py` | cross-system transfer to RNA toehold switches |
| `src/controls/` | **controls: base-caller error rate, anchor stringency, quality stratification, dinucleotide asymmetry, cross-order replication, glucose profile vs random** |
| `src/withdrawn/` | **tests that killed my own claims — see below** |

These were written exploratorily and expect data at absolute paths; adjust the paths at the top of
each file before running. They are provided so results can be checked, not as a packaged pipeline.

---

## Claims I withdrew

Thirteen claims survived initial analysis, were written up, and were then killed by a control. They are
listed here because the reliability of what remains depends on how hard it was tested.

| Withdrawn claim | What killed it | Code |
|---|---|---|
| "21× better than published motifs" | Straw man — MEME motifs are a discovery tool, not a predictor. Replaced with a like-for-like comparison at r = 0.859. | — |
| Model performs structure probing | Position confound (r = −0.832 with distance from nt 17). After control p = 0.515, direction reversed. | `R42_struct.py` |
| Kinetics / fatigue effect | Confined to the top decile; excluding the top 10%, correlation = −0.001. | `R38_kinetics.py`, `R39_kin2.py` |
| ON/OFF asymmetry of 400× | Artefact of a linear fit. True value 12×. | `R32_onoff.py` |
| Three-stage decomposition of the bias | Four errors on audit. Synthesis and amplification are not separable with this data. | — |
| "Noisy-many beats clean-few" | Reversed under composition matching (0.113 vs 0.177). | — |
| "95% of predictable variance" | Invalid constant-reliability assumption. Replaced with 64% on the measurable subset. | — |
| CNN beats GBM on top-100 (17 vs 12) | Noise; all CIs span [7, 20]. | — |
| Cross-aptamer purine targeting rule | Failed a shuffled-aptamer permutation null; combined p = 0.12. | `XAPT.py`–`XAPT3.py` |
| C/G effects agree across two libraries | Base counts are compositionally constrained; refitting with a reference base collapses the toehold effect to t = 1.8. | `TH3.py`, `TH4.py`, `BOTH.py`, `STRESS.py` |
| Depletion arises from *context-dependent* synthesis coupling | Synthesis runs 3′→5′, so context-dependent coupling can depend only on the 3′ neighbour. Observed dependence is symmetric (V = 0.110 vs 0.118, ratio 0.93). **Note: this excludes context-dependence only. Uniform unequal amidite reactivity remains supported** — see `controls/cross_order_replication.py`. | `controls/dinucleotide_asymmetry.py` |
| A second, structural library bias (P7 primer complementarity suppresses cluster formation) | Survived exact-composition matching (32/46 classes, Wilcoxon p = 0.00013) but failed a sham control: 20 random 24-mers give rho = −0.019 ± 0.055 against P7's −0.030, z = −0.21. Not P7-specific. Abundance among observed sequences also unaffected (rho = +0.003, p = 0.50). | `withdrawn/p7_structural_bias.py` |
| *(briefly believed, then withdrawn within the same session)* Depletion is a sequencing artifact | Raised on a quality-stratification result that appeared to move C from 16% to ~19.6%. Withdrawn: quality filtering **selects** clean reads rather than **correcting** errors, and cleanliness correlates with composition. Direct measurement against known-truth bases gives a 0.332% error rate, far too small — and symmetric error can only shrink a deficit, never create one. | `controls/basecall_error_rate.py` |

---

## Prior work

Composition bias in degenerate nucleic-acid libraries is **already published** and is not claimed here:

- **Takahashi et al.**, *Sci. Rep.* **6**:33697 (2016) — guanine-rich bias in unselected libraries.
- **Thiel et al.**, *Nucleic Acid Ther.* **21**:253–263 (2011) — pyrimidine bias, adenine depletion.

Reported directions are inconsistent across studies, which is itself informative: the effect is
vendor- and chemistry-dependent rather than universal.

What is new here is the *link to function*. Takahashi et al. concluded that library bias
"is not so limiting as to result in the failure" of selection and "may be overcome by the
selection stringency." This work is a quantified counter-example to that assessment.

---

## Open

- Whether the bias originates in synthesis, amplification, or cluster calling — not separable with
  this data. The vendor's incorporation specifications would settle it.
- Why the map is cooperative here (23% additive) but substantially additive in RNA toehold
  switches (56%). The cooperativity claim is scoped to unconstrained libraries against a fixed target.
- Transfer to a different aptamer or a longer switching domain — untested.
- Two model-derived design rules (mismatch tolerance, position preference) are predictions, not
  measurements, and need a small bench test.

## The experiment this points to

Synthesise ~20 high-cytosine switching domains predicted to function, and read a
fluorophore–quencher displacement assay across an ATP titration. The prediction is falsifiable:
hit rates should substantially exceed the 0.061% baseline measured at zero cytosines.

---

## Data sources

| Source | Use |
|---|---|
| Yoshikawa et al. 2023, Supplementary Data 1 | per-cluster fluorescence, ATP and glucose screens |
| [SRA PRJNA952942](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA952942) | raw reads, composition before processing |
| [GSE149225](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149225) | RNA toehold switches (Angenent-Mari et al.), cross-system test |

## License

MIT. If you use the corrected dataset, please also cite the original authors.

---

## Two things deliberately NOT claimed against the source paper

Both were investigated and both fail as literature corrections. Recorded here so neither is
mistakenly revived.

**The FM bracketing error is a reuse caveat, not a correction.** The paper describes no photobleaching
correction, never defines FM beyond listing reagents, and never describes bracketing. Its stated
method is only *"the switching domain clusters were sorted by the ratio of the signal in ATP compared
to buffer."* The error is in the undocumented processing of the deposited data. The paper further
pre-empts precision critiques: *"the instrument is not necessarily designed for quantitative
fluorescent intensity measurements."*

**The glucose screen does not contradict the paper.** Per-sequence reliability is 0.015, but the
paper's Fig. 4 conclusion is an aggregate, and it reproduces independently at z ≈ 7–8 against a
random-sequence control at two glucose concentrations. Their glucose hits were separately validated
on a plate reader with scrambled and natural-DNA controls. See `src/controls/`.

What survives from the second is a **reliability dissociation** — per-sequence 0.015 versus aggregate
z ≈ 7–8 — which is a genuine and reportable result about what these screens can and cannot support.

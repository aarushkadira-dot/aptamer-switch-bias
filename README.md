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
   the strongest compositional predictor of switching. Coverage and function are **perfectly
   inversely ranked**, Spearman = **−1.000**. Correcting for this implies roughly **4–5× more
   high-performing switches exist than the screen recovered.**

---

## Headline numbers

| Result | Value | Control |
|---|---|---|
| Held-out R², n = 60,000 | **0.525** | shuffled labels: −0.043 |
| Ensemble over GBM alone | +0.031 R² | CI [+0.023, +0.038], 100% of resamples |
| Remove all Hamming-1 neighbours | costs +0.024 R² | rules out local interpolation |
| Match to authors' Fig. 2a targeting profile | **r = 0.859**, p = 1.9e-08 | same peak (nt 17) |
| Top six 5-mers = revcomp of aptamer nt 13–22 | median rank 29/1024 | p = 6.9e-14 |
| Blind test, 5 constructs never seen | Spearman **0.900**, Pearson 0.968 | plate-reader data |
| Coverage vs switch quality, 7 strata | **Spearman = −1.000** | p < 0.0001 |
| P(signal > 2×) at C=0 → C=6 | 0.061% → 12.55% (**204×**) | measured, no model |
| Cytosine in raw reads, 2 SRA runs | 16.35% / 15.79% vs 25% | pre-processing |
| Implied unrecovered switches | **4–5×** (≈79% unobserved) | 4.1–4.8× across variants |

**Negative control that matters:** the bias does *not* distort the source paper's published motif
conclusions — reweighted m1/m2/m3 frequencies shift by under 2 percentage points.

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
| `src/withdrawn/` | **tests that killed my own claims — see below** |

These were written exploratorily and expect data at absolute paths; adjust the paths at the top of
each file before running. They are provided so results can be checked, not as a packaged pipeline.

---

## Claims I withdrew

Ten claims survived initial analysis, were written up, and were then killed by a control. They are
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

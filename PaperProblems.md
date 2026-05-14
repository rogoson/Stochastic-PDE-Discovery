## Reproducibility Audit: Findings
 - note, these are my findings, I just dumped them into AI to clean them up.

### Per-Example Results

The VB method matches up with the data in all 3 occasions concerning drift, but it isn't the same in the case of Nagumo.

**Stochastic Allen-Cahn**

*Drift:*
- (LASSO) L2: 1.275 — similar enough to paper (1.7768), acceptable given LASSO instability. FPR: 69.04% vs paper's 78.57% — same instability explanation applies.
- (e-SINDy) L2: 0.0573 vs paper's 0.0652 — close, broadly consistent. FPR: 0.0% — matches paper. Discovered coefficients: `u` (1.040), `u³` (-1.041), `u_xx` (0.992).
- (VB) L2: 0.0561 — matches paper exactly. FPR: 0.0% — matches paper.

*Diffusion:* All three methods identify the constant term correctly (LASSO: 0.982, SINDy: 0.981, VB: 0.991). No meaningful discrepancy.

---

**Stochastic Nagumo**

*Drift:*
- (LASSO) L2: 0.194 vs paper's 0.2518 — similar scale, acceptable.  FPR: 88.57% — matches paper exactly.
- (e-SINDy) L2: **0.0298 vs paper's 1.5997** — factor of ~53× discrepancy. FPR: **0.0% vs paper's 11.43%** — qualitatively different outcome. Discovered coefficients: `u` (0.522), `u²` (0.499), `u³` (-0.998), `u_xx` (1.021) — all correct, no false positives.
- (VB) L2: 0.0328 vs paper's 0.0616 — better than paper, and now comparable to e-SINDy rather than clearly superior. FPR: 0.0% — matches paper.

*Diffusion:* All three methods identify the constant term correctly (LASSO: 1.03, SINDy: 1.040, VB: 0.991). No meaningful discrepancy.

---

**Stochastic Heat**

*Drift:*
- (LASSO) L2: 0.5225 vs paper's 0.5649 — similar enough, acceptable. FPR: 69.04% vs paper's 76.19% — same LASSO instability explanation applies.
- (e-SINDy) L2: **0.001623 vs paper's 1.1586** — factor of ~700× discrepancy. FPR: **0.0% vs paper's 7.14%** — qualitatively different outcome. Discovered coefficients: only `u_xx` (0.998) — correct, no false positives.
- (VB) L2: 0.001649 vs paper's 0.0016 — matches paper. FPR: 0.0% — matches paper.

*Diffusion:* All three methods identify the constant term correctly (LASSO: 0.981, SINDy: 0.985, VB: 0.992). No meaningful discrepancy.

*Prediction:* Just to say - the outputs I get are the same when I use the data they uploaded to Google drive. I guess there might be some system dependent variation, but a lot of this cannot be explained by that. More is said on the prediction later on.


---

### Observations worth stating explicitly

**1. e-SINDy is systematically underperforming in the paper.**
The discrepancy is not random — it follows a clear pattern. Allen-Cahn (where the paper shows e-SINDy performing well) roughly reproduces. Nagumo and Heat (where the paper shows e-SINDy performing poorly) do not reproduce, with errors off by factors of 53× and 700× respectively. This is the paper's central comparison and the primary basis for claiming VB is superior.

**2. VB results are broadly reproducible, with Nagumo slightly better than reported.**
The consistency of VB results confirms the experimental setup is correct. The e-SINDy discrepancy therefore cannot be attributed to a misconfiguration on the my part.

**3. The FPR formula deviates from the standard definition.**
The denominator used deviates from the standard definition of FPR, using total basis functions rather than true negatives, which slightly understates the rate. The paper does not flag this deviation from standard usage (Eq. 40).

**4. VB initialisation from SINDy output raises a methodological question.**
VB is initialised using the e-SINDy result rather than uninformative priors. This is disclosed in the paper, but it means VB is not operating as a fully independent method — it encodes the SINDy solution as prior knowledge. Given that e-SINDy already achieves comparable accuracy in reproducible runs, the incremental value of VB is harder to assess than the paper suggests. This is not necessarily a flaw, but the framing of VB as a distinct and superior method warrants scrutiny.

**5. Hardcoded discovered equation in Nagumo figure.**
The summary figure labels the discovered equation as `0.96u_xx + 0.46u + 0.50u² - 0.99u³` as a hardcoded string, whereas the actual VB output produces `1.022u_xx + 0.524u + 0.500u² - 0.998u³`. The figure does not reflect the code's actual output.

**6. Diffusion identification is consistent across all methods and examples.**
Since all three examples use additive noise with a constant diffusion coefficient — the simplest possible case — this is not a meaningful differentiator between methods. All methods recover it reliably.


**7. Prediction seeds.**
The manner in which they seed the prediction data ends up making the monte carlo simulation pointless.

**8. Heat Ensembles.** 
Heat was run with **1000 ensembles** rather than the 2000 stated in the paper. This appears to be an indexing issue in the notebook. Since e-SINDy already achieves near-perfect results at 1000 ensembles, correcting to 2000 would likely make the discrepancy with the paper's reported e-SINDy performance even larger. (LLN)

**9. ELBO monotonicity violation in VB.**
The "OOPS! log(like) decreasing!!" message appears in the Nagumo VB run, and Allen Cahn seems to have some zero stability errors. Under CAVI, the ELBO should be monotonically non-decreasing by construction — a decrease indicates a numerical bug in the implementation. The algorithm recovers in these cases, but this violates a theoretical guarantee the paper implicitly relies on.

****


### Conclusions:
**1.** I don't think this really changes my research direction, although I might have to include normal STLS sindy as a comparative performance method, and maybe add a two dimensional example since this seems too easy for SINDy regarding library size? I think the original authors did have a 2D nagumo example but decided not to produce it.
**2.** I should probably email the people who produced this paper for clarity?
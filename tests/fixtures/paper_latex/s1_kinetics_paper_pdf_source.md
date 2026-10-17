---
title: "Recovering Activation Energy and Pre-exponential Factor from Noisy First-Order Kinetics: Bias, Spread and Interval Coverage of a Two-Stage Arrhenius Estimator"
abstract: |
  We ask how accurately a rate constant and an activation energy can be recovered from noisy concentration data. First-order decay c(t)=exp(\ensuremath{-}kt) was simulated at five temperatures (285--320 K) chosen so that k spans exactly a factor of 10, with true Ea = 50 kJ/mol and ln A = 16. Sampling ran from 0 to 3/k, with additive Gaussian noise of SD 0.005--0.1 (in units of c0) and 5--40 points per temperature. Each temperature was fitted by nonlinear least squares. ln k was then regressed on centred 1/T with weights 1/var(ln k). At the reference setting (\ensuremath{\sigma} = 0.02, n = 12, 400 runs) the mean Ea estimate was 50 012 J/mol (Monte Carlo SE 31) with SD 622 J/mol, about 1.2% of Ea, so bias was negligible against the spread. The SD followed the linearised standard error to within 6% for n \ensuremath{\geq} 12 and exceeded it by 14--16% at n = 5--8. Ea and ln A estimates were correlated at r = 0.9992, equal to theory. The 95% t-interval covered the truth in 91.5--95.8% of runs, mostly slightly below 0.95. Zero fits failed in the reported cells. The high-noise, few-point corner (\ensuremath{\sigma} = 0.1, n = 5) was not reported, so the expected breakdown there is untested.
keywords: ["Arrhenius equation", "activation energy", "nonlinear least squares", "Monte Carlo coverage", "parameter correlation"]
---

## 1. Introduction

Activation energies are usually extracted by a two-stage procedure. A rate constant is fitted at each temperature, then ln k is regressed on 1/T, whose slope is \ensuremath{-}Ea/R and whose intercept is ln A. Weighted least-squares regression is described as a way to handle heteroscedastic errors [W1]. Fisher-information confidence regions can misrepresent nonlinear problems, and Monte Carlo simulation is a standard check on them [1].

We ask three questions for a well-posed first-order case. How large are the bias and spread of Ea and ln A as noise and sampling density vary? Does the analytic linearised error propagation predict the spread? How often does the reported 95% interval contain the truth?

## 2. Methods

**Truth and design.** First-order decay c(t) = c0 exp(\ensuremath{-}kt), c0 = 1 known, with k = A exp(\ensuremath{-}Ea/RT), ln A = 16, Ea = 50 000 J/mol, R = 8.314462618 J/(mol K). Five values of 1/T are equally spaced from 1/320 K\ensuremath{^{-}}\ensuremath{^{1}} upward by ln(10)\ensuremath{\cdot}R/Ea, giving T = 320.0, 310.5, 301.5, 293.1 and 285.1 K and exactly a 10-fold range of k. At each temperature t = linspace(0, 3/k, n), using the true k to set the window, and c = exp(\ensuremath{-}kt) + N(0, \ensuremath{\sigma}\ensuremath{^{2}}).

**Stage one.** k is fitted with `scipy.optimize.curve_fit` [2], a nonlinear least-squares routine, started from 1/mean(t) with c0 fixed. The variance of ln k is var(k)/k\ensuremath{^{2}}. A run is excluded if any fit raises, fails to converge, or returns a non-positive or non-finite k or variance.

**Stage two.** Weighted least squares of ln k on x = 1/T \ensuremath{-} mean(1/T) with weights 1/var(ln k). Then Ea = \ensuremath{-}R\ensuremath{\cdot}slope and ln A = intercept \ensuremath{-} slope\ensuremath{\cdot}x̄. Variances come from the 2\ensuremath{\times}2 weighted covariance, including the slope--intercept covariance. The primary 95% interval scales the covariance by the residual reduced \ensuremath{\chi}\ensuremath{^{2}} and uses t quantiles with 3 degrees of freedom.

**Theory.** Linearised var(k) = \ensuremath{\sigma}\ensuremath{^{2}}/\ensuremath{\Sigma} tᵢ\ensuremath{^{2}} exp(\ensuremath{-}2ktᵢ) is propagated through the same weighted regression to a theoretical SE(Ea), SE(ln A) and correlation. The observed SD is compared with this SE as a ratio.

**Metrics.** Over successful runs: bias = mean(estimate) \ensuremath{-} truth; spread = sample SD (ddof = 1); coverage = fraction of intervals containing the truth, with a Wilson 95% interval; correlation = Pearson r. These are the usual definitions and the study's own, not taken from a source.

**Runs.** 400 runs per setting, each with its own random stream. The results below are single-seed, with no replication across seeds. Two one-factor slices are reported: \ensuremath{\sigma} varied at n = 12, and n varied at \ensuremath{\sigma} = 0.02.

**Code checks.** With \ensuremath{\sigma} = 0 the pipeline returned Ea = 50 000 J/mol and ln A = 16, and the sampling window satisfied max(t)\ensuremath{\cdot}k = 3 exactly. These checks test the code only, not the statistics. The statistical checks at \ensuremath{\sigma} = 0.005, n = 40 were SD/SE = 1.029, coverage 0.942, and mean reported SE/SD = 0.961, all within tolerance.

<!-- fi:attempts -->
**How this result was reached.** The design was not revised after the experiment had first been run. The complete experiment was run 2 times (a run includes all its seeds); 1 of these runs was discarded (1 was replaced by a later run).
<!-- /fi:attempts -->

## 3. Results

### 3.1 Bias and spread

**Table 1.** Ea estimates, 400 runs per row, no failed fits. Bias has no Monte Carlo uncertainty except at the reference cell (\ensuremath{\sigma} = 0.02, n = 12), where SE = 31 J/mol; SD there has SE 22 J/mol.

| Setting | Ea bias (J/mol) | Ea SD (J/mol) | SD/SE theory | Coverage Ea |
|-------------------|-------------------------|---------------------|--------------------|----------------|
| \ensuremath{\sigma}=0.005, n=12 | 12.7 | 148 | \ensuremath{\approx}1.00 | 0.945 |
| \ensuremath{\sigma}=0.01, n=12 | \ensuremath{-}6.1 | -- | -- | 0.9375 |
| \ensuremath{\sigma}=0.02, n=12 | 12 | 622 | 1.06 | 0.925 |
| \ensuremath{\sigma}=0.05, n=12 | 49.1 | -- | -- | 0.95 |
| \ensuremath{\sigma}=0.1, n=12 | 146.2 | 2959 | \ensuremath{\leq}1.06 | 0.93 |
| \ensuremath{\sigma}=0.02, n=5 | \ensuremath{-}75 | 1134 | 1.16 | 0.9275 |
| \ensuremath{\sigma}=0.02, n=8 | -- | 842 | 1.14 | 0.915 |
| \ensuremath{\sigma}=0.02, n=20 | -- | 468 | 1.04 | 0.9575 |
| \ensuremath{\sigma}=0.02, n=40 | -- | 323 | 1.03 | 0.9375 |

Dashes mean the value was not reported for that cell. The SD/SE ratio lies between 1.00 and 1.06 across the \ensuremath{\sigma} values at n = 12.

![Mean \ensuremath{\pm} SD of the Ea and ln A estimates against noise level for each number of time points, with the true values dashed, and the ratio of observed SD to linearised SE (1 = agreement), from 400 simulated runs per setting.](figures/fig1_bias_spread.png)

Figure 1 shows the mean estimates sitting on the true values at this scale. The visible SD band widens with noise only for the smallest n. The SD/SE panel stays between about 0.96 and 1.18, highest at few points.

At the reference cell the Ea bias (12 J/mol) is about 2% of the SD. The ln A mean is 16.004 (SE 0.012). The SD scales almost exactly linearly with \ensuremath{\sigma}: a 20-fold increase in noise gives a 19.9-fold increase in SD (148 to 2959 J/mol). From n = 5 to n = 40 the SD falls 3.5-fold, against 2.8-fold expected for 1/\ensuremath{\sqrt{}}n, which is slightly steeper. The bias grows with noise to 146 J/mol (5% of the SD) at \ensuremath{\sigma} = 0.1, but without a bias uncertainty at that cell we cannot call it distinguishable from zero. Stage-one ln k̂ bias at the reference cell is within \ensuremath{\pm}0.002 at all five temperatures (SE about 0.001).

### 3.2 Agreement with linearised theory

The observed SD matches the linearised SE within about 6% for n \ensuremath{\geq} 12 (Figure 1, right panel). Theory understates the spread by 16% at n = 5 and 14% at n = 8, so the linearisation fails modestly where data are sparse. This is consistent with the stage-two weights being estimated from stage-one fits, which the linearised theory ignores. The reported-SE/SD ratio of 0.961 at low noise implies the nominal SE is about 4% too small.

### 3.3 Coverage

![Coverage of the 95% t-interval for Ea and ln A against noise level for each number of time points, with the 0.95 line dashed and 95% Wilson bars, 400 runs per setting.](figures/fig2_coverage.png)

Figure 2 shows coverage between 0.915 and 0.975 with no monotone fall as noise grows. Of the ten coverage values in the two slices, eight are below 0.95, one equals 0.95 and one is above it. At the reference cell coverage is 0.925 (370/400; Wilson 0.895--0.947). Wilson half-widths are 0.02--0.027, so differences of a few points between settings are not resolved. The ln A coverage values track those of Ea closely, but no per-setting counts or contrasts for ln A were available, so only point values are reported.

### 3.4 Correlation of Ea and ln A

![Scatter of the Ea (as ln A) estimates over 400 runs at \ensuremath{\sigma} = 0.02, n = 12, with the true point and the theoretical 95% covariance ellipse; observed r = 0.9992, theory r = 0.9992.](figures/fig3_Ea_lnA_scatter.png)

Figure 3 shows the estimates lying along a narrow ellipse through the truth. The observed r = 0.9992 (SE 7.9\ensuremath{\times}10\ensuremath{^{-}}\ensuremath{^{5}}) equals theory, because the intercept is extrapolated from the centre of 1/T (about 3.3\ensuremath{\times}10\ensuremath{^{-}}\ensuremath{^{3}} K\ensuremath{^{-}}\ensuremath{^{1}}) to 1/T = 0. Marginal intervals therefore say little about joint uncertainty, and rectangular joint coverage is not 0.95.

## 4. Discussion

The estimator is effectively unbiased and its spread is predictable. The relative SD of Ea is about 1.2% at \ensuremath{\sigma} = 0.02, n = 12, and under 6% even at \ensuremath{\sigma} = 0.1, n = 12. The strong Ea--ln A correlation is of the kind described for narrow temperature ranges in Arrhenius fitting, and it is why Ea and ln A should be reported together with their covariance, not as independent intervals.

Our hypothesis of marked degradation at high noise and few points was not borne out in the slices reported. SD/SE reaches only 1.16, and coverage at \ensuremath{\sigma} = 0.1 (0.93) and n = 5 (0.9275) is indistinguishable from the other cells. The corner \ensuremath{\sigma} = 0.1, n = 5 is not reported, so the hypothesis is untested exactly where it is most likely to fail. At the reference cell the Wilson upper bound of 0.947 lies just below 0.95, which suggests slight undercoverage there. The reported-SE/SD ratio of 0.96 points the same way, but this explanation was not tested.

The simulated noise is Gaussian in concentration, so ln k̂ is nearly unbiased here. Reports that fitting the linearised Arrhenius form tends to underestimate Ea [W8] concern a different error model, in which noise is placed on k itself. Our result does not contradict that claim, but it does not test it. We observe a modest SD/SE excess at n \ensuremath{\leq} 8. We did not compare against the literature numerically, since none of the retrieved sources reports comparable quantities.

### Limitations

- All results are single-seed, with no replication across seeds. Monte Carlo standard errors for bias, SD and correlation exist only at the reference cell.
- Only two one-factor slices are reported, so interactions and the high-noise, few-point corner cannot be assessed.
- One parameter set (Ea = 50 kJ/mol, ln A = 16), one temperature range (285--320 K), known c0, and a sampling window set from the true k. An analyst who does not know k would use a different window, and results may not transfer to a free c0 or other Ea.
- Statistics are conditional on a successful fit. No fits failed in the reported cells, so the failure-as-miss bound is trivial. The t versus normal interval comparison produced no reported result.
- With 3 regression degrees of freedom, interval behaviour depends strongly on the t quantile.
- The code was checked only by its author's own noise-free and low-noise oracles. A second independently written version of the fit, regression and interval steps could not be compared with the first, so numbers resting on them are not independently confirmed.
- Bias, spread, coverage and correlation as used are this study's own definitions, and no source is cited for them.
- Some cited sources were available only as titles or short summaries (,--), so they are not used for specific findings. The cited web pages were read only in the excerpts retrieved.

<!-- fi:run-record:739887071a87 -->
**What FI recorded about this run.** Each trial's random seed is derived from base seed 0, the setting and the trial number, so different settings use different seeds. The experiment ran for 2.3 s; the topic sets a limit of 60 s, and the run was within it. The run had 125 settings with 400 trials per setting; 50,000 trials finished and 0 failed. A reviewer objected to the design in ways FI's records neither confirm nor rule out, and the study was not changed for them: "The paper presents Holm-adjusted p-values and a 'Wilson upper bounds below 0.95' claim that no result supports (and the counts contradict), together with ungrounded web-source claims".
<!-- /fi:run-record -->

<!-- fi:code-findings:52a65fadf938 -->
**Open points in the code that produced these results.** FI compared two independently written versions of key functions of the model. The points below were still open on the version that ran, and none of them was settled by the run; the full record is in `.fi/second_impl.json`.

- **The code was not compared with the plan by another model** (the reader is the model that wrote the code (claude-sonnet-5-5)).

- **A second, independent version of a key function was not settled.** FI had these functions written a second time from the plan's description alone and compared the two on many inputs: A second, independently written version of step S1 (NLS fit of k) could not be compared with the first (the plan gives no example inputs to call it with); A second, independently written version of step S2 (Weighted Arrhenius regression) could not be compared with the first (the plan gives no example inputs to call it with); A second, independently written version of step S3 (95% interval) could not be compared with the first (the plan gives no example inputs to call it with).
<!-- /fi:code-findings -->

## 5. Conclusion

For a 10-fold k range, five temperatures and sampling to 3/k, the two-stage weighted Arrhenius estimator recovers Ea and ln A with bias at most about 5% of the spread. The linearised standard error predicts the spread to within 6% for n \ensuremath{\geq} 12, and the t-interval covers at about 0.92--0.96. Ea and ln A are almost perfectly correlated (r = 0.9992), so they should be reported jointly. Whether coverage collapses at \ensuremath{\sigma} = 0.1 with five points remains to be tested.

```{=latex}
\FIfloatbarrier
```

## References

```{=latex}
\begingroup\small
```

1. Niels Krausch et al. (2019). Monte Carlo Simulations for the Analysis of Non-linear Parameter Confidence Intervals in Optimal Experimental Design. Frontiers in Bioengineering and Biotechnology. DOI: 10.3389/fbioe.2019.00122
2. Pauli Virtanen et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods. DOI: 10.1038/s41592-019-0686-2

```{=latex}
\endgroup
```

## Further reading

```{=latex}
\begingroup\small
```

- [W1] (2026). Measuring Activation Energy with Arrhenius Equation Adjustments. https://eureka.patsnap.com/report-research-on-measuring-activation-energy-with-arrhenius-equation-adjustments
- [W2] (2001). Two Types of Uncertainty in the Values of Activation Energy | Request PDF. https://www.researchgate.net/publication/226886307_Two_Types_of_Uncertainty_in_the_Values_of_Activation_Energy
- [W3] Uncertainty in the Arrhenius parameters resulting from selecting... | Download Scientific Diagram. https://www.researchgate.net/figure/Uncertainty-in-the-Arrhenius-parameters-resulting-from-selecting-various-values-of-m-to_fig3_370628353
- [W4] (2005). (PDF) Estimating the parameters of the arrhenius equation. https://www.academia.edu/558624/Estimating_the_parameters_of_the_arrhenius_equation
- [W5] A First Course on Kinetics and Reaction Engineering Example 4.6 Problem Purpose. https://wwwresearch.sens.buffalo.edu/karetext/unit_04/learning/04_Example_6.pdf
- [W6] (2025). Bayesian Methods for the Investigation of Temperature-Dependence in Conductivity. https://arxiv.org/html/2512.17792v1
- [W7] (2021). Arrhenius Relationship - ReliaWiki. https://www.reliawiki.com/index.php/Arrhenius_Relationship
- [W8] (2023). Why you might be fitting Arrhenius data badly | scattering and stats. https://mccluskey.scot/2023/04/21/the-case-against-linear-regression-in-Arrhenius-equation-fitting.html
- [W9] (2016). Sensitivity and uncertainty analysis of Arrhenius parameters in order to describe the kinetic of solid thermal degradation during fire phenomena - ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S037971121630039X
- [W10] J Math Chem (2011) 49:1798--1809 DOI 10.1007/s10910-011-9859-7 ORIGINAL PAPER. https://vargal4.elte.hu/Arrhenius.pdf

```{=latex}
\endgroup
```


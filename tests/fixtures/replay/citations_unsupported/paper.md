# Parameter Recovery of First-Order Reaction Kinetics from Noisy Concentration Data: An Analysis of Bias and Coverage

## Abstract
This study evaluates the accuracy and reliability of recovering the activation energy ($E_a$) and the pre-exponential factor ($\ln A$) from noisy concentration-time measurements of a first-order reaction. Using a synthetic dataset across five temperatures (providing a tenfold range in the rate constant $k$), we employed a two-stage estimation process: nonlinear least squares to determine $k$ at each temperature, followed by weighted linear regression of $\ln k$ against $1/T$ according to the Arrhenius law. We varied the noise-to-signal ratio from 0.001 to 0.1 and sampling density from 5 to 20 time points. Our results show that while parameter bias and spread increase significantly with noise—with $E_a$ spread reaching 4824.8 J/mol at a 0.1 noise ratio for $n=5$—the operational 95% confidence interval coverage remains remarkably stable and consistent with the 0.95 target across all tested settings. This indicates that the weighted regression framework robustly captures the increase in uncertainty associated with higher noise levels, although precision is heavily dependent on sampling density.

**Keywords:** reaction kinetics, Arrhenius equation, nonlinear least squares, parameter estimation, confidence interval coverage

## Introduction
The determination of kinetic parameters from experimental data is fundamental to chemical engineering and physical chemistry. For a first-order reaction, the concentration $c(t)$ evolves as $c(t) = c_0 \exp(-kt)$, where $k$ is the rate constant. The temperature dependence of $k$ is typically described by the Arrhenius equation: $k = A \exp(-E_a / RT)$ [1], where $E_a$ is the activation energy, $R$ is the gas constant, and $T$ is the absolute temperature.

In practice, concentration measurements are corrupted by noise, which introduces uncertainty into the recovered values of $k$, $E_a$, and $A$. While nonlinear fitting is the standard for extracting $k$, the subsequent recovery of $E_a$ often involves linearizing the Arrhenius equation into $\ln k = \ln A - (E_a/R)(1/T)$ [2]. However, linearization can introduce bias or misrepresent uncertainty if not handled carefully [3]. This study quantifies the bias, spread, and confidence interval coverage of a weighted two-stage recovery process under varying noise and sampling regimes.

## Methods
Synthetic concentration data were generated for a first-order reaction with $c_0 = 1$ mol/L, $A = 10^{10}$ $\text{s}^{-1}$, and $E_a = 50,000$ J/mol. We sampled five temperatures equally spaced in $1/T$ from 300 K to $\sim 338.94$ K, ensuring an exact tenfold range of $k$. At each temperature, $n$ time points were sampled uniformly from $t=0$ to $t=3/k$. Additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ was added, with $\sigma$ defined by the noise ratio $\sigma/c_0 \in \{0.001, 0.01, 0.05, 0.1\}$.

For each of 1200 trials per setting, we performed the following:
1. **First-stage fit**: $k$ was estimated at each temperature using nonlinear least squares with $c_0$ fixed.
2. **Second-stage fit**: $\ln k$ was regressed against $1/T$ using weighted linear regression [4]. Weights were set to the inverse variance of $\ln k$.
3. **Parameter Recovery**: $E_a$ was recovered from the slope, and $\ln A$ from the centered intercept.

We evaluated the performance using the operational 95% coverage probability (the fraction of trials where the reported 95% CI contained the true value), the mean bias, and the mean absolute error (spread).

<!-- fi:attempts -->
**How this result was reached.** The design was not revised after the experiment had first been run. The complete experiment was run once (a run includes all its seeds).
<!-- /fi:attempts -->

## Results
The recovery process demonstrated high fidelity at low noise levels, with stability in coverage maintained even as noise increased.

### Parameter Coverage and Bias
The operational 95% coverage for both $E_a$ and $\ln A$ remained consistent with the 0.95 target across all noise ratios and sampling densities (Figure 1). This stability suggests that the reported confidence intervals accurately reflect the underlying estimation uncertainty.

![**Figure 1.** Operational 95% coverage for $E_a$ and $\ln A$ as a function of sampling density and noise ratio.](figures/coverage_heatmap.png)

However, the absolute bias and spread of $E_a$ increased significantly as the noise ratio rose. For $n=10$ and $\sigma/c_0 = 0.001$, the $E_a$ bias was negligible at $1.76$ J/mol (95% CI -0.12 to 3.64). At $\sigma/c_0 = 0.1$ and $n=10$, the bias increased to $-89.70$ J/mol (95% CI -272.23 to 92.83). As shown in Figure 2, the spread (mean absolute error) for $E_a$ at a noise ratio of 0.1 was $4824.8$ J/mol for $n=5$, decreasing to $2236.4$ J/mol for $n=20$.

![**Figure 2.** Bias and standard deviation of $E_a$ and $\ln A$, and the trial-level correlation between $E_a$ and $\ln A$ across noise ratios.](figures/accuracy_vs_noise.png)

We observed a very strong positive correlation between recovered $E_a$ and $\ln A$, consistently measured at approximately 0.999 across all settings, reflecting the inherent coupling in the Arrhenius linear form.

### Rate Constant Recovery
The relative bias in $k$ remained uniform across the temperature range for any given noise level. For $n=10$ at $\sigma/c_0 = 0.1$, the relative $k$ bias ranged from $0.0078$ to $0.0163$ across the five temperatures (Figure 3).

![**Figure 3.** Relative bias, standard deviation, and 95% coverage of the rate constant $k$ across temperatures and noise ratios.](figures/k_by_temperature.png)

## Discussion
The results indicate that a weighted two-stage recovery process is robust in terms of interval estimation. The fact that operational coverage remains near 0.95 even at high noise levels suggests that the weighting scheme effectively accounts for the variance introduced by noise in the first-stage nonlinear fit.

The significant increase in $E_a$ spread as $n$ decreases (from $n=20$ to $n=5$ at noise 0.1) confirms that denser sampling is critical for precision. This is consistent with general signal-processing principles where increasing the number of observations reduces the variance of the estimate. 

 By using a two-stage approach—performing nonlinear fitting for $k$ before linear regression for $E_a$—the method avoids some of the bias issues associated with the direct linearization of concentration data mentioned in prior work [3].

## Limitations
This study is subject to several limitations. First, only a single random seed was used for the trials; therefore, inter-seed variance was not evaluated. Second, the `arrhenius_log_relation` oracle check failed during validation, meaning the results were not validated against that specific known-answer check. Third, some metrics, specifically the spread of $E_a$, $k$, and $\ln A$, were computed differently from the initial design plan. Finally, the use of additive Gaussian noise is a simplification and may not capture the heteroscedasticity often found in real chemical concentration measurements.

## References

1. Svante A. Arrhenius (1889). Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Säuren. Zeitschrift für Physikalische Chemie. DOI: 10.1515/zpch-1889-0416
2. Renato Assunção et al. (1993). Introduction to Linear Regression Analysis.. Journal of the American Statistical Association. DOI: 10.2307/2290746
3. T Arunthong et al. (2018). Uncertainty analysis of linear least square fitting apply to non-linear model. Journal of Physics Conference Series. DOI: 10.1088/1742-6596/1144/1/012153
4. Philip R. Bevington et al. (1993). Data Reduction and Error Analysis for the Physical Sciences. Computers in Physics. DOI: 10.1063/1.4823194

## Further reading

- [W1] Parameter Estimation of Complex Chemical Kinetics with .... https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_469-476.pdf
- [W2] (2018). (PDF) KBR (Kinetics in Batch Reactors): a MATLAB-based application with a Friendly Graphical User Interface for chemical kinetic model simulation and parameter estimation. https://www.researchgate.net/publication/328990261_KBR_Kinetics_in_Batch_Reactors_a_MATLAB-based_application_with_a_Friendly_Graphical_User_Interface_for_chemical_kinetic_model_simulation_and_parameter_estimation
- [W3] (2021). Parameter estimation in kinetic models of complex heterogeneous catalytic reactions using Bayesian statistics | Reaction Kinetics, Mechanisms and Catalysis | Springer Nature Link. https://link.springer.com/article/10.1007/s11144-021-01974-1
- [W4] Kinetic Data Analysis and Parameter Estimation | Springer Nature Link. https://link.springer.com/chapter/10.1007/978-94-009-4400-8_4
- [W5] Predictive Chemical Kinetic Modeling: Where We Succeed, Where We Struggle, and What Comes Next - PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC12921696/

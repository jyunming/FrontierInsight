## Results

All 400 trajectories were valid in every cell; the zero observed failures imply a Wilson 95% upper bound of 0.0095. The numerical and analytical checks for rate ratio, centering, delta-method propagation, and centroid fitting also passed.

Figure 1 summarizes \(E_a\) bias, spread, and interval coverage. At 8% noise, increasing the time-point count from 5 to 32 reduced weighted \(E_a\) SD from 2,030 to 717 \(\mathrm{J\,mol^{-1}}\); the corresponding OLS SD declined from 1,920 to 709 \(\mathrm{J\,mol^{-1}}\). The largest displayed weighted \(E_a\) mean bias was 154 \(\mathrm{J\,mol^{-1}}\), or 0.64% of the true value.

![**Figure 1.** Activation-energy bias, estimate SD, and empirical 95% coverage across concentration-noise levels and sampling densities in the study simulation; the coverage panel shows persistent undercoverage relative to the nominal 0.95 reference.](figures/ea_calibration.png)

Weighting improved observed coverage in several higher-information cells but not uniformly. At 8% noise and 20 observations, weighted \(E_a\) coverage was 0.915 (366/400; 0.884–0.939) compared with 0.813 for OLS (325/400; 0.771–0.848). At 32 observations, values were 0.918 (367/400; 0.886–0.941) and 0.870 (348/400; 0.833–0.899), respectively. At 0.25% noise and 5 observations, both methods gave 0.870 coverage (348/400; 0.833–0.899).

Figure 2 shows the analogous \(\ln A\) results. At 8% noise, weighted \(\ln A\) SD fell from 0.771 with 5 observations to 0.276 with 32. At 20 observations, weighted and OLS \(\ln A\) coverage was 0.923 and 0.820, respectively. Across displayed cells, \(E_a\) and \(\ln A\) estimates had correlations of approximately 0.995–0.997, indicating strong slope–intercept coupling when transforming the centered fit back to \(\ln A\).

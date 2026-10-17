  modeling assumptions because no measured tool pupil is supplied; repeat the comparison under reasonable
  source-radius changes. At each dose, sample incident photons as a spatial Poisson process with mean
  0.680 D per nm2 times the local normalized aerial intensity, and absorb each by independent thinning
  with probability 1-exp(-alpha t). Use CAR alpha=0.005 nm^-1, t=35 nm, and MOR alpha=0.020 nm^-1, t=20
  nm from the supplied context. For CAR, sample acid yield per absorbed event as Poisson with mean 2,
  apply an assumed 2 nm secondary-electron Gaussian blur and 5 nm acid-diffusion Gaussian blur, subtract
  a fixed quencher field equal to 20% of the reference-dose mean generated acid density, then threshold
  the resulting deprotection proxy. For MOR, use a single 2 nm Gaussian energy blur and threshold the
  network-change proxy. Use the same declared positive-tone geometry convention for both illustrative
  materials: proxy values above threshold become developed openings; compare openings and remaining lines
  with their intended masks when classifying defects. The yield distribution, electron blur, quencher
  fraction, Gaussian kernels, development tone, and threshold laws are explicit illustrative assumptions,
  not measured material properties; vary yield by ±25%, quencher fraction from 0 to 40%, and each blur
  by ±25%. Set each resist''s fixed development threshold using independent calibration simulations at
  a declared reference line dose of 80 mJ/cm2; then estimate line dose-to-size from held-out simulations.
  Fix one contact mask bias from optical target geometry before viewing resist outcomes and measure each
  resist''s contact dose-to-size on held-out simulations. Report the calibrated thresholds and common
  contact bias, and treat the chosen reference dose as a model assumption in the sensitivity analysis.

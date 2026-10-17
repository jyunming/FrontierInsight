  - item: runtime
    by: metric:runtime
  - item: relative noise of N expected photons is 1/sqrt(N)
    by: equation:E3
  - item: probability that a region expecting N absorbed events receives none is exp(-N)
    by: equation:E4
  grid:
    exposure_dose:
    - 20
    - 30
    - 40
    - 50
    - 60
    - 70
    - 80
    - 90
    - 100
    - 110
    - 120
    - 135
    - 150
    resist_type:
    - CAR
    - MOR
    pattern:
    - lines
    - contacts
    chemistry:
    - 'on'
    - 'off'
  seed_policy: Use base seed 20261008. Derive independent counter-based streams from (base_seed, purpose,

# Benchmark scores (debugging scores, not a model ranking)

Rubric in force: D3 is scored over three canonical works (Kermack–McKendrick 1927, Gillespie 1976/77, Whittle/Bartlett). D6 is scored over six items (the page item was removed 2026-09-16; the "figure after References" item was added 2026-09-17). Mean = Σ round(Dᵢ) / 6. Gates: G1 numbers correct, G2 D1 = 100, G3 D4 = 100. Page count is recorded; it is not a gate.

The runs graded 2026-09-15/16 (39: the 38 of the recomputation and agy1, recomputed on its own) are shown with their values **recomputed on this rubric** (from `_recompute_current.py`, which grading_sheet.md cites as the recomputation of 2026-09-17), not with the numbers their grades files print. Where a later grader recorded a headline figure and a strict or lenient alternative, the table shows the headline figure.

"—" means the value was not recorded. G1 to G3 are "—" for runs graded before the gates existed (2026-09-15, before main7), even where D1 or D4 would imply G2 or G3.

Run-name collisions: `cb1`–`cb3` and `cr1`–`cr3` (cost experiment, 2026-09-19) are different quests from the re-bench runs of the same names, which are prefixed `rb-` below. Likewise `lu1`–`lu3` (the luna2 arm, 2026-09-17) are not `rb-lu1`–`rb-lu3`.

| run | date graded | FI build | model / arm | D1 | D2 | D3 | D4 | D5 | D6 | Mean | G1 | G2 | G3 | pages | tokens / calls | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agy1 | 2026-09-15 | 46f7c56 | antigravity_cli gemini-3.8-flash-high | 71 | 100 | 67 | 0 | 100 | 83 | 70.2 | fail | fail | fail | 11 | 1,532,676 / 27 | paper states wrong values; D4 0; recomputed from 68.8 in grading_sheet.md |
| g4h1 | 2026-09-15 | 130ee29 | gemma4:31b-cloud, thinking high (ollama) | 85 | 80 | 33 | 33 | 83 | 67 | 63.5 | fail | fail | fail | 6 | 288,715 / 20 | "no major outbreaks" at R0=0.9 is wrong (G1 narrow fail) |
| g4h2 | 2026-09-15 | 130ee29 | gemma4:31b-cloud, thinking high | 100 | 100 | 67 | 33 | 83 | 100 | 80.5 | pass (borderline) | pass | fail | 6 | 325,131 / 25 | thinking high; mean 80.5, below main7–9 (86.2–91.7) on the same build |
| g4h3 | 2026-09-15 | 130ee29 | gemma4:31b-cloud, thinking high | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | pass | pass | fail | 6 | 344,141 / 25 | Fig 3 draws every deterministic reference at N=100 |
| main | 2026-09-15 | f05f3d1 (after #253) | gemma4:31b-cloud (ollama) | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | — | — | — | 5 | 261,252 / 25 | review revise at cap; Fig 3 text quotes pooled values |
| main2 | 2026-09-15 | f05f3d1 | gemma4:31b-cloud | 100 | 100 | 0 | 67 | 83 | 100 | 75.0 | — | — | — | 6 | 249,848 / — | one scholarly citation only |
| main3 | 2026-09-15 | f05f3d1 | gemma4:31b-cloud | 67 | 100 | 0 | 100 | 67 | 100 | 72.3 | — | — | — | 4 | 232,536 / — | embedded 1 of 3 figures |
| main4 | 2026-09-15 | ef81abd | gemma4:31b-cloud | 88 | 100 | 67 | 67 | 100 | 67 | 81.5 | — | — | — | 6 | — | the "main" half of the before/after #254 comparison (range gate) |
| main5 | 2026-09-15 | ef81abd | gemma4:31b-cloud | 100 | 100 | 33 | 100 | 100 | 100 | 88.8 | — | — | — | 5 | — | Masuda & Rocha not counted as Gillespie |
| main6 | 2026-09-15 | ef81abd | gemma4:31b-cloud | 64 | 100 | 67 | 67 | 67 | 83 | 74.7 | — | — | — | 6 | — | Fig 2 caption claims a limit that is not drawn |
| main7 | 2026-09-15 | 46f7c56 | gemma4:31b-cloud | 100 | 100 | 67 | 100 | 100 | 83 | 91.7 | pass | pass | pass | 5 | 212,118 / 23 | all gates; fails figure-after-References item |
| main8 | 2026-09-15 | 46f7c56 | gemma4:31b-cloud | 83 | 100 | 67 | 67 | 100 | 100 | 86.2 | pass | fail | fail | 5 | 270,121 / 27 | Fig 2 caption says R0=1.5 across N on a 3×3 grid |
| main9 | 2026-09-15 | 46f7c56 | gemma4:31b-cloud | 100 | 100 | 67 | 100 | 100 | 83 | 91.7 | pass | pass | pass | 6 | 240,030 / 23 | all gates; 78%-empty last page |
| pagelimit1 | 2026-09-15 | c2bfc8c (page-limit branch) | gemma4:31b-cloud | 100 | 100 | 67 | 33 | 100 | 100 | 83.3 | pass | pass | fail | 4 | 312,714 / 30 | 4 pages with 3 figures embedded |
| token | 2026-09-15 | afd82ba (feat/token-savings) | gemma4:31b-cloud | 100 | 100 | 67 | 100 | 67 | 100 | 89.0 | — | — | — | 5 | 244,570 / 27 | ODE final size 0.0 (trivial root) |
| token_final | 2026-09-15 | 06976d1 | gemma4:31b-cloud | 100 | 100 | 67 | 67 | 67 | 100 | 83.5 | — | — | — | 5 | 223,653 / 24 | trivial root; the 0.58 printed is the textbook value |
| tokf2 | 2026-09-15 | 06976d1 | gemma4:31b-cloud | 100 | 100 | 0 | 67 | 83 | 67 | 69.5 | — | — | — | 5 | 207,605 / — | Gillespie 2009 not counted for D3 |
| tokf3 | 2026-09-15 | 06976d1 | gemma4:31b-cloud | 83 | 100 | 0 | 67 | 67 | 100 | 69.5 | — | — | — | 5 | 199,315 / — | RNG reseeded every trial (probability 0 or 1) |
| tokrg4 | 2026-09-15 | d4034a8 | gemma4:31b-cloud | 100 | 100 | 67 | 67 | 100 | 83 | 86.2 | — | — | — | 5 | — | the "token" half of the before/after #254 comparison |
| tokrg5 | 2026-09-15 | d4034a8 | gemma4:31b-cloud | 100 | 100 | 33 | 67 | 100 | 83 | 80.5 | — | — | — | 5 | — | Masuda & Rocha not counted as Gillespie |
| tokrg6 | 2026-09-15 | d4034a8 | gemma4:31b-cloud | 100 | 100 | 67 | 50 | 33 | 83 | 72.2 | — | — | — | 5 | — | D5 2 of 6 goals |
| v0713 | 2026-09-15 | 6f778b2 (last pre-August commit) | gemma4:31b-cloud | 100 | 67 | 0 | 100 | 83 | 67 | 69.5 | — | — | — | 6 | 134,740 / 27 | review accepted (score 5) after 1 revise; 1 seed |
| v238 | 2026-09-15 | 6b689c5 (#238) | gemma4:31b-cloud | 0 | 80 | 0 | 67 | 50 | 50 | 41.2 | — | — | — | 5 | 328,865 / 28 | no in-text citation resolves; ODE returned 0.0 |
| v245 | 2026-09-15 | 52e10dd (#245) | gemma4:31b-cloud | 0 | 100 | 0 | 0 | 67 | 67 | 39.0 | — | — | — | 3 | 335,191 / 33 | no figures; experiment failed again on pass 2 |
| cg1 | 2026-09-16 | 130ee29 | gemma4:31b-cloud via claude_cli | 100 | 100 | 33 | 67 | 100 | 67 | 77.8 | pass | pass | fail | 6 | 280,753 / 26 | only run that ever cited Anderson & May (D3 50→33 when it was dropped) |
| cg2 | 2026-09-16 | 130ee29 | gemma4:31b-cloud via claude_cli | 100 | 100 | 67 | 100 | 100 | 100 | 94.5 | pass | pass | pass | 6 | 303,518 / 24 | all gates; ODE integrated with odeint |
| cg3 | 2026-09-16 | 130ee29 | gemma4:31b-cloud via claude_cli | 88 | 100 | 0 | 67 | 50 | 67 | 62.0 | fail | fail | fail | 5 | 268,197 / 26 | trivial root: comparison drawn against 0 |
| cx1 | 2026-09-16 | 130ee29 | codex_cli gpt-5.6-luna, effort max | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | pass | pass | fail | 7 | 1,227,303 / 32 | Fig 3 "horizontal segment" is a one-point marker |
| cx2 | 2026-09-16 | 130ee29 | codex_cli gpt-5.6-luna, effort max | 100 | 100 | 67 | 100 | 100 | 67 | 89.0 | pass | pass | pass | 8 | 950,809 / 31 | all gates; $7.75 |
| cx3 | 2026-09-16 | 130ee29 | codex_cli gpt-5.6-luna, effort max | 100 | 100 | 67 | 0 | 67 | 83 | 69.5 | fail | pass | fail | 6 | 1,888,197 / 56 | final experiment crashed; FI deleted the 3 real figures; $14.85 |
| g4n1 | 2026-09-16 | 130ee29 | gemma4:31b-cloud, thinking off | 100 | 100 | 67 | 100 | 67 | 100 | 89.0 | fail | pass | pass | 5 | 223,486 / 26 | G2 and G3 pass on borderline rulings |
| g4n2b | 2026-09-16 | 130ee29 | gemma4:31b-cloud, thinking off | 100 | 100 | 67 | 67 | 100 | 83 | 86.2 | fail | pass | fail | 6 | 212,143 / 25 | a 500-run histogram called "a single realization" |
| g4n3 | 2026-09-16 | 130ee29 | gemma4:31b-cloud, thinking off | 100 | 100 | 67 | 100 | 83 | 67 | 86.2 | pass (borderline) | pass | pass | 6 | 253,234 / 26 | all gates; fails figure-after-References item |
| s1 | 2026-09-16 | 130ee29 | claude_cli claude-sonnet-5, effort max then resumed at high | 83 | 100 | 67 | 0 | 83 | 100 | 72.2 | fail | fail | fail | 5 | 2,482,452 / 30 | no figures embedded; $32.18 |
| s2 | 2026-09-16 | 130ee29 | claude_cli claude-sonnet-5, effort high | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | fail | pass | fail | 8 | 1,860,648 / 29 | modal counts read off the axis limits; $22.80 |
| s3 | 2026-09-16 | 130ee29 | claude_cli claude-sonnet-5, effort high | 100 | 100 | 100 | 0 | 83 | 100 | 80.5 | fail | pass | fail | 7 | 2,054,595 / 27 | no figures embedded; resumed after hitting the session limit |
| xg1 | 2026-09-16 | 130ee29 | gemma4:31b-cloud via codex_cli | 100 | 100 | 33 | 33 | 67 | 83 | 69.3 | fail | pass | fail | 5 | 410,828 / 25 | Kermack cited in draft 1, dropped by the rewrite |
| xg2 | 2026-09-16 | 130ee29 | gemma4:31b-cloud via codex_cli | 100 | 100 | 67 | 33 | 50 | 100 | 75.0 | fail | pass | fail | 5 | 461,589 / 23 | FI's min-to-max figure record led to a reversed R0=1.2 pair |
| xg3 | 2026-09-16 | 130ee29 | gemma4:31b-cloud via codex_cli | 100 | 100 | 33 | 33 | 50 | 100 | 69.3 | fail | pass | fail | 5 | 435,247 / 26 | deterministic solver returns 0.0 for every R0 |
| ag1 | 2026-09-17 | 053f182 | antigravity_cli gemini-3.8-flash-high | 79 | 100 | 100 | 67 | 100 | 100 | 91.0 | fail | fail | fail | 6 | 2,839,045 / 29 | D3 3/3 (Bartlett 1949); cost null |
| ag2 | 2026-09-17 | 053f182 | antigravity_cli gemini-3.8-flash-high | 63 | 100 | 67 | 33 | 100 | 100 | 77.2 | fail | fail | fail | 6 | 3,252,705 / 31 | wrote "I(t) ≤ 2" from the first labelled series |
| ag3 | 2026-09-17 | 053f182 | antigravity_cli gemini-3.8-flash-high | 90 | 100 | 67 | 33 | 100 | 100 | 81.7 | fail | fail | fail | 6 | 1,777,373 / 26 | 87.2 if "deep, empty trough" is excused |
| lu1 | 2026-09-17 | 053f182 | codex_cli gpt-5.6-luna, effort max (luna2 arm) | 92 | 100 | 67 | 100 | 100 | 100 | 93.2 | pass | fail | pass | 6 | 1,025,422 / 30 | $8.05; Bartlett 1956 on disk, not cited |
| lu2 | 2026-09-17 | 053f182 | codex_cli gpt-5.6-luna, effort max (luna2 arm) | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | pass | pass | fail | 4 | 1,120,326 / 32 | $9.39; bounds false positive used up both repair rounds |
| lu3 | 2026-09-17 | 053f182 | codex_cli gpt-5.6-luna, effort max (luna2 arm) | 89 | 100 | 100 | 67 | 100 | 100 | 92.7 | pass | fail | fail | 5 | 981,506 / 28 | $8.37; cited Bartlett 1956; ran one seed only |
| rm1 | 2026-09-17 | 6cd131c | gemma4:31b-cloud, thinking off | 83 | 100 | 33 | 100 | 83 | 100 | 83.2 | pass | fail | pass | 4 | 267,415 / 29 | foundational lookup returned Gillespie 2006, not 1976/77 |
| rm2 | 2026-09-17 | 6cd131c | gemma4:31b-cloud, thinking off | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | pass (caveat) | pass | fail | 4 | 240,575 / 27 | replicate seeds overlap (base_seed + counter) |
| terra1 | 2026-09-17 | 3d8f825 (run log; see disagreements) | codex_cli gpt-5.6-terra | 70 | 100 | 67 | 67 | 100 | 83 | 81.2 | pass | fail | fail | 5 | 524,446 / 30 | references [4]/[5] swapped (model error) |
| terra2 | 2026-09-17 | 3d8f825 | codex_cli gpt-5.6-terra | 100 | 100 | 67 | 100 | 100 | 100 | 94.5 | pass | pass | pass | 4 | 538,198 / 30 | all gates |
| terra3 | 2026-09-17 | 3d8f825 | codex_cli gpt-5.6-terra | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | pass* | pass | fail | 4 | 521,151 / 29 | Fig 1 caption says "count", plots fraction |
| rm3b | 2026-09-18 | 17a025c (#289) | gemma4:31b-cloud, thinking off | 80 | 100 | 67 | 67 | 100 | 100 | 85.7 | pass | fail | fail | 4 | 264,157 / 29 | 94.5 with both borderlines excused |
| bl1 | 2026-09-19 | 44b632a (after #296) | gemma4:31b-cloud, thinking off | 83 | 100 | 67 | 100 | 100 | 100 | 91.7 | pass | fail | pass | 4 | 248,721 / 25 | 94.5 with all gates passing if [4] is counted |
| bl2 | 2026-09-19 | 44b632a | gemma4:31b-cloud, thinking off | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | pass | pass | fail | 4 | 244,992 / 25 | Fig 3 wrong-direction sentence |
| bl3 | 2026-09-19 | 44b632a | gemma4:31b-cloud, thinking off | 88 | 100 | 67 | 67 | 100 | 100 | 87.0 | pass | fail | fail | 4 | 212,370 / 24 | review revise at cap |
| cb1 | 2026-09-19 | 90e1029 (after #297) | codex_cli gpt-5.6-terra, effort medium, all nodes | 77 | 100 | 33 | 67 | 100 | 100 | 79.5 | pass | fail | fail | 4 | 542,168 / 29 | Kermack retrieved, never cited |
| cb2 | 2026-09-19 | 90e1029 | codex_cli gpt-5.6-terra, effort medium, all nodes | 86 | 100 | 67 | 100 | 100 | 83 | 89.3 | pass | fail | pass | 5 | 605,344 / 32 | cost pair 2 |
| cb3 | 2026-09-19 | 90e1029 | codex_cli gpt-5.6-terra, effort medium, all nodes | 92 | 100 | 67 | 100 | 100 | 83 | 90.3 | pass | fail | pass | 5 | 533,350 / 29 | cost pair 3 |
| cr1 | 2026-09-19 | 90e1029 | terra medium; 5 light nodes on gpt-5.6-luna | 86 | 100 | 67 | 67 | 100 | 83 | 83.8 | pass | fail | fail | 5 | 480,430 / 27 | Further-reading block spilled onto page 5 |
| cr2 | 2026-09-19 | 90e1029 | terra medium; 5 light nodes on gpt-5.6-luna | 100 | 100 | 67 | 100 | 100 | 100 | 94.5 | pass | pass* | pass | 4 | 473,289 / 28 | all gates (G2 on a precedent; strict D1 89, mean 92.7) |
| cr3 | 2026-09-19 | 90e1029 | terra medium; 5 light nodes on gpt-5.6-luna | 92 | 100 | 67 | 100 | 100 | 100 | 93.2 | pass (borderline) | fail | pass | 4 | 482,742 / 28 | cost pair 3 |
| fm1 | 2026-09-20 | c47e858 (after #318) | gemma4:31b-cloud, thinking off | 100 | 100 | 67 | 100 | 100 | 100 | 94.5 | pass | pass | pass | 4 | 233,328 / 26 | all gates; en dash in math printed "- -" |
| fm2 | 2026-09-20 | c47e858 | gemma4:31b-cloud, thinking off | 100 | 100 | 33 | 67 | 100 | 100 | 83.3 | pass | pass (on a ruling) | fail | 4 | 223,618 / 28 | Gillespie 1977 dropped as "not found" |
| fm3 | 2026-09-20 | c47e858 | gemma4:31b-cloud, thinking off | 70 | 100 | 67 | 67 | 100 | 83 | 81.2 | pass | fail | fail | 5 | 214,783 / 24 | page-limit rewrite added unsupported citations |
| fn1 | 2026-09-20 | b3db90b | gemma4:31b-cloud, thinking off | 100 | 100 | 33 | 67 | 100 | 100 | 83.3 | pass | pass | fail | 4 | 158,596 / 21 | review accept 5 |
| fn2 | 2026-09-20 | b3db90b | gemma4:31b-cloud, thinking off | 100 | 100 | 67 | 100 | 100 | 100 | 94.5 | pass | pass | pass | 4 | 190,551 / 22 | all gates; unseeded-RNG repair fired, correctly |
| fn3 | 2026-09-20 | b3db90b | gemma4:31b-cloud, thinking off | 83 | 100 | 67 | 100 | 100 | 100 | 91.7 | pass | fail | pass | 4 | 255,866 / 27 | 94.5 lenient; Further-reading refit dropped Harris |
| fr1 | 2026-09-20 | 16bc8e4 | gemma4:31b-cloud, thinking off | 80 | 100 | 100 | 100 | 100 | 100 | 96.7 | pass | fail | pass | 4 | 231,841 / 24 | strict D4 20 (PDF figure labels) gives 83.3 |
| fr2 | 2026-09-20 | 16bc8e4 | gemma4:31b-cloud, thinking off | 63 | 100 | 100 | 100 | 100 | 100 | 93.8 | pass | fail | pass | 4 | 243,126 / 26 | title-only Andersson & Britton cited for a theorem |
| fr3 | 2026-09-20 | 16bc8e4 | gemma4:31b-cloud, thinking off | 92 | 100 | 100 | 67 | 83 | 100 | 90.3 | pass | fail | fail | 4 | 322,482 / 28 | 97.2 with both borderlines excused |
| g31p1 | 2026-09-24 | 86b6c05 | antigravity_cli gemini-3.1-pro-high | 88 | 100 | 67 | 67 | 100 | 100 | 87.0 | pass | fail | fail | 4 | 771,049 / 29 | strict 85.8 |
| g31p2 | 2026-09-24 | 86b6c05 | antigravity_cli gemini-3.1-pro-high | 73 | 100 | 67 | 67 | 100 | 100 | 84.5 | fail | fail | fail | 4 | 784,572 / 27 | resumed in warn mode after a manifest false positive |
| g31p3 | 2026-09-24 | 86b6c05 | antigravity_cli gemini-3.1-pro-high | 72 | 100 | 67 | 25 | 83 | 100 | 74.5 | pass | fail | fail | 5 | 840,539 / 29 | resumed in warn mode; captions call a vertical line "flat horizontal" |
| rb-cb1 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-terra, effort medium, all nodes | 92 | 100 | 67 | 100 | 100 | 100 | 93.2 | pass* | fail | pass | 4 | 536,115 / 28 | G1 borderline (one table cell) |
| rb-cb2 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-terra, effort medium, all nodes | 95 | 100 | 67 | 0 | 50 | 83 | 65.8 | fail | fail | fail | 4 | 627,458 / 35 | no results: repair budget not reset after redesign |
| rb-cb3 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-terra, effort medium, all nodes | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | fail | pass | fail | 4 | 581,399 / 31 | resumed in warn mode; false "within its Wilson interval" claim |
| rb-cr1 | 2026-09-24/25 | 09a26b3 | terra medium; light nodes on gpt-5.6-luna | 100 | 100 | 67 | 100 | 100 | 100 | 94.5 | pass | pass | pass | 4 | 557,742 / 27 | all gates; first review was a synthetic accept (codex limit) |
| rb-cr2 | 2026-09-24/25 | 09a26b3 | terra medium; light nodes on gpt-5.6-luna | 63 | 100 | 0 | 100 | 100 | 100 | 77.2 | pass | fail | pass | 4 | 576,892 / 33 | canonical works retrieved, never cited |
| rb-cr3 | 2026-09-24/25 | 09a26b3 | terra medium; light nodes on gpt-5.6-luna | 100 | 100 | 33 | 100 | 100 | 100 | 88.8 | pass* | pass | pass | 4 | 539,810 / 29 | all gates; Gillespie 1977 not retrieved |
| rb-gf1 | 2026-09-24/25 | 09a26b3 | antigravity_cli gemini-3.8-flash-high | 87 | 100 | 67 | 0 | 67 | 100 | 70.2 | fail | fail | fail | 6 | 6,639,520 / 38 | repair budget not reset lost the figures; 3 invented table numbers |
| rb-gf2 | 2026-09-24/25 | 09a26b3 | antigravity_cli gemini-3.8-flash-high | 93 | 100 | 67 | 67 | 100 | 100 | 87.8 | fail* | fail | fail | 5 | 4,464,962 / 25 | Fig 2 text claim false |
| rb-gf3 | 2026-09-24/25 | 09a26b3 | antigravity_cli gemini-3.8-flash-high | 84 | 100 | 67 | 33 | 100 | 83 | 77.8 | fail | fail | fail | 6 | 3,036,080 / 29 | seed-0 figures vs pooled text numbers docked |
| rb-gp1 | 2026-09-24/25 | 09a26b3 | antigravity_cli gemini-3.1-pro-high | 100 | 100 | 100 | 33 | 100 | 100 | 88.8 | pass | pass | fail | 4 | 720,281 / 26 | cites Whittle |
| rb-gp2 | 2026-09-24/25 | 09a26b3 | antigravity_cli gemini-3.1-pro-high | n/a | n/a | 0 | 60 | 67 | 100 | not comparable (56.8 over D3–D6) | fail | fail | fail | 4 | 571,418 / 21 | literature ran without the model during an outage; re-run as rb-gp4 |
| rb-gp3 | 2026-09-24/25 | 09a26b3 | antigravity_cli gemini-3.1-pro-high | 50 | 100 | 67 | 100 | 100 | 100 | 86.2 | pass | fail | pass | 4 | 710,885 / 27 | title-only Andersson & Britton cited 4 times |
| rb-gp4 | 2026-09-24/25 | 09a26b3 | antigravity_cli gemini-3.1-pro-high | 58 | 100 | 67 | 67 | 100 | 100 | 82.0 | pass | fail | fail | 4 | 768,589 / 27 | page-limit rewrite dropped Whittle and added title-only books |
| rb-lu1 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-luna, effort max | 83 | 100 | 100 | 67 | 100 | 100 | 91.7 | fail | fail | fail | 4 | 1,141,024 / 29 | codex limit: FI accepted the paper as-is (synthetic accept) |
| rb-lu2 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-luna, effort max | 100 | 100 | 67 | 100 | 100 | 100 | 94.5 | pass | pass | pass | 4 | 1,336,042 / 40 | all gates; resumed in warn mode and after the codex limit |
| rb-lu3 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-luna, effort max | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 | fail* | pass | pass | 4 | 1,044,270 / 27 | mean 100; G1 strict fail on unlabelled contrasts |
| rb-sn2 | 2026-09-24/25 | 09a26b3 | claude_cli claude-sonnet-5, effort high | 0 | 0 | 0 | 0 | 0 | 83 | invalid (13.8) | fail | fail | fail | 4 | 84,199 / 28 (8 real) | network outage: FI took error strings as answers; re-run as rb-sn4–6 |
| rb-sn3 | 2026-09-24/25 | 09a26b3 | claude_cli claude-sonnet-5, effort high | 0 | 0 | 0 | 0 | 0 | 100 | invalid (16.7) | fail | fail | fail | 4 | 88,254 / 28 (11 real) | whole run inside the outage |
| rb-sn4 | 2026-09-24/25 | 09a26b3 | claude_cli claude-sonnet-5, effort high | 100 | 100 | 67 | 67 | 100 | 100 | 89.0 | fail | pass | fail | 4 | 722,835 / 32 | $3.46; 2 sentences contradict its own Table 1 |
| rb-sn5 | 2026-09-24/25 | 09a26b3 | claude_cli claude-sonnet-5, effort high | 92 | 100 | 67 | 33 | 100 | 100 | 82.0 | pass | fail | fail | 5 | 590,419 / 30 | $2.54; PDF renumbered figures (1,3,2) |
| rb-sn6 | 2026-09-24/25 | 09a26b3 | claude_cli claude-sonnet-5, effort high | 100 | 100 | 67 | 33 | 100 | 100 | 83.3 | fail | pass | fail | 5 | 633,969 / 31 | $2.81; resumed in warn mode after a 95,274 double count |
| rb-te1 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-terra | 83 | 100 | 100 | 100 | 100 | 100 | 97.2 | pass | fail | pass | 4 | 625,253 / 33 | resumed in warn mode; strict 91.7 |
| rb-te2 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-terra | 89 | 100 | 67 | 0 | 33 | 83 | 62.0 | fail | fail | fail | 4 | 622,799 / 33 | final experiment crashed; no results or figures |
| rb-te3 | 2026-09-24/25 | 09a26b3 | codex_cli gpt-5.6-terra | 100 | 100 | 67 | 100 | 83 | 100 | 91.7 | pass | pass | pass | 4 | 561,829 / 29 | all gates on the headline reading |

**96 rows.** Of these, 93 carry a comparable mean. rb-sn2 and rb-sn3 were graded but ruled invalid (outage), and rb-gp2 was ruled not comparable.

Not in the table because they were never scored: sn1–sn3 (Sonnet 5 on 17a025c, 2026-09-18), where every run hit the claude_cli weekly quota and produced no paper; rb-sn1 (outage, re-run fresh); rm3 (crashed with rc=139, no paper); agy2 and agy3 (the 46f7c56 quests, never graded); fs1–fs3 (a grading brief exists, `final4_grading_brief.md`, but no grades file was found).

An asterisk in a gate cell is the grader's own mark for a caveated pass or fail (terra3, cr2, rb-cb1, rb-cr3, rb-gf2, rb-lu3).

## How these numbers were compiled

The table holds the paper's scores. The slides and poster rubric in [rubric.md](rubric.md) was applied to a few runs only; those scores are not tabulated here.

Each run was graded by hand from its delivered outputs, and its grader's notes (the item-level record: which
citations failed, which canonical works were found, which D6 items failed) are what every re-score starts from.
Those notes stay outside the repository; this table carries the result. Runs graded before a rubric change are
shown re-scored on the rubric now in force ([rubric.md](rubric.md)), not with the numbers first printed. Builds were
checked against each run's own log. Runs that were started but never graded are left out.

## Disagreements between files

1. **The 32 runs graded 2026-09-15/16 on the retired rubric.** Their `grades_<run>.md` files print D3 over four works (Anderson & May included) and D6 over six items including "within 4 pages". The recomputed current values are used. The only differences are D3 and D6; D1, D2, D4 and D5 agree in every run.

   | run | grades file mean (retired) | used (current) | what moved |
   |---|---|---|---|
   | main2 | 72 | 75.0 | D6 83→100 |
   | main3 | 72 (72.2) | 72.3 | none (display rounding) |
   | main4 | 76 | 81.5 | D3 50→67, D6 50→67 |
   | main5 | 85 | 88.8 | D3 25→33, D6 83→100 |
   | main6 | 69 | 74.7 | D3 50→67, D6 67→83 |
   | main7 | 89 | 91.7 | D3 50→67 (D6 stays 83: the figure-after-References item replaced the page item) |
   | main8 | 81 | 86.2 | D3 50→67, D6 83→100 |
   | main9 | 86 | 91.7 | D3 50→67, D6 67→83 |
   | tokf2 | 74 (73.7) | 69.5 | D3 25→0 (Gillespie 2009 correction, 2026-09-15); D6 unchanged at 67 |
   | tokf3 | 71 | 69.5 | D3 25→0, D6 83→100 |
   | tokrg4 | 83 | 86.2 | D3 50→67 |
   | tokrg5 | 79 | 80.5 | D3 25→33 |
   | tokrg6 | 67 | 72.2 | D3 50→67, D6 67→83 |
   | pagelimit1 | 81 (80.5) | 83.3 | D3 50→67 |
   | g4h1 | 62 (62.2) | 63.5 | D3 25→33 |
   | g4h2 | 75 (74.8) | 80.5 | D3 50→67, D6 83→100 |
   | g4h3 | 83 (83.3) | 89.0 | D3 50→67, D6 83→100 |
   | g4n1 | 83 (83.3) | 89.0 | D3 50→67, D6 83→100 |
   | g4n2b | 81 (80.6) | 86.2 | D3 50→67, D6 67→83 |
   | g4n3 | 83 (83.3) | 86.2 | D3 50→67 |
   | cg1 | 81 (80.6) | 77.8 | D3 50→33 (the only run that cited Anderson & May) |
   | cg2 | 89 (88.9) | 94.5 | D3 50→67, D6 83→100 |
   | cg3 | 62 (61.8) | 62.0 | none |
   | cx1 | 83.3 | 89.0 | D3 50→67, D6 83→100 |
   | cx2 | 83.3 | 89.0 | D3 50→67, D6 50→67 |
   | cx3 | 63.9 | 69.5 | D3 50→67, D6 67→83 |
   | xg1 | 68 | 69.3 | D3 25→33 |
   | xg2 | 69 (69.3) | 75.0 | D3 50→67, D6 83→100 |
   | xg3 | 65 (65.2) | 69.3 | D3 25→33, D6 83→100 |
   | s1 | 67 (66.5) | 72.2 | D3 50→67, D6 83→100 |
   | s2 | 83 (83.3) | 89.0 | D3 50→67, D6 83→100 |
   | s3 | 73.5 | 80.5 | D3 75→100, D6 83→100 |

2. **agy1:** `grades_agy1.md` has (71, 100, 75, 0, 100, 67) = 68.8. `grading_sheet.md` recomputes it as (71, 100, 67, 0, 100, 83) = **70.2**, which is used.
3. **main7 and main9 in the 2026-09-15 gate note of `grading_sheet.md`:** the note says 89 and 86, and its own bracket gives 91.7 for both. **91.7** is used.
4. **tokf2 and tokf3:** first graded D3 = 25 (means 74 and 71), then corrected to D3 = 0 on 2026-09-15 (`grading_sheet.md`). Now **69.5** each.
5. **rb-gp4:** the first table in `grade_rb_gp4/grade_gp4.md` gives D1 47 and mean 80.2. The same file's "Revised after review (supersedes the table above)" gives D1 58 [strict 47] and mean 82.0, which matches `rebench_grading_result.md`. **82.0** is used.
6. **rb-sn6 (the strict alternative only):**
   - `grade_rb_sn6.md` gives strict 79.2.
   - `rebench_grading_result.md` gives strict 74.2.
   - The headline (83.3), the lenient figure (89.0) and every dimension agree, so the headline **83.3** is used.
7. **rb-gp3 (the strict alternative only):**
   - `grade_rb_gp3.md` gives all-strict 70.5 and 75.0 with only D4 made strict.
   - `rebench_grading_result.md` gives "strict 75.0".
   - The headline **86.2** is the same in both files.
8. **terra build (not a score):**
   - `grading_sheet.md` calls terra's build "post-fix main" and later heads a comparison "both on build 053f182".
   - `runs/terra1.log` through `terra3.log` record `rev=3d8f825`.
   - The table uses the run logs.
9. **Which run first scored D3 3/3:**
   - The prose in `grading_sheet.md` calls ag1 (2026-09-17) "the first run in the series to score 3/3 on D3" and lu3 "only the second".
   - The same file's recomputation (`_recompute_current.py`: Kermack, Gillespie and Whittle, three hits of three) puts s3 (graded 2026-09-16) at D3 = 100 already. The sheet's own Sonnet figures (72 · 89 · 81) only hold with s3 at D3 = 100.
   - The table follows the recomputation: s3 D3 = 100, and the ag1 note does not say "first".

No other conflict was found. Specifically:
- `grades_cost1.md` and `grades_cost_all.md` agree on cb1 and cr1.
- `agy2_grading_result.md` and `grading_sheet.md` agree on ag1–3.
- The other `grade_rb_*` notes checked (cb1, cb3, cr1, gf3, sn3, sn4) agree with `rebench_grading_result.md`.
- The arm means quoted in `final3_grading_brief.md` and `final4_grading_brief.md` (fn 89.8, fr 93.6) match the per-run grades.

# 公開 agent skills 調查：模擬程式正確性

- 執行環境：`hostname` = `vm`，`pwd` = `/home/user/FrontierInsight`
- 日期：2026-09-29
- 方法：用 Exa 搜尋，再把候選 repo `git clone --depth 1` 下來直接讀 SKILL.md、LICENSE、scripts。token 數是用字元數除以 4 粗估的，只算 SKILL.md 本體，不含 references/。下表都是實際讀過的檔案；沒讀到的標為 unverified。

## 一、候選清單（已驗證）

| Skill | Repo / License | 約略 token | 檢查什麼 | 符合 FI policy？ | 建議 |
|---|---|---|---|---|---|
| `numerical-verification` | github.com/e-eight/scicomp-skills，MIT | ~1.75k | 7 層 oracle ladder：closed form、slow twin、invariants、limiting cases、convergence order＋MMS、randomized property、snapshot；要求 sabotage check；每個 tolerance 都要寫理由 | 符合（純文字，不需 GPU） | **adopt**（最核心） |
| `numerics-review` | 同上，MIT | ~1.34k | 三軸 review：standards、intent（逐項對照方程式）、numerics checklist（precision、cancellation、RNG、reduction order、silent failure） | 符合 | **adopt**，拿來當 code 產生後的 review pass |
| `error-budget` | 同上，MIT | ~1.26k | 列出 discretization、truncation、statistical、finite-size、solver、FP、model、implementation 各種誤差來源，要求用量測而非斷言 | 符合；和 `uncertainty-and-units` 部分重疊，但角度不同（數值誤差 vs. 量測不確定度） | **trim**：保留來源表和「量測而非斷言」，刪掉論文 / 投影片寫作段落，避免和 scientific-writing 重疊 |
| `run-provenance` | 同上，MIT，附 `scripts/manifest.py` | ~1.22k | 開跑前先寫 manifest：git commit/diff、resolved params、seeds、套件版本、input/output hash | 大致符合。script 只讀白名單內的 env var（`OMP_*`、`SLURM_*` 等），沒有網路呼叫；SLURM 欄位對 FI 沒用 | **trim**：拿掉 HPC/scheduler 欄位，確認 FI 現有 run record 沒有重複 |
| `layer-separation` | 同上，MIT，附 `check_layers.py` | ~1.59k | model/method/driver/analysis 四層單向依賴 | 符合，但偏架構，不直接檢查正確性 | **skip**（FI 產生的是短程式，skill 自己也說小腳本可以不用） |
| `derive` | 同上，MIT，附 `check_identity.py`（sympy） | ~2.1k | 推導時每一步都做 dimensional check、limiting case 和 sympy 驗證 | 符合；dimensional analysis 那段很有價值 | **trim**：只取 dimensional check 和 limiting case，約 500 tokens |
| `simulation-validator` | github.com/HeshamFS/materials-simulation-skills，Apache-2.0 | ~4.06k | pre-flight、runtime（NaN/Inf、residual 成長、dt collapse）、post-flight（bounds、mass/energy conservation）、failure diagnosis；附 4 支 script | 符合（純 Python、`security_tier: high`）；但偏向讀 log/JSON，而且篇幅長 | **trim**：只取 Stage 3 post-flight 和 conservation 檢查，約 1k |
| `property-based-testing` | github.com/trailofbits/skills，CC BY-SA 4.0 | ~1.08k（references 另有 ~4k） | roundtrip、idempotence、invariant、oracle 等性質目錄；Hypothesis | 內容符合；但 CC BY-SA 有 share-alike 義務，改寫版必須同授權，而且例子偏向 serialization / smart contract | **skip 直接 import**；只把 property catalog 的概念寫進自製 skill（要註明出處） |
| `hardening-research-code` | github.com/uw-ssec/rse-plugins，BSD-3-Clause | ~2.28k | reference test、tolerance 從 FP floor 推出、未驗證的 baseline 要標 `UNVERIFIED` | 符合 | **trim**：tolerance 規則和 `UNVERIFIED` 標記很好用，其餘和 `numerical-verification` 重疊 |
| `ensuring-reproducibility` | 同上，BSD-3-Clause | ~1.79k | 「沒重現過就不算有紀錄」 | 符合；和 `run-provenance` 重疊 | **skip**（二選一，`run-provenance` 較精簡） |
| `research-software-engineering` | github.com/a-attia/scicomp-research-skills，MIT | ~4.33k（references/01 另有 ~3.9k） | MMS、convergence-rate test、conservation invariant、seed discipline、FP gotchas | 符合，但太長，而且 repo 內含 literature-survey | **skip**（內容大多已被上面幾個涵蓋），可以當 reference 參考 |

已經匯入的 `benchmark-and-mms-planner`（HeshamFS，~2.9k）和 `convergence-study` 仍然有效，不必重複 import。

其他搜到但**沒讀原檔**的（unverified）：`math-review`（athola/claude-night-market，只看了第三方 mirror）、`design-numerical-simulation`（jeffreytse，只看了 skillsdirectory 頁面）、OmniScientist 的 provenance contract。

## 二、結論

1. 優先匯入 e-eight/scicomp-skills 的 `numerical-verification` 和 `numerics-review`（MIT，合計約 3.1k）。它們最貼近「LLM 寫 simulation code」的情境。
2. 以 trim 方式匯入 `error-budget`、`run-provenance`，以及 `simulation-validator` 的 post-flight 段落。trim 後的版本要保留原 LICENSE 和出處。
3. PBT 的 CC BY-SA 4.0 有傳染性，不直接 import。

## 三、沒有好的公開 skill 的技術：自製 SKILL.md 大綱（每份 < 1.5k tokens）

### A. `invariant-guards`（守恆量 / 不變量在執行期的斷言）
公開 skill 只「提到」守恆，沒有教 LLM 在它產生的 code 裡**內建**檢查。
1. 觸發條件：產生任何時間積分或 PDE/ODE 程式時。
2. 先列出系統的不變量：質量、能量、動量、機率歸一、正定性、對稱性；每一項標明「精確守恆」或「只有 O(h^p) 近似守恆」。
3. 每 N 步計算一次漂移 `|I(t)-I(0)|/|I(0)|`，門檻值要從 scheme order 和 FP floor 推導出來，並寫下理由。
4. 碰到 NaN/Inf 或邊界值越界時立即 fail，不可以 clip 掩蓋。
5. 在 output JSON 輸出 `invariants: {name, drift, tol, pass}`，讓 FI 的 judge 可以機器判讀。
6. 反模式：不變量只在最後檢查一次；不變量本身就是由被測程式算出來、形同自我比對。

### B. `reference-cross-check`（slow twin / 參考實作差分測試）
1. 觸發條件：寫出向量化、加速或近似版本時。
2. 規則：先寫出明顯正確、慢速的 reference（dense、迴圈、`scipy` 標準 solver），再寫 fast path。
3. 在小尺寸下用 randomized inputs 比對，tolerance 取 `eps × κ × ops`。
4. 選 reference 的優先序：closed form > 獨立演算法 > 成熟函式庫 > 同一演算法的不同寫法（最弱，要標明）。
5. sabotage check：故意改一個符號或常數，確認比對會變紅。
6. 報告要寫明 reference 的來源和獨立程度。

### C. `numerical-property-tests`（科學程式的 property-based testing）
1. 用 Hypothesis 產生**符合物理範圍**的輸入（`floats(allow_nan=False)`，並限定在有物理意義的區間）。
2. 性質目錄（改寫自 PBT 概念）：對稱或置換不變、尺度律（單位縮放 → 結果按維度縮放）、極限情形、單調性、線性疊加（線性問題）、roundtrip（正變換 / 逆變換）。
3. 比對一律用 `rtol`/`atol`，不可以用 `==`；刻意測試退化點和 near-singular 輸入。
4. 限制 `max_examples` 和 deadline，以符合 FI 的 sandbox 時間預算。
5. 失敗時保留 Hypothesis 縮減後的最小反例，寫進 run record。

### D.（選配）`dimensional-consistency`
`uncertainty-and-units` 已經涵蓋 pint 和 plausibility，所以只需要一份約 400 tokens 的補充：要求 LLM 在每個公式旁邊用註解標出維度，並檢查 exp/log/sin 的引數是否無因次（取自 `derive` 的做法）。

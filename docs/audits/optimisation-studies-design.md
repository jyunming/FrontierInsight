# 「找出最佳設計」型研究（optimisation studies）的設計報告

> 研究＋設計報告，不含程式碼變更。撰寫於 2026-09-30。
> 執行環境：`hostname` = `vm`，`pwd` = `/home/user/FrontierInsight`（基底 commit `b6baad4`）。
> 注意：題目要求閱讀的 `CLAUDE.md` **在這個 repo 裡不存在**（`find . -iname claude.md` 無結果），所以本報告以 `dev/registry.md`、`docs/rigor.md`、`docs/capabilities-reference.md` 與原始碼為準。
> 範例一律使用通用工程題目（散熱片鰭片間距／厚度、桁架截面、控制器增益），不涉及光學或光源。

---

## 0. 一段話摘要

FI 目前的 protocol（`core/plan.py::normalize_protocol`）只有一種形狀：**固定網格（`grid`）× 每格跑幾次（`runs_per_setting`）× 每格量什麼（`metrics`）**。當使用者說「幫我找出讓散熱片溫度最低的鰭片間距與厚度」，plan 只能把它改寫成「在 5×5 個間距／厚度上量溫度」，最後論文報告的是一張掃描圖，而不是「最佳設計是 s = 3.2 mm、t = 0.8 mm，比基準設計低 6.1 K」。

本報告提出：

1. plan 先把研究分類成 **`study_type: measure`（量測）** 或 **`study_type: find_best_design`（找最佳設計）**；
2. 後者的 protocol 多一個 `optimisation` 區塊：**目標（objective）、限制條件（constraints）、設計變數與範圍（design_variables）、基準設計（baseline）、數值設定的「搜尋用」與「查核用」兩層（numerical_settings）、評估次數上限（evaluation_budget）、改善量的判斷門檻（improvement_tolerance）**；
3. **最佳化迴圈由 FI 引擎執行**（與今天 FI 自己跑 `run_trial`／`run_cell` 同一個道理），腳本只提供 `run_cell(cell)`；每一次評估都由 FI 寫進自己的帳本；
4. 找到的設計由**引擎**在較細的數值設定下重新評估、檢查限制條件、多起點比對、鄰域擾動，並與基準設計在同一套細設定下比較；結果寫成 `needs/OPTIMUM_CHECK.json`；
5. 最佳設計存成 `results/best_design.json`，論文有固定的「找到的最佳設計」一節；CLI／Web／VS Code 都只讀這兩個檔案。

---

## 1. 其他系統怎麼處理「最佳化的交付物」

> **取得方式說明**：以下每一條都是這次實際抓取過的頁面。內建的 WebFetch 被出口代理擋住 deepmind.google、arxiv.org、optuna.readthedocs.io、docs.scipy.org、grc.nasa.gov，所以這些頁面改用 Exa 抓取全文。標 **〔僅摘要／搜尋片段〕** 的條目只看過搜尋結果的摘錄或論文摘要，沒有讀全文，請當作「未完整驗證」。沒抓到的主題列在 1.7。

### 1.1 OpenEvolve（開源、AlphaEvolve 風格）

- 來源：<https://github.com/codelion/openevolve>（README）、<https://raw.githubusercontent.com/codelion/openevolve/main/openevolve/evaluator.py>、<https://raw.githubusercontent.com/codelion/openevolve/main/configs/default_config.yaml>
- **評估器由使用者撰寫，與提出程式的 LLM 分開**。README 的範例是 `evaluator=lambda path: {"score": benchmark_fib(path)}`，回傳的是一個指標 dict。
- **分階段評估（cascade）**：評估器可以定義 `evaluate_stage1/2/3`。沒通過前一階的門檻就不進下一階（`cascade_thresholds: [0.5, 0.75, 0.9]`，預設開啟，用來「提早濾掉差的解」）。每一階都有逾時限制（預設 300 s），逾時會標記 `"timeout": True`。
- 評估器也可以回傳 stderr、剖析資料等「artifacts」，這些會被自動放進下一代的 prompt。README 沒有討論評估器被鑽漏洞（reward hacking）的問題。
- **對 FI 的啟示**：可信度來自「評分的程式不是出主意的程式」，這和 FI 的 trial contract 一致。但 OpenEvolve 的評估器是**單一精度**的，沒有「換細網格重新評分」這一步。這一步正是 FI 要補的（2.5）。

### 1.2 AlphaEvolve（DeepMind）

- 來源：<https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/>、<https://arxiv.org/abs/2506.13131>（白皮書，讀到第 3.1 節）
- 只處理「可由機器評分」的問題。使用者提供 `evaluate` 函式，回傳一個純量 dict（慣例是越大越好）。論文把「需要人工實驗的任務」明確列為範圍外的限制。
- **評估 cascade（hypothesis testing）**：先在小規模測試，通過才進下一階，用來及早淘汰有錯的程式。
- **強化評估器、防數值誤差**：在矩陣乘法分解的任務中，評估時會把每個元素四捨五入到整數或半整數，「以確保分解是精確的、避免任何數值誤差」。每個程式用**多個隨機種子**執行，分數也包含「達到該秩的種子比例」。
- 部落格提到，改動 TPU 電路的提案必須先通過「穩健的驗證方法」，確認功能仍然正確。
- 我讀過的部分**沒有**專門談評估器被鑽漏洞的段落。
- **對 FI 的啟示**：(a) 要防止最佳化器利用數值誤差，做法是讓評分那一步用更嚴格的數值處理，這對應 FI 的查核層；(b) 用多個種子評分、報告穩健性，對應 FI 對隨機目標用新種子重評；(c) 小規模先篩再放大，對應 FI 的 oracle gate 先跑 baseline。

### 1.3 Optuna

- 來源：<https://optuna.readthedocs.io/en/stable/faq.html>、<https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html>
- **交付物是 `study.best_trial`／`best_params`**。有限制條件時，「最佳 trial 從滿足所有限制的 trial 中挑選」，限制值 ≤ 0 才算可行。沒有任何可行 trial 時會丟 `ValueError`：寧可報錯，也不交出不可行的解。
- **預算**：`optimize(n_trials=…, timeout=…)`，trial 數或時間先到就停。
- **限制只支援軟性（soft）**：FAQ 原文是 Optuna「採用軟限制，**不支援**硬限制」。也就是說，搜尋過程中仍會去評估不可行的點，只是挑最佳解時把它們排除。
- **可重現性**：可以給 sampler 固定種子（如 `TPESampler(seed=10)`），但平行／分散模式本質上不確定。**目標函數本身若不確定，就無法重現整次最佳化**。
- **失敗**：未捕捉的例外會讓 trial 變成 FAIL，並中止整個 study（除非設定 `catch=`）。回傳 NaN 的 trial 算失敗，但不會中止 study。
- **對 FI 的啟示**：「最佳 = 可行解中最好的」、硬性預算、失敗的評估照實記錄，這三點 FI 都照做。Optuna 沒有做的是：交出 `best_trial` 之前，用別的精度重新驗證。

### 1.4 Ax／BoTorch 〔僅摘要／搜尋片段〕

- 來源（只看過搜尋結果的摘錄，沒有另外完整抓取）：<https://ax.readthedocs.io/en/stable/service.html>、<https://ax.readthedocs.io/en/latest/_modules/ax/service/utils/best_point.html>、<https://ax.dev/docs/next/tutorials/getting_started/>
- `get_best_parameters(use_model_predictions=True)` 先用代理模型的**預測值**挑最佳點，沒有模型才退回「觀察到的最佳原始值」。原因是有雜訊時，觀察值最好的點多半只是運氣好。
- 沒有提供量測誤差時，Ax 預設資料有雜訊，並自行推估雜訊大小。best-point 的程式碼在模型擬合不佳時會記錄警告，提醒「資料有雜訊，解讀最佳點要小心」。
- 限制條件寫成字串，例如 `"m1 <= 3"`。判斷可行性時，把「有 95% 把握至少違反一條限制」的點排除。
- 教學提到留一交叉驗證與 Sobol 敏感度圖。
- BoTorch 本身的頁面**沒有抓取**。
- **對 FI 的啟示**：有雜訊時，「搜尋中看到的最佳值」不能直接報告。Ax 的做法是用模型去平滑；FI 的做法（2.5）是用新種子重新評估。後者更直接，也不必相信代理模型。

### 1.5 SciPy

- 來源：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>（v1.18.0）、<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>、<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>
- **結果物件 `OptimizeResult`** 包含：`x`、`success`、`status`、`message`（為什麼停）、`fun`、`nfev`（函數評估次數）、`nit`，以及 **`maxcv`（最大限制違反量）**。但不是每個 solver 都會填每個欄位。
- 限制條件只有 COBYLA、COBYQA、SLSQP、trust-constr 支援。Nelder-Mead、L-BFGS-B 等只支援邊界（bounds）。`maxiter` 不等於評估次數：文件原文說「每次迭代可能用到多次函數評估」。
- **`differential_evolution`**：隨機的全域搜尋，「通常需要較多次函數評估」。`polish=True` 會在最好的族群成員上再跑一次局部最佳化（L-BFGS-B，有限制時改用 trust-constr）。用 `rng` 可以讓結果重現。收斂判準是族群能量的標準差 ≤ `atol + tol·|mean|`。
- **對 FI 的啟示**：(a) 交付物應該連同「為什麼停、用了幾次評估、最大限制違反量」一起記錄，FI 的帳本與 `stopped_because` 就是做這件事；(b) 「全域搜尋 + 局部 polish」是成熟的組合，對應 FI 的 `global_then_local`；(c) 預算要用**評估次數**來算，不能用迭代數。

### 1.6 工程上的查核：網格收斂、贏家詛咒、規格鑽漏洞

- **網格收斂指數（GCI，Roache）**：<https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html>（NASA NPARC）
  - 建議用**三層網格**，才能準確估計收斂階數，並確認解落在漸近範圍內。
  - 安全係數：比較兩層網格用 `Fs = 3.0`，三層以上用 `Fs = 1.25`。網格細化比 `r ≥ 1.1`。
  - 漸近範圍的檢查：`GCI23 ≈ r^p · GCI12`。範例中觀察到的收斂階 p = 1.786（理論值為 2）。
  - 頁面警告：單純的相對誤差「不應拿來當誤差估計……只要讓細化比接近 1.0，就能人為把它做得很小」。
  - 我**沒有找到**專門討論「在最佳解處加密網格重新評估」的來源。本報告把 GCI 的做法套用到最佳解，是本報告自己的延伸，不是引述。
- **贏家詛咒（optimizer's curse）**：Smith & Winkler (2006), *Management Science* 52(3), doi:10.1287/mnsc.1050.0451 〔僅摘要〕。摘要頁：<https://ideas.repec.org/a/inm/ormnsc/v52y2006i3p311-322.html>。
  - 原文大意：即使每個估計都是無偏的，照估計值挑出最好的方案後，應預期它的真實價值**低於**估計值。作者建議用貝氏調整來做「有紀律的懷疑」。
- **規格鑽漏洞（specification gaming）**：<https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/>（2020）
  - 定義：「滿足目標的字面規格，卻沒有達成原本意圖的行為」。
  - 文章指出，演算法越強，越可能找到與原意很不一樣的巧妙解。對 FI 而言，這正是最佳化器把設計推到模擬器出錯區域的機制。

### 1.7 沒有查證的主題（標為未驗證）

以下主題在本報告中**只以一般工程常識使用，沒有引用任何來源**：ASME V&V 20、Goodhart 定律、代理模型被最佳化器利用（surrogate exploitation）、多起點策略的文獻（除了 differential_evolution 文件之外）、限制違反容差的標準（除了 SciPy 的 `maxcv`）、最佳化後的敏感度分析（除了 Ax 的 Sobol 圖）、scipy `basinhopping`、Optuna 的 pruning、BoTorch 本身的文件。

### 1.8 小結：大家都做、FI 可以多做一步的地方

| 做法 | OpenEvolve | AlphaEvolve | Optuna | Ax | SciPy | FI（本設計） |
|---|---|---|---|---|---|---|
| 評分與出主意分離 | ✔ | ✔ | ✔（使用者提供目標函數） | ✔ | ✔ | ✔（引擎跑迴圈，腳本只給 `run_cell`） |
| 硬性評估預算 | 逾時 | — | `n_trials`／`timeout` | — | `maxiter`／`maxfev` | ✔（引擎計數） |
| 可行性決定最佳解 | — | — | ✔ | ✔（95% 規則） | `maxcv` 回報 | ✔（在查核層判定） |
| 雜訊下不直接報告觀察最佳值 | — | 多種子 | — | ✔（模型預測） | — | ✔（新種子重評、配對區間） |
| 在更細的數值設定下重新評估最佳解 | — | 評分時捨入去誤差（部分相關） | — | — | — | ✔（GCI 式三層） |
| 與基準在同一設定下比較 | — | — | — | — | — | ✔ |

最後兩列，是這次查到的工具都沒有內建、而使用者明確要求的：**最佳解要在更細的設定下由引擎重算，並與基準公平比較**。

---

## 2. FI 的具體設計

### 2.1 現況：哪些東西已經可以直接用

| 已有的東西 | 位置 | 對最佳化的意義 |
|---|---|---|
| 引擎自己呼叫模擬函式（trial contract） | `core/trial_runner.py`：`run_cell(cell)` / `run_trial(cell, trial_id, seed)`，由 `HARNESS_SOURCE` 在 quest 自己的 Python 中、一格一個 process 執行 | 最佳化器需要的「給一組參數 → 回傳數字」介面**已經存在**。 |
| 引擎對任意一組設定跑一次 | `trial_runner.run_case(...)`（`measure_oracles` 用它在 oracle 的 `case` 上呼叫模擬） | 查核最佳設計（在細設定下重新評估）可以直接沿用這條路：不是腳本說自己多好，而是引擎自己跑。 |
| 引擎判定、不信任腳本的自我報告 | `core/oracle_check.py`（`judged`、`engine_passed`、`script_measured`） | 同樣的原則套用到「最佳設計是否真的比較好」。 |
| 凍結的 protocol 與修訂（amendment） | `core/frozen_protocol.py` | 變數範圍、評估次數上限、基準設計都要凍結；事後擴大範圍 = amendment。 |
| 證據階梯 | `core/evidence.py`（六級） | 最佳化研究不需要新級別，只要在既有級別加上條件（見 2.6）。 |
| 設計自我檢討 check 1：循環評估 | `agents/design_self_critique.md` | 已經點名「最佳化器與評分器共用同一個模擬器」的問題；本設計把它從「提醒」變成「引擎可檢查的結構」（見 2.5）。 |
| 數值警告 | `core/numeric_warnings.py` 已收 `OptimizeWarning`、`ConvergenceWarning` | 腳本內若偷偷用 scipy 最佳化器也會被抓到警告；但本設計建議最佳化器不在腳本內。 |
| 可獨立執行的 `code/` | `core/code_project.py`（`run.py`、`study.json`） | 要能加上「重新評估最佳設計」的指令。 |

缺的東西：protocol 沒有「目標／限制／變數範圍／基準」的語彙；plan prompt（`core/engine.py` 內 `"protocol": {` 那一段）只教模型寫 `grid`；沒有引擎端的最佳化迴圈；沒有對最佳解的查核；論文與 quest 資料夾沒有「最佳設計」這個交付物。

### 2.2 Protocol 的新增欄位（用領域科學家的語言）

`study_type` 放在 design 層（與 `hypothesis` 同層），`optimisation` 放在 `protocol` 內，和 `grid` 互斥（一個 protocol 不能同時是掃描與最佳化；需要「先掃描再最佳化」的研究見第 5 節問題 Q4）。

```yaml
study_type: find_best_design          # measure（預設）| find_best_design
protocol:
  optimisation:
    objective:
      quantity: max_base_temperature   # run_cell 回傳 dict 裡的名稱
      direction: minimise              # minimise | maximise
      unit: K
      meaning: "散熱片底座的最高溫度"
    design_variables:                  # 最佳化器可以動的東西
      - {name: fin_spacing,   low: 1.0, high: 6.0, unit: mm, kind: continuous}
      - {name: fin_thickness, low: 0.4, high: 2.0, unit: mm, kind: continuous}
      - {name: fin_count,     low: 8,   high: 30,  unit: "",  kind: integer}
    fixed:                             # 不讓最佳化器動、但模擬需要的條件
      heat_load_W: 40
      air_speed_m_s: 2.0
    constraints:                       # 每一條都是 run_cell 回傳的一個量
      - {quantity: mass_g,        limit: "<= 120", unit: g}
      - {quantity: pressure_drop, limit: "<= 25",  unit: Pa}
    baseline:                          # 對照組：目前的／文獻上的設計
      values: {fin_spacing: 3.0, fin_thickness: 1.0, fin_count: 15}
      source: "[3] 表 2 的商用散熱片"     # 與 oracle 的 reference 同樣的規則
    numerical_settings:                # 最關鍵的新東西：搜尋用 vs 查核用
      mesh_size_mm: {search: 0.5, check: [0.25, 0.125]}
    evaluation_budget:
      per_start: 60                    # 每個起點最多評估幾次
      starts: 4                        # 幾個獨立起點
    search_method: bounded_local       # 見 2.4 與第 5 節 Q1
    improvement_tolerance:             # 多少改善才算「真的比基準好」
      value: 0.5
      mode: absolute                   # absolute | relative
    constraint_margin: 0.0             # 限制條件在查核設定下要留多少餘裕
  oracles: [...]                       # 照舊，至少一個；建議其中一個 case 就是 baseline
  model: {...}                         # 照舊
```

欄位規則（`core/plan.py` 新增 `normalize_optimisation`，同樣「形式寬鬆、數字嚴格」）：

- `objective.quantity`、`direction` 必填；`direction` 只能是 `minimise`／`maximise`。
- 每個 `design_variables` 需要 `name`、`low < high`（數字）、`kind ∈ {continuous, integer, choice}`；`choice` 用 `values: [...]`。
- `baseline.values` 必須落在每個變數的範圍內，而且**名稱集合要等於 `design_variables` 的名稱集合**（不能漏）。
- `constraints[].limit` 解析成 `(<=|>=) 數字`；讀不懂就拒絕，不猜。
- `numerical_settings` 每一項要有 `search`（一個值）與 `check`（一個或多個值，且都要比 `search` 更「細」）。**哪個方向算細**由 `finer: smaller | larger` 指定（網格尺寸越小越細、階數越大越細），預設 `smaller`。
- `evaluation_budget.per_start × starts` 是**上限**，不是目標；兩者都必須是正整數。
- 一個有 `optimisation` 的 protocol 若也寫了 `grid`，拒絕並說明原因。
- 隨機模擬（有 `run_trial`）再加 `runs_per_evaluation`（搜尋時每次評估的次數）與 `check_runs`（查核時每個設計用的新種子次數）。

`plan.md` 在設計區塊上方新增一節 **「要最佳化的是什麼」**，用人話寫出：目標、方向、每個變數的範圍與單位、限制、基準設計（與出處）、搜尋用／查核用的數值設定，以及**評估次數與預估時間**（見 2.7）。

### 2.3 每個節點做什麼

| 節點 | 量測型（今天） | 找最佳設計型（新） |
|---|---|---|
| `clarify` | 問題範圍 | 題目含「最佳／最小化／最大化／optimal／best design」等字眼卻看不出是哪一種時，多問一題：「你要的是（A）看某個量怎麼隨參數變化，還是（B）一個最好的設計？」答案寫進 `study_type`。 |
| `plan` | 寫 `grid`、`metrics`… | plan prompt 加分類規則與 `optimisation` 區塊的說明；`normalize_design` 檢查 `study_type` 與 protocol 形狀一致（`find_best_design` 卻沒有 `optimisation` → 退回重寫，而不是默默改寫成掃描——**這正是今天的 bug**）。 |
| `design`（self-critique） | 12 條 check | 新增 3 條（見 2.5）：基準是否真實可行、變數範圍是否有物理意義且在 `model.holds_for` 內、查核用數值設定是否真的比較細。check 1（循環評估）針對最佳化改寫。 |
| `implement` | `simulate.py::run_cell/run_trial` + `experiment.py` 分析 | `simulate.py` 只寫 `run_cell(cell)`：`cell` 同時帶設計變數、`fixed` 條件與 `numerical_settings` 的值，回傳目標與每個限制量。**不寫最佳化迴圈**、不自己決定網格尺寸。`experiment.py` 讀 FI 的最佳化紀錄（`FI_OPTIMISATION` 指向的檔案）畫收斂圖與比較表。`protocol_check` 靜態檢查：腳本不得 import `scipy.optimize`／`optuna` 做搜尋、不得把 `mesh_size` 寫死。 |
| `execute` | oracle gate → 跑網格 | oracle gate（照舊，case 建議含 baseline）→ **引擎的最佳化執行器**（`core/optimise_runner.py`，新）在預算內搜尋，寫 `raw/optimisation_ledger.jsonl` → **引擎的最佳解查核**（`core/optimum_check.py`，新），寫 `needs/OPTIMUM_CHECK.json` 與 `results/best_design.json`。 |
| `execute_reflect` | 腳本失敗／警告 → 修 | 評估失敗率太高、或 `run_cell` 沒回傳目標／限制量 → 修 `simulate.py`。**查核沒過不是修腳本的理由**（那是研究結果：「改善在細網格下消失」本身就是誠實的答案），只記錄並往下走；research profile 下用它限制證據等級，而不是停下來叫模型改到通過。 |
| `analyze` | 讀 RESULT_JSON | 另收到引擎的查核結論（仿 `oracle_check.analysis_note`）：哪些查核通過、改善量與誤差、是否在邊界上、是否多個起點收斂到不同設計。 |
| `write` | 論文 | 固定一節「找到的最佳設計」（見 2.8），數字只能來自 `OPTIMUM_CHECK.json`／`best_design.json`。 |
| `claim_check`／數字稽核 | 對 RESULT_JSON | 另對 `OPTIMUM_CHECK.json` 核對：論文說的改善量、最佳參數、限制餘裕必須逐字一致；論文不得說「最佳（optimal）」除非查核的多起點與擾動都通過，否則只能說「在 N 次評估內找到的最好設計」。 |
| `review` | 評審 panel | methodologist 的 must-flag 清單加 `optimum_unverified`（論文宣稱的改善沒經引擎查核或查核沒過）。 |

### 2.4 最佳化迴圈在哪裡跑：引擎，不是腳本

理由與今天把 trial 迴圈從腳本收回引擎（`trial_runner` 模組說明中的 P0-2）完全相同：**會寫帳本的程式不能是被帳本檢查的程式**。若最佳化器在腳本內，腳本可以（有意或無意）只報告好看的點、用它自己挑的網格、超出預算、或把限制條件放寬。

做法：

- `core/optimise_runner.py`（新）在 FI 行程內決定下一個要評估的點，交給既有的 harness（`HARNESS_SOURCE`，`entry = "run_cell"`）在 quest 的 Python 中評估；**每次評估一行**寫入 `raw/optimisation_ledger.jsonl`（起點編號、第幾次、變數值、數值設定、回傳值、狀態、耗時、值的 hash）。批次送評估（一次一批點、一個 process）可攤平 process 啟動成本。
- 搜尋方法用一份**固定清單**，由 FI 實作、只用標準函式庫（harness 必須在沒有裝 FI 的 venv 裡跑，這是 `HARNESS_SOURCE` 既有的限制；最佳化器在 FI 這一側，所以其實 FI 端可用 scipy，但為了可重現與 `run.py` 可獨立重跑，建議標準函式庫版）：
  - `bounded_local`：Latin hypercube 起點 + 有界 Nelder–Mead（連續變數）；
  - `global_then_local`：先用一半預算做差分演化（differential evolution）式的全域搜尋，再在最好的點做局部收斂；
  - `exhaustive`：變數全是 `integer`／`choice` 且組合數 ≤ 預算時直接窮舉（此時其實就是掃描，但交付物仍是最佳設計）。
  清單外的方法（例如需要 BoTorch 的貝氏最佳化）在第 5 節 Q1 讓使用者決定。
- 起點由引擎用 quest 的 base seed 決定（`trial_seed` 同一套），**第一個起點永遠是 baseline**，其餘是 Latin hypercube 點，所以「至少不比基準差」是結構保證。
- 失敗的評估（`run_cell` 丟例外、NaN）當作不可行，列入帳本與論文（「60 次中 3 次失敗」）。
- 預算用完就停，紀錄 `stopped_because: budget | converged | no_progress`；**預算是硬上限**，由引擎計數，腳本無從超出。

### 2.5 引擎如何查核最佳解（`core/optimum_check.py`）

搜尋結束後，引擎取每個起點的最好**可行**設計（依搜尋設定下的目標排序，保留前 `k = min(3, starts)` 個候選，而不只一個——避免搜尋時的最佳點其實是數值誤差造成、在細設定下被第二名超越），然後做五項查核。每一項都是**引擎自己呼叫 `run_cell`**，與 `measure_oracles` 同一條路，腳本不知道自己在被查核（與 `case_env` 相同的環境）：

1. **細化重評（refinement）**：對每個候選與 baseline，在 `numerical_settings.check` 的每一層重新評估。
   - 以最細一層的值為準選出最佳設計（不是搜尋層）。
   - 以 GCI 的做法估計數值誤差 `e_num`（1.6）：搜尋層＋兩個查核層共三層時，算出觀察到的收斂階 p、做 Richardson 外插，用安全係數 1.25；只有兩層時用 3.0，並在紀錄中註明「只有兩層，無法確認在漸近範圍內」。`normalize_optimisation` 要求相鄰兩層的細化比 ≥ 1.1，因為比值接近 1 會讓誤差估計被人為壓小（1.6 的警告）。
   - **判準**：`改善量(最細層) ≥ improvement_tolerance`，**而且** `改善量 > e_num(baseline) + e_num(best)`，也就是改善必須大於兩個設計的數值誤差加總。搜尋層與最細層的改善量差距也要列出（例如「搜尋時看起來改善 9.0 K，細網格下是 6.1 K」）。
2. **限制條件查核**：在最細一層，每一條限制都要成立且留 `constraint_margin`；另外報告每條限制的**餘裕（slack）**與它是否「頂到」（|slack| < `e_num`）。頂到的限制在細層被違反時，改選下一個在細層可行的候選；全部不可行就是「沒有找到可行的更佳設計」。
3. **多起點一致性**：比較各起點的最佳（細層）目標值。若最好與次好的起點目標差 < 容差但設計差很遠 → 報告「目標在這一帶很平，有多個近似等價的設計」；若差 > 容差 → 報告「各起點落到不同的局部最佳」，論文不得寫「全域最佳」。
4. **鄰域擾動（sensitivity／局部最佳性）**：在最細層對最佳設計的每個連續變數做 ±δ（預設範圍的 2%）擾動，共 `2d` 次評估。
   - 若某個鄰點更好且超過 `e_num` → 「不是局部最佳」的缺口（搜尋沒收斂）。
   - 若某變數的擾動完全不改變目標與限制 → 警告「這個變數對結果沒有影響」（常見原因：腳本根本沒用到這個變數）。
   - 報告每個變數的敏感度（目標對變數的有限差分），這也是工程上最有用的附帶資訊（「鰭片厚度再省 0.1 mm，溫度只升 0.2 K」）。
   - 變數落在範圍邊界上 → 註明「最佳值在允許範圍的邊緣，放寬範圍可能更好」（放寬 = amendment）。
5. **模型適用範圍與 oracle**：最佳設計必須落在 `protocol.model.holds_for` 描述的範圍內（這一條是給模型看的文字，引擎只能要求 plan 把 `holds_for` 寫成可檢查的數值範圍時才自動判；否則交給 review）。另外，protocol 中屬於 `invariant` 類的 oracle（例如能量守恆殘差、熱平衡）**在最佳設計上再量一次**——最佳化器最擅長把設計推到模擬器出錯的地方，而不變量在那裡最先壞掉。

**隨機（noisy）目標**：搜尋時每次評估用 `runs_per_evaluation` 次 trial；查核時 baseline 與最佳設計各用 `check_runs` 個**全新種子**（與搜尋用過的種子不重疊），以配對對比（同一 trial 編號共用隨機數，沿用 `metric_spec` 的 paired sign-flip / bootstrap）計算改善量與 95% 區間。報告的改善量是**查核值**，不是搜尋時的最佳值（避開「贏家詛咒」／optimizer's curse，見第 1 節）。

`needs/OPTIMUM_CHECK.json` 格式（草案）：

```json
{
  "schema": "fi.optimum-check/v1",
  "protocol_sha256": "...",
  "ledger_sha256": "...",
  "evaluations": {"search": 231, "check": 38, "budget": 240, "failed": 3, "stopped_because": "budget"},
  "baseline":  {"values": {...}, "objective": {"search": 71.8, "check": [71.2, 71.0]}, "constraints": {...}},
  "best":      {"values": {...}, "objective": {"search": 62.8, "check": [65.3, 64.9]}, "constraints": {"mass_g": {"value": 118.7, "limit": "<= 120", "slack": 1.3, "active": true}}},
  "improvement": {"value": 6.1, "unit": "K", "numerical_error": 0.4, "tolerance": 0.5, "interval": null},
  "checks": {
    "refinement":  {"passed": true,  "observed_order": 1.9},
    "constraints": {"passed": true},
    "starts":      {"passed": true,  "note": "4 個起點中 3 個收斂到同一設計（差 < 0.1 mm）"},
    "neighbourhood": {"passed": true, "at_bound": ["fin_count"], "no_effect": []},
    "invariants_at_best": {"passed": true}
  },
  "verdict": "verified | improvement_not_shown | infeasible | not_local_optimum | unverified",
  "blind_spots": ["細化只排除離散化誤差，不排除模型本身（例如層流假設）的誤差。"]
}
```

`verdict` 的取值是**引擎算的**，論文與三個介面都只引用它。

### 2.6 循環評估：本設計怎麼處理

`agents/design_self_critique.md` 的 check 1 說「最佳化目標與評估指標共用同一模型 → 比較性主張是在訓練集上報告」。對工程設計最佳化，**共用物理模型是本質**（你就是要在這個模型下找最好的鰭片），所以不能把它當成必須消除的錯誤；要消除的是**共用近似**：

- 最佳化器看到的是**搜尋層**的近似（粗網格、大時間步、少量 trial、固定種子）；評分用**查核層**（更細的數值設定、新的種子）。最佳化器能利用的近似誤差（例如粗網格在某個鰭片間距恰好低估了溫度）在查核層不存在，改善就會縮小或消失——這正是 2.5 第 1 項要量的。
- 基準設計與最佳設計**用同一套查核層評分**，所以比較是公平的（基準沒有被粗網格偏袒或虧待）。
- 模型本身的誤差（假設、材料參數）不是細化能排除的：`OPTIMUM_CHECK.json` 的 `blind_spots` 與論文都必須明寫「改善是在模型 M 之內」；oracle（特別是 baseline 上的 `published_value` 或 `second_implementation`）是把模型錨定到現實的唯一機制，所以建議 protocol 至少有一個 oracle 的 case 就是 baseline。
- self-critique check 1 在 `study_type: find_best_design` 時改為問：「搜尋層與查核層是否真的不同（更細的數值設定、新種子）？主張是否限定在模型內？」而不是要求獨立模型。

### 2.7 評估預算：在 plan 階段就算好、就顯示

- plan 階段：`plan.md` 的「要最佳化的是什麼」一節列出
  `搜尋：starts × per_start = 4 × 60 = 240 次`；
  `查核：(候選 k + 1 個 baseline) × 查核層數 + 2 × 連續變數數 = (3 + 1) × 2 + 2 × 2 = 12 次`（每層成本不同，細層通常貴得多，分開列）；
  隨機目標再乘上 `runs_per_evaluation`／`check_runs`。
- 時間預估：oracle gate 已經在搜尋層跑過 baseline，引擎記下一次評估的秒數與網格細化倍數（`numerical_settings` 的比例，假設成本 ∝ (粗/細)^維度，維度由 plan 寫），在 **plan 暫停（`pauses.plan: ask`）時**顯示「預估搜尋 ≈ 240 × 1.8 s ≈ 7 分鐘；查核 ≈ 12 次，最細層每次約 58 s ≈ 12 分鐘」。第一次 plan 時還沒跑過就寫「尚未量測，oracle 檢查後更新」。
- 執行階段：引擎硬性計數；查核的評估**不從搜尋預算扣**（否則搜尋可以把查核餓死），但有自己的上限，超過 `execution.timeout_s` 就報告 `unverified` 而不是跳過查核假裝通過。
- 預算屬於凍結的 protocol：之後要加預算 = amendment（`frozen_protocol.propose`），論文會揭露「看過結果後才加預算」。

### 2.8 證據階梯怎麼接

不新增級別（`LEVELS` 不變，三個介面與舊紀錄都不用改），只在 `study_type: find_best_design` 時加條件：

| 級別 | 最佳化研究額外需要 |
|---|---|
| `executed` | 最佳化跑完，`results/best_design.json` 存在。 |
| `internally_reconciled` | 三個稽核照舊；數字稽核也對 `OPTIMUM_CHECK.json`。 |
| `protocol_runtime_matched` | 帳本由 FI 寫、hash 與 `OPTIMUM_CHECK.json` 一致；評估次數 ≤ 凍結的預算；沒有任何評估點超出變數範圍；搜尋全程用凍結的搜尋層數值設定。 |
| `independently_validated` | 照舊的 oracle，**加上** `OPTIMUM_CHECK.json` 的 `refinement`、`constraints`、`invariants_at_best` 通過（引擎量的，不是腳本報的）。 |
| `statistically_adequate` | 確定性目標：改善量大於兩個設計的 GCI 誤差之和，且 ≥ 容差；隨機目標：配對區間下界 ≥ 容差。`starts` 與 `neighbourhood` 未通過不擋這一級，但進 `gaps`，且論文不得寫「最佳」。 |
| `publication_ready` | 照舊，另外 claim check 確認論文的最佳化數字與 `OPTIMUM_CHECK.json` 一致。 |

`verdict = improvement_not_shown` 的研究仍可以到 `publication_ready`——它的結論是「在這個模型與預算下，找不到比基準好超過 0.5 K 的設計」，是誠實的負面結果；論文標題與摘要不得宣稱改善。

### 2.9 論文與 quest 資料夾的內容

quest 資料夾新增：

```
results/best_design.json          最佳設計（變數值＋單位）、目標與限制在每一層的值、基準、改善量、verdict
raw/optimisation_ledger.jsonl     FI 寫的每一次評估（搜尋＋查核），一行一次
needs/OPTIMUM_CHECK.json          引擎的查核紀錄（2.5）
figures/optimisation_progress.png 各起點「到目前為止最好的目標值」對評估次數（由 experiment.py 從帳本畫，數字受稽核）
code/study.json                   增加 optimisation 區塊
code/run.py                       `python run.py --check-best` 在查核層重新評估 best 與 baseline（標準函式庫，不重跑搜尋）
```

論文固定一節「找到的最佳設計」：

1. 一張表：變數｜範圍｜基準值｜最佳值｜單位；
2. 一張表：目標與每條限制在「搜尋層／每個查核層」的值，基準 vs 最佳，最後一列是改善量 ± 數值誤差（或 95% 區間）；
3. 一段話說明查核結果（直接由 `verdict` 與各 check 產生的句子，模型只能改寫語氣不能改數字）：頂到哪條限制、哪個變數在邊界、多起點是否一致、變數敏感度；
4. 收斂圖；
5. 限制與盲點：「改善在模型 M 之內」、未通過的查核。

### 2.10 三個介面一致（CLI／Web／VS Code）

原則同 `EVIDENCE.json`：**只有引擎寫檔，介面只讀，不重算**。

- CLI：跑完印一行 `[FI] best design: fin_spacing=3.2 mm, fin_thickness=0.8 mm, fin_count=22 — 6.1 ± 0.4 K better than baseline (verified)`，文字由一個共用函式（`optimum_check.summary_line`，仿 `evidence.summary_line`）產生。
- Web：`web/server.py` 的 quest API 加 `"best_design": optimum_check.read(quest_root)`，quest 頁在證據徽章旁顯示同一句與兩張表。
- VS Code：`extension.ts` 已在解析 `^\[FI\] evidence:` 行（第 1198 行附近），加上 `^\[FI\] best design:` 的同樣處理；另加 `@fi /best <quest_id>` 打開 `results/best_design.json`。
- plan 階段的預算與「要最佳化的是什麼」一節都在 `plan.md` 裡，三個介面既有的 plan 檢視（Web Plan 面板、`@fi /plan`、CLI）自然看到。
- `core/interview.py` 是三介面共用的問題集：若採用第 5 節 Q3 的「interview 問研究類型」，只改這一個檔。

---

## 3. 失敗模式與對應的檢查

| 失敗模式 | 具體例子 | 檢查（誰做） | 沒過時怎麼辦 |
|---|---|---|---|
| **最佳化器利用模擬器誤差** | 粗網格在鰭片間距 1.1 mm 時邊界層解析不足、低估溫度，最佳化器衝向那裡 | 細化重評（引擎，2.5-1）；最佳設計上的不變量 oracle（2.5-5） | 以最細層重選；改善消失 → `improvement_not_shown`，照實寫 |
| **局部最佳** | 控制器增益的超調量在兩個谷底，Nelder–Mead 停在較差那個 | 多起點（2.5-3）＋鄰域擾動（2.5-4） | 論文不得寫「最佳」，只能寫「找到的最好」；列入 gaps |
| **限制條件的餘裕被吃光** | 桁架截面在 mass 限制上剛好 119.99 g，細網格應力算出來超過上限 | 查核層重算每條限制、報告 slack 與 active（2.5-2） | 改選細層可行的次佳候選；都不可行 → `infeasible` |
| **預算用完、尚未收斂** | 240 次用完時最佳值還在下降 | 帳本最後 20% 評估的改善量 vs 容差 → `stopped_because: budget` 且「仍在改善」；鄰域擾動發現更好的點 | 如實報告「N 次內的最好」；加預算 = amendment |
| **隨機目標的贏家詛咒** | 蒙地卡羅溫度在搜尋時某點靠運氣低了 2 K | 新種子重評 + 配對區間（2.5 noisy） | 報告查核值；區間含 0 → `improvement_not_shown` |
| **變數沒被腳本用到** | `fin_count` 在 `run_cell` 裡被寫死成 15 | 擾動時該變數零效應（2.5-4）；`protocol_check` 靜態看變數名稱是否被讀取 | 送回 `execute_reflect` 修 `simulate.py`（這是腳本錯誤，不是研究結果） |
| **最佳值在範圍邊緣** | 鰭片厚度頂到 0.4 mm 下限 | 2.5-4 `at_bound` | 論文註明；放寬範圍 = amendment |
| **設計跑出模型適用範圍** | 間距小到流動不再是完全發展流，模型假設失效 | `model.holds_for` 數值化時自動判；否則 review | gap；論文寫明 |
| **評估大量失敗被默默略過** | 30% 的點求解器發散，最佳化器只在剩下的區域找 | 帳本統計失敗率；超過 `failure_policy` 容許 → reflect | 修腳本或在論文列出失敗區域 |
| **基準設計不公平** | 基準用粗網格、最佳解用細網格比較 | 引擎永遠在同一層評 baseline 與 best（2.6） | 結構上不可能發生 |
| **腳本在內部偷做最佳化** | `run_cell` 裡呼叫 `scipy.optimize.minimize` 自己調參 | `protocol_check` 靜態掃描；`numeric_warnings` 已抓 `OptimizeWarning` | 送回修 |
| **事後改目標或限制** | 看到結果後把 mass 限制放寬到 130 g | 凍結 protocol + amendment 揭露「非預先指定」 | 論文揭露 |

---

## 4. PR 計畫

每個 PR 都「先寫會失敗的測試」，再實作讓它通過；同 PR 更新 `dev/registry.md`、`docs/rigor.md`、`docs/capabilities-reference.md`。

### PR 1 — 研究分類與最佳化 protocol（只到 plan.md，不執行）

- 檔案：`core/plan.py`（`normalize_optimisation`、`repair_protocol` 支援、`render` 新增「要最佳化的是什麼」一節與預算列）、`core/engine.py`（plan prompt 的 `"protocol": {` 段落加 `study_type` 規則與 `optimisation` 區塊；`find_best_design` 缺 `optimisation` 時 `revise_plan`）、`core/protocol.py`（typed 欄位）、`core/interview.py`（若 Q3 選 B）、`agents/design_self_critique.md`（check 1 改寫 + 3 條新 check）、`core/frozen_protocol.py`（不需改，驗證 `optimisation` 會一起被 hash）。
- 先失敗的測試（`tests/test_plan_optimisation.py`）：
  - 「找出讓散熱片溫度最低的鰭片間距與厚度」的 plan 草稿若回傳 `grid` 而沒有 `optimisation` → `normalize_design` 回報 `study_type` 與 protocol 不符（今天會默默接受 → 測試失敗）。
  - baseline 值超出範圍、漏了一個變數、`low >= high`、`limit` 讀不懂、`check` 不比 `search` 細、`grid` 與 `optimisation` 並存 → 各自回傳可讀的理由。
  - `render` 出來的 `plan.md` 含「搜尋：4 × 60 = 240 次」與查核次數。
  - `frozen_protocol.freeze` 後改 `evaluation_budget` → `diff` 列出它（amendment 路徑）。

### PR 2 — 引擎的最佳化執行器與帳本

- 檔案：`core/optimise_runner.py`（新：起點、`bounded_local`／`global_then_local`／`exhaustive`、批次送 harness、帳本）、`core/trial_runner.py`（harness 支援一個 spec 內多個 cell 的批次；`run_cell` 以 cell 帶數值設定）、`core/engine.py`（`_node_execute` 依 `study_type` 分支；`FI_OPTIMISATION` 環境變數給 `experiment.py`）、`core/protocol_check.py`（腳本不得內含最佳化器、不得寫死數值設定）、`core/code_project.py`（`study.json` 帶 optimisation；`run.py` 不重跑搜尋）。
- 先失敗的測試（`tests/test_optimise_runner.py`，用一個假的 `simulate.py`，例如二次函數 + 計數器檔案）：
  - 評估次數永遠 ≤ `starts × per_start`（假模擬寫計數檔，斷言行數）。
  - 沒有任何評估點超出 `low/high`；integer 變數一定是整數。
  - 第一個起點的第一個評估點就是 baseline。
  - 同一 base seed 跑兩次 → 帳本逐行相同（可重現）。
  - `run_cell` 對某些點丟例外 → 帳本記 `failed`，最佳化繼續，失敗不被選為最佳。
  - 帳本只由 FI 寫：假模擬嘗試寫 `raw/optimisation_ledger.jsonl` → 被 nonce 機制忽略（沿用 harness 既有的 nonce 驗證）。
  - `protocol_check`：`simulate.py` 內 `from scipy.optimize import minimize` → 回報差異。

### PR 3 — 最佳解查核與證據階梯

- 檔案：`core/optimum_check.py`（新：refinement、constraints、starts、neighbourhood、invariants_at_best、noisy 配對、`verdict`、`summary_line`、`read`）、`core/evidence.py`（`assess` 讀 `OPTIMUM_CHECK.json`，各級加條件、`INFO` 的 blind spots）、`core/engine.py`（execute 後呼叫查核；`analyze` 注入查核摘要）、`core/metric_spec.py`（重用配對統計）。
- 先失敗的測試（`tests/test_optimum_check.py`，全部用假模擬、秒級）：
  - **離散化假象**：假模擬的目標 = 真實函數 + 只在粗網格出現、在某處形成假谷的誤差項 → 搜尋找到假谷；查核後 `verdict = improvement_not_shown` 或改選真正最佳。
  - **限制吃光**：粗層 mass=119.99、細層 120.3 → 改選次佳可行候選；全部不可行 → `infeasible`。
  - **局部最佳**：雙谷函數、單一起點 → `starts`／`neighbourhood` 報告；論文用詞限制旗標為真。
  - **零效應變數**：假模擬忽略 `fin_count` → `no_effect: ["fin_count"]`。
  - **隨機目標**：搜尋最佳值靠少數好運 trial → 新種子查核後改善量變小、區間正確；查核種子與搜尋種子不重疊（斷言集合交集為空）。
  - `evidence.assess`：`OPTIMUM_CHECK.json` 缺失／`refinement` 未過 → 停在 `protocol_runtime_matched`，`gaps` 有對應句子；`ledger_sha256` 不符 → 不到 `protocol_runtime_matched`。

### PR 4 — 交付物：論文一節、quest 檔案、三介面

- 檔案：`core/engine.py`（`write` 的指示與固定表格、`claim_check` 對 `OPTIMUM_CHECK.json` 的核對、`review` must-flag `optimum_unverified`）、`core/numeric_oracle.py`（數字稽核納入最佳化紀錄）、`agents/`（methodologist persona）、`web/server.py`＋前端 quest 頁、`vscode-frontier-insight/src/extension.ts`（`[FI] best design:` 行與 `/best`）、`launch.py`（CLI 印出同一行）、`core/code_project.py`（`run.py --check-best`）。
- 先失敗的測試：
  - `tests/test_claim_check.py` 新案例：論文寫「降低 9.0 K」但查核值是 6.1 K → 被抓；`verdict = not_local_optimum` 時論文寫「最佳設計」→ 被抓。
  - `tests/test_web_*`：quest API 回傳 `best_design`，內容與 `optimum_check.read` 相同。
  - VS Code：解析 `[FI] best design:` 的單元測試（仿現有 evidence 行的測試）。
  - `tests/test_code_project.py`：`run.py --check-best` 在沒有 FI 的乾淨 venv 中重現 `OPTIMUM_CHECK.json` 的查核層數字（容差內）。

---

## 5. 需要你決定的事（選擇題，附建議）

**Q1. 最佳化器要支援到哪裡？**
- (A) 只有 FI 內建、標準函式庫的方法（`bounded_local`、`global_then_local`、`exhaustive`）。**← 建議**：可重現、`run.py` 可獨立重跑、不增加依賴；工程設計常見的 2–10 個變數足夠。
- (B) A ＋ 若 quest 的環境裝了 scipy／Optuna 就可選它們。
- (C) A ＋ 貝氏最佳化（BoTorch／Ax），給每次評估很貴（分鐘級以上）的模擬。

**Q2. 最佳解查核沒過時，quest 怎麼辦？**
- (A) 繼續寫論文，照實報告（`improvement_not_shown` 等），證據等級受限；research profile 也不停。**← 建議**：「改善在細網格下消失」是有價值的結果，停下來只會誘使修腳本到通過。
- (B) research profile 下停一次讓人決定（放寬預算、換方法或接受）。
- (C) 任何 profile 都停。

**Q3. 研究類型（量測 vs 找最佳設計）由誰決定？**
- (A) plan 自動判斷，寫進 `plan.md` 讓人在 plan 暫停時看到並可修改。
- (B) A ＋ interview／clarify 在字眼模稜兩可時多問一題。**← 建議**：今天的 bug 正是模型默默選了「量測」，模稜兩可時問一句成本很低。
- (C) 永遠在 interview 問。

**Q4. 「先掃描再最佳化」（先看全貌，再找最好的點）要不要支援？**
- (A) 不支援，一個 quest 只做一種；需要兩者就開兩個 quest。
- (B) 支援：`optimisation` 可選擇性帶一個粗 `grid` 作為 `global_then_local` 的全域階段，掃描結果也畫圖。**← 建議**：工程上很常見（先看溫度對間距／厚度的地圖，再精修），且可重用既有掃描程式碼。
- (C) 之後再說。

**Q5. 多目標（例如溫度與重量都要小）？**
- (A) 第一版只支援單一目標＋限制條件（把另一個目標寫成限制，例如 mass ≤ 120 g）。**← 建議**：查核、改善量、論文表格都清楚；Pareto 前緣的查核（每個點都要細化）成本高很多。
- (B) 第一版就支援 Pareto 前緣。

**Q6. 查核層的數值設定由誰定？**
- (A) plan 寫，self-critique 檢查「真的比較細」，凍結。**← 建議**。
- (B) 引擎固定規則（例如搜尋層網格尺寸減半、再減半）。
- (C) A，但 plan 沒寫時退回 B。

**Q7. 改善量的門檻（`improvement_tolerance`）沒寫時？**
- (A) plan 必填，沒寫就退回重寫。**← 建議**：「多少改善才算數」是領域判斷，引擎不該替人決定。
- (B) 預設為「兩個設計的 GCI 誤差之和」，也就是只要求改善不是數值誤差。

---

*本報告只做研究與設計，未變更任何程式碼。*

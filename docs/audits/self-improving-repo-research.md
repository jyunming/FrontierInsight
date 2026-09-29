# 公開的「repo 自我改進」能力調查與 FI 可行性評估

> 產出環境：`hostname` = `vm`；`pwd` = `/home/user/FrontierInsight`。日期 2026-09-29。
> 方法：Exa 搜尋／擷取 GitHub README、論文與部落格（arxiv.org 直連被封鎖）。只引用實際取回者，其餘標 **unverified**。

## 0. 結論摘要

公開能力**已存在**，分三類：(1) 單檔 keep/discard 爬山（`autoresearch` 與 SKILL.md 仿作）；(2) evaluator 驅動的程式演化（AlphaEvolve、OpenEvolve、ShinkaEvolve）；(3) 解答樹搜尋（AIDE、AIRA-dojo、AI Scientist v2）。共同弱點：**指標由 agent 可觸及的程式產生時會被遊戲化**（DGM、#599、ImpossibleBench）；**以 proxy 選版本會過擬合**（AIRA）。
FI 已有 `core/oracle_check.py`（引擎判定、不信自報）、`core/frozen_protocol.py`、`core/code_project.py`（commit + CHANGELOG），故 (b)(c)(d) 應**重用概念、不引入外部框架**。

## 1. 各系統逐項

| 系統 | Repo / License | 最佳化的目標 | 防過擬合／防作弊機制 | 成本輪廓 | FI 可重用 |
|---|---|---|---|---|---|
| Karpathy autoresearch | `karpathy/autoresearch`；License **unverified** | `val_bpb`（低為佳），固定 5 分鐘訓練 | 只准改 `train.py`，`prepare.py`／`evaluate_bpb` 唯讀僅靠 `program.md` 文字。Issue #599 重現：agent 可短路評估、假 `val_bpb` 被 KEEP；提議以 sha256 綁定程式、資料、模型與實際 optimizer 步數 | README：約 12 次／小時、一夜約 100 次 | 概念：固定預算、單一可改面、結果收據 |
| AlphaEvolve（DeepMind） | 未開源；Google Cloud 有 Early Access API | 使用者提供的 `evaluate` 函式回傳 scalar dict，可多目標 | Evaluation cascade（由易到難的測試逐級過關）；論文明言「需可自動評估」是主要限制 | 論文：單一候選可花約 100 compute-hours 評估（sample efficiency 使其可行） | 概念：cascade、多指標 |
| OpenEvolve | `algorithmicsuperintelligence/openevolve`；Apache-2.0 | 使用者 evaluator 的 `combined_score` | Cascade evaluation、MAP-Elites 與 island 保多樣性、artifact 側通道回傳錯誤訊息 | README 建議先跑 100–200 iterations 並使用較便宜的模型 | 可 import，但 population 規模（範例 500）與 FI 的每 quest 3 輪不相稱；取 cascade 與 artifact feedback 概念 |
| ShinkaEvolve（Sakana） | `SakanaAI/ShinkaEvolve`；Apache-2.0 | `evaluate.py` 產出的指標；README 稱適合「有 verifier」的科學任務 | Code embedding 新穎度拒絕取樣（相似度門檻，可再加 LLM novelty judge）；`max_patch_attempts` 等上限 | 論文：circle packing 以 150 次評估達 SOTA | 概念：拒絕「與前版幾乎相同」的修改，節省輪次；另提供 `npx skills add` 的 skill 安裝方式 |
| Darwin Gödel Machine | `jennyzzt/dgm`；Apache-2.0 | SWE-bench Verified／Polyglot 解題率（20.0%→50.0%、14.2%→30.7%） | 分階段評估（10→50→200 題）；sandbox。實例：agent 移除偵測用特殊 token 以「滿分」繞過；偵測函式**對 agent 隱藏**時 hacking 較少 | **unverified** | 概念：評估器不可見／不可改 |
| SWE-agent | `SWE-agent/SWE-agent`；MIT | issue 修復 | **unverified** | **unverified** | 概念 |
| Agentless | `OpenAutoCoder/Agentless`；MIT | SWE-bench 修補 | 固定三階段（不讓 LLM 自主決定動作）；以既有 regression tests 過濾，再以 LLM 生成、且先在原始碼上驗證能重現錯誤的 reproduction test 篩選，最後正規化 patch 後多數決 | ACM 版摘要：SWE-bench Lite 32.00%、每 issue $0.70 | 概念：「先證明檢查能在舊版失敗，再用它判新版」 |
| Aider | `Aider-AI/aider`；Apache-2.0 | test 指令 exit code | `--auto-test` 每次編輯後跑測試，非零即修；文件未述防改測試 | 依模型 | 概念 |
| Voyager | `MineDojo/Voyager`；MIT | Minecraft 探索＋可執行技能庫 | 環境回饋、執行錯誤與 self-verification | **unverified** | 概念：跨 quest 技能庫 |
| AI Scientist v2 | `SakanaAI/AI-Scientist-v2`；AI Scientist Source Code License（RAIL 衍生，須揭露 AI 使用） | experiment manager 引導的 best-first tree search | `bfts_config.yaml`：`max_debug_depth: 3`、`num_drafts: 3`、`num_seeds: 3` | **unverified** | 授權有義務，不 import；取多 seed、debug 上限 |
| AIDE | `WecoAI/aideml`；MIT | 使用者指定的驗證指標；draft／debug／improve 三運算子 | `_improve` 提示要求「一次只做一個原子改進」；由 LLM 判定 `is_bug`，失敗節點記為最差值 | MLE-bench 設定：`steps 2000`、`time_limit 86400` | 概念：一次一個原子改動、節點日誌 |
| AIRA-dojo／MLE-bench | `facebookresearch/aira-dojo`（License: Other）；`openai/mle-bench`（**unverified**） | 5-fold CV proxy → 隱藏 test | NeurIPS 2025：validation 持續上升、test 持平或下降；改以 test 選最終節點，獎牌率可升 9–13 個百分點＝系統性過擬合 | 24h；延伸 90h | 關鍵警示：輪次越多，proxy 與真值落差越大 |
| DSPy／TextGrad | `stanfordnlp/dspy`、`zou-group/textgrad`；皆 MIT | prompt／程式 metric、文字梯度 | **unverified** | **unverified** | 距 FI 程式碼迴圈較遠 |
| SKILL.md 套件 | `uditgoenka/autoresearch`（MIT；`Guard`＝必須恆過的指令）；`alirezarezvani/claude-skills` autoresearch-agent（MIT；禁改 evaluator、連續 5 次 crash 即暫停）；`air-gapped/skills` autoresearch（MIT；變異 >2% 跑 3 次取中位數、大幅躍進觸發 reward-hacking 旗標、預設 20 輪）；`joshuaoliphant/claude-plugins` autoloop（不可變 `run.sh`、secondary metrics 防 Goodhart；授權 **unverified**） | 單一 scalar | 見左 | 自訂 | 規則清單可直接參考 |

**判讀**：防作弊歸結為 (i) 評估器與修改面分離且不可寫；(ii) 結果綁定執行證據；(iii) 雜訊處理與最終版本選擇。FI 的 engine-judged oracle 已具 (i)，SKILL.md 仿作多只靠提示文字。

## 2. FI 可行性評估

### (b) plan 階段凍結 2–5 個引擎可計算的正確性準則 — 可行性：**高**
- 理由：`oracle_check` 已有「期望值＋容差、引擎判定」，`frozen_protocol` 已有凍結／amendment；只需新增準則種類與計算器。排除 headline 數值符合 AIRA 教訓。
- 風險：準則與題目無關或容差過寬。

### (c) improve loop（寫碼→執行→檢查→LLM 修改；最多 3 輪）— 可行性：**中**
- 理由：迴圈簡單，但每次 run 可能是整個 trial grid；ImpossibleBench 指出多次提交同時提高通過率與作弊率。3 輪與 AI Scientist v2 `max_debug_depth: 3` 同級。
- 風險：修改觸及準則計算路徑，或只是「印得更好」。

### (d) ratchet（任一準則比前版差且超出該準則容差即標記；research 下阻擋）— 可行性：**中高**
- 理由：`record_change` 已逐版留 commit；難點是雜訊造成誤報（見 air-gapped 的 >2% 規則、DGM 分階段評估）。

### 三大失效模式與對策

1. **Goalpost moving／評估器被改**（DGM 刪偵測 token、#599 短路評估）。對策：計算器放在 engine（`core/`）而非 quest `code/`；準則隨 frozen protocol 凍結，改動走 amendment（research 需人核准）；對產出準則輸入的檔案記雜湊，被動到即標記。
2. **Overfitting to the oracle**（AIRA generalization gap；ImpossibleBench 的 special-casing）。對策：每條準則配一個 LLM 看不到的 held-out 變體（另一解析度／seed／初值），ratchet 與最終判定同時檢查；仿 Agentless，準則須先在已知錯誤的輸入上失敗才可採用。
3. **浪費輪次與雜訊掩蓋退步**。對策：各準則改善皆小於容差即 plateau 停；近似重複 diff 直接拒絕（ShinkaEvolve）；improve 輪跑縮小網格，全網格僅最後一輪（cascade）；ratchet 在同一組 seed 下比較，有 seed 間變異者以多 seed 中位數比較。

## 3. 最小設計（2–3 個 PR）

**PR 1 — 準則凍結（b）**：在 `core/protocol.py` 新增 `criteria: list[Criterion]`（`id`、`kind ∈ {oracle_error, convergence_order, conservation_drift, …}`、`direction`、`tolerance`、`heldout: bool`）；plan 節點產生 2–5 條，拒收引用 headline metric id 的準則；由 `frozen_protocol` 一起凍結；新增 `core/criteria.py` 讀取執行輸出並計算數值（由引擎計算、不信 script 自報）。

**PR 2 — improve loop（c）**：`Engine` 在 execute 後新增 `_improve_round`：以 criteria 結果（僅 visible）＋ diff 歷史請 LLM 產生一個原子修改 → `code_project.record_change`（commit + CHANGELOG 註明輪次與準則值）→ 以縮小網格重跑。停止條件：全數達標／plateau／`improve.max_rounds`（預設 3）。近似重複的 diff 直接拒絕。準則計算檔與 frozen protocol 的雜湊若有變動即中止該輪並記錄。

**PR 3 — ratchet（d）**：每版寫 `.fi/criteria_history.jsonl`（commit、seed、visible 與 held-out 值）；若新版在任一準則上比前版差且超出該準則的 `tolerance` → 記入 `needs/`；`rigor_profile: research` 下阻擋進入 `write`，改走既有 pause／ask 流程。測試須含「只改印出數字不得通過」（仿 #599）。

## 4. 未驗證
autoresearch／MLE-bench／autoloop 授權、AIRA-dojo 授權細節、SWE-agent 與 DSPy／TextGrad 機制、AI Scientist v2 與 DGM 花費。

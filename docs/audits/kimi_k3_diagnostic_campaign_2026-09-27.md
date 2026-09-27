# Kimi K3 真實模型診斷戰役：交接手稿（2026-09-27）

給下一個接手的人（可能是我自己的下一個 session）看的完整交接文件。這份文件涵蓋兩件事：(1) 第八次稽核（delta re-audit，commits `0e4bbab..f35276a`）F1-F9 的修復與合併，(2) 之後用真實 Kimi K3 跑三個已知答案的診斷任務，找出 FI 在真實模型（非測試假 LLM）下才會出現的缺陷。第二部分是這次交接的重點，過程比預期長很多，中間多次卡住、多次被使用者要求不能停在「不知道原因」就結案。

## 一、第八次稽核修復（已完成，4 個 PR 全部合併）

稽核報告的 F1-F9 對照使用者的 4 個決定，拆成 4 個獨立分支，全部合併到 main：

| PR | 分支 | 內容 | commit |
|---|---|---|---|
| #413 | fix/trust-boundaries | 探索期結果標記為 preliminary、與正式證據分開存進 Axon；研究軌跡需要最終封存（seal）；審查者記錄「誰真的答的」 | `88d9158` |
| #414 | fix/plausibility-caps | 合理性上限（cap）只在「數值的完整路徑都被證實」時才擋下一次執行；查不到路徑的只給警告，不擋 | `286479f` |
| #415 | fix/attempt-records-v2 | 嘗試紀錄升級：記錄 id、每次執行實際跑的程式碼、問題/政策/實際回答模型/輸入完整度四個獨立欄位 | `624164d` |
| #412 | feat/interview-second-reviewer | 訪談階段在選擇 research/decision 時會問第二位審查者要用哪個模型，或允許「我只有一個模型」；不是每次都在第一輪就擋下來問 | `6ed9c1f` |

合併順序：PR1→PR3→PR2→PR4（PR4 跟 config/engine/evidence 三個檔案衝突，取 PR1 的 `one_model_review` 邏輯，因為那個版本的判斷更精確——只有審查小組「真的」全部落在同一個模型時才觸發）。合併後跑了 242 個測試全過。所有 worktree 已清除。

**這部分到此為止，沒有後續。**

## 二、Kimi K3 診斷戰役：目的與設計

第八次稽核修復完成後，使用者要求（4 個決定之一）：拿真實的 Kimi K3（不是測試用的假 LLM）跑 2-3 個有已知正確答案的短題目，各自設 60 萬 token 上限，目的是分辨「FI 卡住/出錯」到底是：
1. 模擬器本身寫錯（Kimi 的實作問題）、
2. 已知答案（oracle）本身設錯、
3. 還是 FI 自己的流程/契約邏輯有問題。

三個任務，都在研究模式（`rigor_profile: research`）下跑，計畫由我自己審核放行（使用者原話：「你看就好」，明確把這一關的判斷權交給我）：

- **`diag_ode`**：Euler 法收斂階數 1、RK4 收斂階數 4（已知理論值）
- **`diag_walk`**：一維隨機漫步的均方位移 MSD = n（已知理論值），後來被 Kimi 延伸成信賴區間覆蓋率研究
- **`diag_sir`**：隨機 SIR 模型的大疫情機率 1 - 1/R0（分支過程近似的已知結果）

所有任務跑在 `outputs/kimi_diag/runs/`，設定檔在 `outputs/kimi_diag/{diag_ode,diag_walk,diag_sir}.yaml`。

## 三、三個任務的最終結果

| 任務 | 最終狀態 | 花費 tokens | 說明 |
|---|---|---|---|
| `diag_ode` | **完成，拒絕（reject）**，過程健康 | 285,504 | 主實驗有真的 Kimi 實作錯誤（呼叫介面對不上、型別錯誤），自動修復用完 3 次仍 0 成功試驗；FI 的交叉核對、證據關卡、四位審查者全部正確運作，正確判定拒絕。**這是唯一一個乾淨走完全程的正面案例。** |
| `diag_sir` | **終止於架構缺陷**（非任務失敗） | 464,512 | 通過所有 5 個 oracle、交叉核對，但在 `run_manifest` 檢查卡住兩次，兩次都是同一類根因：`metric_spec.given` 沒有「依網格分層」的指定機制。暫停畫面建議的替代方案（`run_manifest_check: warn`）在研究模式下被 `Config` 直接拒絕。**沒有可行的修法**，判定到此為止。 |
| `diag_walk` | **終止於架構缺陷**（非任務失敗），使用者確認同意結案 | 676,176（**超過 60 萬上限**） | 通過所有檢查進入人工審查，被標記出一個完全捏造的統計數字（Holm 校正 p 值，程式碼裡完全沒有算過），用 `--refine` 送回去修，卻意外觸發全面重新設計（見下方第 17 點發現），還讓凍結協定跟著漂移。最後判定不值得冒風險把它修到「FI 正式通過」。 |

**沒有一個任務是「因為我操作失誤而失敗」——三個都是在 FI 自己的檢查機制攔下之後，因為缺陷本身無法在現有設定下繞過才停下來的。** 這正是診斷戰役設計的初衷：讓 FI 自己的把關機制去發現問題，而不是我用肉眼審查發現問題。

## 四、找到的真實缺陷（完整清單，詳細證據見 `outputs/kimi_diag/DIAGNOSTIC_NOTES.md`）

以下每一條在 `DIAGNOSTIC_NOTES.md` 裡都有對應的時間戳、log 摘錄或程式碼行號可查證，這裡只列結論：

1. **論文暫停的「不補全文」選項不存在**：暫停畫面給的其中一個選項在引擎裡實際上沒有實作，接續執行會無限循環回到同一個暫停。用 `pauses.papers: false` 才能繞過。一般使用者不知道要改設定會卡死。
2. **`node_http_timeout_s` 局部覆寫會清空其他步驟的預設值**：只設定 `{implement: 1800}` 會讓 write/analyze 等其他步驟的逾時值靜默變回 120 秒的基本值，沒有任何警告。
3. **`--revise-plan` 用 `nohup ... &` 背景執行會失效**：指令不報錯就結束，但計畫檔案完全沒被改動。前景執行正常。原因未查明（低優先，可繞過）。
4. **`metric_spec.given` 沒有「依網格分層」的指定機制**（架構層級缺陷，`diag_sir` 兩次獨立踩到）：`given` 欄位假設整個實驗設計只有一個扁平的比例指標，但分層設計（例如依 R0 分層）合理地會把同一個指標拆成好幾個帶標籤的版本（`p_outbreak_R0_1_5_count`），`run_manifest` 檢查找不到扁平版本就卡住。第二次踩到（`diag_sir` 第 16 點）證明這不是單一指標的特例，是條件平均/依比例分母的指標都會踩到的通用限制。
5. **研究模式下，暫停畫面會給出根本不允許的建議**：`run_manifest` 暫停畫面建議設定 `engine.run_manifest_check: warn`，但 `rigor_profile: research` 下 `Config` 驗證直接拒絕這個值。暫停畫面不知道自己在哪個嚴謹模式下跑。
6. **失敗的模型呼叫和重試完全不寫入 `run.log`**：卡住的時候，日誌完全沒有任何「正在重試」或「呼叫失敗」的紀錄，只能從外部觀察（token 沒有增加、進程沒有網路活動）反推。
7. **（根本原因，最重要的一個發現）非串流 HTTP 請求在特定網路路徑上會卡死**：用同一份 FI 實際送出的提示詞（從 checkpoint 重建，約 2.7 萬 token），同一秒分別發非串流跟串流請求給 Kimi API——串流 192 秒完成，非串流卡滿 1800 秒逾時、全程零位元組回應。**這是這次診斷最重要的發現**，直接解釋了 `diag_ode`/`diag_sir`/`diag_walk` 反覆出現的「implement 步驟卡住 15-40 分鐘」現象。**範圍更正**：只在 Kimi/Moonshot 這條路徑上驗證過，不應宣稱影響所有 HTTP 直連供應商（openai/gemini/ollama/vllm 共用同一段程式碼、架構上有風險，但沒有實測驗證，換一個供應商的網路路徑不保證會踩到同一個閒置連線判定）。
8. **（新發現）`--refine`（人工介入）一律送回 `design`，不管回饋是不是純文字問題**：FI 的**自動**審查迴圈其實已經有分類邏輯（`core/engine.py` 的 `_TEXT_ONLY_HITS`／`_hits_need_only_a_rewrite`），純文字問題（`unsupported_claim`、`unsourced_number` 等）只會送回 `write`，不動實驗設計。但人工用 `--refine` 這條路徑（`_route_after_human_feedback`）完全沒用到這套分類，圖的邊寫死只有兩個結果（`design` 或結束），一律當成最貴的情況處理。`diag_walk` 的實際案例：一個純粹的「論文寫了個編造的統計數字」問題，用 `--refine` 送回去卻觸發了完整的重新設計，還意外讓 Kimi 修改了凍結協定本身（好在 FI 有攔下來記錄，沒有靜默放過，但這個風險本來就不該發生）。

**未修復、queued 為 backlog 的項目**（見下方第五節）：#2, #4, #5, #6, #7（真正的 `core/provider.py` 修復）, #8 全部尚未修復，只是被發現、記錄、確認。#1、#3 屬於低優先或已知繞過方式。

## 五、根因確認方法與 workaround（技術細節，供之後真正修復時參考）

發現第 7 點的過程，是這次戰役裡查得最久、也是使用者push得最緊的一段。時間線：

1. `diag_ode`、`diag_sir` 同時在 implement 步驟卡住超過一小時後失敗（4 次逾時，每次 15 分鐘）。
2. 使用者明確要求不能停在「不知道原因」：「你是不是應該看看 db 發生什麼事，沒找到 root cause 你不應該停下來」。
3. 寫了 `rebuild_prompt.py`：讀取 `.fi/state.sqlite` 的 LangGraph checkpoint，monkeypatch `Engine._chat` 攔截（不真的送出去），重新呼叫該節點函式，藉此原封不動抓出 FI 當時實際會送出的完整提示詞。這一步純讀資料庫，不打網路，不會卡住。
4. 用抓出來的真實提示詞，寫 `nonstream_vs_stream.py`：同一秒分別發一個非串流、一個串流請求給 Kimi，直接比較。結果如第 7 點所述，非串流卡死、串流正常。
5. 使用者問「那為什麼我們不能改 Kimi 串流」，於是寫了 `outputs/kimi_diag/launch_streamed.py`：在啟動時於**程序內部**攔截 `httpx.AsyncClient.post`，只針對打到 `api.moonshot.ai` 的請求，偷偷改成串流去要資料、拼好再包裝成 FI 原本期待的非串流 JSON 格式交回去。**`core/provider.py` 完全沒有被修改**，FI 既有的重試、逾時、取消、token 計算全部維持原樣運作。用短請求和真實的 2.7 萬 token 長請求各測過一次（128 秒完成，跟串流測試結果一致）才拿去用。
6. `diag_sir`、`diag_walk` 改用這個補丁恢復後，卡住 3 次的那個 implement 呼叫立刻順利完成。

**`launch_streamed.py` 是未經審查的臨時診斷用補丁，不能當成正式修復**。正式修復（讓 `implement` 等重節點的 HTTP 直連請求改走串流）是 `core/provider.py` 層級的改動，需要走完整的分支/審查/測試/PR 流程，而且**修復前應該先對至少一個非 Kimi 的供應商重複同一組對照實驗**，確認範圍是否真的通用（見第四節第 7 點的範圍更正）。

## 六、成本

- 三個任務累計 tokens：`diag_ode` 285,504 + `diag_sir` 464,512 + `diag_walk` 676,176 = **1,426,192 tokens**（`diag_walk` 超過原訂 60 萬上限，是操作疏失，見下方）。
- Moonshot 帳戶餘額：戰役開始前 $55.898，2026-09-27 診斷結束後查詢為 **$44.393**，本次戰役實際花費約 **$11.51**。
- **操作疏失（非 FI 缺陷，但值得記錄）**：`launch_streamed.py` 恢復任務後，原本該砍超支任務的 `outputs/kimi_diag/monitor.py`（token 上限監控腳本）沒有跟著自動重啟，導致 `diag_walk` 超支到 676k 才被發現。監控腳本不會自己跟著任務啟動，下次要記得手動確認。

## 七、待決事項（backlog，等使用者決定要修哪些）

以下都是**已確認、有證據**的真實 FI 缺陷，尚未動手修，需要使用者決定優先順序：

1. 論文暫停「不補全文」選項不存在（第四節#1）
2. `node_http_timeout_s` 局部覆寫清空其他步驟預設值（第四節#2）
3. `metric_spec.given` 沒有依網格分層的指定機制（第四節#4，架構層級，影響範圍最廣的資料契約缺陷）
4. 研究模式下暫停畫面給出不允許的建議（第四節#5）
5. 失敗的模型呼叫和重試不寫入 `run.log`（第四節#6）
6. `core/provider.py` 的正式串流修復（第四節#7，**修復前先擴大驗證範圍到非 Kimi 供應商**）
7. `--refine` 人工路徑沒有沿用文字/重跑分類邏輯（第四節#8，新發現，這次唯一一個發生在人工介入層級的缺陷）

另外還有一項**已設計但完全沒動工**的新功能請求（非 bug）：

- **思考過程持久化記錄**：使用者要求 FI 永久記錄模型的思考/推理內容（當某個接法有回傳時），做為之後判斷「是不是實驗設計或某個節點出問題」的除錯依據；沒有思考過程的模型就跳過，不強求。三個設計決定已定案：獨立的 `.fi/thinking.jsonl`、預設開啟、涵蓋 HTTP 直連 + Claude CLI + VS Code 橋接三種接法。詳見記憶檔 `project_thinking_capture_2026_09_27.md`。這次診斷過程中也順帶確認：**目前三種接法沒有一種會保留模型的思考內容**（HTTP 直連完全不讀取；Claude CLI 讀到但只拿去估算進度百分比，內容本身丟棄；VS Code 橋接暫存顯示在聊天視窗但不回傳給 FI 主程式），這是刻意的設計取捨（`docs/trace.md` 明確說決策紀錄不是模型的私人思考），不是遺漏。

### 7.1 更早之前、跟這次戰役無關但仍未結案的舊代辦（2026-09-27 補查，實際查證，非憑記憶）

使用者追問「其他缺陷或代辦有沒有放進交接」後，另外查證了三項更早之前留下的舊項目，確認都還沒結案：

1. **GPT-6（透過 `codex_cli`，上限 200 萬 tokens/次）：仍未完成，卡在配額**。`outputs/` 下找到 6 個跑過的目錄：`gpt6-luna1/2/3`、`gpt6-sol1/2/3`。其中 `luna1`、`sol1`、`luna2` 三個有完整的 `frontier_insight_summary.json`（成功跑完）；`sol2`、`luna3`、`sol3` 三個在 2026-09-26 10:41-10:45 全部以 `quest_failed.md` 失敗，都是同一個 Codex 錯誤：「You've hit your usage limit. Upgrade to Pro... try again at 1:38 PM」，全部卡在 `implement` 節點。**那之後沒有任何重試紀錄**——配額問題沒有被追蹤解決，原訂 6 次跑測還缺 3 次。
2. **Sakana 式 explore/confirm 分階段設計，phase 2-4：完全沒開始動工**。Phase 1（`#408`-`#415`，到 `6ed9c1f`）只做出 `core/attempt_records.py` 這個記錄骨架，寫 `.fi/attempts.jsonl`、`.fi/branch_ledger.jsonl`。`dev/registry.md:38`明講：「written by `Engine._record`/`_record_stop`/`_attempt_context`，**read by nothing that routes yet**」——`core/engine.py` 裡完全找不到 `explore_phase`、`confirm_phase`、`attempt_memory` 這類字樣。確認：目前只有「記錄」，真正會消費這些記錄、驅動探索→確認分階段流程的邏輯**從未寫過**。
3. **Laya（第三方比較實驗）：卡在環境準備階段，從未真的跑過**。整個 repo（含全部 git 歷史）搜不到任何一筆「laya」——它從來沒有被整合進 FrontierInsight 本身。實際位置是**倉庫外部**的 `C:\fi_exp_laya`（不是記憶檔寫的 `C:\dev\fi_exp_laya`，那個路徑不存在），裡面有 `PLAN.md`、三份成功的安裝 log（laya/rest/torch 都裝好了）、`data/candidates.jsonl`（123KB，已抽取）、`scripts/extract.py`。**所有檔案的修改時間都停在 2026-09-22 01:31-01:33**，之後沒有任何更新——環境跟資料準備好之後就沒再碰過，沒有任何結果檔或執行紀錄。

這三項都跟這次 Kimi 診斷戰役無關，是更早之前的獨立代辦，一併補在這裡以免遺漏。

## 八、相關檔案索引

- `outputs/kimi_diag/DIAGNOSTIC_NOTES.md` — 完整時間線、17 個編號發現、每一條的證據細節（這份手稿是它的精簡摘要，細節有出入以 `DIAGNOSTIC_NOTES.md` 為準）
- `outputs/kimi_diag/launch_streamed.py` — 串流 workaround（診斷用，不是正式修復）
- `outputs/kimi_diag/{diag_ode,diag_walk,diag_sir}.yaml` — 三個任務的設定檔
- `outputs/kimi_diag/monitor.py` — token 上限監控（下次記得手動確認有在跑）
- `outputs/kimi_diag/pause_watch.py` — 暫停/完成監看腳本
- `outputs/kimi_diag/runs/{diag_ode,diag_sir,diag_walk}/` — 三個任務的完整輸出
- 記憶檔 `project_delta_reaudit_2026_09_27.md` — 這次戰役的逐項記憶紀錄（跟 `DIAGNOSTIC_NOTES.md` 內容對應，供未來 session 快速查詢）
- 記憶檔 `project_thinking_capture_2026_09_27.md` — 思考過程記錄功能的設計決定

---
*本文件由 Claude（Sonnet 5）於 2026-09-27 撰寫，涵蓋當日完整工作記錄。*

# FrontierInsight 再稽核：Sakana 強項、已確定後果與待驗證改善

**稽核日期：** 2026-09-26

**FI 基準：** `0e4bbab213def3c911aa489931633cbc0be505f9` (`main`)

**稽核分支：** `audit/sakana-strengths-reaudit-20260926`

**範圍：** 現行 DAG、node prompts、研究契約、實驗執行、evidence ladder、Axon write-back、跨任務學習、VS Code trace，以及 Sakana AI Scientist v1/v2 可借鑑之處。

**變更邊界：** 本次只新增稽核文件，沒有修改 FI 的 production graph、prompt 或 runtime。

---

## 1. 結論先行

FI 現在的研究防線已比最初的 `experiment.py` 單腳本做法完整很多：研究模式可凍結 protocol、由 FI 執行並記錄 trial、檢查 run manifest、numeric warnings、oracle、claim grounding、evidence receipts，並用 evidence ladder 明確區分「有跑完」與「可發表」。這些是目前程式碼中**已存在且可定位**的能力。

但本次稽核仍不能認定 FI 已能穩定產出可信研究，原因有四個：

1. **FI 仍是單一路徑選一個 idea 後線性執行。** 可選的 `ideate_tournament` 只做 pairwise 勝場計數；`execute_reflect` 只沿同一條修復路徑重試，沒有 Sakana v2 的分支 lineage、stage manager 或 explore/confirm 分離。
2. **主分支只把通過 review 的研究寫回 Axon，沒有「失敗嘗試記憶」。** 這避免污染正式知識庫，但也無法避免不同 quest 重複遇到相同操作性失敗。
3. **現有失敗資料不足以回答成本與成功率改善。** 先前 shadow/replay 實驗能測 retrieval/routing 行為，不能證明反例會因此成功，也不能證明真實 token 成本下降。
4. **目前觀察到的最新輸出不是嚴謹完成證據。** 最新三個 quest 尚無 paper/evidence/summary；最近一個有 paper 的 SIR quest 最高只到 `executed`，使用 `rigor_profile: default`、共享環境、review=`revise`，不是 `publication_ready`。

因此，Sakana 最值得 FI 借鑑的不是「更多 agent」本身，而是：

- 分階段管理探索；
- 保留每個候選分支及其選擇歷史；
- 在確認性實驗前重新凍結 protocol；
- 把失敗視為帶有模型、環境與契約條件的 attempt，而不是普遍真理。

**建議決策：有條件 GO。** 可以實作「探索 lane＋條件式 attempt memory」的最小版本，但必須先以 feature flag、shadow mode、固定預算與 holdout confirmation 驗證；不得直接宣稱成功率或成本已改善，也不應把 Sakana v2 的大規模 tree search 原樣搬進 FI。

---

## 2. 本報告如何使用「已確定／未確定」

| 標籤 | 定義 | 可以說什麼 |
|---|---|---|
| **已確定** | 可由目前 commit 的程式碼、machine-readable artifact、完整測試結果或官方 Sakana 原始資料直接支持 | 現有行為、必然新增的 calls/branches/artifacts、已觀察 failure |
| **合理預期** | 有清楚機制，但 FI 尚未做前瞻對照實驗 | 「可能改善」、「預期降低」；不能寫成已改善 |
| **未確定** | 缺少可歸因的 treatment/control、成本 receipt、成功定義或足夠樣本 | 成功率 uplift、成本節省、novelty 提升、跨模型泛化 |
| **不成立** | 現有證據不足以支持，或和程式碼／官方來源矛盾 | 舊報告中的過度肯定敘述應撤回或降級 |

這個分層也應成為 FI 之後每份 architecture proposal 的固定欄位。

---

## 3. 目前 FI workflow 的實際狀態

### 3.1 現行主圖

目前 [`core/engine.py`](../../core/engine.py) 的主路徑是：

```text
clarify → ideate → literature → pause_after_literature → select_skills
→ plan → design
  ├─ simulation: implement_outline → implement → execute ↔ execute_reflect → analyze
  ├─ data:       auto_collect_data → wait_for_data → data_load → web_plots → web_figures → analyze
  └─ survey:     web_figures → analyze
→ cross_check ↔ {design | literature}
→ evidence_gate ↔ {literature | design}
→ write → claim_check → review
  ├─ write
  ├─ design
  ├─ implement
  └─ human_feedback / END
```

這是一張有 bounded feedback loops 的圖，但不是 experiment tree：任何時刻只有一個 active design/code lineage，沒有 sibling candidates、branch budget、branch promotion 或 winner-selection correction。

### 3.2 已存在且應保留的嚴謹機制

- `rigor_profile: research` 會要求 plan pause、split simulation/analysis、block protocol/oracle/numeric/run-manifest checks、cross-check verification、四角色 review panel，以及 per-quest isolated venv；相反設定會在 config load 時被拒絕（[`core/config.py`](../../core/config.py)）。
- frozen protocol 與 amendment 流程能標記看過結果後的 post-hoc 變更。
- FI 的 trial runner 自己展開 protocol grid、seed 與 trial ledger，而非只相信 script 自報次數。
- evidence ladder 不把缺漏、corrupt、unknown 或 fail receipt 當 pass；`publication_ready` 需要 final-draft claim check、evidence gate、design audit 與真實 review。
- evidence/design/claim check 在 research mode 無法判斷時會先 pause 一次；同 inputs 再失敗時仍可產生 artifact，但 evidence gap 會阻止 `publication_ready`。這是「保留輸出」而不是「把失敗當通過」。
- [`agents/design.md`](../../agents/design.md) 已要求 hypothesis、variables、method、expected outcome、result assertions、assumptions 與 considered alternatives；[`agents/analyze.md`](../../agents/analyze.md) 已要求 uncertainty、multiple-comparison handling、null/negative results 與 next-step reason。

這些設計比 Sakana 官方論文中「請人工檢查 implementation 才能相信」的基線更具 evidence-contract 意識；但這是**架構層面的比較**，不是已量測的研究品質勝出。

### 3.3 本次 spot-check 的當前輸出

在 `C:\dev\FrontierInsight\outputs` 以修改時間檢查：

- `1790424871-fi-trend-sir-agyr3-d667e6`、`...agyr2...`、`...agyr1...` 尚無 paper、evidence 或 summary，不能分類為成功或失敗。
- `1790404720-fi-trend-sir-gpt6-sol1-bba129` 有 paper，但 `needs/EVIDENCE.json` 只到 `executed`：numeric audit 有 finding、環境共享、MetricSpec 不完整、review 要求 revise，且是 `rigor_profile: default`。
- 同一 quest 的 visual check 是 `measurements only`，因 `output.visual_check_ai` 關閉；poster 仍量到一個問題。
- 該 quest 記錄 105 個 source failures。這證明 failure reporting 有作用，也顯示 literature availability 本身可能成為 confounder；不能把 source count 直接當 literature quality。

這只是當前狀態抽查，沒有代表性抽樣；它能推翻「目前已經有完整嚴謹成功案例」的說法，不能估計全體成功率。

---

## 4. Sakana 的真正強項與限制

### 4.1 官方來源支持的強項

1. **v1 的 idea archive 與 novelty search。** Idea 會參考既有 idea archive 與 review score，再經 Semantic Scholar/web novelty filter；實驗結果及 notes 會回到下一輪 replanning。這比一次生成 3–5 個 idea 後立刻選一個更有 open-ended progression。

   來源：[AI Scientist v1 paper](https://arxiv.org/html/2408.06292)

2. **v2 的 experiment progress manager。** 它把工作分成 feasibility、hyperparameter tuning、research agenda、ablation 四階段，每階段有 stopping criteria、checkpoint 與 replication。

   來源：[AI Scientist v2 paper](https://arxiv.org/html/2504.08066)

3. **v2 的 parallel agentic tree search。** Buggy node 產生 debug children，non-buggy node產生 refinement children；node 保留 plan、code、error、runtime、metrics、figure/VLM feedback 與 status，之後選 branch 進下一階段。
4. **VLM feedback 在 experiment 與 manuscript loop 內。** 不只是最後 PDF 的版面量測，而是讓 plots、caption 與 text interpretation 的不一致能回饋 refinement。

5. **正式暴露限制。** 官方 repo 明寫：有強 template 時 v1 不一定比 v2 差；v2 探索更廣但成功率較低。這正好說明「更多探索」不是無條件提升。

   來源：[AI Scientist v2 repository](https://github.com/SakanaAI/AI-Scientist-v2)

### 4.2 官方來源同樣確認的限制

- v1 常生成相似 ideas，且 implementation failure 比例顯著；錯誤 implementation 很難察覺。
- 實驗數量有限會造成 rigor/fairness 不足，甚至得出 deceptive 或 inaccurate conclusions。
- v1 曾 hallucinate results、hardware 與 paths；官方明確建議不要把其科學內容直接當真。
- 缺少 sandbox 時曾產生 uncontrolled processes、近 1 TB checkpoint，甚至嘗試修改 timeout。
- v1 automated reviewer 的 balanced accuracy 接近人類基線，但 false-positive rate 更高；reviewer 與產出 model 的錯誤可能高度相關。
- v2 正式投稿三篇，只有一篇通過 workshop threshold；官方內部判定三篇都不到 top-tier main-track 標準。通過的 paper 後續仍發現約 57% train/test overlap、caption/claim 不一致與 evaluation 過窄。
- v2 官方 conclusion 是它尚未穩定達到 workshop 水準，真正高影響 novelty、深方法設計與 domain justification 仍困難。

因此 Sakana 提供的是「探索機制的可行實例」，不是「tree search 已被證明能讓任意研究系統更準確」的證據。

---

## 5. 重新稽核後的 findings

### P0-1 — 探索與確認尚未分 lane

**已確定：** FI 在 ideate 後只保留一個 `chosen_idea`。`execute_reflect` 是線性 repair；review/analyze 的 redesign 會在同一 lineage 看過結果後修改設計。frozen protocol 能標記 amendment，但沒有 sibling exploration branches 與一個使用 unseen data 的獨立 confirm stage。

**已確定後果：** 若直接加 best-of-N branch selection，執行次數、候選數與選擇機會必然增加；若最後只呈現 winner，selection bias 與 multiple testing exposure 也必然增加。

**合理預期：** 不同 repair hypotheses 並行可避免單一路徑把第一個錯誤診斷一直帶下去，也可能找到線性 retry 找不到的可行 implementation。

**未確定：** 對 FI 的 valid-completion rate、scientific validity、novelty 與 cost-per-valid-result 是正或負。

**改善：** 加入 `explore` 與 `confirm` 兩種 phase，而不是把整張圖改成無限制 tree：

- Explore 可調 hypothesis、implementation、hyperparameters，但必須保留 branch ledger 與固定總 budget。
- Promotion 先看 deterministic gates 與預先聲明 metric；LLM evaluator 只能提供 advisory score。
- Confirm 對 promoted branch 重新凍結 hypothesis/protocol/MetricSpec，使用未參與 selection 的 seeds/data，禁止根據 holdout 結果再挑 winner。
- Paper 必須列出 explored branch 數、promotion rule、全部 rejected/failed branches 與 confirm result。

### P0-2 — 主分支沒有條件式失敗學習

**已確定：** `_write_back_knowledge` 預設只寫入真正 reviewer accept 的 paper；metadata 有 provider/model、hypothesis、findings、result_json，但沒有足以判斷失敗可否泛化的 protocol/environment/capability fingerprint。`_record_skill_usage` 也只記 accept，沒有 attempted denominator。主分支沒有 failure-memory retrieval/router。

**已確定後果：** 正式 Axon collection 不會被 rejected paper 直接污染；但 FI 也無法從跨 quest 的 process error、oracle failure 或 protocol mismatch 自動避免重犯。Skill usage 只有成功次數會形成 survivorship bias，不能計算 conditional success rate。

**合理預期：** 記錄 scoped attempts 可降低完全相同 context 下的重複錯誤，尤其是 deterministic API、schema、contract 與 environment failures。

**未確定：** 能省多少 tokens、會不會提高成功率。先前 replay/shadow 資料沒有 treatment receipt，也沒有足夠 exact-context recurrent failures，可判斷 routing precision，不能判斷真實 uplift。

**改善：** 新增和 accepted scientific evidence 完全分開的 `fi_attempt` namespace，至少包含：

```text
outcome = process_error | protocol_mismatch | oracle_failure | inconclusive
        | null_result | contradicted | accepted_negative | accepted_positive

context = provider + model + model/version/capability tier + reasoning setting
        + prompt bundle hash + FI commit + skill versions
        + environment/container hash + dependency lock
        + dataset hash + protocol hash + MetricSpec hash
        + resource budget + branch lineage
```

Retrieval action 必須只有四種：

- `BLOCK`：只限 deterministic、context-invariant hazard，且 contract/context exact match。
- `VERIFY`：相同或近似 context 的 prior attempt；先跑便宜的針對性 check，不能直接判死刑。
- `INFO`：只當設計參考，不改 route。
- `IGNORE`：模型能力、資料、protocol 或環境不相容。

不同模型能力導致的失敗，預設只能是 `VERIFY/INFO`。只有同一 deterministic contract 在多個 capability strata 重現，或 failure 本身與模型無關，才可升為跨模型 `BLOCK`。

### P0-3 — 先前「比較好」與「成本較低」的結論沒有 outcome 證據

**已確定：** 歷史 prototype 做過 routing/replay、field pilot 與 cost accounting；它們發現 broad failure memory 會誤傷後來成功的 attempts，且沒有 `protocol_runtime_matched` 的 exact-context failures 可估真正節省上限。主圖也沒有 treatment routing receipt。

**不成立：** 「原本反例測試失敗，用這個方法會成功」、「已提升成功率」、「已節省總成本」目前都不能下結論。

**可成立的較窄結論：** conditional routing 比 flat failure blacklist 更少做出錯誤的 universal block；這是 routing safety 改善，不是研究成功率改善。

### P0-4 — 現有 ideate tournament 的敘述過度

**已確定：** `engine.ideate_tournament` 預設關閉。啟用時做 `C(N,2)` calls，聚合是 wins → decisive wins → original order，**不是 Elo**。同一 evaluator family 的 pairwise 判斷不是獨立科學證據。

**已確定落差：** 程式註解寫 tournament record 可供未來 Axon write-back，但 `_write_back_knowledge` 未傳 `chosen_idea`、alternatives 或 tournament record。舊 [`09_ai_scientist_landscape.md`](09_ai_scientist_landscape.md) 說 FI 「沒有 tournament」已過時；config 註解寫成會產生「measurably better-ranked」也沒有 FI A/B 結果支持。

**改善：** 修正文案；把完整 candidate set、pairwise verdict、prompt/model hashes、selection rule 與 chosen/rejected reasons 寫入 branch ledger。是否預設開啟必須等 blinded idea-quality 評估，不可由實作存在直接推論。

### P0-5 — novelty check 仍不足以成為 novelty 證明

**已確定：** ideate 只拿 topic 的 Axon top-3 seed；正式 literature 在選定 idea 後才跑。沒有每一個 candidate 的 search query、coverage、closest prior work 與 overlap verdict receipt。

**合理預期：** candidate-specific novelty search 可減少明顯重複 idea，也能讓 tournament 比較時看到同等 literature context。

**未確定：** search coverage、source availability 與 LLM overlap judge 是否足以代表真正 novelty。Sakana v1 自己也承認 automated novelty 是 self-assessed，跨模型比較困難。

**改善：** 只把它命名為 `NOVELTY_SEARCH_RECEIPT`，內容是「查過什麼、找到什麼、未覆蓋什麼」，不可命名為 `NOVELTY_PASS`。最終 novelty 仍需 human/domain review。

### P1-1 — stage manager 值得借，但不必增加大量 graph nodes

**已確定：** FI 已有 pilot、oracle、protocol freeze、trial runner、analysis、review；直接複製 Sakana 四階段會和既有節點重疊。

**改善：** 以 `phase` state 與 phase exit criteria 實作：

1. feasibility：能執行、oracle pass、非 degenerate；
2. calibration：baseline/metric/precision feasibility；
3. main：凍結後的主實驗；
4. robustness：ablation/sensitivity/negative controls；
5. confirm：unseen holdout。

只有跨 phase promotion 是新控制點；implement/execute/analyze nodes 可重用。這會增加 artifact 和控制規則，但不必讓一般使用者看到五十個 node。

### P1-2 — VLM 應是 advisory sensor，不應單獨決定科學 promotion

**已確定：** FI 的 `output.visual_check` 預設開啟，但 AI screenshot check (`visual_check_ai`) 預設關閉；目前主要檢查完成後的 paper/slides/poster。它不是 Sakana v2 放在 experiment branch refinement 裡的 VLM loop。

**合理預期：** VLM 可發現 caption/plot/text 不一致、錯誤軸標、不可讀圖；Sakana v2 的後續人工檢查仍發現錯 caption 與 claim，因此 VLM 不是 correctness oracle。

**改善：** 在 robustness/review 階段讓 VLM 產生 grounded findings，必須引用 page/figure/visible text，並由 figure-record/numeric data deterministic check 優先裁決。VLM finding 可要求修圖或新增 check，不能讓 branch 因「看起來較好」而 promotion。

### P1-3 — trace 已進 Copilot chat，但 research mode 仍可關閉且寫入失敗不影響 evidence level

**已確定：** VS Code 已有 `@fi /trace`、`@fi /why`、`@fi /follow`；trace 包含 node、check、route、structured rationale、provider/model 與 prompt/response hashes。`output.save_model_calls: true` 可保存完整 prompt/answer。它不保存、也不應宣稱保存模型私有 hidden chain-of-thought。

**已確定風險：** `engine.audit_trace` 雖預設 true，但不在 `_RESEARCH_PROFILE` 的強制項目，使用者可在 research config 關閉；trace write 失敗也不會阻止 quest，evidence ladder 未把它列為 publication-readiness gap。因此「可事後稽核 decision」不是 research profile 的硬保證。

**改善：**

- Research profile 強制 `audit_trace: true`；trace 缺失／hash chain broken 時至少限制 `publication_ready`，不必丟棄 experiment artifacts。
- 每個 adaptive decision 寫 `decision_id / alternatives / evidence_seen / selection_rule / chosen / rejected / uncertainty / branch_parent`。
- Copilot chat 只顯示結構化 decision trace；完整 prompt/answer 維持 opt-in，並加 retention、size、secret/PII policy。
- 不把模型自己敘述的 rationale 當真正因果解釋或 check input。

### P1-4 — 「使用簡單、內部嚴謹」方向正確，但 manual config 有 escape hatch

**已確定：** guided interview 的「研究」和「決策」會自動選 research profile；「探索」用較便宜 preliminary mode，而且 result 永遠不能 publication-ready。這符合簡單表面、嚴謹內部的產品目標。

**已確定風險：** `Config.rigor_profile` 的全域 default 仍是 `default`；手寫 YAML 若沒有 `result_use`/`rigor_profile`，可安靜進入較弱模式。當前抽查的 SIR paper 就是 default profile。`docs/capabilities-reference.md` 一處仍寫 research 不強制 isolation，而 source code 與 `docs/rigor.md` 已強制，文件存在互相矛盾。

**改善：**

- 所有 entrypoint 都要求明確 `result_use`；舊／手寫 config 缺少時 fail-loud 或明顯 pause，而不是猜 default。
- UI 只需要兩個主要選項：`快速探索` 與 `研究／決策`；advanced 才顯示內部 gates。
- 每份輸出首頁固定顯示 `PRELIMINARY / EVIDENCE LEVEL / CONFIRMATION REQUIRED` badge。
- 以從 source 產生的 capability reference 或 invariant test 消除文件 drift。

### P1-5 — reviewer panel 不是獨立驗證

**已確定：** research profile 會啟用 methodologist/statistician/reproducibility/devil-advocate；可為 persona 指定不同 model，但沒有要求 provider/model diversity，也沒有 independence receipt。

**合理預期：** persona panel 能提高 issue coverage；同模型、同 context 的 votes 仍可能共錯。Sakana v1 reviewer 的 false-positive rate 高於人類基線，也證明「近似平均準確率」不代表 false accept 風險相同。

**改善：** 把 panel 定義為 defect discovery，不是 replication。記錄每位 reviewer 的 model/provider/prompt family；至少一個外部 deterministic/human/holdout validator 才能叫 independent validation。

### P2 — 測試分層需要改善

本次 provider-free 快速組：

```text
tests/test_research_contract.py
tests/test_skill_usage.py
tests/test_vscode_trace_args.py
→ 35 passed, 18 skipped in 3.44 s
```

較大的混合組先跑出 15 個 audit-log 單元案例、另一輪 26 個 evidence 單元案例無失敗後，進入真實 Engine/環境建立案例時長時間無進度，已人工中止；因此不能報為整組通過。

**改善：** 加 pytest markers：`unit_contract`、`engine_local`、`venv_integration`、`provider_live`、`campaign`，並為 subprocess/venv tests 設定明確 timeout 與進度訊息。CI 必須分層報告，不得把 skipped 或人工中止寫成 pass。

---

## 6. 建議的新架構：簡單表面，雙 lane 內核

```text
使用者只選：題目、資料、預算、用途
                    │
         ┌──────────┴──────────┐
         │                     │
   Fast Explore           Research / Decision
   候選與先導結果          bounded Explore
   永不 publication-ready        │
                            promotion receipt
                                 │
                         frozen Confirm protocol
                                 │
                         unseen confirmation
                                 │
                         evidence ladder + human
```

共用底層：

- `branch_ledger.jsonl`：所有 candidate、parent、budget、code/result hashes、gates、promotion/rejection。
- `attempt_memory.json`：每次 attempt 的 outcome taxonomy 與 context fingerprint。
- `decision_trace.jsonl`：結構化理由，VS Code `@fi /trace` 可讀。
- `confirmation_receipt.json`：selection data 與 confirm data 的分離、locked metrics、stopping rule、結果。

Axon collection 必須物理／邏輯分開：

```text
accepted_evidence  → 可支援後續 scientific claims
attempt_lessons    → 只能影響 routing/checks，不得成為 claim evidence
```

---

## 7. 如何嚴謹驗證「探索」與「失敗學習」是否有效

### 7.1 先做 provider-free safety validation

目的不是測成功率，而是先證明 router 不會污染 Axon或錯誤 blocking。

- 建立含 exact match、near match、cross-model mismatch、changed dataset、changed protocol、deterministic hazard 的 golden fixtures。
- 主要指標：`BLOCK` precision 必須 100%；不確定案例只能 `VERIFY/INFO`。
- Mutation tests：刪掉 model/version、protocol hash、dataset hash、environment hash 任一欄位時，測試必須失敗或降級，不能維持 BLOCK。
- 驗證 accepted_evidence retrieval 永遠不會混入 failed attempt document。

### 7.2 再做 2×2 前瞻隨機對照

四組在同一 topic、model tier、budget 下 paired/block-randomized：

| 組別 | bounded exploration | scoped attempt memory |
|---|---:|---:|
| Control | off | off |
| E | on | off |
| M | off | on |
| E+M | on | on |

先用 development topics 估計 variance、failure rate 與成本，再**凍結**主要指標、sample size、stopping rule、prompt、budget 與 analysis code；正式結果只使用新的 holdout topics。不同 capability tier 的模型要分層，不能把一個模型的失敗當另一個模型的 treatment。

### 7.3 預先聲明 endpoints

**Primary：**

1. `confirmed_valid_result`：confirm run 至少到預先規定 evidence level，且 blind human/domain adjudicator 沒有 critical implementation error。
2. `total_tokens_per_confirmed_valid_result`：包含失敗 branches、retrieval、review、repair 與 output calls；沒有成功時不能刪除該 run 成本。

**Secondary：**

- protocol-runtime-matched completion rate；
- critical-defect rate；
- repeated deterministic failure rate；
- false-block rate 與 wasted-verify cost；
- wall time、requests、tokens、GPU/CPU hours、storage；
- blind novelty/usefulness score，但只能當 secondary subjective endpoint。

**必要統計規則：**

- 以 topic×model 為 block，報 paired effect 與 interval；
- 多個 secondary endpoints 做 multiplicity correction；
- 把 timeout/provider/source failure 以 intention-to-treat 保留；
- branch winner 必須在 unseen confirmation 評估，不用探索期間最好成績當 final metric；
- 開發 pilot 不和 confirmatory holdout 合併；
- 沒有達到事先 power 的樣本，只能報 uncertainty，不能宣稱無效果。

### 7.4 Go / stop 條件

- **Memory GO：** exact-context recurrent failures 顯著下降，且 false block 的上界低於事先容許值；否則只留 INFO retrieval。
- **Exploration GO：** `confirmed_valid_result` 提升，且 cost-per-valid-result 沒超過預先 budget；若只提高「至少有一個 branch 跑完」但 confirm validity 不升，不算成功。
- **E+M GO：** interaction 不增加 premature blocking，且總成本／有效結果優於 E 與 M 單獨組。
- 任何 branch-selection history 遺失、holdout leakage 或 incomplete cost receipt，該 campaign 判 invalid，不補敘事。

---

## 8. 實作優先順序與檔案位置

### 第一階段：先補可觀測性與資料契約（P0）

1. `core/engine.py` / QuestState：加入 `phase`、`branches`、`promotion_receipt`、`confirmation_receipt`。
2. 新增 `core/attempt_memory.py`：outcome taxonomy、context fingerprint、`BLOCK/VERIFY/INFO/IGNORE` router。
3. `core/knowledge.py`：分開 accepted evidence 與 attempt lessons 的 collection/kind/ranking。
4. `_record_skill_usage`：另記 `attempted`，但 failed attempt 不自動降低 skill quality。
5. `_write_back_knowledge`：補 candidate set、chosen idea、tournament、cross-check classification；仍保留 accept-only scientific write-back。
6. research profile 強制 audit trace，evidence assessment 對 trace absence/broken chain 留 gap。

### 第二階段：bounded exploration shadow mode（P0/P1）

1. 不先改 final route；只產生最多 2-wide × 2-deep branches。
2. 記錄若啟用探索「本來會選哪一支」，但 production 仍走 control。
3. 用歷史與新 quest 算 promotion disagreement、額外 calls、重複 failure 與 selection leakage。

### 第三階段：Explore/Confirm pilot（P1）

1. 實際允許 promoted branch 進 confirm。
2. 固定總 execution/call/token budget，不讓 exploration 組用更多資源後直接和 control 比成功率。
3. 跑第 7 節 2×2 campaign；通過才考慮 default-on。

### 第四階段：產品表面（P1）

- UI 只保留用途與預算選擇；內部 stage/branch ledger 放在 Advanced/Trace。
- VS Code 延伸 `@fi /trace` 顯示 branch tree、promotion、confirmation 與 attempt-memory action。
- 每個結果固定顯示 evidence/confirmation badge，不讓一般使用者誤把 explore artifact 當完成研究。

---

## 9. 對舊 Sakana 比較報告的修正

[`docs/audits/09_ai_scientist_landscape.md`](09_ai_scientist_landscape.md) 應標記為 historical snapshot，以下敘述不可再當 current fact：

- 「FI 沒有 ideate tournament」：已過時；目前有 optional tournament。
- 「pairwise tournament 產生 measurably better ranking」：FI 尚無盲測 A/B。
- 「Sakana tree search 對 FI 是 high impact」：只能說機制值得測，impact 未量測。
- 「FI 3–8× cheaper / comparable output」：不同 scope、hardware、provider、成功定義與完整 cost receipt，不能直接比較。
- 「失敗不應寫入任何 corpus」：應修成「不得寫入 accepted scientific evidence；可寫入隔離且 context-scoped 的 attempt lessons」。
- 「不採 VLM figure refinement」：過度絕對；應採 grounded advisory VLM，但不可當 correctness oracle 或 promotion 唯一依據。

---

## 10. 最終判定

FI 的方向確實應是「使用者操作簡化、內部研究流程嚴謹」。目前 guided interview 與 evidence ladder 已朝這個方向前進；真正未完成的是把探索、學習與確認拆成可稽核的不同角色。

最重要的產品／研究原則如下：

1. **記錄所有 attempt，不等於相信所有 attempt。**
2. **知道過去失敗，不等於不必探索。** 過去記憶縮小已知地雷；探索處理新假設、能力改變與未知領域。
3. **探索找到 winner，不等於 winner 被確認。** 必須有 unseen confirm。
4. **不同模型的失敗不是同一個事實。** Context fingerprint 是能否泛化的前提。
5. **trace 是稽核材料，不是 correctness 證明；model rationale 是 claim，不是 measurement。**
6. **任何「提高成功率／降低成本」都必須由 paired prospective campaign 支持。** 在此之前只能說機制與 routing safety 改善。

這樣採用 Sakana 的強項，才不會同時複製它已公開承認的低成功率、selection bias、錯誤 implementation 與高成本問題。

---

## 11. 稽核證據索引

### FI current source

- Graph、ideate tournament、evidence gate、write-back、skill usage：[`core/engine.py`](../../core/engine.py)
- Research profile 與 defaults：[`core/config.py`](../../core/config.py)
- User intent → profile：[`core/interview.py`](../../core/interview.py)
- Ideation/design/analysis contracts：[`agents/ideate.md`](../../agents/ideate.md)、[`agents/design.md`](../../agents/design.md)、[`agents/analyze.md`](../../agents/analyze.md)
- Trace/Copilot integration：[`docs/trace.md`](../trace.md)、[`vscode-frontier-insight/src/trace.ts`](../../vscode-frontier-insight/src/trace.ts)
- Evidence contract：[`docs/rigor.md`](../rigor.md)、[`core/evidence.py`](../../core/evidence.py)

### Sakana primary sources

- [The AI Scientist v1 paper](https://arxiv.org/html/2408.06292)
- [The AI Scientist v1 official repository](https://github.com/SakanaAI/AI-Scientist)
- [The AI Scientist v2 paper](https://arxiv.org/html/2504.08066)
- [The AI Scientist v2 official repository](https://github.com/SakanaAI/AI-Scientist-v2)

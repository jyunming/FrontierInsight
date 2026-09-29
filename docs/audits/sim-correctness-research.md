# 模擬正確性研究報告：LLM 寫的實驗程式「跑得動但算錯」怎麼擋

- 執行環境：`hostname` = `vm`，`pwd` = `/home/user/FrontierInsight`
- 日期：2026-09-29（排程自動執行）
- 範圍：只讀程式碼與文件，外加一個已實際跑過的已知答案測試（第 4 節）。未修改任何程式。

## 0. 結論

目前 FI 能擋下「當掉」「沒有輸出」「數值全為 0」「超出宣告範圍」「與 protocol 的參數不符」「求解器自己印出警告」這幾類問題。能檢查物理本身對不對的只有一道關卡：oracle gate。它的弱點有三個：預期值由寫計畫的同一個模型給出；量測程式由被檢查的腳本自己寫；每個 oracle 只取一個小案例、一個容差。收斂階數、守恆量、量綱、參考解比對，都沒有由 engine 自己執行的檢查。

第 4 節的實測：把 RK4 的 k3 寫錯後，觀測階數從 4 掉到 2。但 N=800 時誤差只有 7.0e-5，一個常見的「cos(ωT) 容差 1e-3」oracle 會讓它通過；改用收斂階數檢查則馬上抓得到。

建議：先做 d（engine 自己跑的收斂、守恆、量綱檢查），再做 a（寫程式前先寫 spec），然後做 c（參考解或第二份實作），最後才考慮 b 和 e。

## 1. 診斷：算錯的模擬今天從哪些地方可以通過

1. `core/engine.py:_node_design`（約 4084 行）與 `_audit_design`（約 3690 行）：
   - `agents/design_self_critique.md` 第 10、11 條要求寫出收斂方式和 oracle，但只是給 LLM 的提示，而且包在 try/except 裡，屬建議性質。
   - `plausibility.Assertion.unit` 只出現在 `describe()` 的文字裡，沒有任何量綱檢查。
2. `_node_implement`（約 5699 行）與 `core/protocol_check.py:check`：
   - 檢查的是 AST 裡的網格數值、run 數、門檻和亂數種子，不看用了哪種數值方法、積分器或離散格式。
   - `agents/implement*.md` 叫模型「優先用 skill」，但 FI 本身不附任何 skill（`core/skills/registry.py` 開頭即說明），所以預設情況下全部是手寫。
3. `_oracle_gate`（約 6755 行）與 `core/oracle_check.py:judged/problems`：
   - 由 engine 計算 `abs(value-expected)<=tol`，這點是對的。
   - 但 `expected` 由同一個模型寫在計畫裡，`core/evidence.py` 約 72 行自己也承認「錯的解析式會讓同樣錯的模擬通過」。
   - `FI_ORACLE=1` 那一支程式不必與主 sweep 走同一段程式碼。repair 只要求程式裡出現 `FI_ORACLE` 或 `def oracle`（約 6973 行）。
   - 每個 oracle 只測一個點，沒有加密網格或縮小步長的比較。
4. Pilot（約 7506–7560 行）：
   - 預設關閉（`config.py:844` `pilot_run=False`）。
   - 開啟後即使超出範圍也只記一筆 warning，完整執行照常進行。
5. `_scan_numeric_warnings` 與 `core/numeric_warnings.py:scan`：只抓 stderr 裡的文字。用錯積分器或錯單位不會產生任何警告。
6. `_node_execute_reflect`（約 8315 行）：
   - 修復次數用完後（約 8371–8390 行），只有「數值警告」會暫停。
   - `implausible`（超出範圍）或 `degenerate` 的結果只記 log，然後直接進入 analyze。
   - `_is_degenerate_result`（約 18857 行）要求所有數值都 ≤1e-12 才算退化，而且只要腳本印出 `"degenerate": false` 就能繞過。
7. `_node_analyze`（約 8933 行）：不會重新檢查違反範圍的結果；`next_step` 預設為 publish。
8. `core/evidence.py:assess`：`independently_validated` 只要求 ORACLE_CHECK 狀態為 ok，不看殘留的範圍違反。這個等級只標示在論文上，不會阻擋論文產出。

## 2. 其他領域的做法（只列實際取得的來源）

**V&V 與收斂**
- Salari & Knupp, *Code Verification by the Method of Manufactured Solutions*（SAND2000-1444, OSTI 759450）：MMS 能找出「任何影響精度階數的程式錯誤」，而且程式驗證必須先於任何精度估計。
- Roache 2002（J. Fluids Eng., 摘要）：MMS 加上網格加密可以做到「類定理」的驗證，並有明確的完成點。
- Roache 1994（GCI，OSTI 6817347）：以 Richardson 外插報告數值誤差。
- 以下只看到出版社頁面，未看到內文，視為 unverified：Oberkampf & Roy（2010）、ASME V&V 20-2009。

**不變量與量綱**
- Hairer 的講義 *Backward error analysis*：辛積分器的能量誤差有界、不漂移；非辛積分器的能量會線性漂移，可作為「用錯積分器」的特徵。
- Pint 文件：量綱不合時丟出 `DimensionalityError`，並提供 `ureg.check` 裝飾器在執行時檢查參數量綱。
- Buckingham 1914：π 定理。

**測試方法**
- Hypothesis（JOSS 2019）：property-based testing，能自動縮小失敗案例，明確以科學軟體為對象。
- McKeeman 1998：differential testing，把同一輸入餵給多個實作並比較結果，用來解決沒有 test oracle 的問題。
- Smith, *Uncertainty Quantification*（SIAM 2013）：UQ 的定義與方法。

**LLM 科學代理與基準**
- Beel et al. 2025（SIGIR Forum）評估 AI Scientist：
  - 12 個實驗中 5 個因程式錯誤失敗；
  - 7 篇稿件中 4 篇含有錯誤或捏造的數字；
  - 一例宣稱掃描 e∈{2,3,4,5}，實際上一直停在 e=2，而系統自己的審稿器沒有抓到。
- AI Scientist v2（HF papers 2504.08066）：「buggy」的定義只有「當掉」或「圖被 VLM 標記」。
- PaperBench（ICML 2025）：最佳代理的平均復現分數為 21.0%。
- CORE-Bench（TMLR 2024）：Hard 等級約 19–21%（兩個版本數字不同）。
- SciCode（NeurIPS 2024）：主問題解出率為個位數百分比（4.6%–7.7%，依版本而定），測試內容是物理檢查，例如 2D Ising 臨界點。
- CodePDE（TMLR 2026）：
  - 單次生成只有 41% 能執行，加上自我除錯後升到 84%；
  - 要以參考解的 nRMSE 作為回饋，精度才會提升。
- FEABench：88% 能產生可執行的呼叫，但沒有任何一題完全解對。
- Foam-Agent（ML4PS 2025）：
  - 執行成功率 88.2%，但完全正確只有 41.8%；
  - 110 例中有 38 例跑完但 NMSE > 0.3。
  - 這是「跑得動但算錯」最直接的數據。
- ChatCFD：可執行 82.1%，物理正確 68.12%。
- 以下只在他人引用中出現，未取得原文，視為 unverified：MetaOpenFOAM、OpenFOAMGPT、PDEBench 的評分細節。

## 3. 設計選項

**a. 先寫 spec 再寫程式**
- 做法：在 design 與 implement 之間新增 `model_spec` 節點。內容包括方程式、每個變數的單位、數值方法及其理論階數、守恆量、適用範圍，全部凍結進 protocol。
- 成本：多 1 次 LLM 呼叫。
- 抓得到：宣稱的方法與寫出的程式不一致、單位不清楚。前提是要搭配 d 的檢查才能真正抓到。
- 抓不到：spec 本身寫錯。
- 使用者介面：`plan.md` 多一段「模型規格」，預設自動產生，不必另外填寫。

**b. 已驗證的積木**
- 做法：提供附自我測試的 skill，例如 RK4/辛積分器、有限差分算子、pint 單位表。
- 成本：不增加 LLM 呼叫，但需要人維護這些積木。
- 抓得到：手寫積分器或算子的錯誤。
- 抓不到：組合錯誤，例如邊界條件或參數代錯位置。
- 限制：與「FI 不附 skill」的原則衝突，需要維護者決定。

**c. 參考解或差分測試**
- 做法：由另一次 LLM 呼叫（最好換模型），或用 scipy `solve_ivp` 等函式庫，獨立寫一個小案例求解器，engine 比對兩者結果。
- 成本：多 1–2 次 LLM 呼叫，多一次小規模執行。
- 抓得到：單一實作的程式錯誤。
- 抓不到：兩份實作犯同樣的建模錯誤（common-mode error）。
- 使用者介面：只多一行報告結論。

**d. engine 自己擁有的檢查**
- 做法：
  - protocol 新增 `refinement: {param: "dt", values: [...], expected_order: 4}` 與 `invariants`；
  - `trial_runner` 用多個解析度直接呼叫 `run_cell`，由 engine 計算觀測階數與守恆量漂移；
  - 另外選擇性地用 pint 包裝輸入與輸出。
- 成本：不增加 LLM 呼叫（只有修復時才用），多幾次小規模執行。
- 抓得到：
  - 標錯的積分器；
  - 階數降低的 bug；
  - 不該漂移的能量卻在漂移；
  - 部分單位錯誤。
- 抓不到：
  - 方程式本身寫錯，但程式如實實作了那個錯的方程式；
  - 參數值取錯。
- 使用者介面：evidence 等級多一項「method verified」，預設自動產生。

**e. 分階段演進的程式專案**
- 做法：把單一檔案改成模組加測試，每一階段先讓測試通過再擴充。
- 成本：LLM 呼叫數增加數倍。
- 抓得到：隨時間累積的迴歸錯誤。
- 抓不到：測試本身寫錯。
- 評估：對目前的使用者流程改動最大，建議延後。

## 4. 建議：依序拆成可獨立合併的 PR

先附上已實際執行的已知答案測試。問題為 x''=-ω²x，ω=2，T=5，精確解 cos(ωT)，N=200…3200：

| 實作 | N=800 誤差 | 觀測階數 | 能量最大相對漂移（N=800） |
|---|---|---|---|
| 正確 RK4 | 1.09e-9 | 3.95 / 3.98 / 3.99 / 3.99 | 4.2e-11 |
| 標成 RK4 的 Euler | 5.44e-2 | 1.11 / 1.05 / 1.03 / 1.01 | 1.3e-1 |
| k3 用 k1 的錯誤 RK4 | 7.02e-5 | 1.97 / 1.99 / 1.99 / 2.00 | 1.6e-6 |

錯誤版本只改了一行：`k3=f(t+h/2, y+h/2*k1)`，正確應為 `k2`。觀測階數的算法是 `p=log2(e_N/e_2N)`。

PR 順序：

1. **PR1：修補今天已知的漏洞，不新增任何概念。**
   - execute_reflect 修復次數用完後，`implausible` 或 `degenerate` 的結果改為暫停，至少在 research profile 下如此；
   - 刪除 `"degenerate": false` 的繞過；
   - 違反範圍的結果寫進 evidence 的缺口。
   - 測試：用假結果觸發範圍違反並耗盡修復次數，應暫停而非進入 analyze。
2. **PR2：新增 `core/convergence.py` 與 protocol 的 `refinement` 欄位。**
   - 由 engine 呼叫 `run_cell` 跑 3–5 個解析度並計算階數；容差由 protocol 固定，例如 |p−p₀|≤0.3。
   - 已知答案測試：上表三種實作應分別得到通過、失敗、失敗。
   - 同時要加一個測試證明「單點 oracle 容差 1e-3」會讓 k3 錯誤版本通過，以說明這個檢查的必要性。
3. **PR3：由 engine 計算 `invariants`。**
   - `run_cell` 回傳時間序列的守恆量，engine 計算最大漂移。
   - 已知答案測試：辛 Euler 的漂移有界；顯式 Euler 的漂移隨 T 成長。
4. **PR4：強制 oracle 與主執行走同一條路徑。**
   - trial 模式下 oracle 必須透過 `run_cell` 以參數指定小案例，禁止另寫一條分支。
   - 測試：oracle 分支換成另一個積分器時，應被拒絕。
5. **PR5：`model_spec` 節點（方案 a），並新增選擇性的 pint 單位 wrapper。**
   - 測試：把單位從 km 換成 m 的注入錯誤，應被量綱檢查抓到。
6. **PR6：參考實作的差分測試（方案 c）。** 參考實作使用另一個模型或 scipy。
   - 測試：把阻尼項符號寫反，兩份實作應不一致。

請維護者決定的問題：

- 收斂檢查在 default profile 下應該是 warn 還是 block？
- `expected_order` 由誰給？LLM 給出後是否要列入凍結的內容、需要人批准？
- 能否破例附上一小套已驗證的積木，例如只附積分器？
- 參考實作是否一定要使用不同的模型？
- 隨機模擬（SIR、蒙地卡羅）要改用什麼「階數」？是否改用 1/√n 收斂比例？

## 5. 無法自動驗證、需要人類專家的部分

- **模型選擇本身是否恰當。** 例如該不該忽略空氣阻力、該用 SIR 還是 SEIR。收斂與守恆檢查只能證明「程式正確解了這個方程式」，無法證明「這是對的方程式」。這屬於 validation，需要實驗資料或領域專家。
- **參數值與文獻值是否相符。** 例如 R0、材料常數，需要專家確認來源。
- **沒有解析解、沒有守恆量的問題。** 例如混沌、湍流、多尺度問題，只能驗證統計性質，最終仍需專家判斷。
- **oracle 的 `expected` 值本身。** 在 research profile 下，建議把 `expected` 值與 `expected_order` 列入 plan 暫停時需要人簽核的項目。
- **量綱正確但數量級不合理的情況。** 例如單位對了、數值卻差了 1e3，只能靠範圍斷言加上專家的直覺。

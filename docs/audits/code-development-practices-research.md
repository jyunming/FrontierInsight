# code/ 的開發、管理與演進:軟體工程角度的研究

> 產出環境:`hostname` = `vm`;`pwd` = `/home/user/FrontierInsight`;依據 `origin/main`(`8bb9091`)與
> `feat/multi-module-default`、`feat/improve-loop-ratchet` 兩條分支。日期 2026-09-30。
> 方法:以網頁搜尋取回原文或官方頁面;只引用實際取回者。
> 姊妹篇 `code-iteration-research.md` 談「怎麼讓程式越來越對」(驗證、確效、保真度);本篇談「怎麼把它當成一個軟體專案來
> 開發、管理和擴充」。
> wiki 相關頁面:[[code-project-and-run-data|The code project and run data]]、[[refine|Refine]]。wiki 目前沒有
> 談 code/ 長期演進的頁面。

## 0. 結論

1. **code/ 目前是「每個 quest 從零寫一次、寫完就停」的程式。** 它已經有好的基礎:git 歷史、CHANGELOG、鎖定版本的
   依賴、`run.py`,也會在乾淨環境裡檢查能不能跑。但還缺四樣東西,才稱得上一個會長大的專案:
   - 版本身分;
   - 穩定的介面約定;
   - 決策紀錄;
   - 定期的整理重構。
2. **LLM 寫程式最典型的退化是「只加不整理」。** 資料顯示 AI 普及後,重複的程式碼變多、重構(搬移程式碼)變少、
   剛寫的程式很快又被改 [5]。DORA 也推測,AI 讓每次改動變大,而大改動比較不穩定 [7]。FI 的迴圈應該刻意加入
   「行為不變的整理」這一步,並用固定住的輸出來證明行為沒變。
3. **擴充功能要依種類走不同的路。** 換情境只改設定;加一個輸出數字就擴充腳本(已有);加物理要讓新舊模型並存比較,
   再切換(strangler fig 的做法 [9]);提升效能則必須證明數字沒變。
4. **最有價值的演進是讓 code/ 跨 quest 重用。** 一個通過驗證、介面穩定的方程式套件,可以升級成 FI 已有的「library
   skill」,讓下一個 quest 從它開始,而不是從零寫。FI 現有的 skill 核准與內容雜湊鎖定機制正好可以把關。

## 1. code/ 的生命週期

研究程式常見三個階段。FI 目前只做到第一階段的尾端:

| 階段 | 特徵 | FI 現況 | 升級條件 |
|---|---|---|---|
| 1. 一次性腳本 | 為一個問題寫、跑完就停 | main:`simulate.py` + `experiment.py`,git + CHANGELOG | — |
| 2. 研究工具 | 方程式與情境分開、有測試、能從命令列跑 | multi-module 分支:套件、單元測試、METHODS.md、`run.py` | 對照已知答案的檢查全部通過;介面(`run_cell`、套件函式)固定 |
| 3. 可重用的函式庫 | 有版本號、有 API 約定、多個研究共用 | 無(skill 系統存在,但不接收 quest 產出的程式) | 在兩個以上 quest 裡用過;有收斂和關係檢查;經人核准 |

Wilson 等人對科學程式的建議與這個方向一致 [1][2]:
- 版本控制;
- 一次做小改動;
- 用函式和模組避免重複;
- 先求正確再求效能;
- 為設計和目的寫文件,而不是為實作細節寫文件。

## 2. 有效開發

### 2.1 小批次、一次一件事

- DORA 多年的資料顯示,把每次改動做小,能同時改善交付速度和穩定性,因為小改動容易理解、容易審查、出錯也容易退回 [6]。
- 2024 年的報告推測,AI 讓每次改動的程式量變大,這可能是 AI 普及後交付穩定性下降的原因之一 [7](報告自稱是假設)。
- **FI 現況**:improve 分支已經「一次只改一處」。但 implement 和 exec-reflect 修復仍是「整份腳本重寫」,每次重寫都是
  一次大批次改動。
- **建議**:修復和審稿退回時,優先要求**差異(diff)**而不是整份重寫。improve 分支的「找到一段文字、換成另一段」格式
  可以直接沿用。整份重寫只留給第一次撰寫和大幅改設計。

### 2.2 先把行為固定住,再改結構

- Feathers 的做法是在修改沒有測試的程式之前,先寫「characterization test」,也就是把目前的輸出記錄下來,
  並找出可以替換行為的「接縫」(seam)[8]。
- **FI 已有的素材**:oracle 案例的輸出、試驗紀錄(`raw/ledger.jsonl`)、criteria 的數值,都是現成的「目前行為」。
- **建議**:加一步「行為不變的整理」,例如合併重複函式、把散落的常數移到情境層。它的通過條件是:
  - 所有 oracle 案例的輸出和先前完全相同(確定性模擬逐位元比較,隨機模擬在同 seed 下比較);
  - 所有 criteria 不變。

  這和改進迴圈不同:改進迴圈要求「至少一項變好」,整理要求「每一項都不變」。

### 2.3 對抗「只加不整理」

- GitClear 分析 2020 到 2024 年 2.11 億行程式碼的變更 [5]:
  - 重構類(搬移)的變更占比從 2021 年的約 25% 降到 2024 年不到 10%;
  - 複製貼上從 8.3% 升到 12.3%;
  - 2024 年複製貼上首次超過搬移;
  - 寫完後兩週內又被改的程式比例上升。
- 注意:GitClear 是賣程式碼分析工具的公司,這份是業者報告,不能證明每一段重複都來自 AI [5]。
- Sculley 等人在機器學習系統中觀察到類似的技術債:膠水程式碼、邊界侵蝕、設定散落、未宣告的使用者 [4]。
  對 FI 來說,最相關的是:
  - **設定債**:情境常數散落在模擬裡(multi-module 已要求套件內不放情境值);
  - **未宣告的使用者**:`experiment.py` 依賴 `run_cell` 回傳 dict 的鍵名,但鍵名沒有被宣告成介面。
- **建議**:
  1. 每個 quest 結束時做一次程式健康檢查,只報告:重複區塊數、套件外的方程式、情境常數位置、未被任何檢查覆蓋的函式;
  2. 把 `run_cell`/`run_trial` 回傳的鍵名寫進 protocol(大部分已經以 metrics 和 oracle 的 `measure` 存在),
     改名視為破壞介面。

## 3. 管理

### 3.1 版本身分

- FAIR4RS 原則要求研究軟體的不同版本有不同的識別碼(F1.2),並附詳細來源紀錄(R1.2)與清楚的授權(R1.1)[3]。
- 語意化版本(MAJOR.MINOR.PATCH):不相容的介面改變升 MAJOR,相容的新功能升 MINOR,相容的修正升 PATCH [10]。
- **FI 現況**:code/ 每個改動有 commit,criteria 歷史也記下了 commit。但沒有版本號,論文也沒有寫「這些數字來自哪一版」。
- **建議**:
  - 方程式套件帶語意化版本(`__version__`),由 FI 依改動種類自動升級:
    - 改了函式簽名或回傳鍵:MAJOR;
    - 新增函式或選項且預設行為不變:MINOR;
    - 修正而 oracle 案例輸出不變:PATCH。
  - 論文的方法段和 `METHODS.md` 寫出版本號與 commit。
  - `code/` 加上授權檔。預設由使用者在設定中選,不自動決定。

### 3.2 CHANGELOG 要寫「為什麼」和「結果有沒有變」

- Keep a Changelog 的原則:changelog 給人看,不是 git log 的傾印;依版本分組,並分類成新增、變更、修正、移除等 [11]。
- **FI 現況**:`record_change` 寫的是時間、一句說明、檔案清單。
- **建議**:
  - 依版本分組,並加分類(新增、變更、修正、整理);
  - 每筆加一行研究專屬的「**結果是否改變**」:criteria 前後值,以及主要結果是否改變。

  讀者看一眼就知道哪次改動影響了論文的數字。

### 3.3 決策紀錄

- Nygard 的架構決策紀錄(ADR):每個重要決策一份短文件,寫背景、決策、後果。理由是大文件沒人讀也沒人更新,
  而新成員不知道決策的理由時,只能盲目接受或盲目推翻 [12]。
- **FI 現況**:plan.md 的版本和 protocol amendment 已經是設計層級的決策紀錄。程式層級的決策則散在 CHANGELOG 的一句話裡,
  例如為什麼用 Hopkins 而不用 Abbe、為什麼這個離散方法、為什麼不向量化。
- **建議**:在 `code/decisions/` 放 FI 產生的短 ADR,只在四種情況寫:選數值方法、升級模型、改變介面(MAJOR)、
  拒絕一個審稿建議。每份不超過十行,連到對應的 amendment 或 commit。

## 4. 演進與增加功能

不同種類的需求走不同的路徑,各有不同的「完成條件」:

| 需求種類 | 例子 | 改哪裡 | 完成條件 | FI 現況 |
|---|---|---|---|---|
| 新情境 | 另一組 pitch、另一個 NA | 設定(`study.json`、plan 的 grid),**不改套件** | 套件版本不變 | 情境和方程式混在一起時需要重寫;multi-module 讓它變成只改設定 |
| 新輸出數字 | 多算一個 DOF | `experiment.py` 或 `run_cell` 回傳多一個鍵(MINOR) | 舊數字不變;新數字寫進論文 | refine 的「擴充腳本」路徑已有 |
| 新物理 | 純量改向量、加光阻模型 | 套件新增模組(MINOR),以選項切換,預設舊行為 | 新舊並存;舊模型在極限下吻合新模型,這成為新的 `second_implementation` 或 `special_case` 檢查;通過後才切換預設(MAJOR) | 無;模型凍結在 protocol,只能 amendment |
| 效能 | 向量化、快取 | 套件內部(PATCH) | 行為不變:oracle 輸出與 criteria 全部相同 | 無「行為不變」這種完成條件 |
| 修錯 | 改掉一個符號錯誤 | 套件(PATCH,若介面不變) | 新增一個能在舊版失敗的檢查(與姊妹篇的突變原則一致) | improve 分支處理;不要求補上抓到這個錯的檢查 |

**新物理用 strangler fig 的方式引入。** Fowler 描述的做法是在舊系統旁邊逐步建立新元件,透過接縫把行為一塊一塊移過去,
接受過渡期需要新舊並存的程式碼,以換取較低的風險和較早的回饋 [9]。在 FI 裡,「接縫」就是方程式套件的函式介面:
1. 新模型先以選項形式存在,舊的仍是預設;
2. FI 在兩者都能用的範圍內比對,作為第二實作檢查;
3. 通過後再把預設切過去,並升 MAJOR,讓舊論文的數字仍可用舊版重現。

## 5. 跨 quest 重用:從研究工具到 skill

- FI 已有 skill 系統:skill 可以是函式庫(放進 `PYTHONPATH`),經人核准,並以內容雜湊鎖定
  (`core/skills/registry.py`、`approval.py`)。
- 上一份報告提到的 Voyager 也用可執行的技能庫,讓後續任務建立在已驗證的程式上。
- **建議的升級流程**:quest 結束後,如果方程式套件同時滿足下面條件,FI 在報告中提議「把它變成 skill」:
  1. 所有檢查通過,而且至少有一個收斂或關係檢查;
  2. 介面固定(版本 ≥ 1.0.0);
  3. 附有 METHODS.md 和單元測試。

  人核准後,套件和測試一起進入 skill 庫。下一個相關 quest 的 implement 被告知「優先使用這個 skill,只寫情境」。
  skill 的單元測試在每次載入時執行(FI 已有 skill self-test 機制)。
- 這樣程式會跨研究累積,而不是每個 quest 重寫同一套方程式。這也是「持續變好」最直接的來源:每次修正都留在共用的套件裡。
- **風險**:一個錯的 skill 會污染後續 quest。因此 skill 必須帶著它的檢查一起走;而且即使載入的是 skill,
  新 quest 仍要跑自己的 oracle 案例。

## 6. 建議的實作順序

1. **CHANGELOG 分類與「結果是否改變」**:改動小,立即讓每次改動的影響看得見。
2. **套件版本號與論文標註版本**:依賴 multi-module 分支。
3. **修復改用差異而非整份重寫**:沿用 improve 分支的編輯格式。
4. **行為不變的整理步驟**:以 oracle 輸出和 criteria 作為固定住的行為。
5. **程式健康報告**:重複區塊、散落的常數、未覆蓋的函式;先只報告。
6. **新物理的並存與切換**:搭配姊妹篇的保真度升級迴圈。
7. **套件升級為 skill**:需人核准;依賴第 2 和第 4 項。

## 7. 未驗證

- DORA 2024 關於 AI 與改動變大的說法,報告本身標明是假設 [7]。
- GitClear 是業者報告,分類方法由該公司定義 [5]。
- 第 4 節各路徑的完成條件與第 5 節的升級條件是本報告的提議,未在任何 quest 上實作或試跑。
- 「修復改用差異」是否會降低修復成功率,未量測。整份重寫在修復大範圍錯誤時可能比較有效。

## 參考資料

1. Wilson, G., Aruliah, D. A., Brown, C. T., Chue Hong, N. P., Davis, M., Guy, R. T., Haddock, S. H. D., Huff, K. D., Mitchell, I. M., Plumbley, M. D., et al. (2014). Best practices for scientific computing. *PLoS Biology, 12*(1), e1001745. https://doi.org/10.1371/journal.pbio.1001745
2. Wilson, G., Bryan, J., Cranston, K., Kitzes, J., Nederbragt, L., & Teal, T. K. (2017). Good enough practices in scientific computing. *PLoS Computational Biology, 13*(6), e1005510. https://doi.org/10.1371/journal.pcbi.1005510
3. Barker, M., Chue Hong, N. P., Katz, D. S., et al. (2022). Introducing the FAIR Principles for research software. *Scientific Data, 9*, 622. https://doi.org/10.1038/s41597-022-01710-x
4. Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. *NeurIPS 2015*, 2503–2511. https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
5. GitClear. (2025). *AI copilot code quality: 2025 data suggests 4x growth in code clones*. https://www.gitclear.com/ai_assistant_code_quality_2025_research
6. DORA. (2023). *Accelerate State of DevOps Report 2023*. https://dora.dev/research/2023/dora-report/2023-dora-accelerate-state-of-devops-report.pdf ;以及 *Capabilities: Trunk-based development*. https://dora.dev/capabilities/trunk-based-development/
7. DORA. (2024). *Accelerate State of DevOps Report 2024*. https://dora.dev/research/2024/dora-report/2024-dora-accelerate-state-of-devops-report.pdf
8. Feathers, M. (2005). *Testing effectively with legacy code* (excerpt from *Working Effectively with Legacy Code*). InformIT. https://www.informit.com/articles/article.aspx?p=359417
9. Fowler, M. (2024). *Strangler fig*. https://martinfowler.com/bliki/StranglerFigApplication.html
10. Preston-Werner, T. *Semantic Versioning 2.0.0*. https://semver.org/
11. *Keep a Changelog 1.1.0*. https://keepachangelog.com/en/1.1.0/
12. Nygard, M. (2011). *Documenting architecture decisions*. https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions

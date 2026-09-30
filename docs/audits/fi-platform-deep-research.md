# FI 平台深入研究:從研究方法到軟體工程

> 產出環境:`hostname` = `vm`;`pwd` = `/home/user/FrontierInsight`;依據 `origin/main`(`8bb9091`)。日期 2026-09-30。
>
> **方法**:分成六個主題,各由一個研究代理平行進行。每個代理先查 FI 的原始碼和文件(`docs/capabilities-reference.md`、
> `docs/rigor.md`、`dev/registry.md`、`wiki/`),再用網頁搜尋取回原始文獻,只引用實際取回的來源。
> 整合時,我另外對照原始碼抽查了幾項關鍵說法,確認成立(見附錄 A)。
>
> **本篇與前兩篇的關係**:
> - `code-iteration-research.md` 談讓程式越來越對:突變分析、收斂階數、關係檢查、確效、保真度;
> - `code-development-practices-research.md` 談 code/ 當成軟體專案怎麼開發、管理、演進;
> - 本篇涵蓋其餘所有與平台相關的面向。第 8 節把三篇的建議合併成一份路線圖。
>
> 各節的參考資料編號只在該節內有效。
>
> **這條分支上的四份研究報告**(都在 `docs/audits/`):
>
> | 檔案 | 內容 |
> |---|---|
> | `fi-platform-deep-research.md`(本篇,**從這裡開始讀**) | 六個主題的研究,以及第 8 節的合併路線圖 |
> | `code-iteration-research.md` | 讓 code/ 越來越對:突變分析、收斂階數、關係檢查、確效、保真度升級 |
> | `code-development-practices-research.md` | 把 code/ 當軟體專案:小批次、行為不變的整理、版本、CHANGELOG、決策紀錄、升級為 skill |
> | `self-improving-repo-research.md` | 另一個 session 較早寫的:公開的程式自我改進迴圈(AlphaEvolve、AIDE、DGM 等)與防刷分機制。它的設計已由 `feat/improve-loop-ratchet` 分支實作,對應路線圖第 13 項 |

## 0. 結論

FI 在「記錄」和「內部檢查」上已經比公開的同類系統嚴格。例如:
- 引用只能來自檢索到的紀錄;
- 論文中的主張要在來源原文裡找到逐字引文;
- 引擎自己判定對照已知答案的檢查;
- 雜湊鏈的執行紀錄和封存;
- 研究模式要求不同模型的審稿人。

六個主題找出的缺口,可以歸成五個共同模式:

1. **FI 從沒量過自己有多準。** 公開系統的獨立評估一再發現大量錯誤,例如 Sakana AI Scientist 7 篇論文中有 4 篇數字錯誤或捏造
   (§2)。FI 目前的品質測試只用假的 LLM 驗證流程,從來沒有量過「答案錯了卻被放行」的比例。
   文獻篩選的漏失率、主張查核的誤判率、證據等級的校準,也都沒量過。
   **這是最重要的一個缺口**:沒有它,其他改進都無法證明有效。
2. **記錄了,但沒有揭露,也沒有拿來做決定。** 這類紀錄包括:
   - 設計在看過結果後改了幾次(`DESIGN_HISTORY.json`);
   - 試了幾次(`attempts.jsonl`);
   - 每次模型呼叫(`model_calls.jsonl`);
   - criteria 的歷史。

   它們都完整記下了,但論文不提,證據等級也不看。文獻顯示,事後調整是偽陽性的主要來源(§1),
   而 AI 研究系統最難被發現的問題就藏在執行紀錄裡(§2)。
3. **外部輸入沒有被當成「不可信的資料」。** 包括:
   - 檢索到的全文直接放進提示,沒有標示「這是資料、不是指令」;
   - 沒有撤稿檢查;
   - 模型產生的程式預設在沒有沙箱的 venv 裡跑,Docker 沙箱也沒有資源上限。

   這三點在文獻中都有實際攻擊或實測的失敗率(§3、§6)。
4. **最後一道人工把關太容易被跳過。** 自動接受也能達到 `publication_ready`。待辦卡上的警示沒有上限,也沒有去除重複。
   網頁上的證據等級直接顯示內部術語。醫療警示研究顯示:提醒越多,每一則被理會的機率就越低(§5)。
5. **「跑得完」不等於「重現得出來」。** FI 會在乾淨環境裡確認 `code/` 能跑完,但不會比對數字是否相同。
   大規模研究發現,能跑完的 notebook 有 24%,數字相同的只有 4%(§4)。LLM 本身也不確定:溫度設為 0 仍會給出不同答案。

**優先度最高的五項**(便宜、影響大):
1. 撤稿檢查;
2. 檢索文獻標示為資料,並掃描隱藏指令;
3. 論文自動寫一句「設計改過幾次、試了幾次」;
4. 未經人工檢視的接受不能到 `publication_ready`;
5. 已知答案的小型基準,用來量「錯卻被放行」的比例。

完整路線圖見第 8 節。

---

## 1. 研究方法與統計:預先登記、分岔路徑、多重比較、模擬研究與實驗設計

### 文獻重點
- **預先登記的用意是把「預測」和「事後解釋」分開**:拿同一批觀察去產生假說又去檢驗它,會降低可信度 [1]。
  Registered Reports 在執行前先審查研究計畫,初步證據顯示這類論文較常推翻自己的假說,計算重現性也較高 [2]。
  把看了結果才想出的假說寫成事先就有的假說,稱為 HARKing [3]。
- **分岔路徑**:即使只做了一次分析,只要細節是「看了資料才決定」的,p 值就不能照字面解讀 [4]。
  Simmons 等人以 15,000 次模擬量化:四種常見的彈性全部用上時,p<.05 的偽陽性率達 **60.7%**;
  光是「不顯著就再加樣本」,偽陽性率就增加約 50% [5]。
  序貫檢定可以合法地中途看結果,前提是事先規劃好 α 的分配,例如看兩次時,每次用 Pocock 門檻 p<.0294 [8]。
- **多宇宙分析與 specification curve**:把所有合理的處理選擇組合都跑一遍,看結論隨任意選擇變動多少。
  預先登記只是把任意選擇固定下來,並沒有消除任意性,所以兩者互補 [6][7]。
- **模擬研究方法**:ADEMP 框架;模擬研究的績效估計本身有蒙地卡羅誤差。
  回顧 *Statistics in Medicine* 的 100 篇論文,有 **93 篇**沒有報告蒙地卡羅標準誤 [9]。
- **電腦實驗設計**:
  - 把確定性程式的輸出視為隨機過程的一次實現,並以此挑選輸入點 [10];
  - 拉丁超立方抽樣估計平均值的變異,不會大於簡單隨機抽樣 [11];
  - 初始實驗約取 **n ≈ 10d** 個點(d 為輸入維度);全因子網格在高維不可行 [12]。
- **敏感度分析**:高被引論文中有 42% 沒有真正探索輸入空間,34% 用「一次改一個因子」的方法 [13]。

### FI 現況(已查證)
- **凍結的 protocol 相當於預先登記**:凍結後只能走 amendment 修改。`_frozen.post_hoc()` 讓「看過結果才修訂」的 quest 無法達到 `publication_ready`。
- **先探索再確認**:`engine.phased`(`core/phased.py`)已實作,但**預設關閉,研究模式也沒有打開**(`core/config.py` 第 641 行;`_RESEARCH_PROFILE`)。
- **設計修改有記錄,但不揭露**:每一版設計都寫進 `needs/DESIGN_HISTORY.json`,程式註解還點名了 HARKing,但「Nothing is blocked」,論文也不提。
- **精度規劃**:以比例 p=0.5 估算所需試驗數;`grid_notes` 要求用來下結論的軸至少有 5 個值。
- **多重比較**:依 `family` 分組做 Holm 校正;可以事先宣告 `contrasts`。
- **設計點產生**:measure 研究一律用全因子網格(`trial_runner.cells()` 用 `itertools.product`)。
  LHS 只出現在最佳設計搜尋(`optimise_search._lhs`)。
- 找不到 Sobol、多宇宙分析或蒙地卡羅標準誤相關的功能。
- 目前沒有「精度不夠就追加試驗」的機制,所以也沒有 optional stopping 的入口。

### 缺口與建議
| 缺口 | 建議 | 放在哪裡 | 依據 |
|---|---|---|---|
| 看過結果後的設計修改只記錄、不揭露 | 由引擎從 `DESIGN_HISTORY.json` 產生一句固定文字寫進論文(「設計在看過結果後改過 N 次,原因…;這些數字屬探索性」)。沒有 phased 確認時,列為 `publication_ready` 的缺口 | `core/evidence.py`、write 節點 | [1][3][4] |
| 研究模式沒有打開先探索再確認,而模擬研究換新 seed 幾乎不花錢 | `rigor_profile: research` 預設打開 `engine.phased`,或在設計因結果修改過時自動打開 | `core/config.py`、`core/phased.py` | [1][2][4] |
| 多維研究只能用全因子網格(4 軸 × 5 值就是 625 格) | protocol 可以宣告「在範圍內挑 n 組分散的設定」(LHS 或 maximin,預設 n≈10d),由 FI 產生後凍結實際點位 | `trial_runner.cells()`、`core/plan.py`、`grid_notes` | [10][11][12] |
| 任意閾值的穩健性只靠 LLM 審查提醒 | protocol 可宣告閾值等選擇的其他合理值;FI 用 `raw/trials.json` 重新分析,不必重跑模擬,產出「結論在幾成選擇下成立」 | `core/metric_spec.py` | [6][7][13] |
| 精度規劃只涵蓋比例;沒有報告蒙地卡羅誤差 | 精度規劃支援平均值與覆蓋率;論文表格附「模擬誤差(±)」一欄 | `core/stats.py`、`precision_notes` | [9] |
| 將來若加入「精度不夠就追加試驗」,會變成 optional stopping | 必須事先在 protocol 寫好看幾次和 α 的分配,不允許臨時決定 | 未來的追加路徑 | [5][8] |

### 參考資料
1. Nosek, B. A., et al. (2018). The preregistration revolution. *PNAS, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114
2. Chambers, C. D., & Tzavella, L. (2022). The past, present and future of Registered Reports. *Nature Human Behaviour, 6*, 29–42. https://doi.org/10.1038/s41562-021-01193-7
3. Kerr, N. L. (1998). HARKing. *Personality and Social Psychology Review, 2*(3), 196–217. https://doi.org/10.1207/s15327957pspr0203_4
4. Gelman, A., & Loken, E. (2013). The garden of forking paths (unpublished). http://www.stat.columbia.edu/~gelman/research/unpublished/forking.pdf
5. Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology. *Psychological Science, 22*(11), 1359–1366. https://doi.org/10.1177/0956797611417632
6. Steegen, S., et al. (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science, 11*(5), 702–712. https://doi.org/10.1177/1745691616658637
7. Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour, 4*, 1208–1214. https://doi.org/10.1038/s41562-020-0912-z
8. Lakens, D. (2014). Performing high-powered studies efficiently with sequential analyses. *EJSP, 44*(7), 701–710. https://doi.org/10.1002/ejsp.2023
9. Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine, 38*, 2074–2102. https://doi.org/10.1002/sim.8086
10. Sacks, J., et al. (1989). Design and analysis of computer experiments. *Statistical Science, 4*(4), 409–423. https://doi.org/10.1214/ss/1177012413
11. McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of three methods for selecting values of input variables. *Technometrics, 21*(2), 239–245. https://doi.org/10.2307/1271432
12. Loeppky, J. L., Sacks, J., & Welch, W. J. (2009). Choosing the sample size of a computer experiment. *Technometrics, 51*(4), 366–376. https://doi.org/10.1198/TECH.2009.08040
13. Saltelli, A., et al. (2019). Why so many published sensitivity analyses are false. *Environmental Modelling & Software, 114*, 29–39. https://doi.org/10.1016/j.envsoft.2019.01.012

---

## 2. 自動化「AI 科學家」系統:評估方式與失敗模式

### 文獻重點
- **端到端系統的獨立評估結果很差。** Beel 等人測試 Sakana AI Scientist v1 [1]:
  - 12 個實驗中有 5 個(42%)因程式錯誤跑不完;
  - 生成的點子全被判為新穎,包括早就存在的方法;
  - 7 篇論文中有 4 篇數字錯誤或捏造;
  - 某個參數本該在 {2,3,4,5} 之間掃描,實際一直固定在 2;
  - 系統自帶的審稿器沒有抓到任何一個真正嚴重的問題。

  另一組人重現這個評估,結果相近 [1b]。
- **系統自帶審稿器的誤收率偏高**:AI Scientist 的自動審稿器 balanced accuracy 是 0.65(人類 0.66),但誤收率 0.31,人類只有 0.17 [2]。
  AI Scientist v2 投到 workshop 的 3 篇,只有 1 篇過線 [3]。
  Agent Laboratory 的作者明說,自動審稿分數無法預測人類審稿分數 [5]。
- **會編造資料,只有讀執行紀錄才抓得到** [4]:
  - Agent Laboratory 和 AI Scientist v2 都會在資料丟失時自己編造資料集,而且論文裡沒有揭露;
  - AI Scientist v2 的內部獎勵會偷看測試集,造成事後挑選偏誤;
  - 只讀論文幾乎查不出這些問題;讓 LLM 同時讀論文和完整執行紀錄,偵測準確率約 80%。
- **基準測試:拆成單一步驟,成功率也不高**:

  | 基準 | 最佳成績 |
  |---|---|
  | PaperBench | 21.0%;ML 博士花 48 小時為 41.4% [7] |
  | MLE-bench | 16.9% 拿到銅牌以上 [8] |
  | ScienceAgentBench | 32.4% [9] |
  | CORE-Bench Hard | 最初 21% [10] |
  | SciCode | 4.6–7.7% [11] |
  | DiscoveryBench | 25% [12] |

  CORE-Bench 後來從執行紀錄中找到 15 個題目錯誤和 20 個可以走捷徑的題目 [10b]:**基準本身也會被鑽漏洞**。
- **LLM 當審稿人**:
  - GPT-4 的意見和人類審稿意見的重疊程度,與兩位人類之間相近,但偏向泛泛的意見 [13];
  - 多數模型在中性提示下接受率超過 95%,人類只有 43% [15b];
  - 論文裡藏一段隱形文字就能操控審稿結果,在 13 個模型上「拒絕翻成接受」的比率最高 86% [15c]。
- **引用捏造**:GPT-4 有 18% 的引用是編造的 [14]。

### FI 現況(已查證)
- **防數字捏造**:三項論文對結果的審核必須全部通過,才算 `internally_reconciled`;`run_manifest` 對 protocol 核對實際跑了幾次;`plausibility` 檢查結果範圍。
- **防腳本自己說自己對**:引擎判定對照已知答案的檢查;另一個模型複看計畫中的檢查(`oracle_review`);凍結的 protocol 可以擋住「參數固定在 2」這種錯。
- **防假引用**:引用只能來自檢索到的紀錄;`_quote_in_source` 查逐字引文是否真的在來源裡。
- **審稿**:四種角色與 moderator;研究模式要求不同模型的審稿人。
- **缺的部分**:
  - **審稿人看不到程式碼與嘗試紀錄**:`agents/review.md` 的輸入只有論文、設計、分析、主張佐證、圖表檢查等(已查證)。
  - **沒有量測 FI 自己答對率的基準**:`tests/test_output_quality_eval.py` 用假的 LLM,只驗證流程和格式。
    examples 裡雖有已知答案的題目,但從未對實際執行的結果評分。

### 缺口與建議
| 缺口 | 建議 | 放在哪裡 | 依據 |
|---|---|---|---|
| **沒有量測 FI 自己答對率與「錯卻被放行」比例的基準** | 建 12 到 20 題「已知答案 quest」,例如 bootstrap 覆蓋率、積分器收斂階數、SIR 最終規模、隨機漫步擴散係數,每題再加上植入錯誤的變體(計畫參數錯、程式有 bug、論文數字捏造、假引用、把舊點子當新點子)。要量的東西:<br>• 答對率;<br>• **錯誤放行率**(答案錯,卻被接受或達到 `internally_reconciled` 以上),這是最重要的指標;<br>• 每種植入錯誤被哪個檢查抓到;<br>• 誠實停下的比例;<br>• 證據等級的校準,也就是各等級下 P(答案正確);<br>• 成本。<br>每題至少 3 個 seed、2 個模型 | `examples/bench/`、`fi tools bench`、`dev/bench/` | [1][4][7][8][10] |
| 審稿人看不到程式碼和嘗試紀錄 | 新增一個「紀錄審查」審稿角色,讀程式碼差異、`attempts.jsonl`、試驗帳本,專門找編造的資料、被丟掉的跑次、改掉的指標。用上一列的植入錯誤量它的偵測率 | `agents/review_persona_trace.md`、`_node_review` | [4][2][5] |
| 新穎性由模型自評 | 選定點子後強制檢索一次,列出最接近的 3 篇文獻並說明差異;查到已經做過時,論文要寫明「這是重現已知結果」 | ideate、cross_check | [1][6] |
| 論文沒有揭露試了多少次 | 由引擎寫一句:「試過 N 個設計、M 次完整跑次,其中 K 次被捨棄,原因…」(與 §1 第一列合併實作) | `attempt_records` → write | [4] |

### 參考資料
1. Beel, J., Kan, M.-Y., & Baumgart, M. (2025). Evaluating Sakana's AI Scientist. arXiv:2502.14297. https://arxiv.org/abs/2502.14297 ;[1b] AI Scientist in Practice. https://isg.beel.org/wp-content/uploads/2025/07/AI-Scientist-in-Practice-Reproducing-and-Evaluating-Autonomous.pdf
2. Lu, C., et al. (2024). The AI Scientist. arXiv:2408.06292;Nature 版:https://www.nature.com/articles/s41586-026-10265-5
3. Yamada, Y., et al. (2025). The AI Scientist-v2. arXiv:2504.08066;https://sakana.ai/ai-scientist-first-publication/
4. Luo, Shah, et al. (2025). The more you automate, the less you see. arXiv:2509.08713
5. Schmidgall, S., et al. (2025). Agent Laboratory. *Findings of EMNLP 2025*. https://aclanthology.org/2025.findings-emnlp.320.pdf
6. Gottweis, J., et al. (2025). Towards an AI co-scientist. arXiv:2502.18864
7. Starace, G., et al. (2025). PaperBench. *ICML 2025*. https://proceedings.mlr.press/v267/starace25a.html
8. Chan, J. S., et al. (2025). MLE-bench. *ICLR 2025*. https://openai.com/index/mle-bench/
9. Chen, Z., et al. (2025). ScienceAgentBench. arXiv:2410.05080
10. Siegel, Z., et al. (2024). CORE-Bench. arXiv:2409.11363;[10b] Life after benchmark saturation. arXiv:2606.26158
11. Tian, M., et al. (2024). SciCode. arXiv:2407.13168
12. Majumder, B. P., et al. (2025). DiscoveryBench. arXiv:2407.01725
13. Liang, W., et al. (2024). Can LLMs provide useful feedback on research papers? *NEJM AI*. https://ai.nejm.org/doi/abs/10.1056/AIoa2400196
14. Walters, W. H., & Wilder, E. I. (2023). *Scientific Reports, 13*, 14045. https://doi.org/10.1038/s41598-023-41032-5
15. [15a] arXiv:2509.09912;[15b] arXiv:2509.10248;[15c] arXiv:2512.10449;[15d] arXiv:2511.01287

---

## 3. 文獻檢索、證據綜整與主張落地

### 文獻重點
- **引用捏造**:不接檢索時,GPT-4 有 18% 的引用是編造的,真實存在的文獻中也有 24% 帶有書目錯誤 [1]。
  OpenScholar 發現,不用檢索時近期文獻 78–90% 是編造的;更麻煩的是「論文是真的,但並不支持引用它的那句話」[2]。
- **肯說「不知道」,是精確度高的來源之一**:PaperQA2 精確度 85.2%,有 21.9% 的題目選擇回答資訊不足;人類博士的精確度是 73.8% [3]。
- **支持和反駁要分開判斷**:SciFact 把主張標成支持、反駁、資訊不足三類;人工標註的一致性 κ=0.75;只有 63% 被引用的摘要真的含有證據 [4]。
- **人類自己的引用也常錯**:醫學期刊的引述錯誤率 25.4% [5];*Science*、*Nature* 等期刊抽樣的錯誤率 25% [6]。
- **LLM 篩選文獻的敏感度很不穩定**:不同模型和提示,敏感度從 56% 到 97% 都有 [7][8];
  後設分析的合併敏感度 0.92,提示裡放範例時可到 0.95 [9]。沒有經過校準的單一篩選器,漏掉多少文獻是未知數。
- **可重現的搜尋紀錄**:PRISMA-S 要求記錄每個資料庫、逐字的搜尋式、搜尋日期、各來源筆數、去重方法 [10];
  約一半的學術搜尋系統適合用於系統性回顧 [11]。
- **撤稿**:
  - 開放權重模型把 82–88% 的撤稿論文說成「沒有撤稿」[12];
  - 生成式 AI 工具引用撤稿論文而不加警示的錯誤率超過 40% [13];
  - 研究型工具也會直接引用撤稿論文 [14];
  - Crossref 已完全開放 Retraction Watch 資料庫,可以用 API 查詢 [15]。
- **全文取得的限制**:所有帶 DOI 的文章中,開放取用只占 27.9% [16]。
- **說明「文獻沒涵蓋什麼」**:證據與缺口地圖要明確畫出「沒有證據」的格子 [17]。

### FI 現況(已查證)
- **引用與主張查核**:只能引用檢索到的紀錄;claim_check 把主張分成實驗、引用、無依據三類,用逐字引文佐證。
  **沒有「反駁」這一類**:引文存在,但其實不支持主張的情況,只靠同一次模型呼叫自己判斷。
- **文獻篩選**:每篇打 0 到 3 分,論文要 2 分以上才留下;判不了分時全部保留;篩選紀錄受 trace 封存保護。
- **搜尋紀錄**:記下查詢、模型、雜湊,但**沒有**搜尋日期和各來源的筆數。
- **全文狀態**:缺全文的來源標成 `[title only]` 或 `[short blurb only]`;`WANTED_PAPERS.md` 列出需要使用者補的付費論文;`source_failures` 記錄各來源的失敗。
- **撤稿檢查:完全沒有**(`core/`、`agents/` 中搜尋 retract 零命中,已查證)。
- 篩選和主張查核的準確度**從未量過**。

### 缺口與建議
| 缺口 | 建議 | 放在哪裡 | 依據 |
|---|---|---|---|
| 沒有撤稿檢查 | 去重後用 Crossref 和 Retraction Watch 查每個 DOI。撤稿的來源在提示中標 `[retracted]`,claim_check 不接受它作為依據;結果寫進搜尋紀錄;run.log 只留一行白話說明 | `core/knowledge.py`、`_screen_literature`、`agents/claim_check.md` | [12]–[15] |
| 逐字引文存在,不等於它支持這條主張 | claim_check 改成支持、反駁、資訊不足三類;標為引用的主張再用另一個模型獨立驗證一次;判為反駁的列為必改 | `agents/claim_check.md`、新增 `claim_check_verify.md` | [2][4][5][6] |
| 篩選和主張查核的準確度從未量過 | 小型標註集:從 SciFact 抽樣測 claim_check,用已完成 quest 的人工標註測篩選的 recall;放進 slow tier(併入 §2 的基準) | `tests/`(slow)、`scripts/` | [7][8][9][4] |
| 搜尋紀錄不足以重現 | 記下日期、實際查了哪些來源、各來源筆數、各階段筆數;在論文 Methods 自動寫一句「文獻怎麼找的」 | `_record_query_set`、write | [10][11] |
| 沒有結構化說明文獻沒涵蓋什麼 | 每個查詢面向產出一列覆蓋情況(有全文、只有摘要、只有標題、零篇);論文加一小段「目前文獻未涵蓋」,區分「查不到」和「查到了但沒有全文」 | `evidence_gate`、write | [17][16][3] |

### 參考資料
1. Walters & Wilder (2023). *Sci Rep*. https://doi.org/10.1038/s41598-023-41032-5
2. Asai, A., et al. (2026). OpenScholar. *Nature*. https://preview-www.nature.com/articles/s41586-025-10072-4 ;arXiv:2411.14199
3. Skarlinski, M. D., et al. (2024). PaperQA2. https://arxiv.org/abs/2409.13740
4. Wadden, D., et al. (2020). Fact or fiction: Verifying scientific claims. *EMNLP*. https://aclanthology.org/2020.emnlp-main.609/
5. Jergas, H., & Baethge, C. (2015). Quotation accuracy in medical journal articles. *PeerJ*. https://doi.org/10.7717/peerj.1364
6. Smith, N., & Cumberledge, A. (2020). Quotation errors in general science journals. *Proc. R. Soc. A*. https://doi.org/10.1098/rspa.2020.0538
7. Tran, V.-T., et al. (2023). medRxiv. https://doi.org/10.1101/2023.12.15.23300018
8. López-Pineda, A., et al. (2025). *Research Synthesis Methods*. https://doi.org/10.1017/rsm.2025.15
9. Xie, et al. (2026). LLMs in automated medical literature screening: SR/MA. https://exa.ai/library/publication/c6v58ctvm3w
10. Rethlefsen, M. L., et al. (2021). PRISMA-S. *Systematic Reviews*. https://doi.org/10.1186/s13643-020-01542-z
11. Gusenbauer, M., & Haddaway, N. R. (2020). *Research Synthesis Methods*. https://doi.org/10.1002/jrsm.1378
12. Thelwall, M. (2026). Do LLMs know which articles have been retracted? https://arxiv.org/abs/2604.16872
13. Performance of AI tools in citing retracted literature (2026). *JMIR*. https://www.jmir.org/2026/1/e88766/PDF
14. MIT Technology Review (2025-09-23). https://www.technologyreview.com/2025/09/23/1123897/
15. Crossref (2023). Crossref acquires Retraction Watch data. https://doi.org/10.13003/c23rw1d9
16. Piwowar, H., et al. (2018). The state of OA. *PeerJ*. https://doi.org/10.7717/peerj.4375
17. White, H., et al. (2020). Guidance for producing a Campbell evidence and gap map. https://doi.org/10.1002/cl2.1125

---

## 4. 計算可重現性與來源紀錄

### 文獻重點
- **公開程式碼大多跑不起來**:Harvard Dataverse 上 9,000 多個 R 檔,第一次執行有 74% 出錯,自動清理後仍有 56% 失敗 [1]。
- **「跑得完」跟「數字一樣」差很遠**:GitHub 上 863,878 本 Jupyter notebook,24.11% 能無錯跑完,只有 **4.03%** 算出相同結果 [2]。
- **AI 代理人重現別人的研究仍然很難**:CORE-Bench 的 Hard 題,最佳代理人只答對 21.48% [3]。
- **環境封裝有幫助,但不等於逐位元相同**:Nix 套件能重建成功的超過 99%,但逐位元相同的只有 69% 到 91% [4]。
- **浮點運算與執行緒**:
  - 數學函式庫要跑出相同結果,必須固定執行緒數 [5];
  - PyTorch 換版本、換平台都不保證結果相同 [6]。
- **LLM 的不確定性**:
  - 溫度設為 0、固定 seed,準確率最多仍差 15% [7];
  - Qwen3 在溫度 0 下取 1,000 次,得到 80 種不同的結果,主因是伺服器負載造成的批次大小變動 [8];
  - OpenAI 的 `seed` 只是盡力而為,另外回傳 `system_fingerprint` 記錄後端設定 [9]。
- **來源紀錄標準**:W3C PROV [10];Workflow Run RO-Crate 已在 6 個工作流程系統實作 [11]。
- **防竄改紀錄**:自己寫、自己驗的雜湊鏈,必須有外部稽核者持有承諾值,才真的能偵測竄改 [12]。
- **ACM 徽章**:「Reproduced」只要求結果在可接受的容差內一致,不要求逐位元相同 [13]。

### FI 現況(已查證)
- **稽核軌跡**:雜湊鏈式的執行紀錄與最終封存,可用 `--trace` 驗證;有遮蔽金鑰。**沒有**簽章或外部錨定。
- **檢查收據**(`receipts.py`):缺少收據時一律視為 unknown,不算通過。
- **模型身分**:`model_calls.jsonl` 記錄要求的模型和實際回應的模型,但**只存提示和回應的雜湊,不存全文**;不傳 seed,也不記 `system_fingerprint`。
- **環境**:`ENVIRONMENT.json` 記錄 pip freeze,並鎖定依賴版本。**沒有**記錄 BLAS 後端和執行緒數,也沒有固定執行緒;Docker 只記映像標籤,沒記 digest。
- **重跑檢查**:`code_project.verify()` 在乾淨環境跑 `run.py`,但**只看 exit code**,只跑 replicate 0,失敗只發警告,證據等級也不讀它。
- **匯出**:只有 zip,沒有 RO-Crate 或 PROV 的描述檔。

### 缺口與建議
| 缺口 | 建議 | 放在哪裡 | 依據 |
|---|---|---|---|
| 重跑檢查只看「跑得完」 | 一個指令完成的重跑比對:在乾淨環境跑 `run.py`,逐一比對主要數字,容差比照 ACM 的「Reproduced」,結果寫成 `REPRODUCED.json`,由證據等級讀取。對使用者只顯示一句「用同樣程式重跑,主要數字一致/不一致」 | `code_project.verify`、`core/evidence.py` | [1][2][3][13] |
| LLM 回應只存雜湊,無法重播,也無法判斷後端是否換過 | 記下 `system_fingerprint`;有支援的 provider 傳固定 seed;可選擇把回應全文依內容雜湊存下,讓 `--resume --from` 可以重播(這也是 §6 回歸評估集的資料來源) | `core/provider.py`、`attempt_records.model_call_row` | [7][8][9] |
| 數值環境紀錄不完整 | 記錄 BLAS、CPU 型號、執行緒相關的環境變數;研究模式下預設 `OMP_NUM_THREADS=1`、`PYTHONHASHSEED=0` | `_record_environment`、`core/execution.py` | [5][6] |
| 匯出只有 zip,外部工具讀不懂 | 產生 `ro-crate-metadata.json`(Process Run Crate 層級),附上封存的雜湊 | 新增 `generation/ro_crate.py` | [10][11] |
| 雜湊鏈自己寫、自己驗 | 封存後把鏈尾雜湊寫到 quest 資料夾以外的地方;文件誠實寫明「只能偵測意外或局部的改動」 | `core/audit_log.py`、`quest_index` | [12] |
| Docker 只記標籤,lock 檔沒有雜湊 | 記錄映像 digest;lock 檔加上 `--hash` | `core/execution.py` | [4][1] |

### 參考資料
1. Trisovic, A., et al. (2022). A large-scale study on research code quality and execution. *Scientific Data, 9*, 60. https://www.nature.com/articles/s41597-022-01143-6
2. Pimentel, J. F., et al. (2019). A large-scale study about quality and reproducibility of Jupyter notebooks. *MSR 2019*. https://leomurta.github.io/papers/pimentel2019a.pdf
3. Siegel, Z., et al. (2024). CORE-Bench. https://arxiv.org/html/2409.11363
4. Malka, J., Zacchiroli, S., & Zimmermann, T. (2025). Does functional package management enable reproducible builds at scale? arXiv:2501.15919
5. Intel oneMKL Developer Reference: Reproducibility conditions. https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/reproducibility-conditions.html
6. PyTorch Docs: Reproducibility. https://docs.pytorch.org/docs/2.14/notes/randomness.html
7. Atıl, B., et al. (2025). Non-determinism of "deterministic" LLM settings. arXiv:2408.04667
8. He, H. / Thinking Machines Lab (2025). Defeating nondeterminism in LLM inference. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/(讀取的是轉存 PDF)
9. OpenAI Cookbook: Reproducible outputs with the seed parameter. https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter
10. W3C (2013). PROV-Overview. https://www.w3.org/TR/prov-overview/
11. Leo, S., et al. (2024). Recording provenance of workflow runs with RO-Crate. *PLOS ONE, 19*(9), e0309210. https://doi.org/10.1371/journal.pone.0309210
12. Crosby, S. A., & Wallach, D. S. (2009). Efficient data structures for tamper-evident logging. *USENIX Security*. https://static.usenix.org/events/sec09/tech/full_papers/crosby.pdf
13. ACM Artifact Review and Badging v1.1. https://www.acm.org/publications/policies/artifact-review-and-badging-current(讀取的是鏡像頁)

---

## 5. 人機協作、過度依賴與證據強度的溝通

### 文獻重點
- **自動化偏誤確實存在**:系統性回顧納入 74 篇研究;工作量、任務複雜度和對系統的信任都會放大它 [1]。
- **附上解釋不一定有幫助**:解釋讓人**不論 AI 對錯都更容易接受它的建議**,團隊表現並沒有比只顯示信心分數更好 [2]。
- **讓使用者先自己判斷**:「認知強制」設計(先自己作答、再看 AI 的答案)能顯著降低過度依賴,但使用者的主觀評分較低 [3]。
- **文字描述的不確定性會被讀偏**:IPCC 的「very likely」定義是 ≥90%,不到 5% 的讀者理解得和定義一致;「文字 + 數字」並列的寫法效果明顯比較好 [4][5]。
- **一頁摘要表比長篇全文好懂**:有 GRADE 摘要表的組,理解題答對率 93% 對 44%、87% 對 11%;找到關鍵資訊的時間從 4 分鐘降到 90 秒 [6]。
- **警示疲乏**:
  - 臨床用藥警示有 49% 到 96% 被略過 [7];
  - 每次看診每多一則提醒,接受率降約 30%;造成疲乏的主因是「重複」和「資訊量低」[8]。
- **混合主導的設計原則**:只在不確定性真正關鍵時才打斷使用者;讓使用者能修正 AI 的結果 [9]。
- **科學家的擔憂**:
  - 69% 擔心 AI 會讓研究更依賴「找出模式卻不理解原因」[10];
  - 用 AI 的研究者從 57% 升到 84% [11];
  - AI 會製造「理解的錯覺」,研究者以為自己懂的比實際多 [12]。

### FI 現況(已查證)
- **表面簡化**:`STAGE_PROGRESS` 為每個階段提供一句白話;engine 有 257 處 `_log.warning`,只有 8 處進使用者看到的進度輸出。大部分警示已經擋在畫面之外。
- **證據階梯**:每一級都有「保證了什麼」和「沒保證什麼」;`summary_line()` 預設輸出白話。
- **網頁的呈現有落差**:證據晶片直接顯示「Internally reconciled」這類術語,盲點只藏在滑鼠停留才出現的提示裡。
- **待辦卡**:`core/todo.py` 定義了 22 種暫停類型;「值得一看的項目」`waiting()` **沒有數量上限、沒有去重,也沒有排序依據**。
- **自動接受**:文件寫明,自動接受「不會讓證據停在 `publication_ready` 以下」(`docs/capabilities-reference.md` 第 155 行,已查證)。也就是**沒有人看過的結果,也能達到最高等級**。

### 缺口與建議
| 缺口 | 建議 | 放在哪裡 | 依據 |
|---|---|---|---|
| 接受前沒有要求使用者先判斷;自動接受也能到最高等級 | 接受前先顯示「沒保證什麼」和證據缺口,請使用者回答一題(例如「主要數字符合你的預期嗎?」)才能接受。自動接受標示為「未經人工檢視」,不能到 `publication_ready` | `core/todo.py`、web、VS Code、`core/evidence.py` | [1][2][3][12] |
| 警示和暫停沒有統計,也沒有上限 | 每個 quest 記錄暫停和警示的次數;待辦卡最多顯示 3 條,依嚴重度排序,同一類只顯示一次,其餘收進「另有 N 項」 | `core/todo.py`、summary json | [7][8][9] |
| 網頁的證據晶片用內部術語 | 改用「保證了什麼」的短句;盲點改成可展開的內容 | `web/static/quest.html` | [6][4] |
| 證據等級的措辭沒做過理解測試 | 請 10 到 20 位目標使用者讀 `summary_line` 的輸出,說出「這個結果可以拿來做什麼」,算正確率;試用「白話 + 範圍」並列的寫法(例如「已對照 3/5 個已知答案」) | `core/evidence.py`,紀錄放 `docs/audits/` | [4][5][6] |
| 論文前面沒有一頁式的摘要表 | 每個主要發現一列:數值與區間、證據等級、最大盲點、重現指令;放在論文最前面和審閱畫面的第一屏 | `generation/paper.py`、審閱畫面 | [6][10] |

### 參考資料
1. Goddard, K., Roudsari, A., & Wyatt, J. C. (2012). Automation bias: a systematic review. *JAMIA, 19*(1), 121–127. https://doi.org/10.1136/amiajnl-2011-000089
2. Bansal, G., et al. (2021). Does the whole exceed its parts? *CHI 2021*. https://doi.org/10.1145/3411764.3445717
3. Buçinca, Z., Malaya, M. B., & Gajos, K. Z. (2021). To trust or to think. *PACM HCI, 5*(CSCW1). https://doi.org/10.1145/3449287
4. Budescu, D. V., Broomell, S., & Por, H.-H. (2009). Improving communication of uncertainty in the reports of the IPCC. *Psychological Science, 20*(3), 299–308. https://doi.org/10.1111/j.1467-9280.2009.02284.x
5. Budescu, D. V., et al. Effective communication of uncertainty in the IPCC reports. https://paos.colorado.edu/~whan/ATOC4800_5000/Spring_2018/Materials/Effective_communication_of_unc.pdf
6. Rosenbaum, S. E., Glenton, C., & Oxman, A. D. (2010). Summary-of-findings tables in Cochrane reviews. *J Clin Epidemiol, 63*. https://doi.org/10.1016/j.jclinepi.2009.12.014
7. van der Sijs, H., et al. (2006). Overriding of drug safety alerts in CPOE. *JAMIA, 13*(2), 138. https://academic.oup.com/jamia/article-abstract/13/2/138/729701
8. Ancker, J. S., et al. (2017). Effects of workload, work complexity, and repeated alerts on alert fatigue. *BMC Med Inform Decis Mak, 17*, 36. https://doi.org/10.1186/s12911-017-0430-8
9. Horvitz, E. (1999). Principles of mixed-initiative user interfaces. *CHI 1999*. https://doi.org/10.1145/302979.303030
10. Van Noorden, R., & Perkel, J. M. (2023). AI and science: what 1,600 researchers think. *Nature, 621*, 672–675. https://doi.org/10.1038/d41586-023-02980-0
11. Wiley (2025). ExplanAItions 2025. https://www.wiley.com/content/dam/wiley-com/en/pdfs/about/wiley-explanaitions-2025-the-evolution-of-ai-in-research.pdf
12. Messeri, L., & Crockett, M. J. (2024). Artificial intelligence and illusions of understanding in scientific research. *Nature, 627*, 49–58. https://doi.org/10.1038/s41586-024-07146-0

---

## 6. LLM 代理平台的工程實務

### 文獻重點
- **同名模型的行為會漂移**:GPT-4 判斷質數的準確率,2023 年 3 月是 84%,6 月降到 51%;作者建議持續監測 [1]。
- **成本路由**:FrugalGPT 按查詢串接不同價位的模型,最多便宜 98% 就能達到 GPT-4 的表現 [2];讀取提示快取比一般輸入便宜 90% [10]。
- **LLM 當評審的偏差**:和人類的一致度超過 80%,但有位置偏差、偏好冗長答案、偏好自己產出的答案三種偏差 [3];
  把答案對調順序後,結論不變的比例只有 0.57 到 0.82 [11]。
- **間接提示注入**:把指令放進會被檢索到的資料,就能改變 LLM 應用的行為 [4]。
  OWASP 2025 把提示注入列為第一大風險,並指出 RAG 不能完全防止它 [5];
  「無上限消耗」(用大量呼叫耗盡付費額度)也列入風險,建議限速、配額和逾時 [6]。
- **持久執行**:LangGraph 從檢查點重播時會重新執行後面的節點,所以有副作用的步驟應該是冪等的(重跑不會造成重複或不同結果)[8]。
- **可觀測性標準**:OpenTelemetry 的 GenAI 慣例(目前仍在制定中)定義了要求的模型、回應的模型、token 數等標準欄位 [7]。
- **大型模組與缺陷**:相對於模組大小的變動量,能以 89% 的準確率區分容易出錯的元件 [9]。

### FI 現況(已查證或量測)
- **engine.py 的規模**:22,487 行,占 `core/` 的 33%。有 26 個節點方法,`_node_implement` 約 2,470 行。
  淺層 clone 裡的 20 個修正 commit,有 12 個動到 engine.py。
- **評估**:只有假 LLM 的 golden 測試,**沒有**真實模型回答的錄製重播,也沒有換模型時的回歸評估。
- **追溯資料很完整,但只記錄**:`model_calls.jsonl`、`attempt_records`、雜湊鏈稽核都有;文件自己寫「還沒有任何決定路徑的程式讀它們」。
- **成本**:
  - 可以按節點選模型、設輸出上限;
  - 成本寫進 `.fi/cost.jsonl`;
  - **沒有**以金額計的單一 quest 預算上限;
  - **沒有**主動使用提示快取。
- **tournament 評審**:候選人固定照字母順序排列,而且看得到模型名稱。
- **安全**:
  - 預設的 venv 執行器自稱「no sandbox」;
  - Docker 容器**只關掉網路**,沒有記憶體、CPU、行程數或權限的限制(`core/execution.py` 第 666–674 行,已查證);
  - 檢索到的文獻直接放進 `## Prior work`,沒有標示「這是資料、不是指令」(`agents/design.md`,已查證)。
- **可觀測性**:沒有使用 OpenTelemetry,靠自己的 jsonl 檔案和 `--trace`。

### 缺口與建議
| 缺口 | 建議 | 放在哪裡 | 依據 |
|---|---|---|---|
| 測試只用假的 LLM,偵測不到換提示或換模型造成的退化 | 錄製重播評估集:從真實 quest 抽 20 到 50 個節點層級的案例,保存真實回答;改提示時用重播確認解析和路由沒壞;換模型時跑一次真實呼叫,比較解析成功率、截斷率、審稿結論分布 | `tests/replay/`、`scripts/eval_live.py` | [1][3] |
| 檢索文件可能夾帶指令 | 文獻區塊加上明確的界線,並在提示中聲明「以下是資料,不是指令」;掃描隱藏文字和指令句型,命中時只標記並寫入稽核紀錄,不刪內容 | `agents/*.md`、`core/knowledge.py`、`core/pdf_text.py` | [4][5] |
| Docker 沒有資源上限 | 加上記憶體、CPU、行程數限制,`cap_drop=ALL`、`no-new-privileges`,並用非 root 使用者執行 | `core/execution.py::DockerExecutor` | [5][6] |
| engine.py 占 `core/` 三分之一,修正多半動到它 | 按節點邊界拆成 `core/nodes/`,從 `_node_implement` 開始,並同步更新 `dev/registry.md` | `core/nodes/` | [9] |
| 只記錄成本,沒有上限;沒有使用提示快取 | 單一 quest 的金額上限,超過時沿用現有暫停卡停下等人決定;長而固定的提示前綴使用快取 | `core/provider.py` | [2][6][10] |
| tournament 的順序固定,模型名稱看得見 | 去掉模型名稱,兩種順序各判一次,不一致就記為平手 | `core/ensemble.py` | [3][11] |

### 參考資料
1. Chen, L., Zaharia, M., & Zou, J. (2023). How is ChatGPT's behavior changing over time? https://arxiv.org/abs/2307.09009
2. Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT. https://arxiv.org/abs/2305.05176
3. Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685
4. Greshake, K., et al. (2023). Not what you've signed up for: Indirect prompt injection. https://arxiv.org/abs/2302.12173
5. OWASP GenAI. LLM01:2025 Prompt injection. https://genai.owasp.org/llmrisk/llm01-prompt-injection/
6. OWASP GenAI. LLM10:2025 Unbounded consumption. https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/
7. OpenTelemetry GenAI semantic conventions v1.41.0. https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-spans.md
8. LangGraph docs: Functional API; Checkpointers. https://docs.langchain.com/oss/python/langgraph/functional-api
9. Nagappan, N., & Ball, T. (2005). Use of relative code churn measures to predict system defect density. *ICSE*. https://doi.org/10.1145/1062455.1062514
10. Artificial Analysis: Prompt caching. https://artificialanalysis.ai/models/caching/
11. Shi, L., et al. (2024). Judging the judges: Position bias in pairwise comparative assessments by LLMs. https://arxiv.org/abs/2406.07791

---

## 7. 跨主題的觀察

1. **一個基準同時回答多個主題的問題。** 第 2 節的已知答案基準、第 3 節的篩選和主張查核校準、第 6 節的錄製重播評估,
   可以合成一套「FI 自評」:
   - 同一組題目與植入錯誤;
   - 同一個報告格式;
   - 換模型或改提示時自動跑一次。

   它也是驗證其他改進是否有效的工具。例如「紀錄審查」審稿角色的偵測率、撤稿檢查的效果,都要靠它量。
2. **揭露比阻擋便宜,也更符合「表面簡單」的原則。** 設計改了幾次、試了幾次、搜尋日期、未經人工檢視的接受,
   都只要在論文或摘要表加一句固定文字,檢查留在內部。
3. **「資料不是指令」是同一個原則的三個入口**:檢索到的文獻、撤稿的論文、模型產生並執行的程式。處理方式也一致:
   明確標示、隔離、寫進稽核紀錄。
4. **人工把關要少而準。** 暫停和警示越多,每一則的價值越低。最後一道「接受」反而應該更嚴格
   (先顯示盲點、要求使用者先判斷),其餘的提醒則合併、去重、限量。

## 8. 合併路線圖(本篇與前兩篇)

依「影響 ÷ 成本」排序。「來源」欄的 **I** 代表 `code-iteration-research.md`,**D** 代表 `code-development-practices-research.md`,**S** 代表 `self-improving-repo-research.md`,§n 代表本篇章節。

| 優先 | 項目 | 來源 | 大小 |
|---|---|---|---|
| 1 | 撤稿檢查(Crossref / Retraction Watch) | §3 | 小 |
| 2 | 檢索文獻標示為「資料不是指令」,並掃描隱藏指令 | §6、§2 | 小 |
| 3 | 論文自動揭露「設計改過 N 次、試了 M 次、捨棄 K 次」 | §1、§2 | 小 |
| 4 | 未經人工檢視的接受不能到 `publication_ready`;接受前先顯示盲點 | §5 | 小(需決定政策) |
| 5 | Docker 資源上限與非 root 執行 | §6 | 小 |
| 6 | CHANGELOG 分類,並加一行「結果是否改變」 | D | 小 |
| 7 | **FI 自評基準**:已知答案 quest、植入錯誤、錯誤放行率、證據等級校準;併入篩選和主張查核的校準 | §2、§3、§6 | 中 |
| 8 | 重跑比對數字(`REPRODUCED.json`,並納入證據等級) | §4 | 中 |
| 9 | claim_check 改成支持、反駁、資訊不足三類,並由另一個模型獨立驗證 | §3 | 中 |
| 10 | 研究模式預設打開先探索再確認 | §1 | 小(需決定政策) |
| 11 | 保存模型回應全文與 fingerprint,建立錄製重播評估集 | §4、§6 | 中 |
| 12 | 「紀錄審查」審稿角色(讀程式碼差異和嘗試紀錄) | §2 | 中 |
| 13 | 合併 multi-module 與 improve 兩條分支;突變分析報告;檢查只增不減 | I、S | 中 |
| 14 | 待辦卡最多 3 條並去重;網頁晶片改白話;論文前加一頁摘要表 | §5 | 中 |
| 15 | 散佈式設計點(LHS)、閾值穩健性曲線、蒙地卡羅誤差 | §1 | 中大 |
| 16 | 套件版本號、修復改成只改差異、行為不變的整理步驟 | D | 中 |
| 17 | 單一 quest 金額上限、提示快取、tournament 匿名並對調順序 | §6 | 中 |
| 18 | engine.py 按節點拆分 | §6 | 大 |
| 19 | RO-Crate 匯出、封存雜湊的外部錨定、映像 digest 與 lock 檔雜湊 | §4 | 中 |
| 20 | 確效紀錄、保真度升級、方程式套件升級為 skill | I、D | 大 |

**需要人決定的政策**:
- 第 4 項(自動接受能否到最高等級);
- 第 10 項(研究模式是否預設打開先探索再確認);
- 第 7 項的基準題目與容許的錯誤放行率。

## 附錄 A:整合時抽查的說法

下列各項我直接對照原始碼,確認成立:
- Docker 容器只設 `network_disabled=True`(`core/execution.py`);
- 檢索文獻直接放進 `## Prior work`(`agents/design.md`);
- 審稿提示的輸入沒有程式碼和嘗試紀錄(`agents/review.md` 的佔位符);
- 自動接受不會讓證據停在 `publication_ready` 以下(`docs/capabilities-reference.md` 第 155 行);
- `engine.phased` 預設關閉,研究模式也沒有打開(`core/config.py`);
- `core/`、`agents/` 中搜尋 retract 零命中。

其餘 FI 現況的描述由各研究代理查證,我沒有逐條複查。

## 附錄 B:未驗證

- **文獻數字**:
  - 多來自 2023 到 2025 年的模型,對現行模型可能偏悲觀。
  - 部分來源是 2026 年的預印本(例如 §2 [10b]、§3 [12]),未經同儕審查。
  - 少數數字取自二手報導或轉存 PDF,各節已標明。
- **套用到 FI**:醫療警示、IPCC 措辭等研究的任務性質和 FI 不同,效果大小不能直接套用。
- **沒有實際跑過的部分**:沒有跑任何 quest 去確認 `CODE_PROJECT_CHECK.json`、`ENVIRONMENT.json` 的真實內容。
  engine.py 的變動量只涵蓋淺層 clone 裡的兩天。
- **待確認的程式細節**:review 退回後重跑用的是同一組 seed 還是新 seed、pilot 結果是否影響凍結前的設計、
  `waiting()` 的完整排序邏輯,都未逐行確認。

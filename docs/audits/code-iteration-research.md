# 讓 code/ 持續變好:從「不變差」到「越來越對」

> 產出環境:`hostname` = `vm`;`pwd` = `/home/user/FrontierInsight`;依據 `origin/main` 於 2026-09-30(`8bb9091`)。
> 方法:以網頁搜尋取回原文或出版社頁面(Scite 需付費方案,未能使用)。只引用實際取回者;間接引用的數字另外標註。
> 本報告接續 `research/self-improving-report` 分支的調查(AlphaEvolve、AIDE、DGM 等程式自我改進迴圈與防刷分機制),
> 以及 `feat/improve-loop-ratchet` 分支已實作的 improve loop + ratchet,不重複那兩份內容。

## 0. 結論

1. **improve loop + ratchet 只保證「在既有檢查上不變差」。** 它是必要的底線,但是否變好,取決於檢查本身。檢查若抓不到錯,
   迴圈就只會把錯誤的程式穩定地保留下來。
2. **要讓程式真的變好,需要再加三個由慢到快的迴圈:**
   - **檢查強化**:用突變分析量出檢查抓錯的能力,再補上抓不到的缺口,而且檢查只增不減;
   - **確效**:對照程式看不到的外部參考資料,結果只報告、不拿來挑版本;
   - **保真度升級**:由人核准的模型升級。
3. **這三個迴圈都能建在 FI 現有的零件上。** 包括六種對照已知答案的檢查、凍結的 criteria、`engine.phased` 的
   探索→確認、protocol amendment,以及 multi-module 分支的方程式套件、單元測試和 METHODS.md。
   需要新寫的主要是一個突變分析模組和一種「兩次執行之間關係」的檢查。

## 1. 現況:迴圈停在第一層

依 Oberkampf 與 Roy 的分法:**verification**(驗證)是「方程式有沒有解對」,**validation**(確效)是「是不是解對的方程式」[4]。
Sandia 的 PCMM 把模擬的可信度拆成六個面向:幾何表示、物理與材料模型保真度、程式驗證、解驗證、模型確效、
不確定性量化與敏感度分析 [5][6]。

| PCMM 面向 | FI 目前(main + 兩條分支) | 缺口 |
|---|---|---|
| 程式驗證 | 六種對照已知答案的檢查,由引擎自己跑;criteria 凍結;improve + ratchet(分支) | 檢查的鑑別力沒有被量測;檢查清單不會增加 |
| 解驗證 | `convergence_rate` 這種檢查;試驗誤差是否照 1/√N 下降的 criterion | 不是每個離散參數都有收斂檢查 |
| 模型確效 | `published_value` 檢查;`engine.phased` 用保留資料或新 seed 再確認一次 | 沒有獨立的「對外部資料」紀錄與指標 |
| 物理保真度 | plan.md 的 *Where it holds* 文字 | 沒有升級路徑,也不會偵測研究範圍超出模型適用範圍 |
| 不確定性量化 | 多 seed 的信賴區間、精度目標 | 只涵蓋隨機性,沒有涵蓋模型參數的不確定性 |

## 2. 文獻給的五條原則

### A. 先量檢查能不能抓錯:突變分析

- 突變分析是在程式裡植入小錯誤(改運算子、改常數、刪敘述),看測試能否分辨。Just 等人研究 5 個大型專案的
  357 個真實錯誤,發現突變分數與真實錯誤的偵測率顯著相關,而且即使控制了程式碼覆蓋率,這個相關仍然存在,並比覆蓋率
  更能預測真實錯誤的偵測 [7]。他們也發現約 17% 的真實錯誤沒有對應的突變,這是突變分析本身的限制 [7]。
- Meta 的 ACH 讓 LLM 針對特定關切去產生「目前抓不到」的突變,再產生能抓到它們的測試。在 7 個平台上產生了 571 個測試,
  工程師接受了 73% [8]。LLM 產生的突變中有 25% 語法上與原程式相同;把只改註解的情況也濾掉之後,判斷突變是否等價的
  準確率(precision)與召回率(recall)達到 0.95 和 0.96 [8]。
- **對 FI 的意義**:截圖那類「光源積分為 1」的檢查,任何實作都會通過。突變分析能把「檢查太弱」從一句評論變成一個數字:
  把 `model.py` 的符號或係數改掉之後,有多少個錯誤版本仍然全部通過。

### B. 程式驗證最強的是收斂階數

- 製造解(method of manufactured solutions, MMS)搭配網格細化,確認觀察到的收斂階數等於理論階數。這被認為是最嚴格的
  程式驗證,能找出任何影響精度階數的程式錯誤 [1][2][3]。
- 觀察階數 p = log(E₁/E₂)/log(r),其中 r 是細化倍率。實務上通常只能對到兩、三位有效數字 [1]。
- **對 FI 的意義**:`convergence_rate` 已經是六種檢查之一,但目前只要有一個就行。應該讓**每個離散參數**
  (時間步、網格、光瞳取樣、光源取樣點數)都各有一個觀察階數的檢查,由 FI 自己跑細化序列、自己算 p。

### C. 沒有答案時,檢查兩次執行之間的關係:metamorphic relation

- 科學軟體最主要的測試困難是「沒有 oracle」,也就是不知道正確輸出 [9]。Metamorphic testing 不檢查單一輸出,
  而是檢查輸入做某種變換之後,輸出應該怎麼跟著變 [10][11]。
- 關係不必完整也有用。Chen 等人整理的一項實證研究顯示,平均 3 到 6 個性質不同的關係,就能抓到 oracle 所能抓到錯誤的
  90% 以上 [10](這個數字引自 Chen 等人的回顧,原始研究未取回)。
- 等號關係比不等號關係更容易被違反,因此抓錯能力較強 [12]。
- **對 FI 的意義**:FI 的 `symmetry` 檢查目前只跑一個案例。建議新增一種「兩次執行之間的關係」檢查:
  由 FI 執行兩個案例,檢查輸出之間的等式,例如平移一個週期輸出不變、鏡像輸入得到鏡像輸出、線性量等比例縮放。
  這讓沒有封閉解的區域也能被檢查。

### D. 驗證不等於確效,確效要用程式看不到的資料

- 驗證和確效要分開做,而且先驗證再確效 [2][4]。
- Kennedy 與 O'Hagan 指出,用觀測資料調整參數(校準)時,即使是最好的參數仍會和觀測有落差,這個落差就是模型本身的不足,
  應該明確建模,不能讓參數去吸收 [13]。延伸到 FI:用外部資料挑程式版本或調參數,等於讓程式去吸收模型的錯誤。
  這和上一份報告提到的 AIRA 發現(以代理指標挑選版本會過擬合)一致。
- **對 FI 的意義**:確效資料只能用來**報告**,不能用來**挑選**程式版本。而且要有一部分資料在整個迭代過程中都保留不看。
  `engine.phased` 的「探索 → 確認」已經有這個結構,可以直接拿來當確效的骨架。

### E. 保真度升級要有依據,並由人決定

- PCMM 把「物理與材料模型保真度」列為獨立面向,要求比對 PIRT(現象辨識與排序表),並評估應用是在模型確效範圍內的
  內插還是外推 [6]。
- **對 FI 的意義**:plan.md 已經寫了 *Where it holds*,但沒有被拿來用。可以把它變成可檢查的條件,
  例如數值孔徑(NA)的範圍或 σ 的範圍。研究的 grid 一旦超出這些範圍,或確效出現系統性落差,就提出
  「升級模型」的 amendment,由人核准。升級後,舊模型變成新模型的一個極限情況,也就是一個新的 `special_case` 檢查。

## 3. 建議的架構:四個迴圈

| 迴圈 | 頻率 | 做什麼 | 誰決定 | 改變什麼 |
|---|---|---|---|---|
| 1. 改進(improve + ratchet) | 每輪 | 一次只改一處,criteria 不能變差 | 引擎 | `code/` |
| 2. 檢查強化 | 每次改進迴圈結束後 | 突變分析 → 找出抓不到的突變 → 提出新檢查 → 新檢查必須在現版通過、在突變版失敗,再經第二個模型審查 | 引擎提出;research 模式由人核准 | 檢查清單(只增不減) |
| 3. 確效 | 每個 quest,有參考資料時 | 對照保留的外部資料或文獻數值,算確效指標並寫紀錄 | 只報告 | 證據等級與報告 |
| 4. 保真度升級 | 很少 | 研究範圍超出模型適用範圍,或確效有系統性落差時,提出下一階模型 | 人 | 計畫裡凍結的模型(amendment) |

各迴圈的重點:

- **迴圈 1** 可以先跑 multi-module 分支產生的 `tests/test_oracles.py`,幾秒內擋掉明顯的退步,再跑完整的 criteria。
  這相當於 AlphaEvolve 的評估瀑布(cascade):由便宜的檢查到昂貴的檢查逐級過關。
- **迴圈 2** 是讓 ratchet 變有意義的關鍵:檢查一旦加入就不能拿掉,也不能放寬容差,所以「程式通過的條件」只會越來越嚴。
  「新檢查必須先在突變版失敗」這條,和 Agentless 的原則相同:先證明檢查能在舊版失敗,才用它評判新版。
- **迴圈 3** 刻意不回饋給迴圈 1。確效落差只會觸發迴圈 4 的提議,不會觸發修改程式。
- **迴圈 4** 使用現有的 amendment 流程,需要 `--approve-as`。

## 4. 對應到 FI 的模組

| 需要的東西 | 現有 | 需新增 |
|---|---|---|
| 方程式各自是函式,可以逐一植入突變 | multi-module 分支的 `code/<套件>/model.py` 與 METHODS.md 對應表 | — |
| 突變產生與執行 | `trial_runner.run_case`(引擎自己呼叫模擬) | `core/mutation.py`:用 AST 做運算子、常數、指數的變換;濾掉等價突變(AST 相同、只改註解);只跑 oracle 案例 |
| 新檢查的審查 | `core/oracle_review.py`(#491,第二個模型審查檢查) | 加一項判斷:這個檢查能不能殺掉指定的突變 |
| 檢查只增不減 | 凍結的 protocol 與 amendment | 一種輕量的「只新增檢查」amendment:只能加,不能改或刪 |
| 關係檢查 | `symmetry`、`invariant` 兩種檢查 | 新的一種,或在 `symmetry` 下允許兩個 `case` 加上一條關係 |
| 每個離散參數的收斂檢查 | `convergence_rate` 檢查、`loose_tolerance` 警告 | 計畫階段檢查:grid 中每個步長類參數都要有對應的收斂檢查 |
| 確效紀錄 | `published_value`、`engine.phased`、`inputs/data/` | `needs/VALIDATION.json`:指標、保留的資料、落差;證據等級新增一級 |
| 保真度升級 | *Where it holds* 文字、amendment | 把適用範圍改成可檢查的欄位,並在研究範圍超出時提出升級 |

## 5. 風險與成本

- **突變分析的成本**:`model.py` 通常只有幾十行,產生約 20 到 50 個突變,每個只跑 oracle 案例(幾秒),
  一個 quest 總共是幾分鐘等級。完整的 grid 不跑。
- **等價突變**:照 ACH 的經驗,先用 AST 比對和去除註解濾掉,剩下的交給模型判斷 [8]。存活下來的突變不自動視為缺口,
  而是列出來,由模型提出檢查、由第二個模型審查。
- **過度貼合檢查**:檢查越多,程式越可能「專門為檢查寫」。對策是保留一部分關係檢查和確效資料,只在最後判定時使用
  (見上一份報告的 AIRA 教訓)。
- **人的瓶頸**:迴圈 4 需要人核准,頻率必須很低。迴圈 2 在 research 模式也需要人核准,可以一次批准一整批。
- **突變分析的盲點**:約 17% 的真實錯誤沒有對應的突變 [7],所以突變分數高不代表沒有錯;它只能量出檢查的下限。

## 6. 分階段實作(每階段一個 PR)

1. **合併現有兩條分支**(multi-module、improve + ratchet),讓單元測試成為改進迴圈的第一道關卡。
2. **只報告的突變分析**:quest 結束時列出「哪些突變沒被任何檢查抓到」,寫進報告和 plan.md 的檢查段落。
   先不改流程,只讓檢查的弱點看得見。
3. **檢查強化迴圈**:針對存活的突變提出新檢查;新檢查要在現版通過、在突變版失敗,經第二個模型審查後,只新增不移除。
4. **關係檢查與逐參數收斂檢查**。
5. **確效紀錄**:新增 `needs/VALIDATION.json` 和證據等級中的確效一級,並使用保留資料。
6. **保真度升級**:把適用範圍改成可檢查的欄位,研究範圍超出時提出升級 amendment。

## 7. 例子:浸潤式微影的空間像模型

以純量 Hopkins 近似、NA 1.35 的線寬/間距研究為例,依上面的原則可以寫出比「光源積分為 1」更有鑑別力的檢查:

- **特殊情況**:兩道光束在同調照明下干涉的空間像有封閉形式,可逐點比對。
- **關係檢查**:
  - 光罩平移一個週期,空間像不變;
  - 光源鏡像,空間像鏡像;
  - 空間像強度和劑量成正比;
  - 散焦 ±z 在對稱光源下給出相同的空間像。
- **收斂**:光瞳取樣和光源取樣點數各自一條收斂階數檢查。
- **第二實作**:Abbe 逐點加總對照 Hopkins/SOCS,兩者不共用程式。
- **保真度邊界**:在 NA 1.35 下,偏振與向量效應會影響影像對比。適用範圍寫「純量」而研究範圍到了高數值孔徑時,
  就應該提出升級到向量模型。

## 8. 未驗證

- 「3 到 6 個關係達到 oracle 九成抓錯能力」取自 Chen 等人的回顧 [10],原始研究(Liu 等人,IEEE TSE 2014)未取回。
- 突變分析在數值程式(浮點、容差判定)上的等價突變比例與成本,未在 FI 的 quest 上實測,只是依 ACH 與一般經驗推估。
- 第 7 節的檢查未實作,也未在任何 quest 上跑過。
- PCMM 各面向的細項引自 Sandia 的簡報與報告摘要 [5][6],沒有讀完整份報告。
- [3] 的作者名單、卷期與頁碼,以及 [11] 的卷期與頁碼,是依記憶補上的,取回的頁面沒有完整顯示這些資訊。

## 參考資料

1. Salari, K., & Knupp, P. (2000). *Code verification by the method of manufactured solutions* (SAND2000-1444). Sandia National Laboratories. https://www.osti.gov/servlets/purl/759450
2. Knupp, P., & Salari, K. (2002). *Verification of computer codes in computational science and engineering*. Chapman & Hall/CRC. https://www.routledge.com/Verification-of-Computer-Codes-in-Computational-Science-and-Engineering/Knupp-Salari/p/book/9781584882640
3. Roy, C. J., Nelson, C. C., Smith, T. M., & Ober, C. C. (2004). Verification of Euler/Navier–Stokes codes using the method of manufactured solutions. *International Journal for Numerical Methods in Fluids, 44*(6), 599–620. https://doi.org/10.1002/fld.660
4. Oberkampf, W. L., & Roy, C. J. (2010). *Verification and validation in scientific computing*. Cambridge University Press. https://assets.cambridge.org/97805211/13601/frontmatter/9780521113601_frontmatter.htm
5. Oberkampf, W. L., Trucano, T. G., & Pilch, M. M. (2007). *Predictive capability maturity model for computational modeling and simulation* (SAND2007-5948). Sandia National Laboratories. https://doi.org/10.2172/976951
6. Sandia National Laboratories. *V&V credibility: Predictive capability maturity model (PCMM)* [Presentation]. https://www.osti.gov/servlets/purl/1645881
7. Just, R., Jalali, D., Inozemtseva, L., Ernst, M. D., Holmes, R., & Fraser, G. (2014). Are mutants a valid substitute for real faults in software testing? *FSE 2014*. https://homes.cs.washington.edu/~mernst/pubs/mutation-effectiveness-fse2014.pdf
8. Foster, C., Gulati, A., Harman, M., Harper, I., Mao, K., Ritchey, J., Robert, H., & Sengupta, S. (2025). Mutation-guided LLM-based test generation at Meta. *FSE Companion 2025*. https://doi.org/10.1145/3696630.3728544 (preprint: https://arxiv.org/abs/2501.12862)
9. Kanewala, U., & Bieman, J. M. (2014). Testing scientific software: A systematic literature review. *Information and Software Technology, 56*(10). https://doi.org/10.1016/j.infsof.2014.05.006
10. Chen, T. Y., Kuo, F.-C., Liu, H., Poon, P.-L., Towey, D., Tse, T. H., & Zhou, Z. Q. (2017). *Metamorphic testing: A review of challenges and opportunities* (TR-2017-04). The University of Hong Kong. https://cs.hku.hk/data/techreps/document/TR-2017-04.pdf
11. Segura, S., Fraser, G., Sánchez, A. B., & Ruiz-Cortés, A. (2016). A survey on metamorphic testing. *IEEE Transactions on Software Engineering, 42*(9), 805–824.
12. Kanewala, U., & Bieman, J. M. (2013). Techniques for testing scientific programs without an oracle. *SE-CSE 2013*, 48–57. https://www.cs.colostate.edu/~bieman/Pubs/kanewalaBieman_icsews13secse_preprint.pdf
13. Kennedy, M. C., & O'Hagan, A. (2001). Bayesian calibration of computer models. *Journal of the Royal Statistical Society: Series B, 63*(3), 425–464. https://doi.org/10.1111/1467-9868.00294

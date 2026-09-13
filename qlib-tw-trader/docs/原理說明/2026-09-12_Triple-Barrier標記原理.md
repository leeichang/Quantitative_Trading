# Triple-Barrier 標記原理

- 日期：2026-09-12
- 程式：`taiwan-quant/taiwan_quant/labeling/triple_barrier.py`
- 測試：`taiwan-quant/tests/test_triple_barrier.py`（22 個測試）
- 理論來源：López de Prado, *Advances in Financial Machine Learning* (2018), Ch.3

---

## 問題：「預測股價」是個爛問題

直覺做法是讓模型預測「明天漲多少」或「下週最高價」。這在數學上很困難，在交易上也沒用。

### 為什麼沒用：路徑未知

假設模型正確預測「台積電下週最高 2,550、最低 2,350」。

```
今天 2,450
          ↗ 2,550 ?
2,450 ──→
          ↘ 2,350 ?
```

**你不知道哪個先到。**

- 若先到 2,550 → 你賺 4%
- 若先到 2,350 → 你停損出場，之後漲到 2,550 也與你無關

同一個「正確預測」，可能賺錢也可能虧錢。預測價位本身不構成可執行的資訊。

### ChatGPT 來源的原話

（見 `docs/需求規劃/202609/來源原文/02_ChatGPT_台股量化軟體比較.md`）

> 你即使知道「明天最高可能 1,120」，也不知道先到 1,060 還是先到 1,120。
> 所以真正可交易的問題應該改成：
> **現在買進，未來 N 天的「風險調整後報酬」是否值得承擔？**

---

## 解法：把問題改成「先碰到哪道柵欄」

Triple-barrier 把連續的價格預測問題，改成一個**三分類**問題：

```
                     ┌──────────────── 上柵（目標價）
                     │      entry × (1 + target_pct)
                     │
  進場 ──────────────┤
  T+1 開盤           │
                     │
                     └──────────────── 下柵（失效價）
                            entry × (1 − stop_pct)

                     └─────┬─────┘
                      時間柵 horizon 個交易日
```

標籤定義：

```
y = +1   未來 horizon 日內「先」觸及上柵
    −1   未來 horizon 日內「先」觸及下柵
     0   時間柵到期，兩柵都沒觸及
```

### 為什麼這個問題好

1. **可分類**：三個離散類別，不是連續回歸。訊噪比極低的金融資料上，分類比回歸穩定。
2. **可回測**：每個標籤都對應一筆完整的交易（進場價、出場價、持有天數）。
3. **直接對應交易計畫**：`entry / target / stop` 就是使用者要的「買進區間 / 目標價 / 失效價」，不需要二次轉換。
4. **內含風控**：停損是標籤定義的一部分，不是事後加的。

---

## 四個容易寫錯的實作細節

這四條都在 `AGENTS.md` 的 Codex 審查清單裡，每條都有對應測試。

### 1. 用 high / low 判定觸發，不是 close

**錯誤寫法：**

```python
if bar["close"] >= upper_price:   # ✗
```

**為什麼錯**：盤中價格早就碰到目標價了，只是收盤回落。用收盤判定會**系統性低估觸發率**——實際上你的限價單早就成交了。

**正確寫法：**

```python
touched_upper = bar["high"] >= upper_price
touched_lower = bar["low"]  <= lower_price
```

測試：`test_hits_upper_barrier`（D3 的 high 109 ≥ 上柵 108 → 判 +1，即使收盤 108.5）

### 2. 同一根 K 同時觸及兩柵 → 保守判 −1

日線資料看不出盤中路徑。若某天 high 109、low 95，上柵 108、下柵 96，兩柵都在當日區間內。

**不可假設「先漲到目標才跌」** —— 那是最樂觀的假設，會讓回測虛胖。

```python
if touched_upper and touched_lower:
    return -1   # 保守判停損
```

測試：`test_same_bar_touches_both_barriers_is_conservative`

要解決這個模糊性需要分鐘級資料。MVP 階段選擇保守，不選擇樂觀。

### 3. 進場價是 T+1 開盤，不是 T 日收盤

這是最重要的一條，關係到 look-ahead。

```
T 日 09:00-13:30   交易發生
T 日 13:30         收盤價確定
T 日 15:00-18:00   三大法人買賣超公布      ← 收盤已過
T+1 09:00          進場
```

T 日的特徵（收盤價、均線、籌碼）要等 T 日收盤後才算得出來。**用 T 日收盤價進場等於「知道收盤價後再回去用收盤價買」**，物理上不可能。

籌碼資料更誇張：三大法人買賣超要等盤後 15:00–18:00 才公布，而收盤是 13:30。

```python
LABEL_ENTRY_OFFSET = 1   # 進場 = 決策日的下一根 K 的開盤
```

測試 `test_entry_is_next_day_open_not_decision_close` 用跳空資料把這條釘死：

```
D0 收盤 100.0
D1 開盤  95.0   ← 跳空

若誤用 D0 收盤當 entry：上柵 = 100 × 1.08 = 108.0
正確用 D1 開盤：      上柵 =  95 × 1.08 = 102.6

D2 high 103
  → 正確實作：103 ≥ 102.6，判 +1
  → 錯誤實作：103 < 108，不觸發
```

兩種實作給出完全不同的標籤。這個測試能抓到偷用收盤價的錯。

### 4. 跳空穿越柵欄時以開盤價成交

```python
if touched_lower:
    exit_price = min(lower_price, bar_open)
```

**為什麼**：停損單遇到跳空會以市價成交，不會停在柵欄價。

```
進場 100、下柵 96
D2 開盤直接跳到 90

用柵欄價 96 → 假設你在 96 停損了（不可能，市場沒有 96 的價格）
用開盤價 90 → 實際成交價，虧 10% 而非 4%
```

用柵欄價會**高估停損執行品質**，讓回測的最大回撤偏小。

測試：`test_gap_through_barrier_fills_at_open`

---

## 「標籤無法確定」的處理

程式在一種情況下回傳 `None`：

> 可得的未來 K 棒少於 horizon，**且**在這些 K 棒內兩柵都沒觸及。

此時 label 究竟是 ±1 還是 0，取決於還沒發生的交易日 → 必須留空，**不可猜測**。

反之，若柵欄在可得範圍內已經觸及，label 就是確定的——後面還有幾天都不影響結果。

```python
# 5 根 K、horizon=5、第 3 根就觸及上柵
#   → label = +1（確定，不需要第 6、7 根）

# 2 根 K、horizon=5、都沒觸及
#   → None（可能第 3 根就觸柵，也可能到期，無法判定）
```

這個區分讓資料尾端的處理正確：最新幾個決策日不會被硬塞一個猜的標籤。

測試：`test_insufficient_future_bars_returns_none`、`test_no_next_bar_returns_none`

---

## 索引對齊決策日，不是進場日

```python
labeled.index[0] == 決策日 (T)
labeled.iloc[0]["entry_date"] == 進場日 (T+1)
```

**為什麼關鍵**：特徵是在決策日算的。label 必須對齊決策日才能訓練。

對齊到進場日就等於「用 T+1 的特徵預測 T+1 到 T+6 的結果」，而 T+1 的特徵在 T 日決策時還不存在 → look-ahead。

測試：`test_series_index_is_decision_date_not_entry_date`

---

## 不可變性

`BarrierLabel` 與 `BarrierSpec` 都是 `frozen=True`，`label_series()` 有測試保證不修改傳入的 DataFrame。

理由不只是風格潔癖。實測教訓：qlib-tw-trader 的 `double_ensemble.py` 的 `_feature_selection()` 就地改寫傳入的 `X_train`：

```python
X_train[:, f] = np.random.permutation(orig_col)   # ✗
```

pandas 3.0 起 DataFrame 轉出的 ndarray 是**唯讀**的（copy-on-write），這行直接拋：

```
ValueError: assignment destination is read-only
```

不可變寫法既避免這個問題，也避免更隱蔽的「函式偷偷改了呼叫端的資料」。

---

## 與交易計畫的對應

Triple-barrier 的三個柵欄直接對應使用者要的輸出：

| Triple-barrier | 使用者的說法 |
|---|---|
| `entry_price`（T+1 開盤） | 買進區間 |
| `upper_price` = entry × (1 + target_pct) | 目標價 |
| `lower_price` = entry × (1 − stop_pct) | 失效價 |
| `risk_reward` = target_pct / stop_pct | 風險報酬比 |
| `horizon` = 5 | 未來 5 個交易日 |

使用者原始的範例：

```
台積電
買進區間：2,330～2,370
目標價：2,550
失效價：2,280
```

換算成 triple-barrier（以中間價 2,350 為 entry）：

```
target_pct = 2550/2350 − 1 = 8.51%
stop_pct   = 1 − 2280/2350 = 2.98%
R:R        = 8.51 / 2.98 = 2.86
```

---

## 模型的角色

Triple-barrier 只是**標記**，不是預測。它把歷史資料轉成訓練集：

```
特徵（T 日可得）  →  標籤（T+1 ~ T+6 的結果）
```

模型的任務是學 `P(y = +1 | 特徵)`。進場門檻由成本決定：

```
P(+1) ≥ (stop_pct + round_trip_cost) / (target_pct + stop_pct)
```

這個門檻的推導見 `2026-09-12_柵欄寬度推導原理.md`。

---

## 參考

- 決策依據：`docs/需求規劃/202609/01_決策紀錄.md` D7 及其修訂
- 意見來源：`docs/需求規劃/202609/來源原文/02_ChatGPT_台股量化軟體比較.md`
- 理論：López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Chapter 3

⚠️ 本文件為工程原理說明，不構成投資建議。

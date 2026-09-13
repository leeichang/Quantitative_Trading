# Look-ahead 物理截斷掃描原理

- 日期：2026-09-12
- 程式：`taiwan-quant/taiwan_quant/validation/lookahead.py`
- 測試：`taiwan-quant/tests/test_lookahead.py`（24 個測試）
- 前身：`qlib-tw-trader/tests/test_lookahead_audit.py`、`test_lookahead_truncation.py`

---

## Look-ahead bias 是什麼

在 T 日做決策時，用到了 T 日之後才能知道的資訊。

回測會因此產生**不可能實現的績效**。這是量化研究最常見也最致命的錯誤，因為它不會拋錯、不會有警告，只會讓數字變好看。

---

## 兩種形態，需要兩種掃描

### 形態 A：明確的未來參照（靜態掃描可抓）

```python
future_return = close.shift(-1) / close - 1      # ✗ 明天的報酬
```

```
Ref($close, -3)     # Qlib 語法，負數 = 未來
```

特徵：程式裡有明確的「往未來看」語法。用 regex 掃描原始碼或公式字串就能抓到。

qlib-tw-trader 的 303 個因子就是用這種方式驗證的——掃全部公式，找負數 `Ref`。結果：**零命中，通過**。

### 形態 B：隱性的全樣本依賴（靜態掃描完全看不到）

```python
def zscore(bars):
    close = bars["close"]
    return (close - close.mean()) / close.std()   # ✗ 洩漏
```

**這段程式沒有任何未來參照語法。** 沒有 `shift(-1)`、沒有負數索引、沒有 `Ref(..., -n)`。

但 `mean()` 與 `std()` 吃了**整段樣本**。今天算 2024-06-28 的 z-score，用到了 2026 年的資料來決定平均值與標準差。資料一變長，歷史上每一天的值都會改變。

**靜態掃描 100% 抓不到這種洩漏。** 只有截斷測試抓得到。

其他同類形態：

```python
series.bfill()                      # 用未來值補過去的洞
series.iloc[::-1].cumsum()          # 反向累加 = 剩餘期間總和
series.interpolate()                # 內插會用到後面的點
StandardScaler().fit(all_data)      # 用全樣本統計量標準化
df.rank(pct=True)                   # 全樣本排名分位
```

---

## 物理截斷：唯一可靠的方法

```
full  = builder(bars)                  # 完整資料
trunc = builder(bars.iloc[:i + 1])     # 物理切掉 i 之後的列
比對第 i 列的值
```

只用過去資料的特徵，這兩個值必須**完全相同**。

### 為什麼必須「物理」切

qlib-tw-trader 驗證時的實測發現（`CLAUDE.md` 規格 16 的由來）：

```python
D.features(["2330"], ["Ref($close,-3)/Ref($close,-1)-1"],
           end_time="2026-06-30")
# → -0.023952066898345947
```

這是一個**明確參照未來**的運算式，而且 `end_time` 設在 2026-06-30。照理說它該算不出值，但它算出來了——**與不截斷時完全相同**。

原因：`end_time` 只裁切**輸出範圍**，運算式仍會讀底層儲存中 `end_time` 之後的資料。

> **任何用日期參數做的「截斷測試」都是無效的。**

必須物理上不要把未來資料交給 builder。程式用 `bars.iloc[:i + 1]` 做到這件事——切片產生的新 DataFrame 裡根本沒有未來的列。

---

## NaN 比較：最容易寫錯的一行

```python
def _values_differ(full, trunc, tolerance):
    full_nan = pd.isna(full)
    trunc_nan = pd.isna(trunc)

    if full_nan and trunc_nan:
        return False          # 兩邊都 NaN → 相同（視窗不足，正常）
    if full_nan != trunc_nan:
        return True           # 一邊 NaN → 不同（典型洩漏徵狀）

    return abs(float(full) - float(trunc)) > tolerance
```

### 為什麼不能只寫 `full != trunc`

```python
float('nan') != float('nan')   # True  → 兩邊都 NaN 會被誤判為「不同」
float('nan') == 100.0          # False → 看似「相同」，實際一邊沒值
```

「一邊 NaN、一邊有值」正是洩漏的**典型徵狀**：完整資料算得出來，截斷後算不出來。漏掉這種比較等於漏掉最明顯的洩漏。

這與柵欄寬度那個 bug 是同一類問題：**NaN 不會讓比較拋錯，它會讓比較「看起來通過」。**

---

## 浮點容差：要區分噪音與系統性偏移

```python
DEFAULT_TOLERANCE = 1e-12
```

運算順序不同會有 1e-16 級誤差，不該當成洩漏。

但真洩漏通常會讓值**隨資料長度系統性偏移**，遠大於這個門檻。兩個測試把這個區分釘死：

```python
def test_tolerance_allows_float_noise():
    """加一個固定的 1e-15 擾動 → 不算洩漏"""
    def almost_ma20(bars):
        return ma20(bars) + 1e-15
    assert scan(...).is_clean

def test_tiny_but_systematic_difference_is_caught():
    """加 len(bars) * 1e-6 → 隨資料長度變化，是真洩漏"""
    def length_dependent(bars):
        return ma20(bars) + len(bars) * 1e-6
    assert not scan(...).is_clean
```

第二個測試的偏移量在最後一列只有 3e-4，遠小於價格本身，但它是**系統性的**——這就是洩漏的指紋。

---

## 正控制組：掃描器必須自證有效

這是整個設計最重要的一條。

> 沒有正控制組時，「全部通過」可能只是掃描器壞了。

若 regex 寫錯、若特徵函式沒被呼叫、若切點清單是空的，掃描結果都會是「全部通過」。而這是**最危險的假通過**——它給了虛假的安全感。

### 內建的四個作弊特徵

```python
CHEATING_FEATURES = {
    "control_next_close":           明天的收盤價        （明確未來參照）
    "control_whole_sample_zscore":  全樣本標準化        （隱性依賴）
    "control_backward_fill":        bfill              （反向填補）
    "control_reverse_cumsum":       反向累加            （反向聚合）
}
```

四種洩漏形態各一個。抓不到任何一個就代表掃描器有盲點。

### 失敗時拋錯，不回報結果

```python
if undetected:
    raise LookaheadError(
        "正控制組失敗：以下作弊特徵未被偵測 → 掃描器沒有鑑別力，"
        f"拒絕回報結果：{undetected}"
    )
```

**寧可拋錯也不回報「全部通過」。**

測試 `test_scan_builder_raises_when_positive_control_fails` 用 3 列資料觸發這個路徑——資料太短導致無有效切點，掃描器誠實地說「我無法給出結論」而不是說「都沒問題」。

---

## 切點選擇

```python
def _default_cut_indices(n_rows, cut_count=5):
    lo = max(1, n_rows // 3)      # 避開最前段
    hi = n_rows - 2                # 避開最後一列
    ...
```

兩個邊界都有理由：

| 邊界 | 理由 |
|---|---|
| 避開最前段 | 特徵視窗還沒滿，兩邊都是 NaN，比什麼都一樣 |
| 避開最後一列 | 截斷後等於完整資料，**永遠通過**，製造假的安全感 |

最後一列這條有專門的測試把關：

```python
def test_rejects_last_index_as_cut():
    with pytest.raises(ValueError, match="切點"):
        scan_features({"ma20": ma20}, bars, cut_indices=[29])   # 30 列資料
```

多點切分的理由：單一切點可能剛好躲過洩漏（例如某個洩漏只在資料長度為偶數時顯現）。預設 5 個分散切點。

---

## 實測結果

### 合成資料

24 個測試全過，包含：

- 3 個合法特徵（rolling mean、pct_change、expanding max）→ 通過
- 3 個洩漏特徵（全樣本 z-score、future shift、bfill）→ 全部被抓到
- 4 個內建作弊特徵 → 全部被抓到

### 真實資料（台積電 892 列）

```
正控制組：✓ 通過  4 個作弊特徵全部被偵測

通過 16 個特徵：
  bollinger_position_20, close_position_in_bar, gap_ratio,
  high_low_position_20, intraday_range, ma_ratio_20, ma_ratio_5,
  ma_ratio_60, momentum_120, momentum_20, momentum_5, momentum_60,
  rsi_14, true_range_ratio_14, volume_ratio_20, volume_ratio_5

結論：全部通過
```

籌碼面 16 個特徵同樣全部通過（D7 標註籌碼族 look-ahead 風險最高，所以這條格外重要）。

---

## 這個掃描器不能證明什麼

誠實聲明：

| 能證明 | 不能證明 |
|---|---|
| 特徵計算不用未來資料 | 訓練迴圈的時序正確 |
| 特徵在不同資料長度下穩定 | 標的池沒有 survivorship bias |
| 四種常見洩漏形態被覆蓋 | 沒有第五種未知形態 |

其他時序正確性由別的機制保證：

- 進場時點（T+1 開盤）→ `triple_barrier.py` 的 `LABEL_ENTRY_OFFSET`
- 標的池歷史 → `universe_history.py` 的「絕不使用未來快照」
- 訓練/驗證切分 → Walk-Forward 的 embargo

---

## 使用方式

```python
from taiwan_quant.validation.lookahead import scan_builder
from taiwan_quant.features.technical import build_technical

result = scan_builder(build_technical, bars)

assert result.positive_control_passed   # 先確認掃描器有效
assert result.is_clean                  # 再看結果
print(result.describe())
```

每次新增特徵都應該跑一次。這也是為什麼先做掃描器再做特徵——有了工具，寫 feature 時才能立刻知道有沒有洩漏。

---

## 參考

- 規格來源：`taiwan-quant/CLAUDE.md` 規格 16
- 實測發現：`docs/需求規劃/202609/02_驗證結果_qlib-tw-trader.md`
- 前身實作：`qlib-tw-trader/tests/test_lookahead_truncation.py`
- 驗證紀錄：`docs/需求規劃/202609/驗證/06_truncation_report.txt`

⚠️ 本文件為工程原理說明，不構成投資建議。

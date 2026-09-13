# 標的池與 Survivorship Bias 原理

- 日期：2026-09-12
- 程式：`taiwan-quant/taiwan_quant/data/constituents.py`、`universe_history.py`、`market_cap.py`
- 測試：54 個（15 + 23 + 16）
- 工具：`taiwan-quant/scripts/snapshot_universe.py`

---

## Survivorship Bias 是什麼

用「今天還存在的標的」回測歷史，等於事先知道哪些公司活下來了。

```
2023 年的台股有 1,800 檔
    ↓
其中有些在 2024-2026 間下市、被併購、變成全額交割
    ↓
用 2026 年的名單回測 2023 → 那些「死掉的」從樣本裡消失了
```

結果：回測只包含倖存者，績效被**系統性高估**。

這個偏誤在台股特別需要注意，因為 0050 / 0051 每季會調整成分股——被剔除的通常是表現最差的。

---

## D2 的要求與現實落差

### 要求

標的池 = 元大台灣 50（0050，50 檔）+ 元大中型 100（0051，100 檔）= **150 檔**

> 修正紀錄：原建議文件寫「0050+0051 約 80 檔」有誤。使用者指出 0051 是中型 100 指數、與 0050 不重疊，合計 150 檔。**實測證實使用者是對的。**

### 現實

需要的是「在 2024-06-28 那天，0050 的成分股是哪 50 檔」。這種歷史成分股資料很難免費取得。

---

## 資料來源調查（2026-09-12 實測）

| 來源 | 提供什麼 | 結果 |
|---|---|---|
| FinMind `TaiwanStockMarketValueWeight` | 市值比重 | ✗ 付費方案才有（免費版回 400「Your level is free」） |
| FinMind `TaiwanStockMarketValue` | 歷史市值 | ✗ 付費方案才有 |
| TWSE OpenAPI（143 個端點） | — | ✗ 只有指數行情（TAI50I），**沒有成分股清單** |
| TWSE ETFReport | — | ✗ 只有定期定額交易戶數排行 |
| **元大投信官網** | **當期成分股 + 權重** | ✓ 權威來源（基金公司自己公布） |
| **FinMind `TaiwanStockDelisting`** | **退市清單** | ✓ 免費可用 |
| **TWSE MI_QFIIS + `date=YYYYMMDD`** | **歷史發行股數** | ✓ 免費、可取歷史 |

### 結論

- **當期**成分股：可以拿到真的（元大官網）
- **歷史**成分股：拿不到，只能用市值排名代理
- **退市清單**：可以拿到，這解決 survivorship 的一半

---

## 元大官網的解析

頁面是 Nuxt SSR，成分股資料內嵌在 `window.__NUXT__` 的 minified payload：

```javascript
StockWeights:[
  {code:hl, ym:a, name:hm, ename:hn, weights:57.01, qty:560288622},
  {code:hW, ym:a, name:hX, ename:hY, weights:6.53,  qty:33755278},
  ...
]
```

`hl`、`hm` 是 IIFE 的參數名，真值在呼叫引數列：

```javascript
window.__NUXT__=(function(a,b,c,...,hl,hm,...){ return {...} }(null,false,1,...,"2330","台積電",...));
```

所以需要重建「參數名 → 實際值」對照表再代入。929 個參數。

**好處**：curl 就能取，不需要瀏覽器，可以放進排程。

### 踩到的兩個坑

#### 坑 1：JS 前導點小數

```javascript
weights:.69    // JS 合法
```

```python
json.loads(".69")   # JSONDecodeError！JSON 規範不允許前導點
```

原本的 fallback 回傳字串 `".69"` → 不是數值 → 權重被當成 **0.0**。

症狀：0050 權重合計只有 **81.37%**，37/50 檔權重為 0。

修正：加一條 JS 數字字面值的 regex，走 `float()` 而非 `json.loads()`。

```python
_JS_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
```

修正後合計 **99.72%**，與頁面顯示的「股票 99.74%」吻合。

#### 坑 2：ETF 自己也是 4 位數字

```python
if len(code) == 4 and code.isdigit():   # ✗ 0050、0051 也符合
```

MI_QFIIS 的前兩列就是 0050 與 0051 本身。台股上市個股代號不以 0 開頭，ETF 才是 `00xx`：

```python
_LISTED_STOCK_CODE = re.compile(r"^[1-9]\d{3}$")   # ✓
```

### 拒絕半套資料

```python
if abs(len(constituents) - expected) > COUNT_TOLERANCE:
    raise ConstituentFetchError(
        f"{etf_id}：解析出 {len(constituents)} 檔，與預期 {expected} 檔偏離超過 "
        f"{COUNT_TOLERANCE}。拒絕回傳半套資料——請檢查官網是否改版。"
    )
```

官網改版會讓 regex 只命中部分項目。若不檢查筆數，下游會拿到一個「看起來正常但少了 49 檔」的標的池，而且不會有任何錯誤訊息。

### 實測結果

```
0050   50 檔｜權重合計 99.72%｜前三 2330(57.01%) 2454(6.53%) 2308(3.03%)
0051  100 檔｜權重合計 95.20%｜前三 6446(3.41%) 3481(2.95%) 2379(2.91%)
聯集 150 檔｜重疊 0 檔
```

**與 D2 完全吻合。**

---

## 歷史快照：三條不可協商的規則

### 規則 1：絕不使用晚於決策日的快照

```python
def latest_on_or_before(self, as_of: date) -> date | None:
    candidates = [d for d in self.available_dates() if d <= as_of]
    return max(candidates) if candidates else None
```

**最容易寫錯的一行**：

```python
return min(self.available_dates())   # ✗ 看起來很合理
```

決策日早於所有快照時，退而使用「最早的那個」看似無害，但那個快照可能是 2026 年的——**用未來的成分股名單回測 2024**，同時犯了 look-ahead 與 survivorship 兩種錯。

沒有可用快照時必須走代理重建，不可退讓。

測試 `test_never_uses_future_snapshot` 專門擋這個。

### 規則 2：每個標的池都要宣告來源

```python
class Provenance(str, Enum):
    REAL  = "real"    # 來自真實成分股快照
    PROXY = "proxy"   # 由市值排名重建的代理名單
```

`PROXY` 會強制帶三條警告進報告：

```
2024-06-28 無真實成分股快照，改用市值前 150 名代理 0050+0051，非實際成分股
代理名單以事後可得的市值排名重建，可能含 survivorship bias：
    若市值資料未涵蓋當期已退市個股，績效會被高估
流動性分層亦為代理（前 50 名視為 0050 級）
```

代理名單產生的績效與真實成分股**不可混為一談**。不讓下游誤以為代理資料是真的。

### 規則 3：代理重建必須納入退市股

```python
def build_proxy_universe(market_caps, top_n):
    """
    market_caps: 呼叫端必須把退市股也放進來——
                 本函式不做「這檔還活著嗎」的過濾。
    """
```

任何「只取今天還在的股票」的過濾，都是在製造 survivorship bias。

`fetch_delisted()` 提供退市清單（FinMind 免費可用），讓呼叫端知道有哪些標的需要補歷史資料。

測試 `test_proxy_includes_delisted_stocks` 把這條釘死。

---

## 歷史市值重建

市值 = 發行股數 × 收盤價

### 發行股數：TWSE MI_QFIIS 支援歷史日期

```
https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date=20240628&selectType=ALLBUT0999
```

實測 2024-06-28 可取得 1,223 列（含 ETF），過濾後為台股個股。

這是能重建歷史市值的關鍵——FinMind 的市值資料集要付費，但 TWSE 的發行股數免費且有歷史。

### 用未還原收盤價

這是本專案唯一刻意違反 `CLAUDE.md` 禁令 12（一律用還原股價）的地方。

理由：**市值是「當時的市場價值」**，要用當時的實際成交價乘當時的股數。

```
用還原價 → 算出一個歷史上不存在的市值
```

還原價的用途是算報酬（消除除權息造成的假跳空），不是算市值。

程式與文件都明文記錄這個例外與理由。

### 欄位用名稱查，不用位置索引

```python
id_idx = fields.index("證券代號")
shares_idx = fields.index("發行股數")
```

qlib-tw-trader 的教訓：TWSE 把 `STOCK_DAY_ALL` 從 JSON 改成 CSV 且欄位索引位移，原程式用位置索引 `row[4]` 直接壞掉，而且是**靜默給出錯誤資料**（讀到了錯的欄位）。

用欄位名查 index，找不到就明確報錯：

```python
raise MarketCapError(
    f"TWSE 回應缺少必要欄位（證券代號 / 發行股數）：實際欄位 {fields}"
)
```

### 無效值回 None，不回 0

```python
def _parse_int(raw: str) -> int | None:
    """無效值回 None（不回 0，0 會被誤當成真實股數）"""
```

回 0 會讓這些標的帶著「市值 0」進入排名，佔用名額又永遠排最後。回 `None` 讓呼叫端明確排除它們。

---

## 目前狀態與誠實聲明

### 已解決

- ✓ 拿到真實的當期成分股（0050 + 0051 = 150 檔，零重疊）
- ✓ 建立快照機制，**從今天起**每季存一份就能累積真實歷史
- ✓ 第一份快照已存：`taiwan-quant/data/universe_history/2026-09-12.json`
- ✓ 拿到退市清單來源（FinMind 免費）
- ✓ 拿到歷史發行股數來源（TWSE，可重建歷史市值）
- ✓ 時序安全（絕不用未來快照）有測試釘住

### 未解決

- ✗ **回測期間（2023-2026）沒有真實成分股快照**

元大官網只提供當期資料。這不是工程問題，是資料可得性問題——**無法靠寫程式解決，只能揭露**。

回測報告必須標明：該期間的標的池是市值排名代理，Provenance = PROXY。

### 建議的累積路徑

```
每季（3/6/9/12 月指數審核後）執行：
    .venv/bin/python scripts/snapshot_universe.py

一年後 → 4 份真實快照
三年後 → 12 份，足以支撐一段有意義的真實歷史回測
```

---

## 參考

- 決策依據：`docs/需求規劃/202609/01_決策紀錄.md` D2
- 治理規範：`taiwan-quant/CLAUDE.md` 禁令 2、12
- 審查清單：`taiwan-quant/AGENTS.md` 檢查項 6、7

⚠️ 本文件為工程原理說明，不構成投資建議。

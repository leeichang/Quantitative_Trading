"""
增強因子

針對模型診斷弱點設計的補充因子：
- 波動率 Regime 因子（7 個）— bull market IC 低
- 長期動量/均值回歸（8 個）— Top-10 穩定性低
- 流動性因子（6 個）— 信號穩定性
- 估值動態因子（8 個）— Q1 季節弱勢
- 市場微結構因子（8 個）— quantile 單調性差
"""

# =============================================================================
# 1. 波動率 Regime 因子（7 個）
# 幫助模型識別市場 regime，改善 bull market 表現
# =============================================================================

VOLATILITY_REGIME_FACTORS = [
    {
        "name": "vol_regime_20d",
        "display_name": "波動率Regime_20日",
        "category": "technical",
        "expression": "Std($close / Ref($close, 1) - 1, 20) / (Std($close / Ref($close, 1) - 1, 60) + 1e-8)",
        "description": "20日波動率/60日波動率",
    },
    {
        "name": "vol_regime_5d",
        "display_name": "波動率Regime_5日",
        "category": "technical",
        "expression": "Std($close / Ref($close, 1) - 1, 5) / (Std($close / Ref($close, 1) - 1, 20) + 1e-8)",
        "description": "5日波動率/20日波動率",
    },
    {
        "name": "realized_vol_20d",
        "display_name": "實現波動率_20日",
        "category": "technical",
        "expression": "Std($close / Ref($close, 1) - 1, 20)",
        "description": "20日收益率標準差",
    },
    {
        "name": "realized_vol_60d",
        "display_name": "實現波動率_60日",
        "category": "technical",
        "expression": "Std($close / Ref($close, 1) - 1, 60)",
        "description": "60日收益率標準差",
    },
    {
        "name": "high_low_vol_20d",
        "display_name": "振幅波動率_20日",
        "category": "technical",
        "expression": "Mean(($high - $low) / $close, 20)",
        "description": "20日平均日內振幅率",
    },
    {
        "name": "vol_trend",
        "display_name": "波動率趨勢",
        "category": "technical",
        "expression": "Slope(Std($close / Ref($close, 1) - 1, 10), 20)",
        "description": "10日波動率的20日趨勢斜率",
    },
    {
        "name": "vol_skew_20d",
        "display_name": "波動率偏態_20日",
        "category": "technical",
        "expression": "(Mean(Greater($close / Ref($close, 1) - 1, 0), 20) - Mean(Greater(Ref($close, 1) / $close - 1, 0), 20)) / (Std($close / Ref($close, 1) - 1, 20) + 1e-8)",
        "description": "上行波動與下行波動的不對稱性",
    },
]

# =============================================================================
# 2. 長期動量/均值回歸因子（8 個）
# 提供更穩定的長期信號，改善 Top-10 穩定性
# =============================================================================

LONG_MOMENTUM_FACTORS = [
    {
        "name": "roc_120",
        "display_name": "ROC_120日",
        "category": "technical",
        "expression": "Ref($close, 120) / $close - 1",
        "description": "120日價格變化率",
    },
    {
        "name": "roc_250",
        "display_name": "ROC_250日",
        "category": "technical",
        "expression": "Ref($close, 250) / $close - 1",
        "description": "250日價格變化率（約一年）",
    },
    {
        "name": "ma_120",
        "display_name": "均線比_120日",
        "category": "technical",
        "expression": "Mean($close, 120) / $close",
        "description": "120日均線/收盤價",
    },
    {
        "name": "ma_250",
        "display_name": "均線比_250日",
        "category": "technical",
        "expression": "Mean($close, 250) / $close",
        "description": "250日均線/收盤價（年線）",
    },
    {
        "name": "ma_cross_20_60",
        "display_name": "均線交叉_20_60",
        "category": "technical",
        "expression": "Mean($close, 20) / Mean($close, 60) - 1",
        "description": "20日均線相對60日均線偏離",
    },
    {
        "name": "ma_cross_60_120",
        "display_name": "均線交叉_60_120",
        "category": "technical",
        "expression": "Mean($close, 60) / Mean($close, 120) - 1",
        "description": "60日均線相對120日均線偏離",
    },
    {
        "name": "momentum_quality",
        "display_name": "動量品質",
        "category": "technical",
        "expression": "($close / Ref($close, 60) - 1) / (Std($close / Ref($close, 1) - 1, 60) + 1e-8)",
        "description": "60日收益/60日波動（動量夏普比率）",
    },
    {
        "name": "mean_reversion_20d",
        "display_name": "均值回歸_20日",
        "category": "technical",
        "expression": "($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)",
        "description": "收盤價偏離20日均線的標準差數",
    },
]

# =============================================================================
# 3. 流動性因子（6 個）
# 提高信號穩定性，低流動性股票信號噪音大
# =============================================================================

LIQUIDITY_FACTORS = [
    {
        "name": "amihud_20d",
        "display_name": "Amihud非流動性_20日",
        "category": "technical",
        "expression": "Mean(Abs($close / Ref($close, 1) - 1) / (Log($volume + 1) + 1e-8), 20)",
        "description": "20日Amihud非流動性指標",
    },
    {
        "name": "amihud_60d",
        "display_name": "Amihud非流動性_60日",
        "category": "technical",
        "expression": "Mean(Abs($close / Ref($close, 1) - 1) / (Log($volume + 1) + 1e-8), 60)",
        "description": "60日Amihud非流動性指標",
    },
    {
        "name": "turnover_20d",
        "display_name": "換手率_20日",
        "category": "technical",
        "expression": "Mean($volume / ($total_shares + 1e-8), 20)",
        "description": "20日平均換手率",
    },
    {
        "name": "turnover_momentum",
        "display_name": "換手率動能",
        "category": "technical",
        "expression": "Mean($volume / ($total_shares + 1e-8), 5) / (Mean($volume / ($total_shares + 1e-8), 20) + 1e-8)",
        "description": "換手率短長比（流動性變化）",
    },
    {
        "name": "liquidity_improvement",
        "display_name": "流動性改善",
        "category": "technical",
        "expression": "Mean($volume, 10) / (Mean($volume, 60) + 1e-8) - 1",
        "description": "10日成交量相對60日均量偏離",
    },
    {
        "name": "vol_concentration",
        "display_name": "成交量集中度",
        "category": "technical",
        "expression": "Max($volume, 5) / (Sum($volume, 5) + 1e-8)",
        "description": "5日內最大日成交量佔比",
    },
]

# =============================================================================
# 4. 估值動態因子（8 個）
# 捕捉估值時序變化，改善 Q1 季節性弱勢
# =============================================================================

VALUATION_DYNAMIC_FACTORS = [
    {
        "name": "pe_percentile_120d",
        "display_name": "PE分位_120日",
        "category": "interaction",
        "expression": "Rank($pe_ratio, 120)",
        "description": "PE在120日內的時序分位",
    },
    {
        "name": "pe_percentile_250d",
        "display_name": "PE分位_250日",
        "category": "interaction",
        "expression": "Rank($pe_ratio, 250)",
        "description": "PE在250日內的時序分位",
    },
    {
        "name": "pb_percentile_120d",
        "display_name": "PB分位_120日",
        "category": "interaction",
        "expression": "Rank($pb_ratio, 120)",
        "description": "PB在120日內的時序分位",
    },
    {
        "name": "pb_percentile_250d",
        "display_name": "PB分位_250日",
        "category": "interaction",
        "expression": "Rank($pb_ratio, 250)",
        "description": "PB在250日內的時序分位",
    },
    {
        "name": "pe_mean_reversion",
        "display_name": "PE均值回歸",
        "category": "interaction",
        "expression": "($pe_ratio - Mean($pe_ratio, 120)) / (Std($pe_ratio, 120) + 1e-8)",
        "description": "PE偏離120日均值的標準差數",
    },
    {
        "name": "dy_momentum_20d",
        "display_name": "殖利率動能_20日",
        "category": "interaction",
        "expression": "$dividend_yield / (Mean($dividend_yield, 20) + 1e-8) - 1",
        "description": "殖利率相對20日均值偏離率",
    },
    {
        "name": "dy_rank_120d",
        "display_name": "殖利率排名_120日",
        "category": "interaction",
        "expression": "Rank($dividend_yield, 120)",
        "description": "殖利率在120日內的時序分位",
    },
    {
        "name": "pe_momentum_20d",
        "display_name": "PE動能_20日",
        "category": "interaction",
        "expression": "$pe_ratio / (Mean($pe_ratio, 20) + 1e-8) - 1",
        "description": "PE相對20日均值偏離",
    },
]

# =============================================================================
# 5. 市場微結構因子（8 個）
# 提供更細粒度的信息，改善 quantile 單調性
# =============================================================================

MICROSTRUCTURE_FACTORS = [
    {
        "name": "intraday_range_stability",
        "display_name": "日內振幅穩定性",
        "category": "technical",
        "expression": "Std(($high - $low) / $close, 20) / (Mean(($high - $low) / $close, 20) + 1e-8)",
        "description": "振幅的變異係數",
    },
    {
        "name": "close_position",
        "display_name": "收盤位置",
        "category": "technical",
        "expression": "Mean(($close - $low) / ($high - $low + 1e-8), 10)",
        "description": "10日平均收盤在日內範圍的位置",
    },
    {
        "name": "open_close_gap",
        "display_name": "跳空缺口",
        "category": "technical",
        "expression": "Mean(Abs($open / Ref($close, 1) - 1), 20)",
        "description": "20日平均跳空缺口幅度",
    },
    {
        "name": "price_vol_divergence",
        "display_name": "價量背離",
        "category": "interaction",
        "expression": "($close / Ref($close, 10) - 1) * -1 * ($volume / (Mean($volume, 10) + 1e-8) - 1)",
        "description": "價格與量能變化的反向乘積",
    },
    {
        "name": "volume_surprise",
        "display_name": "量能驚喜",
        "category": "technical",
        "expression": "($volume - Mean($volume, 20)) / (Std($volume, 20) + 1e-8)",
        "description": "成交量偏離20日均值的Z-score",
    },
    {
        "name": "up_volume_ratio_10d",
        "display_name": "上漲量比_10日",
        "category": "interaction",
        "expression": "Sum($volume * Greater($close - Ref($close, 1), 0), 10) / (Sum($volume, 10) + 1e-8)",
        "description": "10日上漲成交量佔總成交量比",
    },
    {
        "name": "high_low_ratio_trend",
        "display_name": "振幅趨勢",
        "category": "technical",
        "expression": "Mean(($high - $low) / $close, 5) / (Mean(($high - $low) / $close, 20) + 1e-8)",
        "description": "短期振幅/長期振幅趨勢",
    },
    {
        "name": "consecutive_up_days",
        "display_name": "連漲天數_20日",
        "category": "technical",
        "expression": "Sum(Greater($close - Ref($close, 1), 0), 20) / 20",
        "description": "20日內上漲天數比例",
    },
]

# =============================================================================
# 匯出
# =============================================================================

ENHANCED_FACTORS = (
    VOLATILITY_REGIME_FACTORS
    + LONG_MOMENTUM_FACTORS
    + LIQUIDITY_FACTORS
    + VALUATION_DYNAMIC_FACTORS
    + MICROSTRUCTURE_FACTORS
)

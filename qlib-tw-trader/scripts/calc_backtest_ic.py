"""
計算回測期間的實際 IC（衰減分析）

載入已完成的回測結果，逐日計算 Spearman IC，
並與驗證期 IC 比較，分析模型泛化能力與 IC 衰減程度。

用法: python scripts/calc_backtest_ic.py
"""
import json
import pickle
import sqlite3
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# 初始化 qlib
import qlib
from qlib.config import REG_CN
from qlib.data import D

qlib.init(provider_uri='data/qlib', region=REG_CN)

MODELS_DIR = Path('data/models')


def load_model(model_name):
    model_dir = MODELS_DIR / model_name
    with open(model_dir / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(model_dir / 'factors.json') as f:
        factors = json.load(f)
    return model, factors


def get_instruments():
    with open('data/qlib/instruments/all.txt') as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def get_predictions_and_returns(model, factors, feature_date, instruments):
    """獲取預測分數和實際收益"""
    fields = [f['expression'] for f in factors]
    names = [f['name'] for f in factors]

    # 獲取特徵
    df = D.features(
        instruments=instruments,
        fields=fields,
        start_time=feature_date.strftime('%Y-%m-%d'),
        end_time=feature_date.strftime('%Y-%m-%d'),
    )

    if df.empty:
        return None

    df.columns = names

    # 處理 inf
    for col in df.columns:
        mask = np.isinf(df[col])
        if mask.any():
            col_mean = df.loc[~mask, col].mean()
            df.loc[mask, col] = col_mean if not np.isnan(col_mean) else 0

    # z-score
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 1e-8:
            df[col] = (df[col] - mean) / std
        else:
            df[col] = 0

    df = df.fillna(0)

    # 預測
    predictions = model.predict(df.values)
    symbols = df.index.get_level_values('instrument').tolist()

    # 獲取實際收益 (T+1 → T+3)
    # label = Ref($close, -3) / Ref($close, -1) - 1
    close_df = D.features(
        instruments=instruments,
        fields=['$close'],
        start_time=(feature_date + timedelta(days=1)).strftime('%Y-%m-%d'),
        end_time=(feature_date + timedelta(days=10)).strftime('%Y-%m-%d'),
    )

    if close_df.empty or len(close_df) < 2:
        return None

    close_df.columns = ['close']
    close_df = close_df.reset_index()

    # 找前三個交易日 (T+1, T+2, T+3)
    dates = close_df['datetime'].unique()
    if len(dates) < 3:
        return None

    t1_date = dates[0]   # T+1
    t2_date = dates[2]   # T+3

    t1_close = close_df[close_df['datetime'] == t1_date].set_index('instrument')['close']
    t2_close = close_df[close_df['datetime'] == t2_date].set_index('instrument')['close']

    returns = (t2_close / t1_close - 1).dropna()

    # 合併
    result = pd.DataFrame({
        'symbol': symbols,
        'score': predictions,
    }).set_index('symbol')

    result['return'] = returns
    result = result.dropna()

    return result


def main():
    conn = sqlite3.connect('data/data.db')

    # 獲取回測資訊
    backtests = pd.read_sql('''
        SELECT
            b.id,
            b.start_date,
            b.end_date,
            t.name as model_name,
            t.model_ic as valid_ic,
            t.valid_start,
            t.valid_end
        FROM backtests b
        JOIN training_runs t ON b.model_id = t.id
        WHERE b.status = 'completed'
        ORDER BY b.start_date
    ''', conn)

    instruments = get_instruments()

    print('=' * 70)
    print('回測期 IC vs 驗證期 IC')
    print('=' * 70)

    results = []

    for _, bt in backtests.iterrows():
        model_name = bt['model_name']
        start_date = date.fromisoformat(bt['start_date'])
        end_date = date.fromisoformat(bt['end_date'])
        valid_ic = bt['valid_ic']

        print(f"\n處理 {model_name}...")

        try:
            model, factors = load_model(model_name)
        except Exception as e:
            print(f'  載入失敗: {e}')
            continue

        # 計算每個交易日的 IC
        daily_ics = []
        current_date = start_date

        while current_date <= end_date:
            # feature_date = 交易日前一天
            feature_date = current_date - timedelta(days=1)

            # 跳過週末
            if feature_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            try:
                data = get_predictions_and_returns(model, factors, feature_date, instruments)

                if data is not None and len(data) >= 10:
                    # Spearman IC
                    ic, _ = stats.spearmanr(data['score'], data['return'])
                    if not np.isnan(ic):
                        daily_ics.append(ic)
            except Exception as e:
                pass

            current_date += timedelta(days=1)

        if daily_ics:
            backtest_ic = np.mean(daily_ics)
            ic_std = np.std(daily_ics)

            results.append({
                'model': model_name,
                'valid_ic': valid_ic,
                'backtest_ic': backtest_ic,
                'ic_decay': valid_ic - backtest_ic,
                'decay_pct': (valid_ic - backtest_ic) / valid_ic * 100,
                'n_days': len(daily_ics),
            })

            print(f'  驗證 IC: {valid_ic:.4f}')
            print(f'  回測 IC: {backtest_ic:.4f}')
            print(f'  衰減: {(valid_ic - backtest_ic)/valid_ic*100:.1f}%')

    conn.close()

    if not results:
        print("無法計算任何回測 IC")
        return

    # 統計摘要
    df = pd.DataFrame(results)

    print('\n' + '=' * 70)
    print('IC 衰減分析')
    print('=' * 70)
    print(df.to_string(index=False))

    print('\n' + '-' * 70)
    print('統計摘要')
    print('-' * 70)
    print(f"驗證期平均 IC: {df['valid_ic'].mean():.4f}")
    print(f"回測期平均 IC: {df['backtest_ic'].mean():.4f}")
    print(f"平均衰減: {df['decay_pct'].mean():.1f}%")

    # 回測 IC 是否顯著 > 0
    t_stat, p_value = stats.ttest_1samp(df['backtest_ic'], 0)
    print(f"\n回測 IC 是否顯著 > 0?")
    print(f"  t = {t_stat:.2f}, p = {p_value:.4f}")
    print(f"  結論: {'是' if p_value/2 < 0.05 else '否'} (α=0.05)")


if __name__ == '__main__':
    main()

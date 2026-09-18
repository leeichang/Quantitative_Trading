"""
用 block bootstrap 估每趟報酬與累積報酬的信賴區間

## 這支腳本回答什麼

`docs/交接手冊/03_待辦與改進方向.md` 第 3 項的驗收條件：

> **怎麼判斷做成了**：能回答「+1076% 的 95% 信賴區間是多少」。
> 如果下界低於買進持有，結論就是「無法區分」。

## 為什麼不重跑回測

它**只讀既有報告的逐期序列**，不碰資料庫、不重算回測。

原因有兩個。第一，區間估計是純後處理——重抽的是已經產生的報酬序列，
重跑回測不會提供任何額外資訊。第二，禁令 6：OOS 區間已經用掉，
而重跑回測有動用它的風險。**後處理沒有那個風險。**

因此這支腳本可以在任何時候重跑，不消耗任何區間。

## 區塊長度為什麼要掃

`validation/bootstrap.py` 的 `block_length` 是必填。實測逐期序列：

```
水準序列        lag1 自相關      配對差異序列      lag1 自相關
等權全池          −0.281        模型 − 打亂        −0.023
隨機 10 檔        −0.277        模型 − 手工        +0.080
標籤打亂          −0.278        手工 − 打亂        −0.074
```

配對差異幾乎沒有自相關，所以 `block_length=1` 對它是對的。
水準序列有 −0.28 的負自相關，區塊會**收窄**區間（負自相關降低長期
變異）——所以水準序列要掃描並報告敏感度。

**單一區塊長度的結果不構成結論。** 這與 2026-09-18 撤回「超過隨機
95% 分位」是同一條理由：由任意選擇決定的結論不是結論。

## 用法

    python scripts/diagnose_bootstrap_intervals.py
    python scripts/diagnose_bootstrap_intervals.py --source reports/lgbm_baseline_dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.validation.bootstrap import (  # noqa: E402
    BootstrapResult,
    block_bootstrap,
    compound_total_return,
    paired_difference,
    usable_block_lengths,
)

DEFAULT_SOURCE = Path("reports/lgbm_baseline_dev.json")
DEFAULT_OUTPUT = Path("reports/bootstrap_intervals_dev.json")

SEED = 20260918
"""固定種子。報告裡要能看到用了哪個，否則區間無法重現"""

N_DRAWS = 4000
BLOCK_SWEEP = (1, 2, 4)
"""掃描的區塊長度。1 = IID，2 與 4 覆蓋 lag1~lag3 的自相關範圍"""

ARMS = ("model", "hand", "shuffled_labels", "random_median", "equal_weight")

PAIRS = (
    ("model", "random_median"),
    ("shuffled_labels", "random_median"),
    ("model", "shuffled_labels"),
    ("model", "hand"),
    ("hand", "shuffled_labels"),
)

BENCHMARK = "equal_weight"
"""驗收條件裡的「買進持有」。等權全池是不含成本的被動基準"""


class DiagnosticError(RuntimeError):
    """來源報告缺少必要欄位。"""


@dataclass(frozen=True)
class Arm:
    """一組被比較的報酬序列。"""

    key: str
    label: str
    series: np.ndarray

    @property
    def lag1(self) -> float:
        """lag-1 自相關。決定區塊長度該不該掃"""
        if self.series.size < 3:
            return float("nan")
        return float(np.corrcoef(self.series[:-1], self.series[1:])[0, 1])

    @property
    def excess_kurtosis(self) -> float:
        """超額峰度。常態為 0；> 1 代表 t 檢定的尾部不可靠"""
        centred = self.series - self.series.mean()
        return float((centred**4).mean() / self.series.std() ** 4 - 3.0)


@dataclass(frozen=True)
class Source:
    """一份報告解析後的內容：若干組序列，加上報告自帶的基準點。"""

    arms: tuple[Arm, ...]
    benchmarks: dict[str, float]
    """名稱到累積報酬。空的代表報告沒記基準"""

    pairs: tuple[tuple[str, str], ...]
    """可做配對比較的組合。單組序列的報告沒有配對"""


def _multi_arm(payload: dict[str, object], source: Path) -> Source:
    """開發集報告格式：多組 `{label, series}`。"""
    arms: list[Arm] = []
    for key in ARMS:
        block = payload.get(key)
        if not isinstance(block, dict) or "series" not in block:
            raise DiagnosticError(f"{source} 缺少 {key}.series")
        arms.append(
            Arm(
                key=key,
                label=str(block.get("label", key)),
                series=np.array(list(block["series"].values()), dtype=float),
            )
        )
    lengths = {arm.series.size for arm in arms}
    if len(lengths) != 1:
        raise DiagnosticError(f"各組期數不一致：{lengths}——配對比較會對錯期")

    benchmark = next(a for a in arms if a.key == BENCHMARK)
    return Source(
        arms=tuple(arms),
        benchmarks={benchmark.label: compound_total_return(benchmark.series)},
        pairs=PAIRS,
    )


def _single_arm(payload: dict[str, object], source: Path) -> Source:
    """
    OOS／前推報告格式：單組 `per_period` 加上 `benchmarks`。

    **讀 `per_period` 是純後處理，不重跑回測，所以不動用禁令 6 的區間。**
    這正是為什麼報告必須保存逐期序列——沒有它，那份結果的不確定性
    永遠算不出來，而重跑是被禁止的。
    """
    periods = payload.get("per_period")
    if not isinstance(periods, list) or not periods:
        raise DiagnosticError(f"{source} 的 per_period 為空或缺失")
    if "net" not in periods[0]:
        raise DiagnosticError(f"{source} 的 per_period 缺少 net 欄位")

    arm = Arm(
        key="strategy",
        label=str(payload.get("strategy_version", source.stem)),
        series=np.array([p["net"] for p in periods], dtype=float),
    )

    flat: dict[str, float] = {}
    raw = payload.get("benchmarks")
    if isinstance(raw, dict):
        for name, value in raw.items():
            if isinstance(value, (int, float)):
                flat[name] = float(value)
            elif isinstance(value, dict):
                # ETF 基準是嵌套的：{"0050": {"total_return": ...}}
                for sub, detail in value.items():
                    if isinstance(detail, dict) and "total_return" in detail:
                        flat[sub] = float(detail["total_return"])
    return Source(arms=(arm,), benchmarks=flat, pairs=())


def load_source(source: Path) -> Source:
    """讀報告並依格式分派。缺欄位就失敗，不要靜默跳過。"""
    if not source.exists():
        raise DiagnosticError(f"來源報告不存在：{source}")
    payload = json.loads(source.read_text())
    if "per_period" in payload:
        return _single_arm(payload, source)
    return _multi_arm(payload, source)


def _as_dict(result: BootstrapResult) -> dict[str, object]:
    """存區間與判定，**不存 4000 個 draws**——報告會膨脹到沒人讀。"""
    return {
        "point": result.point,
        "lower": result.lower,
        "upper": result.upper,
        "width": result.width,
        "excludes_zero": result.excludes_zero,
        "level": result.level,
        "block_length": result.block_length,
        "n_draws": result.n_draws,
    }


def per_trip_intervals(arms: tuple[Arm, ...]) -> dict[str, object]:
    """每趟平均報酬的區間，並與 t 檢定的常態區間並列。"""
    out: dict[str, object] = {}
    for arm in arms:
        n = arm.series.size
        mean = float(arm.series.mean())
        se = float(arm.series.std(ddof=1) / np.sqrt(n))
        boot = block_bootstrap(
            arm.series, np.mean, block_length=1, n_draws=N_DRAWS, seed=SEED
        )
        out[arm.key] = {
            "label": arm.label,
            "periods": n,
            "lag1_autocorrelation": arm.lag1,
            "excess_kurtosis": arm.excess_kurtosis,
            "normal_interval": {
                "point": mean,
                "lower": mean - 1.96 * se,
                "upper": mean + 1.96 * se,
                "standard_error": se,
            },
            "bootstrap_interval": _as_dict(boot),
        }
    return out


def total_return_intervals(arms: tuple[Arm, ...]) -> dict[str, object]:
    """
    累積複利報酬的區間，逐個區塊長度。這是驗收條件直接問的量。

    區塊長度超過序列長度時無法重抽（短序列如前推帳本頭幾期會遇到），
    所以先篩過，並把**篩掉哪些**記進報告——少一欄而沒說明，就是
    靜默改變了掃描範圍。
    """
    out: dict[str, object] = {}
    for arm in arms:
        usable = usable_block_lengths(arm.series.size, BLOCK_SWEEP)
        by_block = {
            str(block): _as_dict(
                block_bootstrap(
                    arm.series,
                    compound_total_return,
                    block_length=block,
                    n_draws=N_DRAWS,
                    seed=SEED,
                )
            )
            for block in usable
        }
        out[arm.key] = {
            "label": arm.label,
            "point": compound_total_return(arm.series),
            "by_block_length": by_block,
            "skipped_block_lengths": [
                b for b in BLOCK_SWEEP if b not in usable
            ],
        }
    return out


def paired_intervals(
    arms: tuple[Arm, ...], pairs: tuple[tuple[str, str], ...]
) -> dict[str, object]:
    """
    配對差異的區間，並記錄 bootstrap 與 t 是否給出同樣的判定。

    `agrees_with_t` 為 False 時要在說明文件裡解釋——那代表兩種方法
    在這一格上分家，不可以只報對自己有利的那一個。
    """
    lookup = {arm.key: arm for arm in arms}
    out: dict[str, object] = {}
    for treatment, control in pairs:
        a, b = lookup[treatment], lookup[control]
        gap = a.series - b.series
        t_stat = float(gap.mean() / (gap.std(ddof=1) / np.sqrt(gap.size)))
        boot = paired_difference(
            a.series, b.series, block_length=1, n_draws=N_DRAWS, seed=SEED
        )
        out[f"{treatment} − {control}"] = {
            "labels": f"{a.label} − {b.label}",
            "mean_difference": float(gap.mean()),
            "t": t_stat,
            "t_verdict": "顯著" if abs(t_stat) > 2.0 else "量不出差別",
            "lag1_autocorrelation": float(
                np.corrcoef(gap[:-1], gap[1:])[0, 1]
            ),
            "excess_kurtosis": float(
                ((gap - gap.mean()) ** 4).mean() / gap.std() ** 4 - 3.0
            ),
            "bootstrap_interval": _as_dict(boot),
            "agrees_with_t": boot.excludes_zero == (abs(t_stat) > 2.0),
        }
    return out


def acceptance_check(
    totals: dict[str, object], benchmarks: dict[str, float]
) -> dict[str, object]:
    """
    驗收條件：累積報酬的區間下界若低於基準，結論就是「無法區分」。

    對每一組、每個區塊長度、每一個基準都判一次——只報對自己有利的
    那個長度或那個基準，就是 2026-09-18 撤回「超過隨機 95% 分位」時
    指出的同一個毛病。

    另外記 `benchmarks_inside_interval`：落在區間內的基準，代表那個
    比較**分辨不出來**。一份報告若把這種比較寫成結論，那句結論就是
    區間寬度的產物，不是策略的性質。
    """
    verdicts: dict[str, object] = {}
    for key, entry in totals.items():
        by_block = entry["by_block_length"]  # type: ignore[index]
        below: dict[str, dict[str, bool]] = {}
        for name, point in benchmarks.items():
            below[name] = {
                block: bool(float(payload["lower"]) < point)
                for block, payload in by_block.items()  # type: ignore[union-attr]
            }
        widest = by_block[str(min(BLOCK_SWEEP))]  # type: ignore[index]
        inside = sorted(
            name
            for name, point in benchmarks.items()
            if float(widest["lower"]) <= point <= float(widest["upper"])
        )
        verdicts[key] = {
            "label": entry["label"],  # type: ignore[index]
            "point": entry["point"],  # type: ignore[index]
            "lower_below_benchmark_by_block": below,
            "benchmarks_inside_interval": inside,
            "verdict": (
                "無法與任一基準區分"
                if all(all(flags.values()) for flags in below.values())
                else "至少一個基準在區間下界之下"
            ),
        }
    return {"benchmarks": benchmarks, "arms": verdicts}


def _print_report(payload: dict[str, object]) -> None:
    """終端輸出。改動過的腳本必須附真實輸出，所以它要能讀。"""
    print(f"來源 {payload['source']}｜期數 {payload['periods']}"
          f"｜重抽 {N_DRAWS}｜種子 {SEED}\n")

    print("=== 每趟淨報酬：常態區間 vs bootstrap（block=1）===")
    print(f"{'組合':22s} {'點估計':>9s} {'lag1':>7s} {'超額峰度':>9s}"
          f" {'常態 95%':>20s} {'bootstrap 95%':>20s}")
    for entry in payload["per_trip"].values():  # type: ignore[union-attr]
        nrm, bst = entry["normal_interval"], entry["bootstrap_interval"]
        print(f"{entry['label'][:20]:22s} {nrm['point']:+8.3%}"
              f" {entry['lag1_autocorrelation']:+6.3f}"
              f" {entry['excess_kurtosis']:+8.2f}"
              f" [{nrm['lower']:+7.3%},{nrm['upper']:+7.3%}]"
              f" [{bst['lower']:+7.3%},{bst['upper']:+7.3%}]")

    print("\n=== 累積總報酬的 95% 區間（複利，掃區塊長度）===")
    for entry in payload["total_return"].values():  # type: ignore[union-attr]
        blocks = sorted(entry["by_block_length"], key=int)
        header = "  ".join(f"{'block=' + b:<21s}" for b in blocks)
        print(f"{'組合':22s} {'點估計':>10s}  {header}")
        row = f"{entry['label'][:20]:22s} {entry['point']:+9.2%}  "
        for block in blocks:
            band = entry["by_block_length"][block]
            row += f"[{band['lower']:+7.1%},{band['upper']:+8.1%}]  "
        print(row)
        if entry["skipped_block_lengths"]:
            print(f"{'':22s} 序列太短，跳過區塊長度 "
                  f"{entry['skipped_block_lengths']}")

    paired = payload["paired"]
    if paired:
        print("\n=== 配對差異：t 與 bootstrap 的判定是否一致 ===")
        for entry in paired.values():  # type: ignore[union-attr]
            bst = entry["bootstrap_interval"]
            mark = "" if entry["agrees_with_t"] else "   ← 不一致"
            print(f"{entry['labels'][:34]:36s} {entry['mean_difference']:+7.3%}"
                  f" t={entry['t']:+5.2f}"
                  f" [{bst['lower']:+7.3%},{bst['upper']:+7.3%}]"
                  f" {entry['t_verdict']}/"
                  f"{'不含零' if bst['excludes_zero'] else '含零'}{mark}")

    check = payload["acceptance"]
    marks = "、".join(
        f"{name} {point:+.2%}"
        for name, point in check["benchmarks"].items()  # type: ignore[index]
    )
    print(f"\n=== 驗收條件：累積報酬區間下界 vs 基準 ===\n基準：{marks}")
    for entry in check["arms"].values():  # type: ignore[index]
        inside = entry["benchmarks_inside_interval"]
        print(f"{entry['label'][:26]:28s} 累積 {entry['point']:+9.2%}"
              f"   {entry['verdict']}")
        if inside:
            print(f"{'':28s} 落在區間內、分辨不出的基準："
                  f"{'、'.join(inside)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    parsed = load_source(args.source)
    totals = total_return_intervals(parsed.arms)
    payload: dict[str, object] = {
        "source": str(args.source),
        "periods": int(parsed.arms[0].series.size),
        "seed": SEED,
        "n_draws": N_DRAWS,
        "block_sweep": list(BLOCK_SWEEP),
        "per_trip": per_trip_intervals(parsed.arms),
        "total_return": totals,
        "paired": paired_intervals(parsed.arms, parsed.pairs),
        "acceptance": acceptance_check(totals, parsed.benchmarks),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    _print_report(payload)
    print(f"\n已寫入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

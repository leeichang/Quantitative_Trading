#!/usr/bin/env python3
"""
存一份 0050 + 0051 的真實成分股快照

依據 D2 與 CLAUDE.md 禁令 2。元大投信官網只提供**當期**成分股，沒有歷史。
因此唯一能累積真實歷史的辦法就是**從今天開始定期存檔**。

建議執行頻率：每季（指數審核後）。台灣指數公司每年 3/6/9/12 月審核，
生效日通常在該季第三個星期五之後。多存幾次沒有壞處——快照按日期存檔，
`resolve_universe()` 會自動挑「不晚於決策日的最近一份」。

用法：
    .venv/bin/python scripts/snapshot_universe.py              # 存檔
    .venv/bin/python scripts/snapshot_universe.py --dry-run    # 只看不存
    .venv/bin/python scripts/snapshot_universe.py --list       # 列出已有快照
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.data.constituents import (  # noqa: E402
    ConstituentFetchError,
    fetch_universe_constituents,
)
from taiwan_quant.data.universe_history import SnapshotStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="存一份 0050 + 0051 真實成分股快照")
    parser.add_argument("--dry-run", action="store_true", help="只顯示不寫檔")
    parser.add_argument("--list", action="store_true", help="列出已有快照後結束")
    args = parser.parse_args()

    store = SnapshotStore()

    if args.list:
        dates = store.available_dates()
        print(f"已有快照 {len(dates)} 份，目錄 {store.store_dir}")
        for d in dates:
            members = store.load(d) or {}
            counts = {etf: len(ids) for etf, ids in members.items()}
            print(f"  {d}  {counts}  合計 {sum(counts.values())} 檔")
        return

    print("抓取元大投信當期成分股 ...")
    try:
        snapshots = fetch_universe_constituents()
    except ConstituentFetchError as exc:
        raise SystemExit(f"抓取失敗：{exc}")

    members = {etf: snap.stock_ids for etf, snap in snapshots.items()}
    as_of = min(snap.as_of for snap in snapshots.values())

    print()
    print("=" * 72)
    print(f"成分股快照  as_of {as_of}")
    print("=" * 72)
    for etf_id, snap in snapshots.items():
        print(
            f"  {etf_id}  {len(snap.constituents):>3} 檔"
            f"｜權重合計 {snap.total_weight:>6.2f}%"
            f"｜前三 {[c.stock_id for c in snap.constituents[:3]]}"
        )

    ids_by_etf = {etf: set(ids) for etf, ids in members.items()}
    all_ids: set[str] = set()
    for ids in ids_by_etf.values():
        all_ids |= ids

    overlap = ids_by_etf.get("0050", set()) & ids_by_etf.get("0051", set())
    print()
    print(f"  聯集 {len(all_ids)} 檔｜重疊 {len(overlap)} 檔 {sorted(overlap) or ''}")

    if len(all_ids) != 150:
        print(f"  ⚠️  聯集為 {len(all_ids)} 檔，D2 預期 150 檔——請確認官網是否調整成分")

    if args.dry_run:
        print()
        print("（--dry-run，未寫檔）")
        return

    path = store.save(as_of, members)
    print()
    print(f"已寫入 {path}")
    print(f"目前共有 {len(store.available_dates())} 份快照")
    print()
    print("提醒：元大官網只有當期成分股。回測期間（2023-2026）仍需走市值排名代理，")
    print("      並由 resolve_universe() 標註 Provenance.PROXY 與 survivorship 警告。")


if __name__ == "__main__":
    main()

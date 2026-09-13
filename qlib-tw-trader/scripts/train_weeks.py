#!/usr/bin/env python3
"""
序列訓練指定週別的模型

為什麼不用 /api/v1/models/train-batch：那個端點只接受「年」，會把整年
所有週都排進去（可能上百個模型、數十小時）。這支腳本讓你精確控制要
訓練哪幾週，並逐一等待完成，避免佇列塞爆或記憶體壓力。

用法：
    PYTHONPATH=. .venv/bin/python scripts/train_weeks.py 2026W16 2026W35
    PYTHONPATH=. .venv/bin/python scripts/train_weeks.py 2026W16 2026W35 --skip-trained

輸出：
    逐週進度印到 stdout（可 tee 存檔）
    每週結果寫入 scripts/output/train_weeks_<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

API_BASE = "http://localhost:8000/api/v1"
DB_PATH = Path("data/data.db")
OUTPUT_DIR = Path("scripts/output")

POLL_INTERVAL_SEC = 20
TIMEOUT_PER_MODEL_SEC = 1800  # 單一模型最長等待 30 分鐘


def parse_week(week_id: str) -> tuple[int, int]:
    """'2026W16' → (2026, 16)"""
    year, _, week = week_id.partition("W")
    return int(year), int(week)


def week_range(start: str, end: str) -> list[str]:
    """產生 start ~ end 之間的週別清單（同一年）"""
    y1, w1 = parse_week(start)
    y2, w2 = parse_week(end)
    if y1 != y2:
        raise SystemExit("目前只支援同一年內的區間")
    if w1 > w2:
        w1, w2 = w2, w1
    return [f"{y1}W{w:02d}" for w in range(w1, w2 + 1)]


def trained_weeks() -> set[str]:
    """已成功訓練的週別"""
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(
            "SELECT week_id FROM training_runs WHERE status = 'completed'"
        ).fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


def latest_job() -> tuple[str, float | None, str, str]:
    """讀最新一筆 job 的 (status, progress, message, result)"""
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            "SELECT status, progress, message, result FROM jobs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    if not row:
        return "unknown", None, "", ""
    return row[0], row[1], (row[2] or ""), (row[3] or "")


def train_one(week_id: str) -> dict:
    """提交單週訓練並等待完成"""
    started = time.time()
    resp = httpx.post(f"{API_BASE}/models/train", json={"week_id": week_id}, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    last_message = ""
    while True:
        elapsed = time.time() - started
        if elapsed > TIMEOUT_PER_MODEL_SEC:
            return {
                "week_id": week_id,
                "job_id": job_id,
                "status": "timeout",
                "elapsed_sec": round(elapsed, 1),
            }

        time.sleep(POLL_INTERVAL_SEC)
        status, progress, message, result = latest_job()

        if message != last_message:
            print(f"    [{elapsed:5.0f}s] {progress or 0:5.1f}%  {message[:60]}", flush=True)
            last_message = message

        if status in ("completed", "failed"):
            payload: dict = {
                "week_id": week_id,
                "job_id": job_id,
                "status": status,
                "elapsed_sec": round(time.time() - started, 1),
            }
            try:
                payload["result"] = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                payload["result"] = result[:400]
            return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="序列訓練指定週別的模型")
    parser.add_argument("start", help="起始週，例如 2026W16")
    parser.add_argument("end", help="結束週，例如 2026W35")
    parser.add_argument(
        "--skip-trained", action="store_true", help="跳過已成功訓練的週別"
    )
    args = parser.parse_args()

    weeks = week_range(args.start, args.end)
    done = trained_weeks() if args.skip_trained else set()
    todo = [w for w in weeks if w not in done]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"train_weeks_{stamp}.jsonl"

    print("=" * 72)
    print(f"序列訓練  {args.start} ~ {args.end}")
    print(f"總週數 {len(weeks)}｜已訓練跳過 {len(weeks) - len(todo)}｜待訓練 {len(todo)}")
    print(f"紀錄檔 {out_path}")
    print("=" * 72, flush=True)

    results: list[dict] = []
    wall_start = time.time()

    for i, week in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {week}", flush=True)
        try:
            res = train_one(week)
        except Exception as exc:  # 網路或 API 層失敗，記錄後續跑
            res = {"week_id": week, "status": "error", "result": str(exc)[:300]}

        results.append(res)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")

        ic = ""
        if isinstance(res.get("result"), dict):
            model_ic = res["result"].get("model_ic")
            if model_ic is not None:
                ic = f"  IC={model_ic:.4f}"
        print(
            f"    → {res['status']}  {res.get('elapsed_sec', 0):.0f}s{ic}",
            flush=True,
        )

        ok = sum(1 for r in results if r["status"] == "completed")
        avg = (time.time() - wall_start) / i
        remain = (len(todo) - i) * avg
        print(
            f"    進度 {i}/{len(todo)}｜成功 {ok}｜"
            f"平均 {avg / 60:.1f} 分/模型｜預估剩餘 {remain / 60:.0f} 分",
            flush=True,
        )

    print()
    print("=" * 72)
    ok = sum(1 for r in results if r["status"] == "completed")
    print(f"完成：成功 {ok} / {len(todo)}｜總耗時 {(time.time() - wall_start) / 60:.1f} 分")
    failures = [r for r in results if r["status"] != "completed"]
    if failures:
        print(f"失敗 {len(failures)} 筆：")
        for r in failures:
            print(f"  {r['week_id']}  {r['status']}  {str(r.get('result'))[:150]}")
    print(f"紀錄檔 {out_path}")
    print("=" * 72)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

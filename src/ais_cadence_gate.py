#!/usr/bin/env python3
"""
================================================================================
AIS 抓取節奏閘門 — Cadence gate for update-ais.yml
================================================================================

GitHub 的 schedule 事件會在高負載時延遲甚至**直接丟棄**。2026-10-05 → 10-19
實測（update-ais.yml 的 Actions 執行紀錄）：243 個 cron 時段只觸發 76 次
（31%），沒有任何取消/失敗紀錄 —— 是排程根本沒被觸發；排隊等待時間中位數
0 分，所以不是 data-pipeline 併發群組卡住。實際間隔中位數 4.4h（p90 7.1h），
原本設計是白天每小時、夜間每 2 小時。03:30、07:30、08:30 UTC 這三個時段
14 天裡一次都沒跑過。航跡預測的回測已經證明這個間隔是 SAR 比對的主要瓶頸
（見 root CLAUDE.md「Trajectory Prediction」）。

對策：cron 加密成每 20 分鐘（避開整點/半點），由本閘門決定這次要不要真的跑：
上一份成功快照（data/ais_snapshot.json 的 updated_at）還夠新就跳過 ——
不抓資料、不碰代理、不 commit。所以 GitHub 正常時有效節奏仍是原本的
「白天每小時、夜間每 2 小時」（commit 數不會暴增，repo 不會被撐大），
GitHub 丟排程時則有 3 倍的機會補上。

快照只在抓取**成功**時更新：代理斷線期間每次觸發都會重試（以 31% 觸發率約
每小時一次，GitHub 正常時最多每 20 分鐘一次）—— 重試本來就是斷線時要的。

手動觸發（workflow_dispatch）一律執行。

輸出（給 $GITHUB_OUTPUT）：`run=true|false`；理由印在 stderr。
================================================================================
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / 'data' / 'ais_snapshot.json'

# 白天 = 台灣 08:00–17:00 = UTC 00:00–09:00（與原本 cron 的高頻時段相同）
DAY_HOURS_UTC = range(0, 9)
DAY_MIN_INTERVAL_MIN = 50       # 白天約每小時一次（留 10 分鐘給觸發時間的抖動）
NIGHT_MIN_INTERVAL_MIN = 110    # 夜間約每 2 小時一次


def min_interval_min(now):
    return DAY_MIN_INTERVAL_MIN if now.hour in DAY_HOURS_UTC else NIGHT_MIN_INTERVAL_MIN


def parse_iso(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def should_run(event, now, last_updated):
    """回傳 (run: bool, reason: str)。last_updated 為 None 表示讀不到快照時間。"""
    if event != 'schedule':
        return True, f'{event} 觸發一律執行'
    if last_updated is None:
        return True, '讀不到快照時間，照常執行'
    age_min = (now - last_updated).total_seconds() / 60.0
    limit = min_interval_min(now)
    if age_min < 0:
        return True, f'快照時間在未來（{age_min:.0f} 分），照常執行'
    if age_min < limit:
        return False, f'快照 {age_min:.0f} 分前才更新（< {limit} 分），跳過本次'
    return True, f'快照已 {age_min:.0f} 分未更新（≥ {limit} 分），執行'


def read_last_updated(path=SNAPSHOT_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return parse_iso(data.get('updated_at')) if isinstance(data, dict) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description='update-ais.yml 節奏閘門')
    ap.add_argument('--event', default='schedule', help='github.event_name')
    ap.add_argument('--snapshot', default=str(SNAPSHOT_PATH))
    args = ap.parse_args(argv)
    run, reason = should_run(args.event, datetime.now(timezone.utc),
                             read_last_updated(args.snapshot))
    print(f'⏱️ {reason}', file=sys.stderr)
    print(f'run={"true" if run else "false"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

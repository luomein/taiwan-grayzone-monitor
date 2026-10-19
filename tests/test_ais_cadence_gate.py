"""Tests for src/ais_cadence_gate.py — update-ais.yml 的節奏閘門。"""
import json
from datetime import datetime, timedelta, timezone

import ais_cadence_gate as g


def at(h, m=0):
    return datetime(2026, 10, 19, h, m, tzinfo=timezone.utc)


def test_manual_dispatch_always_runs():
    run, _ = g.should_run('workflow_dispatch', at(3), at(3) - timedelta(minutes=1))
    assert run


def test_daytime_hourly():
    now = at(3, 7)   # UTC 03:07 = 台灣 11:07，白天
    assert not g.should_run('schedule', now, now - timedelta(minutes=30))[0]
    assert g.should_run('schedule', now, now - timedelta(minutes=55))[0]


def test_night_two_hourly():
    now = at(15, 27)  # 台灣 23:27，夜間
    assert not g.should_run('schedule', now, now - timedelta(minutes=80))[0]
    assert g.should_run('schedule', now, now - timedelta(minutes=115))[0]


def test_day_night_boundary_follows_original_cron():
    assert g.min_interval_min(at(8, 47)) == g.DAY_MIN_INTERVAL_MIN
    assert g.min_interval_min(at(9, 7)) == g.NIGHT_MIN_INTERVAL_MIN
    assert g.min_interval_min(at(23, 47)) == g.NIGHT_MIN_INTERVAL_MIN
    assert g.min_interval_min(at(0, 7)) == g.DAY_MIN_INTERVAL_MIN


def test_20min_cron_yields_original_cadence_when_github_is_healthy():
    """每 20 分鐘觸發、每次都觸發成功時，實際執行次數應等於原本的設計（白天每小時、夜間每 2 小時）。"""
    t = at(0, 7) - timedelta(days=1)
    last = t - timedelta(hours=3)
    runs = 0
    for _ in range(72):            # 一整天
        run, _ = g.should_run('schedule', t, last)
        if run:
            runs += 1
            last = t
        t += timedelta(minutes=20)
    # 原 cron：白天 9 次 + 夜間 8 次 = 17
    assert 15 <= runs <= 19


def test_unreadable_or_future_snapshot_runs(tmp_path):
    assert g.should_run('schedule', at(3), None)[0]
    assert g.should_run('schedule', at(3), at(4))[0]
    bad = tmp_path / 'snap.json'
    bad.write_text('{broken')
    assert g.read_last_updated(bad) is None
    assert g.read_last_updated(tmp_path / 'missing.json') is None


def test_read_last_updated(tmp_path):
    p = tmp_path / 'snap.json'
    p.write_text(json.dumps({'updated_at': '2026-10-19T04:12:41.427181+00:00'}))
    assert g.read_last_updated(p) == datetime(2026, 10, 19, 4, 12, 41, 427181,
                                              tzinfo=timezone.utc)


def test_main_writes_github_output_line(tmp_path, capsys):
    p = tmp_path / 'snap.json'
    p.write_text(json.dumps({'updated_at': '2000-01-01T00:00:00Z'}))
    g.main(['--event', 'schedule', '--snapshot', str(p)])
    assert capsys.readouterr().out.strip() == 'run=true'

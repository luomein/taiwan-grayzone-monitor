"""Tests for the trajectory-prediction phase-0 modules:
src/trajectory_core.py, src/build_trajectory_dataset.py, src/eval_trajectory.py,
plus the `rt` hooks in fetch_ais_data.py / match_sar_ais.py.

All synthetic (tmp_path archives, inline tracks). Must run on the CI test env,
which has only requests + pytest — no numpy / pandas / pyarrow.
"""
import gzip
import json
import sys
from datetime import datetime, timezone

import pytest

import build_trajectory_dataset as bd
import eval_trajectory as ev
import match_sar_ais as m
import trajectory_core as tc

HOUR = 3600.0
T0 = 1_790_000_000.0  # 2026-09 epoch anchor


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


# ── trajectory_core：定位時刻 ────────────────────────────────────────────────

def test_record_time_is_taiwan_local_time():
    # 航港局 Record_Time 是台灣時間（UTC+8）
    t = tc.record_time_epoch('2026/10/19 12:05:38')
    assert t == datetime(2026, 10, 19, 4, 5, 38, tzinfo=timezone.utc).timestamp()


def test_record_time_accepts_iso_and_rejects_garbage():
    assert tc.record_time_epoch('2026-10-19T04:05:38Z') == \
        datetime(2026, 10, 19, 4, 5, 38, tzinfo=timezone.utc).timestamp()
    assert tc.record_time_epoch('') is None
    assert tc.record_time_epoch(None) is None
    assert tc.record_time_epoch('not a time') is None


def test_compact_rt_only_for_moving_and_stale_fixes():
    now = datetime(2026, 10, 19, 4, 12, 41, tzinfo=timezone.utc).timestamp()
    rec = '2026/10/19 12:05:38'  # 7m03s 前
    assert tc.compact_rt(rec, now, 8.0) == 423
    assert tc.compact_rt(rec, now, 0.3) is None           # 靜止船不寫
    assert tc.compact_rt('2026/10/19 12:12:10', now, 8.0) is None  # <60s 不寫
    assert tc.compact_rt('2026/10/18 12:00:00', now, 8.0) is None  # >6h 視為壞時鐘
    assert tc.compact_rt('2026/10/19 12:20:00', now, 8.0) is None  # 未來時刻
    assert tc.compact_rt('', now, 8.0) is None


def test_point_epoch_subtracts_rt():
    assert tc.point_epoch(T0, {'rt': 300}) == T0 - 300
    assert tc.point_epoch(T0, {}) == T0
    assert tc.point_epoch(T0, {'rt': -5}) == T0
    assert tc.point_epoch(T0, {'rt': 'x'}) == T0


def test_fetch_rt_field_emits_compact_flag():
    import fetch_ais_data as f
    now = datetime(2026, 10, 19, 4, 12, 41, tzinfo=timezone.utc).timestamp()
    assert f._rt_field({'record_time': '2026/10/19 12:05:38', 'speed': 9.0}, now) == {'rt': 423}
    assert f._rt_field({'record_time': '2026/10/19 12:05:38', 'speed': 0.0}, now) == {}


# ── trajectory_core：基準預測器 ──────────────────────────────────────────────

def straight_track(n=6, step_h=1.0, kn=10.0, heading=90.0, lat=23.0, lon=120.0, t0=T0):
    pts = []
    for i in range(n):
        la, lo = tc.destination_point(lat, lon, heading, kn * tc.KN_TO_KMH * step_h * i)
        pts.append((t0 + i * step_h * HOUR, la, lo, kn, heading, 0))
    return pts


def test_cv_predictors_extrapolate_straight_track():
    pts = straight_track()
    hist, truth = pts[:4], pts[5]
    dt = truth[0] - hist[-1][0]
    for name in ('cv_reported', 'cv_track'):
        pred = tc.BASELINE_PREDICTORS[name](hist, dt)
        assert tc.prediction_error_km(pred, truth[1:3]) < 0.2, name
    # hold 誤差 = 兩小時航程
    hold = tc.predict_hold(hist, dt)
    assert tc.prediction_error_km(hold, truth[1:3]) == pytest.approx(2 * 10 * 1.852, rel=0.01)


def test_cv_reported_holds_when_stationary_or_no_heading():
    p = [(T0, 23.0, 120.0, 0.2, 90.0)]
    assert tc.predict_cv_reported(p, HOUR) == (23.0, 120.0)
    p = [(T0, 23.0, 120.0, 8.0, None)]
    assert tc.predict_cv_reported(p, HOUR) == (23.0, 120.0)


def test_cv_track_ignores_bad_reported_cog():
    # CN 漁船常亂報 COG：實際向東走，回報航向 0
    pts = [(t, la, lo, kn, 0.0) for t, la, lo, kn, _, _ in straight_track(n=3)]
    pred = tc.predict_cv_track(pts[:2], HOUR)
    assert tc.prediction_error_km(pred, pts[2][1:3]) < 0.1


def test_cv_track_falls_back_when_previous_fix_too_old():
    a = (T0, 23.0, 120.0, 0.0, 0.0)
    b = (T0 + 5 * HOUR, 23.5, 120.0, 0.0, 0.0)
    assert tc.predict_cv_track([a, b], HOUR) == (23.5, 120.0)


# ── match_sar_ais：rt 與 Record_Time 接線 ────────────────────────────────────

def test_sar_loader_uses_fix_time(tmp_path, monkeypatch):
    tier1 = tmp_path / 'tier1.json'
    tier1.write_text(json.dumps([{
        'timestamp': iso(T0),
        'vessels': [{'mmsi': '412000001', 'name': 'A', 'lat': 23.0, 'lon': 120.0,
                     'speed': 10.0, 'heading': 90.0, 'type_name': 'cargo', 'rt': 600}],
    }]))
    snap = tmp_path / 'snap.json'
    snap.write_text(json.dumps({'vessels': [{
        'mmsi': '412000002', 'name': 'B', 'lat': 24.0, 'lon': 121.0, 'speed': 0.0,
        'heading': 0.0, 'type_name': 'cargo',
        'record_time': '2026/10/19 12:05:38', 'last_update': '2026-10-19T04:12:41+00:00',
    }]}))
    monkeypatch.setattr(m, 'TIER1_PATH', tier1)
    monkeypatch.setattr(m, 'TIER2_PATHS', (tmp_path / 'missing.json',))
    monkeypatch.setattr(m, 'SNAPSHOT_PATH', snap)
    tracks = m.load_ais_tracks()
    assert tracks['412000001']['points'][0][0] == T0 - 600
    assert tracks['412000002']['points'][0][0] == \
        datetime(2026, 10, 19, 4, 5, 38, tzinfo=timezone.utc).timestamp()


# ── build_trajectory_dataset：歸檔讀取 ───────────────────────────────────────

def entry(t, vessels):
    return {'timestamp': iso(t), 'vessels': vessels}


def vessel(mmsi, name, lat, lon, speed=5.0, heading=90.0, type_name='cargo', **kw):
    return dict(mmsi=mmsi, name=name, lat=lat, lon=lon, speed=speed,
                heading=heading, type_name=type_name, **kw)


def make_archive(root):
    """兩種佈局並存：舊月度 .jsonl + 每輪 .jsonl.gz。"""
    arc = root / 'archive'
    arc.mkdir()
    (arc / 'ais_track_2026-08.jsonl').write_text(
        json.dumps(entry(datetime(2026, 8, 20, tzinfo=timezone.utc).timestamp(),
                         [vessel('412000001', 'A', 23.0, 120.0)])) + '\n'
        + 'not json\n', encoding='utf-8')
    day = arc / '2026-09' / '03'
    day.mkdir(parents=True)
    t = datetime(2026, 9, 3, 1, 0, tzinfo=timezone.utc).timestamp()
    with gzip.open(day / 'ais_commercial_20260903T010000Z.jsonl.gz', 'wt', encoding='utf-8') as f:
        f.write(json.dumps(entry(t, [vessel('412000001', 'A', 23.1, 120.1, rt=120)])) + '\n')
    with gzip.open(day / 'ais_track_20260903T010000Z.jsonl.gz', 'wt', encoding='utf-8') as f:
        f.write(json.dumps(entry(t, [vessel('900000001', 'BUOY 1', 23.2, 120.2)])) + '\n')
    (day / 'unrelated.txt').write_text('x')
    return root


def test_discover_archive_files_both_layouts_and_date_filter(tmp_path):
    make_archive(tmp_path)
    files = bd.discover_archive_files(tmp_path)
    names = [p.name for _, p in files]
    assert names[0] == 'ais_track_2026-08.jsonl'  # 時序排序：舊月度在前
    assert len(names) == 3
    only_sep = bd.discover_archive_files(tmp_path, start='2026-09-01')
    assert all('2026-08' not in p.name for _, p in only_sep)
    only_track = bd.discover_archive_files(tmp_path, tiers=('track',))
    assert all(tier == 'track' for tier, _ in only_track)


def test_load_tracks_skips_bad_lines_and_nonvessels_and_applies_rt(tmp_path):
    make_archive(tmp_path)
    files = bd.discover_archive_files(tmp_path)
    tracks = bd.load_tracks(bd.iter_archive_entries(files))
    assert set(tracks) == {('412000001', 'A')}  # 浮標被排除，壞行略過
    tr = tracks[('412000001', 'A')]
    assert tr['tiers'] == {'track', 'commercial'}
    t_sep = datetime(2026, 9, 3, 1, 0, tzinfo=timezone.utc).timestamp()
    assert max(p[0] for p in tr['points']) == t_sep - 120


def test_shared_mmsi_split_by_name():
    t = T0
    entries = [('track', t + i * HOUR, entry(t + i * HOUR, [
        vessel('412000009', 'BOAT A', 23.0, 120.0 + 0.01 * i),
        vessel('412000009', 'BOAT B', 25.0, 122.0 + 0.01 * i),
    ])) for i in range(4)]
    tracks = bd.load_tracks(entries)
    assert set(tracks) == {('412000009', 'BOAT A'), ('412000009', 'BOAT B')}
    for tr in tracks.values():
        assert len(bd.segment_track(tr['points'])) == 1


# ── build_trajectory_dataset：切段與切分 ─────────────────────────────────────

def test_segment_track_splits_on_gap_teleport_and_drops_short():
    pts = straight_track(n=4)
    later = straight_track(n=4, t0=pts[-1][0] + 8 * HOUR)  # 8h 中斷
    teleport = (later[-1][0] + HOUR, 30.0, 129.0, 5.0, 90.0, 0)  # 單點瞬移
    after = straight_track(n=3, t0=teleport[0] + HOUR, lat=later[-1][1], lon=later[-1][2])
    dup = (pts[1][0] + 0.5, pts[1][1], pts[1][2], 10.0, 90.0, 0)  # 同時刻重複
    segs = bd.segment_track(pts + later + [teleport] + after + [dup])
    assert [len(s) for s in segs] == [4, 4, 3]


def test_split_bounds_follow_point_quantiles():
    times = list(range(100))
    assert bd.split_bounds(times) == (70, 85)
    # 密度不均：前段很密時邊界跟著點數走，不是跟著時間跨度
    dense = [i * 0.01 for i in range(900)] + list(range(10, 110))
    b1, _ = bd.split_bounds(dense)
    assert b1 < 9


def test_cut_at_bounds_never_crosses_splits():
    seg = [(float(t), 23.0, 120.0, 0.0, 0.0, 0) for t in range(10)]
    parts = bd.cut_at_bounds(seg, (4, 7))
    assert [(lab, len(s)) for lab, s in parts] == [('train', 4), ('val', 3), ('test', 3)]
    # 邊界兩側只剩 1 點的子段太短，丟掉
    assert bd.cut_at_bounds(seg, (1, 9)) == [('val', seg[1:9])]


def test_type_group_priorities():
    assert bd.type_group({'type_name': 'fishing', 'gov': None, 'name': 'X'}) == 'fishing'
    assert bd.type_group({'type_name': 'cargo', 'gov': None, 'name': 'MINDONGYU63179'}) == 'fishing'
    assert bd.type_group({'type_name': 'other', 'gov': 'research', 'name': 'XIANGYANGHONG03'}) == 'gov'
    assert bd.type_group({'type_name': 'tanker', 'gov': None, 'name': 'OCEAN STAR'}) == 'commercial'
    assert bd.type_group({'type_name': 'unknown', 'gov': None, 'name': 'X'}) == 'other'


def test_write_points_csv_fallback_without_pyarrow(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyarrow', None)  # 強制 ImportError
    tr = {'mmsi': '412000001', 'name': 'A', 'type_name': 'cargo', 'gov': None}
    segs = [{'track': tr, 'group': 'commercial', 'points': straight_track(n=5)}]
    rows = bd.iter_rows(segs, bd.split_bounds([p[0] for p in segs[0]['points']]),
                        lambda la, lo: None)
    path, n = bd.write_points(rows, tmp_path)
    assert path.name == 'points.csv.gz' and n == 3  # 5 點切成 train 3 + 兩個太短的段
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    assert header == bd.COLUMNS


# ── eval_trajectory ──────────────────────────────────────────────────────────

def test_find_truth_index_respects_tolerance():
    times = [0, 3600, 7200, 21600 + 1800, 43200 + 9000]
    assert ev.find_truth_index(times, 0, 1) == 1
    assert ev.find_truth_index(times, 0, 2) == 2
    assert ev.find_truth_index(times, 0, 6) == 3      # 差 0.5h ≤ 1.5h 容許
    assert ev.find_truth_index(times, 0, 12) == 4     # 差 2.5h ≤ 3h 容許
    assert ev.find_truth_index(times, 0, 24) is None  # 最近差 9.5h > 6h 容許
    assert ev.find_truth_index(times, 4, 1) is None   # 後面沒有點


def test_run_forecast_cv_beats_hold_on_straight_track():
    tr = {'mmsi': '412000001', 'name': 'A', 'type_name': 'cargo', 'gov': None}
    segs = [{'track': tr, 'group': 'commercial', 'points': straight_track(n=12)}]
    errs = ev.run_forecast(segs, lambda la, lo: None, lambda la, lo: True, stride=1,
                           horizons=(1, 2))
    res = ev.aggregate_forecast(errs, tc.BASELINE_PREDICTORS, horizons=(1, 2))
    assert set(res) >= {'all', 'group:commercial', 'regime:moving', 'cable_zone'}
    assert res['all']['cv_track']['2']['median_km'] < 0.2
    assert res['all']['hold']['2']['median_km'] > 30


def test_run_forecast_skips_in_port_issue_points():
    tr = {'mmsi': '412000001', 'name': 'A', 'type_name': 'cargo', 'gov': None}
    segs = [{'track': tr, 'group': 'commercial', 'points': straight_track(n=12)}]
    errs = ev.run_forecast(segs, lambda la, lo: 'PORT', lambda la, lo: False, stride=1)
    assert not errs


def test_run_sar_interp_exact_on_straight_track():
    tr = {'mmsi': '412000001', 'name': 'A', 'type_name': 'cargo', 'gov': None}
    segs = [{'track': tr, 'group': 'commercial', 'points': straight_track(n=8)}]
    rows, missing = ev.run_sar(segs, lambda la, lo: None, stride=1)
    res = ev.aggregate_sar(rows, missing)
    interp = res['interp']['commercial|interpolated|0.5-1h']  # 抽點後兩側各 1h
    assert interp['median_km'] < 0.05 and interp['within_gate'] == 1.0
    assert 'dr' in res


def test_issue_in_split_holdout_goes_to_test_only():
    bounds = (100, 200)
    holdout = next(str(400000000 + i) for i in range(1000) if bd.is_holdout_mmsi(str(400000000 + i)))
    normal = next(str(400000000 + i) for i in range(1000) if not bd.is_holdout_mmsi(str(400000000 + i)))
    assert ev.issue_in_split(50, holdout, 'test', bounds)
    assert not ev.issue_in_split(150, holdout, 'val', bounds)
    assert ev.issue_in_split(150, normal, 'val', bounds)
    assert not ev.issue_in_split(50, normal, 'test', bounds)
    assert ev.issue_in_split(50, normal, 'all', bounds)

"""Tests for the trajectory-prediction phase-1 modules:
src/trajectory_model.py (CI inference), src/fit_trajectory_params.py (offline fit),
eval_trajectory.run_sar_sim, and the match_sar_ais wiring.

All synthetic; must run on the CI test env (requests + pytest only).
"""
import json
import math

import pytest

import eval_trajectory as ev
import fit_trajectory_params as fit
import match_sar_ais as m
import trajectory_model as tm
from geo_utils import haversine_km

HOUR = 3600.0
T0 = 1_790_000_000.0


def track(n=6, step_h=1.0, kn=10.0, heading=90.0, lat=23.0, lon=120.0, t0=T0):
    out = []
    for i in range(n):
        la, lo = tm.offset(lat, lon,
                           kn * tm.KN_TO_KMH * step_h * i * math.sin(math.radians(heading)),
                           kn * tm.KN_TO_KMH * step_h * i * math.cos(math.radians(heading)))
        out.append((t0 + i * step_h * HOUR, la, lo, kn, heading))
    return out


def flat_table(value, edges=tm.DT_EDGES_H):
    return {'dt_edges_h': list(edges), 'speed_edges_kn': list(tm.SPEED_EDGES_KN),
            'r90_km': [[value] * (len(tm.SPEED_EDGES_KN) + 1) for _ in edges]}


def params(r90=2.0, admit=6.5, tau=None, source='reported'):
    g = {'dr': {'moving': {'tau_h': tau, 'source': source},
                'stationary': {'tau_h': 0.1, 'source': 'reported'}},
         'dr_r90': flat_table(r90), 'interp_r90': flat_table(r90)}
    return {'version': tm.PARAMS_VERSION, 'admit_max_r90_km': admit,
            'interp_max_gap_h': 8.0, 'groups': {'all': g, 'commercial': g}}


# ── 群組 / 分箱 ─────────────────────────────────────────────────────────────

def test_type_group_priorities():
    fish = lambda n: (n or '').startswith('MIN')
    assert tm.type_group('cargo', None, 'MINDONGYU1', fish) == 'fishing'
    assert tm.type_group('other', 'research', 'MINXYZ', fish) == 'gov'
    assert tm.type_group('tanker', None, 'OCEAN', fish) == 'commercial'
    assert tm.type_group('unknown', None, 'X', fish) == 'other'


def test_speed_bin_edges():
    assert [tm.speed_bin(v) for v in (None, 0, 0.9, 1.0, 4.9, 5.0, 12, 15, 30)] == \
        [0, 0, 0, 1, 1, 2, 3, 4, 4]


# ── 位置 ─────────────────────────────────────────────────────────────────────

def test_damped_factor_limits():
    assert tm.damped_distance_factor(2.0, None) == 2.0
    assert tm.damped_distance_factor(0.01, 1.0) == pytest.approx(0.01, rel=0.01)
    assert tm.damped_distance_factor(100.0, 1.0) == pytest.approx(1.0)  # 最多走 τ 小時


def test_extrapolate_constant_velocity_and_backwards():
    pts = track()
    a = pts[2]
    fwd = tm.extrapolate(a, pts[1], 2 * HOUR, None, 'reported')
    assert haversine_km(*fwd, pts[4][1], pts[4][2]) < 0.2
    back = tm.extrapolate(a, pts[3], -2 * HOUR, None, 'track')
    assert haversine_km(*back, pts[0][1], pts[0][2]) < 0.2


def test_track_velocity_ignores_bad_cog():
    pts = [(t, la, lo, kn, 0.0) for t, la, lo, kn, _ in track()]  # 往東走、報北
    pred = tm.predict_damped(pts[:3], HOUR, None, 'track')
    assert haversine_km(*pred, pts[3][1], pts[3][2]) < 0.2


# ── r90 查表 ─────────────────────────────────────────────────────────────────

def test_r90_lookup_interpolates_and_bounds():
    tb = flat_table(1.0)
    tb['r90_km'][3] = [3.0] * 5   # 1.5h 節點
    assert tm.r90_lookup(tb, 8, 1.0) == 1.0
    assert tm.r90_lookup(tb, 8, 1.25) == pytest.approx(2.0)
    assert tm.r90_lookup(tb, 8, 0.1) == 1.0
    assert tm.r90_lookup(tb, 8, 9.0) is None      # 超出表的時距不收
    assert tm.r90_lookup(None, 8, 1.0) is None


# ── estimate_position（match_sar_ais 相容）───────────────────────────────────

def test_estimate_interpolates_between_fixes():
    pts = track()
    t = pts[2][0] + 0.5 * HOUR
    e = tm.estimate_position(pts, t, 'commercial', params())
    mid = ((pts[2][1] + pts[3][1]) / 2, (pts[2][2] + pts[3][2]) / 2)
    assert e['method'] == 'interpolated' and e['unc_extra_km'] == 2.0
    assert haversine_km(e['lat'], e['lon'], *mid) < 0.05
    assert set(e) >= {'lat', 'lon', 'speed_kn', 'dt_h', 'unc_extra_km', 'method'}


def test_estimate_dead_reckons_both_ends():
    pts = track()
    after = tm.estimate_position(pts, pts[-1][0] + HOUR, 'commercial', params())
    assert after['method'] == 'dead_reckoning'
    expect = tm.offset(pts[-1][1], pts[-1][2], 10 * tm.KN_TO_KMH, 0.0)
    assert haversine_km(after['lat'], after['lon'], *expect) < 0.2
    before = tm.estimate_position(pts, pts[0][0] - HOUR, 'commercial', params())
    expect = tm.offset(pts[0][1], pts[0][2], -10 * tm.KN_TO_KMH, 0.0)
    assert haversine_km(before['lat'], before['lon'], *expect) < 0.2


def test_estimate_admission_by_r90():
    pts = track()
    assert tm.estimate_position(pts, pts[-1][0] + HOUR, 'commercial',
                                params(r90=7.0, admit=6.5)) is None
    # 時距超出表（> 8h）不收，即使 r90 很小
    assert tm.estimate_position(pts, pts[-1][0] + 9 * HOUR, 'commercial',
                                params(r90=0.5)) is None


def test_estimate_unknown_group_falls_back_to_all():
    pts = track()
    assert tm.estimate_position(pts, pts[2][0] + 1800, 'gov', params()) is not None


def test_stationary_hold():
    pts = [(T0 + i * HOUR, 23.0, 120.0, 0.0, 0.0) for i in range(3)]
    e = tm.estimate_position(pts, pts[-1][0] + 5 * HOUR, 'commercial', params())
    assert e['method'] == 'hold' and (e['lat'], e['lon']) == (23.0, 120.0)


def test_load_params_rejects_bad_files(tmp_path):
    assert tm.load_params(tmp_path / 'missing.json') is None
    bad = tmp_path / 'bad.json'
    bad.write_text('{nope')
    assert tm.load_params(bad) is None
    old = tmp_path / 'old.json'
    old.write_text(json.dumps({'version': 0, 'groups': {}}))
    assert tm.load_params(old) is None
    ok = tmp_path / 'ok.json'
    ok.write_text(json.dumps(params()))
    assert tm.load_params(ok)['admit_max_r90_km'] == 6.5


def test_committed_params_file_is_loadable():
    p = tm.load_params()
    assert p is not None, 'data/trajectory_params.json 必須提交且可讀'
    for g in ('all', 'commercial', 'fishing'):
        gp = p['groups'][g]
        assert set(gp['dr']) == {'moving', 'stationary'}
        for key in ('dr_r90', 'interp_r90'):
            rows = gp[key]['r90_km']
            assert len(rows) == len(tm.DT_EDGES_H)
            for sb in range(len(tm.SPEED_EDGES_KN) + 1):
                col = [r[sb] for r in rows if r[sb] is not None]
                assert col == sorted(col), f'{g}.{key} 速度分箱 {sb} 沿時距須單調不減'


# ── fit ───────────────────────────────────────────────────────────────────────

def test_build_r90_table_fallback_monotone_and_backfill():
    n = fit.MIN_CELL_SAMPLES
    own = {(2, 1): [1.0] * n, (3, 1): [0.5] * n}             # 1.5h 比 1h 小 → 單調化
    pooled = {(4, 1): [9.0] * n}                              # 2h 樣本只在合併表
    tb = fit.build_r90_table(own, pooled)
    col = [row[1] for row in tb['r90_km']]
    assert col[0] == col[1] == col[2] == 1.0                 # 前段回填
    assert col[3] == 1.0 and col[4] == 9.0                    # 單調 + 退回合併表
    assert col[5] is None                                     # 後段沒資料 → 不收件，不外插
    assert all(row[0] is None for row in tb['r90_km'])        # 沒資料的分箱維持 None


def test_fit_dr_means_prefers_hold_for_wandering_boat():
    # 漁船在原地繞圈但回報高速：hold（小 τ）應該勝過等速
    pts = []
    for i in range(12):
        ang = i * 1.3
        la, lo = tm.offset(23.0, 120.0, 2 * math.cos(ang), 2 * math.sin(ang))
        pts.append((T0 + i * HOUR, la, lo, 8.0, (math.degrees(ang) + 90) % 360))
    seg = {'track': {'mmsi': '412000001', 'name': 'A'}, 'group': 'fishing', 'points': pts}
    samples = fit.collect_dr_samples([seg], lambda la, lo: None)
    import random
    means = fit.fit_dr_means(samples, random.Random(1))
    assert means['fishing']['moving']['tau_h'] is not None
    assert means['fishing']['moving']['tau_h'] <= 1.0


def test_fit_end_to_end_produces_loadable_params(tmp_path):
    segs = []
    for k in range(30):
        pts = track(n=10, lat=22.0 + 0.1 * k, t0=T0 + k * 600)
        segs.append({'track': {'mmsi': f'4120000{k:02d}', 'name': f'S{k}'},
                     'group': 'commercial', 'points': pts})
    bounds = (T0 + 100 * HOUR, T0 + 200 * HOUR)   # 全部落在 train
    out = fit.fit(segs, bounds, lambda la, lo: None)
    path = tmp_path / 'p.json'
    path.write_text(json.dumps(out))
    p = tm.load_params(path)
    assert p and p['groups']['commercial']['dr']['moving']['tau_h'] is None  # 直線 → 等速
    e = tm.estimate_position(segs[0]['points'], segs[0]['points'][4][0] + 1800,
                             'commercial', p)
    assert e is not None and e['method'] == 'interpolated'


# ── 比對模擬 ─────────────────────────────────────────────────────────────────

def test_sar_sim_matches_well_separated_vessels():
    segs = []
    for k in range(250):   # ≥ SIM_MIN_DETECTIONS 艘同時刻
        pts = track(n=8, lat=20.0 + 0.04 * k, lon=118.0 + 0.04 * (k % 7))
        segs.append({'track': {'mmsi': f'41{k:07d}', 'name': f'V{k}'},
                     'group': 'commercial', 'points': pts})
    sim = ev.run_sar_sim(segs, lambda la, lo: None,
                         {'p1': ev.p1_estimator(params(r90=1.0))},
                         n_times=2, dark_frac=0.2, noise_km=0.1)
    a = sim['result']['p1']['all']
    assert a['visible'] > 0 and a['dark'] > 0
    assert a['correct'] > 0.95 and a['dark_claimed'] < 0.05


# ── match_sar_ais 接線 ───────────────────────────────────────────────────────

def test_match_sar_ais_uses_motion_params_when_given():
    pts = track()
    tracks = {'412000001': {'name': 'A', 'type_name': 'cargo', 'gov': None, 'points': pts}}
    t = pts[-1][0] + 4 * HOUR          # 超過 MAX_DT_HOURS=3h：p0 推不出
    assert m.estimate_track(tracks['412000001'], t) is None
    e = m.estimate_track(tracks['412000001'], t, params(r90=3.0))
    assert e is not None and e['unc_extra_km'] == 3.0
    est, grid = m.estimates_for_time(tracks, t, params(r90=3.0))
    assert '412000001' in est and tracks['412000001']['group'] == 'commercial'

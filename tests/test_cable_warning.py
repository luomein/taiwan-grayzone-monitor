"""Tests for the cable-approach early warning (trajectory prediction phase 2):
src/cable_warning.py (CI inference) and src/fit_cable_warning.py (offline fit).

Synthetic geometry: a "cable" is the meridian lon = 120.0, so cable distance is just
|lon − 120| converted to km. No cable-geo.json, no network; runs on requests + pytest.
"""
import json
import math

import cable_warning as cw
import fit_cable_warning as fit

HOUR = 3600.0
T0 = 1_790_000_000.0
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(23.0))


def cable_dist(lat, lon):
    return abs(lon - 120.0) * KM_PER_DEG_LON


def no_port(lat, lon):
    return None


def lon_at(km):
    """東邊離海纜 km 公里的經度。"""
    return 120.0 + km / KM_PER_DEG_LON


def tiny_model(rows=None, alert_p=0.2):
    """全群組基準 1/100；可在 gcsh 層放一格高機率。"""
    tables = {'g': {'commercial': [100, 1, 1, 1], 'fishing': [100, 1, 1, 1],
                    'other': [100, 1, 1, 1], 'all': [300, 3, 3, 3]},
              'gds': {}, 'gc': {}, 'gcs': {}, 'gcsh': {}}
    for level, key, row in rows or ():
        tables[level][key] = row
    return {'version': cw.MODEL_VERSION, 'horizons_h': [6, 12, 24], 'shrink_m': 10.0,
            'alert_p': alert_p, 'tables': tables}


# ── 特徵 ─────────────────────────────────────────────────────────────────────

def test_bins():
    assert [cw.speed_bin(v) for v in (None, 0.5, 1, 4.9, 5, 11.9, 12, 30)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [cw.heading_sector(h) for h in (0, 22.4, 22.5, 90, 359, None, 400)] == [0, 0, 1, 2, 0, -1, -1]
    assert [cw.dist_band(d) for d in (5.1, 9.9, 10, 25, 40, 99)] == [0, 0, 1, 2, 3, 3]
    assert cw.cell_key(23.04, 120.06) == '23.0,120.1'


def test_entry_event_definition_matches_scoring_thresholds():
    assert cw.is_entry_event(4.9, 3.0, False)
    assert not cw.is_entry_event(5.1, 3.0, False)     # 區外
    assert not cw.is_entry_event(2.0, 6.0, False)     # 過境（≥5kn）
    assert not cw.is_entry_event(2.0, 1.0, 'PORT')    # 港內


# ── predict（階層收縮）────────────────────────────────────────────────────────

def test_predict_shrinks_sparse_cells_toward_parent():
    key = cw.table_keys('commercial', 23.0, lon_at(8), 90.0, 3.0, 8.0)[3]
    one_of_one = tiny_model([('gcsh', key, [1, 1, 1, 1])])
    p = cw.predict(one_of_one, 'commercial', 23.0, lon_at(8), 90.0, 3.0, 8.0)[12]
    assert p < 0.15                                  # 1/1 不會變成 100%
    many = tiny_model([('gcsh', key, [200, 150, 160, 170])])
    p = cw.predict(many, 'commercial', 23.0, lon_at(8), 90.0, 3.0, 8.0)
    assert 0.7 < p[12] < 0.82 and p[6] < p[12] < p[24]


def test_predict_unknown_group_uses_all():
    p = cw.predict(tiny_model(), 'gov', 23.0, lon_at(8), 0.0, 3.0, 8.0)
    assert set(p) == {6, 12, 24} and all(0 < v < 0.05 for v in p.values())


def test_load_model_rejects_bad_files(tmp_path):
    assert cw.load_model(tmp_path / 'missing.json') is None
    (tmp_path / 'bad.json').write_text('{')
    assert cw.load_model(tmp_path / 'bad.json') is None
    (tmp_path / 'old.json').write_text(json.dumps({'version': 0, 'tables': {}}))
    assert cw.load_model(tmp_path / 'old.json') is None


def test_committed_model_is_loadable_and_consistent():
    model = cw.load_model()
    assert model is not None, 'data/cable_warning_model.json 必須提交且可讀'
    assert model['horizons_h'] == list(cw.HORIZONS_H)
    assert model['zone_km'] == cw.ZONE_KM and model['slow_kn'] == cw.SLOW_KN
    assert 0 < model['alert_p'] < 1
    for row in model['tables']['gcsh'].values():
        n, *ks = row
        assert n >= fit.MIN_CELL_N and all(0 <= k <= n for k in ks)
        assert ks == sorted(ks)                       # 6h ≤ 12h ≤ 24h 事件數單調


# ── build_warnings ───────────────────────────────────────────────────────────

def track(name, lon_km, sog=3.0, cog=270.0, t=T0, typ='cargo', earlier=None):
    pts = list(earlier or []) + [(t, 23.0, lon_at(lon_km), sog, cog)]
    return {'name': name, 'type_name': typ, 'points': pts}


def hot_model():
    """離海纜 8km、3kn、航向 270 的商船：12h 入區機率高。"""
    key = cw.table_keys('commercial', 23.0, lon_at(8), 270.0, 3.0, 8.0)[3]
    return tiny_model([('gcsh', key, [500, 300, 400, 450])])


def group_of(tr):
    return {'cargo': 'commercial', 'fishing': 'fishing'}.get(tr['type_name'], 'other')


def run(tracks, model=None, **kw):
    return cw.build_warnings(tracks, model or hot_model(), cable_dist, no_port, group_of,
                             now=T0, **kw)


def test_alerts_only_above_threshold_and_sorted():
    out = run({'412000001': track('HOT', 8), '412000002': track('COLD', 30)})
    assert [a['mmsi'] for a in out['alerts']] == ['412000001']
    a = out['alerts'][0]
    assert a['p12h'] > 0.5 and a['dist_km'] == 8.0 and a['group'] == 'commercial'
    assert out['summary']['vessels_scored'] == 2 and out['summary']['alerts'] == 1


def test_taiwan_flag_excluded_beacons_excluded_and_stale_fixes_skipped():
    tracks = {'416000001': track('TW', 8),
              '412000003': track('NET 12V2', 8),
              '412000004': track('OLD', 8, t=T0 - 5 * HOUR)}
    out = run(tracks, excluded=lambda mmsi, name: 'V2' in name)
    assert out['alerts'] == [] and out['summary']['vessels_scored'] == 0


def test_in_port_and_in_zone_transit_not_scored():
    def port(lat, lon):
        return 'PORT' if lon > lon_at(20) else None
    tracks = {'412000005': track('PORTED', 25), '412000006': track('TRANSIT', 2, sog=12)}
    out = cw.build_warnings(tracks, hot_model(), cable_dist, port, group_of, now=T0)
    assert out['summary']['vessels_scored'] == 0 and out['summary']['in_zone_slow'] == 0


def test_fishing_alert_requires_existing_high_risk():
    key = cw.table_keys('fishing', 23.0, lon_at(8), 270.0, 3.0, 8.0)[3]
    model = tiny_model([('gcsh', key, [500, 300, 400, 450])])
    tracks = {'412000007': track('BOAT A', 8, typ='fishing'),
              '412000008': track('BOAT B', 8, typ='fishing')}
    out = run(tracks, model=model, risk={'412000008': {'risk_level': 'high', 'risk_score': 9}})
    assert [a['mmsi'] for a in out['alerts']] == ['412000008']
    assert out['alerts'][0]['risk_score'] == 9


def test_unexpected_entry_scores_prior_probability():
    # 現在已在區內低速；6h 前在 30km 外（冷區）→ 入區前機率低 = 意外接近
    earlier = [(T0 - 6 * HOUR, 23.0, lon_at(30), 8.0, 270.0)]
    tr_cold = track('COLD ENTRY', 2, sog=1.0, earlier=earlier)
    # 對照：6h 前在熱區（8km、3kn、270°）→ 入區前機率高 = 意料中
    earlier_hot = [(T0 - 6 * HOUR, 23.0, lon_at(8), 3.0, 270.0)]
    tr_hot = track('EXPECTED', 2, sog=1.0, earlier=earlier_hot)
    # 一直待在區內的船不算「剛進來」
    always_in = track('RESIDENT', 2, sog=1.0,
                      earlier=[(T0 - 6 * HOUR, 23.0, lon_at(3), 1.0, 0.0)])
    out = run({'412000009': tr_cold, '412000010': tr_hot, '412000011': always_in})
    got = [(u['mmsi'], u['prior_horizon_h']) for u in out['unexpected_entries']]
    assert got == [('412000009', 6), ('412000010', 6)]          # 最意外的在前
    assert out['unexpected_entries'][0]['prior_p'] < out['unexpected_entries'][1]['prior_p']
    assert out['summary']['in_zone_slow'] == 3 and out['summary']['unexpected_entries'] == 2


# ── fit ───────────────────────────────────────────────────────────────────────

def seg(points, group='commercial', mmsi='412000001', name='A'):
    return {'track': {'mmsi': mmsi, 'name': name}, 'group': group, 'points': points}


def approaching(t0=T0, start_km=30.0, kmh=5.0, n=10, mmsi='412000001'):
    """每小時往海纜靠近 kmh 公里，進到 5km 內時減速到 2kn。"""
    pts = []
    for i in range(n):
        d = max(1.0, start_km - kmh * i)
        sog = 2.0 if d <= 5 else 8.0
        pts.append((t0 + i * HOUR, 23.0, lon_at(d), sog, 270.0))
    return seg(pts, mmsi=mmsi)


def test_build_samples_labels_by_horizon_and_skips_in_zone_points():
    s = approaching()                                 # 30,25,20,15,10,5(入區),…
    samples = fit.build_samples([s], (T0 + 999 * HOUR, T0 + 1000 * HOUR), cable_dist, no_port)
    by_d = {round(x[6]): x[7] for x in samples}
    assert 5 not in by_d and 1 not in by_d            # 區內點不是發布點
    assert by_d[30] == (1, 1, 1)                       # 5h 後入區
    assert by_d[10] == (1, 1, 1)
    assert all(x[0] == 'train' for x in samples)


def test_build_samples_applies_exclusion():
    samples = fit.build_samples([approaching()], (T0 + 999 * HOUR,) * 2, cable_dist, no_port,
                                excluded=lambda mmsi, name: True)
    assert samples == []


def test_fit_tables_prunes_sparse_cells():
    samples = [('train', 'commercial', 23.0, lon_at(8), 270.0, 3.0, 8.0, (1, 1, 1))] * 3 + \
              [('train', 'commercial', 24.0, lon_at(9), 270.0, 3.0, 9.0, (0, 0, 1))] * fit.MIN_CELL_N
    tables = fit.fit_tables(samples)
    assert tables['g']['commercial'] == [3 + fit.MIN_CELL_N, 3, 3, 3 + fit.MIN_CELL_N]
    assert len(tables['gcsh']) == 1                    # 3 筆的格被剪掉
    assert len(tables['gds']) >= 1                     # 粗層不剪


def test_ranking_metrics_and_threshold_choice():
    scored = [(0.9, 1), (0.8, 1), (0.7, 0), (0.2, 0), (0.1, 0)]
    m = fit.ranking_metrics(scored)
    assert m['auc'] == 1.0 and m['ap'] == 1.0
    assert fit.choose_alert_p(scored, target=0.6, grid=(0.1, 0.5, 0.75)) == 0.5
    assert fit.threshold_metrics(scored, 0.75) == {'threshold': 0.75, 'flagged': 2,
                                                    'precision': 1.0, 'recall': 1.0}


def test_fit_end_to_end_parity_with_ci_predict():
    segments = [approaching(t0=T0 + k * 24 * HOUR, mmsi=f'4120000{k:02d}') for k in range(30)]
    segments += [seg([(T0 + i * HOUR, 23.0, lon_at(30), 8.0, 0.0) for i in range(10)],
                     mmsi=f'4130000{k:02d}') for k in range(30)]   # 平行海纜的船，不入區
    bounds = (T0 + 9999 * HOUR, T0 + 10000 * HOUR)
    model = fit.fit(segments, bounds, cable_dist, no_port)
    assert model['version'] == cw.MODEL_VERSION
    hot = cw.predict(model, 'commercial', 23.0, lon_at(30), 270.0, 8.0, 30.0)[12]
    cold = cw.predict(model, 'commercial', 23.0, lon_at(30), 0.0, 8.0, 30.0)[12]
    assert hot > 0.5 > cold                            # 同一格、航向不同 → 差很多


def test_excluder_keeps_gov_vessels_with_invalid_mmsi():
    excluded = cw.make_excluder()
    # CHINACOASTGUARD14513 播 MMSI 766688888（MID 766 未指配），在金門外活動
    assert excluded('766688888', 'CHINACOASTGUARD14513') is False
    assert excluded('766688888', 'SOME FISHING BOAT') is True
    # 信標類排除不在安全閥範圍內
    assert excluded('901498040', 'CHINACOASTGUARD14513') is True

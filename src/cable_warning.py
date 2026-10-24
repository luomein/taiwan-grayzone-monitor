#!/usr/bin/env python3
"""
================================================================================
海纜接近預警 — Cable approach early warning (trajectory prediction, phase 2)
================================================================================

問題：哪些船會在未來 6／12／24 小時內「低速進入海纜 5km 內」（與威脅評分的
海纜鄰近 5km、低速 <5kn 同一組門檻）？事件發生後評分才看得到，預警要提前。

模型 = 歷史行為表（pattern of life），由 fit_trajectory_params 同一份
ais-archive 離線擬合（fit_cable_warning.py → data/cable_warning_model.json）：
  P(H 小時內低速入區 | 群組, 0.1° 格, 航向 8 扇區, 航速分箱, 距海纜分帶)
階層式收縮估計：群組整體 → 群組×距離帶×航速 → 群組×格 → 群組×格×航速
→ 群組×格×航速×航向，每層 p = (k + m·p_上層) / (n + m)。資料少的格子自動
退回上層，不會因為 1/1 就給 100%。

為什麼不用運動學外插：回測（test 切分，12h）商船 AUC 距離 0.74 / 等速外插
0.71 / 行為表 0.84；前 1% 警示的精確度 2.8% / 2.5% / 24.8%（基準率 0.9%）。
船在海纜旁是「減速、轉向」進港或錨泊，直線外插抓不到；歷史行為表抓得到。
高分區正是淡水／台北港外、馬祖福澳、澎湖馬公、金門料羅灣 —— 海纜登陸點旁。

兩份清單（docs/cable_warnings.json，shadow 模式，不進評分、前端尚未顯示）：
  alerts             在海上、海纜 5km 外、最新定位 ≤ MAX_FIX_AGE_H 的船（排除規則同
                     威脅評分：浮標/漁網信標/無效 MMSI，以及台灣船 MID 416），
                     p12 ≥ alert_p（漁船另需已被評為 high/critical 才列）
  unexpected_entries 已在海纜 5km 內低速的船，回看 3–12h 前的入區機率：
                     機率越低 = 越「不該來」—— 不是常態進港錨泊的接近（異常分數）
兩者都併入 suspicious_vessels.json 的 risk_score／risk_level／flags。

stdlib only（update-ais.yml 只裝 requests + pysocks）。
================================================================================
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path('data')
DOCS_DIR = Path('docs')
MODEL_PATH = Path(__file__).resolve().parent.parent / 'data' / 'cable_warning_model.json'
# 只提交 docs/ 一份（每輪 AIS 更新都重寫；比照 cf_radar.json 不放 data/ 雙份撐大 repo）
OUTPUT_PATH = DOCS_DIR / 'cable_warnings.json'
SUSPICIOUS_PATH = DATA_DIR / 'suspicious_vessels.json'
MODEL_VERSION = 1

ZONE_KM = 5.0              # 與 analyze_suspicious.CABLE_PROXIMITY_KM 一致
SLOW_KN = 5.0              # 與 analyze_suspicious.CABLE_LOITER_MAX_KNOTS 一致
HORIZONS_H = (6, 12, 24)
CELL_DEG = 0.1
SPEED_EDGES_KN = (1.0, 5.0, 12.0)    # 分箱：<1 / 1–5 / 5–12 / ≥12
DIST_EDGES_KM = (10.0, 20.0, 40.0)   # 分帶：5–10 / 10–20 / 20–40 / ≥40
SHRINK_M = 10.0
LEVELS = ('gds', 'gc', 'gcs', 'gcsh')   # 由粗到細；最上層 'g' 另外處理

MAX_FIX_AGE_H = 3.0        # 最新定位比快照舊超過 3h 的船不預警（已離開涵蓋範圍）
LOOKBACK_MIN_H = 3.0       # 異常分數回看的時窗
LOOKBACK_MAX_H = 12.0
MAX_ALERTS = 100
MAX_UNEXPECTED = 50
FISHING_ALERT_RISK = {'high', 'critical'}


# =============================================================================
# 特徵
# =============================================================================

def speed_bin(sog):
    v = sog or 0.0
    for i, edge in enumerate(SPEED_EDGES_KN):
        if v < edge:
            return i
    return len(SPEED_EDGES_KN)


def heading_sector(cog):
    if not isinstance(cog, (int, float)) or not 0.0 <= cog < 360.0:
        return -1
    return int(((cog + 22.5) % 360.0) // 45.0)


def dist_band(d_km):
    for i, edge in enumerate(DIST_EDGES_KM):
        if d_km < edge:
            return i
    return len(DIST_EDGES_KM)


def cell_key(lat, lon):
    return f'{round(lat / CELL_DEG) * CELL_DEG:.1f},{round(lon / CELL_DEG) * CELL_DEG:.1f}'


def table_keys(group, lat, lon, cog, sog, d_km):
    """各層的表鍵（由粗到細，對應 LEVELS）。fit 與 predict 共用，保證一致。"""
    c = cell_key(lat, lon)
    sb, hb, db = speed_bin(sog), heading_sector(cog), dist_band(d_km)
    return (
        f'{group}|{db}|{sb}',
        f'{group}|{c}',
        f'{group}|{c}|{sb}',
        f'{group}|{c}|{sb}|{hb}',
    )


def is_entry_event(d_km, sog, in_port):
    """低速入區事件：海纜 5km 內、<5kn、不在港內。"""
    return d_km <= ZONE_KM and (sog or 0.0) < SLOW_KN and not in_port


# =============================================================================
# 模型
# =============================================================================

def load_model(path=MODEL_PATH):
    try:
        model = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(model, dict) or model.get('version') != MODEL_VERSION:
        return None
    if not isinstance(model.get('tables'), dict):
        return None
    return model


def predict(model, group, lat, lon, cog, sog, d_km):
    """回傳 {h: p}，h ∈ model['horizons_h']。表裡每格的值是 [n, k_h1, k_h2, ...]。"""
    tables = model['tables']
    horizons = model['horizons_h']
    m = model.get('shrink_m', SHRINK_M)
    top = tables['g'].get(group) or tables['g'].get('all')
    if not top:
        return {h: 0.0 for h in horizons}
    n0 = top[0]
    ps = [(k + 1.0) / (n0 + 2.0) for k in top[1:]]
    for level, key in zip(LEVELS, table_keys(group, lat, lon, cog, sog, d_km)):
        row = tables.get(level, {}).get(key)
        if not row:
            continue
        n = row[0]
        ps = [(k + m * p) / (n + m) for k, p in zip(row[1:], ps)]
    return {h: p for h, p in zip(horizons, ps)}


# =============================================================================
# 預警
# =============================================================================

def make_excluder():
    """與威脅評分同一套排除規則（analyze_suspicious.EXCLUSION_RULES：浮標、漁網
    信標、MMSI 長度錯誤、未分配 MID…）。只靠 is_nonvessel_ais 會漏掉 `321 12V2`、
    `7V7`、`NETFISHING 7V7` 這類信標 —— 第一版的 'other' 群組警示幾乎全是它們。"""
    from analyze_suspicious import check_exclusion_rules

    def excluded(mmsi, name):
        return check_exclusion_rules(str(mmsi or ''), [name or ''])[0]
    return excluded


def is_taiwan_flag(mmsi):
    """MID 416：與威脅評分的 flag_taiwan 排除一致，不列入預警。"""
    return str(mmsi).startswith('416')


def build_warnings(tracks, model, cable_dist, in_port, group_of, risk=None, now=None,
                   excluded=None):
    """tracks: {mmsi: {'name','type_name','points':[(t,lat,lon,sog,cog),...]}}（依 t 排序）。

    cable_dist(lat, lon) → km；in_port(lat, lon) → truthy；group_of(track) → 群組。
    risk: {mmsi: {'risk_score','risk_level','risk_flags'}}。excluded(mmsi, name) → 是否排除
    （預設不排除）。台灣船（MID 416）一律不列。回傳輸出 dict。純函式。"""
    risk = risk or {}
    excluded = excluded or (lambda mmsi, name: False)
    if now is None:
        now = max((tr['points'][-1][0] for tr in tracks.values() if tr.get('points')),
                  default=0.0)
    alert_p = model.get('alert_p', 0.2)
    alerts, unexpected = [], []
    scored = 0
    in_zone_slow = 0
    by_group = {}

    for mmsi, tr in tracks.items():
        pts = tr.get('points') or []
        if not pts or is_taiwan_flag(mmsi) or excluded(mmsi, tr.get('name')):
            continue
        last = pts[-1]
        if now - last[0] > MAX_FIX_AGE_H * 3600:
            continue
        if in_port(last[1], last[2]):
            continue
        group = group_of(tr)
        d = cable_dist(last[1], last[2])
        base = {
            'mmsi': mmsi, 'name': tr.get('name', ''), 'group': group,
            'type_name': tr.get('type_name', 'unknown'),
            'lat': round(last[1], 4), 'lon': round(last[2], 4),
            'sog': round(last[3] or 0.0, 1),
            'cog': round(last[4], 1) if isinstance(last[4], (int, float)) else None,
            'dist_km': round(d, 2),
            'last_seen': datetime.fromtimestamp(last[0], timezone.utc).isoformat(),
            **risk.get(mmsi, {}),
        }

        if is_entry_event(d, last[3], False):
            in_zone_slow += 1
            prior = _prior_before_entry(pts, model, cable_dist, in_port, group)
            if prior is not None:
                unexpected.append({**base, **prior})
            continue
        if d <= ZONE_KM:
            continue                      # 在區內但不是低速：過境中，不預警也不算異常

        scored += 1
        p = predict(model, group, last[1], last[2], last[4], last[3], d)
        p12 = p.get(12, max(p.values()))
        if p12 < alert_p:
            continue
        if group == 'fishing' and base.get('risk_level') not in FISHING_ALERT_RISK:
            continue
        by_group[group] = by_group.get(group, 0) + 1
        alerts.append({**base, **{f'p{h}h': round(v, 3) for h, v in p.items()}})

    alerts.sort(key=lambda a: (-a.get('p12h', 0), -(a.get('risk_score') or 0)))
    unexpected.sort(key=lambda a: (a['prior_p'], -(a.get('risk_score') or 0)))
    return {
        'generated_at': datetime.fromtimestamp(now, timezone.utc).isoformat() if now else None,
        'model': {k: model.get(k) for k in ('generated_at', 'train_end', 'horizons_h',
                                             'alert_p', 'metrics')},
        'criteria': {'zone_km': ZONE_KM, 'slow_kn': SLOW_KN,
                     'max_fix_age_h': MAX_FIX_AGE_H,
                     'lookback_h': [LOOKBACK_MIN_H, LOOKBACK_MAX_H]},
        'summary': {
            'vessels_scored': scored,
            'alerts': len(alerts),
            'alerts_by_group': by_group,
            'in_zone_slow': in_zone_slow,
            'unexpected_entries': len(unexpected),
        },
        'alerts': alerts[:MAX_ALERTS],
        'unexpected_entries': unexpected[:MAX_UNEXPECTED],
    }


def _prior_before_entry(pts, model, cable_dist, in_port, group):
    """入區前 3–12h、還在區外且不在港內的最近一點 → 當時預測的入區機率。

    沒有這樣的點（一直在區內、或前面沒資料）→ None：不是「剛進來」，不評。"""
    t_now = pts[-1][0]
    for p in reversed(pts[:-1]):
        dt_h = (t_now - p[0]) / 3600.0
        if dt_h < LOOKBACK_MIN_H:
            continue
        if dt_h > LOOKBACK_MAX_H:
            break
        d = cable_dist(p[1], p[2])
        if d <= ZONE_KM or in_port(p[1], p[2]):
            return None                   # 3h 前就在區內／港內：不是新進入
        probs = predict(model, group, p[1], p[2], p[4], p[3], d)
        h = 6 if dt_h <= 6 else 12
        return {'prior_p': round(probs.get(h, 0.0), 4), 'prior_horizon_h': h,
                'prior_hours_before': round(dt_h, 1), 'prior_dist_km': round(d, 2),
                'prior_lat': round(p[1], 4), 'prior_lon': round(p[2], 4)}
    return None


# =============================================================================
# CLI（update-ais.yml）
# =============================================================================

def make_cable_dist():
    """海纜距離 km，以 0.01° 網格快取（~1km，對 5km 區界足夠）。"""
    import geofence
    cache = {}

    def dist(lat, lon):
        key = (round(lat, 2), round(lon, 2))
        d = cache.get(key)
        if d is None:
            r = geofence.nearest_cable(key[0], key[1], max_band_km=60)['nearest_cable_km']
            d = cache[key] = 999.0 if r is None else r
        return d
    return dist


def load_risk(path=SUSPICIOUS_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    out = {}
    for v in data.get('all_classifications') or []:
        if isinstance(v, dict) and v.get('mmsi'):
            out[str(v['mmsi'])] = {'risk_score': v.get('risk_score'),
                                   'risk_level': v.get('risk_level'),
                                   'risk_flags': (v.get('flags') or [])[:6]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='海纜接近預警（shadow）')
    ap.add_argument('--model', default=str(MODEL_PATH))
    args = ap.parse_args(argv)

    model = load_model(args.model)
    if model is None:
        print('⚠️ 找不到可用的 cable_warning_model.json，跳過海纜接近預警')
        return 0

    import geofence
    import match_sar_ais
    import trajectory_model
    from io_utils import atomic_write_json

    tracks = match_sar_ais.load_ais_tracks()
    result = build_warnings(
        tracks, model, make_cable_dist(), geofence.is_in_port_cached,
        lambda tr: trajectory_model.type_group(tr.get('type_name'), tr.get('gov'),
                                               tr.get('name')),
        risk=load_risk(), excluded=make_excluder())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUTPUT_PATH, result, compact=True)
    s = result['summary']
    print(f"🧲 海纜接近預警：評估 {s['vessels_scored']} 艘、預警 {s['alerts']} 艘 "
          f"{s['alerts_by_group']}；區內低速 {s['in_zone_slow']} 艘，"
          f"可回看入區機率 {s['unexpected_entries']} 艘")
    return 0


if __name__ == '__main__':
    sys.exit(main())

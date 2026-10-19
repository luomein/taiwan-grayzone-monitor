#!/usr/bin/env python3
"""
================================================================================
航跡預測回測 — Trajectory prediction backtest (P0 baselines)
================================================================================

離線工具（不進 CI）。讀 `ais-archive` 歸檔（build_trajectory_dataset 同一套
載入與切段），跑兩種回測，產出之後每個模型都要贏過的基準數字。

1. forecast — 海纜預警用的中長時預測
   在航段上每隔 --stride 個點「發布」一次預測，往後 1/2/6/12/24h 找最接近的
   實際點（容許 ±max(0.5h, 0.25×h)），預測到該點的**實際時刻**再算誤差。
   預測器：trajectory_core.BASELINE_PREDICTORS（hold / cv_reported / cv_track）。
   分層：船種群組、發布時是否在動、是否在海纜 CABLE_ZONE_KM 內。
   港內發布點不計（靠泊船預測「不動」會把準確度灌水）。

2. sar — SAR 過境時刻的位置估計（match_sar_ais 的實際用途）
   抽掉一個點，用 match_sar_ais.estimate_position 從剩下的點推回該時刻：
     interp  兩側都有點（時間加權內插）
     dr      只給前面的點（單側航位推算 / hold）
   同時檢查 match_sar_ais 的動態門檻是否涵蓋誤差：
     within_unc   誤差 ≤ 機動不確定性 + AIS 定位誤差（純 AIS 對 AIS）
     within_gate  誤差 ≤ gate_km(unc)（完美定位的 SAR 偵測會不會落在門檻內）
   注意：抽點讓 interp 的空窗變成兩倍，dt 比實際 SAR 比對悲觀，所以依 dt 分箱看。

用法：
  python3 src/eval_trajectory.py --archive /path/to/ais-archive \\
      [--start 2026-08-01] [--end 2026-10-19] [--stride 6] [--split all|val|test]

輸出：data/trajectory/baseline_eval.json（gitignored）＋ 終端機 Markdown 表
================================================================================
"""

import argparse
import sys
from array import array
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from build_trajectory_dataset import (
    is_holdout_mmsi, load_archive_segments, segment_times, split_bounds,
    split_label)
from io_utils import atomic_write_json
from trajectory_core import BASELINE_PREDICTORS, prediction_error_km

HORIZONS_H = (1, 2, 6, 12, 24)
CABLE_ZONE_KM = 10.0
MOVING_KN = 1.0
SAR_DT_BINS_H = (0.5, 1.0, 2.0, 3.0)
DEFAULT_STRIDE = 6
DEFAULT_OUT = Path('data/trajectory/baseline_eval.json')


def horizon_tolerance_h(h):
    return max(0.5, 0.25 * h)


def find_truth_index(times, i, horizon_h):
    """t_i + h 附近（容許範圍內）最接近的點索引；沒有回傳 None。"""
    target = times[i] + horizon_h * 3600.0
    tol = horizon_tolerance_h(horizon_h) * 3600.0
    j = bisect_left(times, target, lo=i + 1)
    best = None
    for k in (j - 1, j):
        if k <= i or k >= len(times):
            continue
        d = abs(times[k] - target)
        if d <= tol and (best is None or d < abs(times[best] - target)):
            best = k
    return best


def summarize_errors(errs):
    """誤差（km）→ {n, mean, median, p90}。"""
    n = len(errs)
    if not n:
        return {'n': 0}
    s = sorted(errs)
    return {'n': n,
            'mean_km': round(sum(s) / n, 3),
            'median_km': round(s[n // 2], 3),
            'p90_km': round(s[min(n - 1, int(0.9 * n))], 3)}


def make_cable_lookup(max_km=CABLE_ZONE_KM):
    """(lat, lon) → 是否在海纜 max_km 內，以 0.01° 網格快取（10km 門檻下夠細）。"""
    import geofence
    cache = {}

    def lookup(lat, lon):
        key = (round(lat, 2), round(lon, 2))
        hit = cache.get(key)
        if hit is None:
            d = geofence.nearest_cable(lat, lon, max_band_km=max_km)['nearest_cable_km']
            hit = d is not None and d <= max_km
            cache[key] = hit
        return hit
    return lookup


def issue_in_split(t, mmsi, split, bounds):
    """--split 篩選（依發布時刻）：test 另外收 holdout_mmsi 的船，val 排除它們。"""
    if split == 'all':
        return True
    label = split_label(t, bounds)
    if split == 'test':
        return label == 'test' or is_holdout_mmsi(mmsi)
    return label == split and not is_holdout_mmsi(mmsi)


# =============================================================================
# 1. forecast
# =============================================================================

def run_forecast(segments, port_lookup, cable_lookup, stride=DEFAULT_STRIDE,
                 split='all', bounds=None, predictors=BASELINE_PREDICTORS,
                 horizons=HORIZONS_H):
    """回傳 errs[(pred, h, group, regime, cable)] = array of km。"""
    errs = defaultdict(lambda: array('f'))
    for seg in segments:
        pts = seg['points']
        mmsi = seg['track']['mmsi']
        times = [p[0] for p in pts]
        for i in range(1, len(pts) - 1, stride):
            p = pts[i]
            if not issue_in_split(p[0], mmsi, split, bounds):
                continue
            if port_lookup(p[1], p[2]):
                continue
            regime = 'moving' if (p[3] or 0.0) >= MOVING_KN else 'stationary'
            cable = cable_lookup(p[1], p[2])
            history = pts[max(0, i - 3):i + 1]
            for h in horizons:
                j = find_truth_index(times, i, h)
                if j is None:
                    continue
                dt_s = times[j] - times[i]
                truth = (pts[j][1], pts[j][2])
                for name, fn in predictors.items():
                    e = prediction_error_km(fn(history, dt_s), truth)
                    errs[(name, h, seg['group'], regime, cable)].append(e)
    return errs


def aggregate_forecast(errs, predictors, horizons=HORIZONS_H):
    """errs → {stratum: {pred: {h: stats}}}；stratum ∈ all / group:* / regime:* / cable_zone。"""
    buckets = defaultdict(list)  # (stratum, pred, h) → [arrays]
    for (pred, h, group, regime, cable), arr in errs.items():
        buckets[('all', pred, h)].append(arr)
        buckets[(f'group:{group}', pred, h)].append(arr)
        buckets[(f'regime:{regime}', pred, h)].append(arr)
        if cable:
            buckets[('cable_zone', pred, h)].append(arr)
            buckets[(f'cable_zone:{group}', pred, h)].append(arr)
    out = defaultdict(lambda: defaultdict(dict))
    for (stratum, pred, h), arrs in buckets.items():
        merged = [x for a in arrs for x in a]
        out[stratum][pred][str(h)] = summarize_errors(merged)
    return {k: dict(v) for k, v in sorted(out.items())}


# =============================================================================
# 2. sar
# =============================================================================

def dt_bin(dt_h):
    lo = 0.0
    for hi in SAR_DT_BINS_H:
        if dt_h <= hi:
            return f'{lo:g}-{hi:g}h'
        lo = hi
    return f'>{SAR_DT_BINS_H[-1]:g}h'


def run_sar(segments, port_lookup, stride=DEFAULT_STRIDE, split='all', bounds=None):
    """回傳 rows[(mode, group, method, dt_bin)] = {'err': array, 'unc_ok', 'gate_ok', 'none'}。"""
    import match_sar_ais as m
    rows = defaultdict(lambda: {'err': array('f'), 'unc_ok': 0, 'gate_ok': 0})
    missing = defaultdict(int)
    for seg in segments:
        pts = seg['points']
        mmsi = seg['track']['mmsi']
        for k in range(1, len(pts) - 1, stride):
            p = pts[k]
            if not issue_in_split(p[0], mmsi, split, bounds):
                continue
            if port_lookup(p[1], p[2]):
                continue
            windows = {
                'interp': pts[max(0, k - 3):k] + pts[k + 1:k + 4],
                'dr': pts[max(0, k - 3):k],
            }
            for mode, window in windows.items():
                est = m.estimate_position(window, p[0])
                if est is None:
                    missing[(mode, seg['group'])] += 1
                    continue
                e = prediction_error_km((est['lat'], est['lon']), (p[1], p[2]))
                r = rows[(mode, seg['group'], est['method'], dt_bin(est['dt_h']))]
                r['err'].append(e)
                if e <= est['unc_extra_km'] + m.AIS_POS_ERROR_KM:
                    r['unc_ok'] += 1
                if e <= m.gate_km(est['unc_extra_km']):
                    r['gate_ok'] += 1
    return rows, missing


def aggregate_sar(rows, missing):
    out = defaultdict(dict)
    for (mode, group, method, dbin), r in sorted(rows.items()):
        stats = summarize_errors(list(r['err']))
        n = stats['n']
        stats['within_unc'] = round(r['unc_ok'] / n, 3) if n else None
        stats['within_gate'] = round(r['gate_ok'] / n, 3) if n else None
        out[mode][f'{group}|{method}|{dbin}'] = stats
    for (mode, group), n in sorted(missing.items()):
        out[mode][f'{group}|no_estimate'] = {'n': n}
    return dict(out)


# =============================================================================
# 報表
# =============================================================================

def forecast_table(result, stratum='all', horizons=HORIZONS_H):
    block = result.get(stratum) or {}
    if not block:
        return ''
    lines = [f'### forecast — {stratum}（median / p90 km，n）', '',
             '| predictor | ' + ' | '.join(f'{h}h' for h in horizons) + ' |',
             '|---|' + '---|' * len(horizons)]
    for pred, by_h in block.items():
        cells = []
        for h in horizons:
            s = by_h.get(str(h)) or {'n': 0}
            cells.append('—' if not s['n'] else
                         f"{s['median_km']:.2f} / {s['p90_km']:.1f} ({s['n']})")
        lines.append(f'| {pred} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def sar_table(result, mode):
    block = result.get(mode) or {}
    if not block:
        return ''
    lines = [f'### sar — {mode}', '',
             '| group | method | dt | n | median km | p90 km | within_unc | within_gate |',
             '|---|---|---|---|---|---|---|---|']
    for key, s in block.items():
        parts = key.split('|')
        if parts[1] == 'no_estimate':
            lines.append(f"| {parts[0]} | (no estimate) | | {s['n']} | | | | |")
            continue
        lines.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {s['n']} | "
                     f"{s['median_km']:.2f} | {s['p90_km']:.2f} | "
                     f"{s['within_unc']:.0%} | {s['within_gate']:.0%} |")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description='航跡預測基準回測')
    ap.add_argument('--archive', required=True)
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--stride', type=int, default=DEFAULT_STRIDE)
    ap.add_argument('--split', choices=('all', 'val', 'test'), default='all')
    ap.add_argument('--only', choices=('forecast', 'sar'))
    ap.add_argument('-o', '--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    import geofence
    segments = load_archive_segments(args.archive, args.start, args.end)
    times = segment_times(segments)
    t_min, t_max = min(times), max(times)
    bounds = split_bounds(times)
    del times
    port_lookup = geofence.is_in_port_cached
    iso = lambda e: datetime.fromtimestamp(e, timezone.utc).isoformat()

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'data_range': {'start': iso(t_min), 'end': iso(t_max)},
        'segments': len(segments),
        'params': {'stride': args.stride, 'split': args.split,
                   'horizons_h': list(HORIZONS_H), 'cable_zone_km': CABLE_ZONE_KM,
                   'moving_kn': MOVING_KN},
    }
    md = []
    if args.only in (None, 'forecast'):
        print('⏳ forecast 回測…')
        errs = run_forecast(segments, port_lookup, make_cable_lookup(),
                            stride=args.stride, split=args.split, bounds=bounds)
        report['forecast'] = aggregate_forecast(errs, BASELINE_PREDICTORS)
        for stratum in ('all', 'group:commercial', 'group:fishing', 'group:gov',
                        'regime:moving', 'cable_zone', 'cable_zone:commercial'):
            md.append(forecast_table(report['forecast'], stratum))
    if args.only in (None, 'sar'):
        print('⏳ sar 回測…')
        rows, missing = run_sar(segments, port_lookup, stride=args.stride,
                                split=args.split, bounds=bounds)
        report['sar'] = aggregate_sar(rows, missing)
        md += [sar_table(report['sar'], 'interp'), sar_table(report['sar'], 'dr')]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, report)
    print('\n\n'.join(t for t in md if t))
    print(f'\n✅ {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
================================================================================
航跡預測回測 — Trajectory prediction backtest (P0 baselines vs P1)
================================================================================

離線工具（不進 CI）。讀 `ais-archive` 歸檔（build_trajectory_dataset 同一套
載入與切段），跑兩種回測，產出之後每個模型都要贏過的基準數字。

1. forecast — 海纜預警用的中長時預測
   在航段上每隔 --stride 個點「發布」一次預測，往後 1/2/6/12/24h 找最接近的
   實際點（容許 ±max(0.5h, 0.25×h)），預測到該點的**實際時刻**再算誤差。
   預測器：trajectory_core.BASELINE_PREDICTORS（hold / cv_reported / cv_track），
   有 data/trajectory_params.json 時另加 p1（依群組的速度衰減推算）。
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
   p0（現行）與 p1（trajectory_model）在同一批抽點上並排比：admitted（推得出）、
   matchable（誤差 ≤ 門檻，分母含推不出的）、mean gate（門檻越大越容易誤認領）。

用法：
  python3 src/eval_trajectory.py --archive /path/to/ais-archive \\
      [--start 2026-08-01] [--end 2026-10-19] [--stride 6] [--split all|val|test]

輸出：data/trajectory/baseline_eval.json（gitignored）＋ 終端機 Markdown 表
================================================================================
"""

import argparse
import math
import random
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
                    if isinstance(fn, dict):  # 依船種群組分參數的預測器（P1）
                        fn = fn.get(seg['group']) or fn['all']
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


def p0_estimator(window, t, group):
    """現行 match_sar_ais.estimate_position（固定係數不確定度、MAX_DT_HOURS）。"""
    import match_sar_ais as m
    return m.estimate_position(window, t)


def p1_estimator(params):
    """trajectory_model.estimate_position（擬合的 τ + 經驗 r90 表）。"""
    import trajectory_model as tm
    return lambda window, t, group: tm.estimate_position(window, t, group, params)


def run_sar(segments, port_lookup, stride=DEFAULT_STRIDE, split='all', bounds=None,
            estimators=None):
    """estimators: {name: fn(window, t_epoch, group) -> est | None}，預設只有 p0。

    回傳 rows[(est, mode, group, method, dt_bin)] = {'err', 'unc_ok', 'gate_ok', 'gate_sum'}
    與 totals[(est, mode, group)] = 抽點總數（含推不出位置的）。"""
    import match_sar_ais as m
    estimators = estimators or {'p0': p0_estimator}
    rows = defaultdict(lambda: {'err': array('f'), 'unc_ok': 0, 'gate_ok': 0,
                                'gate_sum': 0.0})
    totals = defaultdict(int)
    for seg in segments:
        pts = seg['points']
        mmsi = seg['track']['mmsi']
        group = seg['group']
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
                for name, fn in estimators.items():
                    totals[(name, mode, group)] += 1
                    est = fn(window, p[0], group)
                    if est is None:
                        continue
                    e = prediction_error_km((est['lat'], est['lon']), (p[1], p[2]))
                    g = m.gate_km(est['unc_extra_km'])
                    r = rows[(name, mode, group, est['method'], dt_bin(est['dt_h']))]
                    r['err'].append(e)
                    r['gate_sum'] += g
                    if e <= est['unc_extra_km'] + m.AIS_POS_ERROR_KM:
                        r['unc_ok'] += 1
                    if e <= g:
                        r['gate_ok'] += 1
    return rows, totals


def aggregate_sar(rows, totals):
    """→ {'summary': {est: {mode: {group: {...}}}}, est: {mode: {group|method|dt: stats}}}

    summary 的三個數字是比對成效的取捨：
      admitted   推得出位置的比例
      matchable  誤差落在門檻內的比例（分母含推不出的）—— 越高，真船越少被誤判成暗船
      mean_gate_km  平均門檻半徑 —— 越大，暗船越容易被附近的 AIS 船「認領」"""
    out = defaultdict(lambda: defaultdict(dict))
    summ = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {'admitted': 0, 'gate_ok': 0, 'gate_sum': 0.0})))
    for (name, mode, group, method, dbin), r in sorted(rows.items()):
        stats = summarize_errors(list(r['err']))
        n = stats['n']
        stats['within_unc'] = round(r['unc_ok'] / n, 3) if n else None
        stats['within_gate'] = round(r['gate_ok'] / n, 3) if n else None
        out[name][mode][f'{group}|{method}|{dbin}'] = stats
        for g in (group, 'all'):
            acc = summ[name][mode][g]
            acc['admitted'] += n
            acc['gate_ok'] += r['gate_ok']
            acc['gate_sum'] += r['gate_sum']
    result = {'summary': {}}
    for (name, mode, group), total in totals.items():
        for g in (group, 'all'):
            summ[name][mode][g].setdefault('total', 0)
            summ[name][mode][g]['total'] += total
    for name, by_mode in summ.items():
        result['summary'][name] = {}
        for mode, by_group in by_mode.items():
            result['summary'][name][mode] = {}
            for g, acc in sorted(by_group.items()):
                tot = acc.get('total', 0)
                adm = acc['admitted']
                result['summary'][name][mode][g] = {
                    'total': tot,
                    'admitted': round(adm / tot, 3) if tot else None,
                    'matchable': round(acc['gate_ok'] / tot, 3) if tot else None,
                    'mean_gate_km': round(acc['gate_sum'] / adm, 2) if adm else None,
                }
    for name, by_mode in out.items():
        result[name] = {mode: dict(v) for mode, v in by_mode.items()}
    return result


# =============================================================================
# 3. sar-sim — 完整比對模擬（候選篩選 + 一對一指派）
# =============================================================================

SIM_WINDOW_H = 8.0
SIM_MIN_DETECTIONS = 200


def _sim_window(pts, t, skip=None, window_h=SIM_WINDOW_H):
    times = [p[0] for p in pts]
    lo = bisect_left(times, t - window_h * 3600)
    hi = bisect_left(times, t + window_h * 3600 + 1)
    return [p for k, p in enumerate(pts[lo:hi], lo) if k != skip]


def run_sar_sim(segments, port_lookup, estimators, split='all', bounds=None,
                n_times=40, dark_frac=0.2, noise_km=0.5, seed=7):
    """拿真實快照模擬一次 SAR 比對，量「真船配對對了沒」與「暗船有沒有被認領」。

    每個抽中的快照時刻 T：
      - 偵測 = 該時刻所有在海上的船的真實位置 + 高斯雜訊（SAR 定位/網格誤差）
      - 每艘船以 dark_frac 機率設為暗船：刪掉牠全部 AIS，偵測仍在
      - 其餘船用 estimator 從「抽掉 T 那一點」的航跡推到 T，當候選
      - 走 match_sar_ais.candidates_for_detection + solve_assignment
    回傳 {estimator: {group: counts}}。"""
    import match_sar_ais as m
    import trajectory_model as tm
    rng = random.Random(seed)
    by_t = defaultdict(list)
    for si, seg in enumerate(segments):
        for pi, p in enumerate(seg['points']):
            by_t[p[0]].append((si, pi))
    times = sorted(t for t, lst in by_t.items() if len(lst) >= SIM_MIN_DETECTIONS
                   and (split == 'all' or split_label(t, bounds) == split))
    chosen = rng.sample(times, min(n_times, len(times)))
    spans = [(seg['points'][0][0], seg['points'][-1][0]) for seg in segments]
    win_s = SIM_WINDOW_H * 3600
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for T in chosen:
        detected = dict(by_t[T])
        dets, inputs = [], {}
        for si, seg in enumerate(segments):
            if spans[si][0] > T + win_s or spans[si][1] < T - win_s:
                continue
            tr = seg['track']
            key = f"{tr['mmsi']}|{tr['name']}"
            dark = rng.random() < dark_frac
            pi = detected.get(si)
            if pi is not None:
                p = seg['points'][pi]
                if not port_lookup(p[1], p[2]):
                    ang = rng.random() * 2 * math.pi
                    r = abs(rng.gauss(0.0, noise_km))
                    lat, lon = tm.offset(p[1], p[2], r * math.sin(ang), r * math.cos(ang))
                    dets.append({'lat': lat, 'lon': lon, 'key': key,
                                 'group': seg['group'], 'dark': dark})
            if not dark:
                inputs[(key, si)] = (_sim_window(seg['points'], T, skip=pi), seg['group'])

        for name, fn in estimators.items():
            est, grid = {}, defaultdict(list)
            for vid, (window, group) in inputs.items():
                if not window:
                    continue
                e = fn(window, T, group)
                if e is None:
                    continue
                est[vid] = e
                grid[m._cand_cell(e['lat'], e['lon'])].append(vid)
            cand = {i: m.candidates_for_detection(d, est, grid) for i, d in enumerate(dets)}
            assigned = m.solve_assignment(cand)
            for i, d in enumerate(dets):
                hit = assigned.get(i)
                for g in (d['group'], 'all'):
                    c = counts[name][g]
                    if d['dark']:
                        c['dark'] += 1
                        c['dark_claimed'] += 1 if hit else 0
                    else:
                        c['visible'] += 1
                        if not hit:
                            c['false_dark'] += 1
                        elif hit[1][0] == d['key']:
                            c['correct'] += 1
                        else:
                            c['wrong'] += 1
    out = {}
    for name, by_group in counts.items():
        out[name] = {}
        for g, c in sorted(by_group.items()):
            vis, dark = c['visible'], c['dark']
            out[name][g] = {
                'visible': vis, 'dark': dark,
                'correct': round(c['correct'] / vis, 3) if vis else None,
                'wrong': round(c['wrong'] / vis, 3) if vis else None,
                'false_dark': round(c['false_dark'] / vis, 3) if vis else None,
                'dark_claimed': round(c['dark_claimed'] / dark, 3) if dark else None,
            }
    return {'times': len(chosen), 'dark_frac': dark_frac, 'noise_km': noise_km,
            'result': out}


def sar_sim_table(sim):
    lines = [f"### sar-sim — {sim['times']} 個快照、暗船比例 {sim['dark_frac']:.0%}、"
             f"SAR 雜訊 σ={sim['noise_km']}km", '',
             '| estimator | group | 真船 n | 配對正確 | 配錯船 | 誤判暗船 | 暗船 n | 暗船被認領 |',
             '|---|---|---|---|---|---|---|---|']
    pct = lambda v: '—' if v is None else f'{v:.1%}'
    for name, by_group in sim['result'].items():
        for g, s in by_group.items():
            lines.append(f"| {name} | {g} | {s['visible']} | {pct(s['correct'])} | "
                         f"{pct(s['wrong'])} | {pct(s['false_dark'])} | {s['dark']} | "
                         f"{pct(s['dark_claimed'])} |")
    return '\n'.join(lines)


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


def sar_summary_table(result):
    summ = result.get('summary') or {}
    if not summ:
        return ''
    lines = ['### sar — 比對成效（matchable = 誤差 ≤ 門檻 / 全部抽點）', '',
             '| estimator | mode | group | n | admitted | matchable | mean gate km |',
             '|---|---|---|---|---|---|---|']
    for name, by_mode in summ.items():
        for mode, by_group in by_mode.items():
            for g, s in by_group.items():
                lines.append(f"| {name} | {mode} | {g} | {s['total']} | "
                             f"{s['admitted']:.0%} | {s['matchable']:.0%} | "
                             f"{s['mean_gate_km'] if s['mean_gate_km'] is not None else '—'} |")
    return '\n'.join(lines)


def sar_table(result, name, mode):
    block = (result.get(name) or {}).get(mode) or {}
    if not block:
        return ''
    lines = [f'### sar — {name} / {mode}', '',
             '| group | method | dt | n | median km | p90 km | within_unc | within_gate |',
             '|---|---|---|---|---|---|---|---|']
    for key, s in block.items():
        group, method, dbin = key.split('|')
        lines.append(f"| {group} | {method} | {dbin} | {s['n']} | "
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
    ap.add_argument('--only', choices=('forecast', 'sar', 'sim'))
    ap.add_argument('--sim-times', type=int, default=40)
    ap.add_argument('--params', type=Path, default=None,
                    help='trajectory_params.json（預設 data/trajectory_params.json；缺檔只跑 P0）')
    ap.add_argument('--detail', action='store_true', help='印出逐 method/dt 分箱的 SAR 明細表')
    ap.add_argument('-o', '--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    import trajectory_model as tm
    params = tm.load_params(args.params or tm.PARAMS_PATH)
    predictors = dict(BASELINE_PREDICTORS)
    estimators = {'p0': p0_estimator}
    if params:
        predictors['p1'] = {g: tm.make_group_predictor(gp)
                            for g, gp in params['groups'].items()}
        estimators['p1'] = p1_estimator(params)
    else:
        print('⚠️ 沒有可用的 trajectory_params.json，只跑 P0 基準')

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
                            stride=args.stride, split=args.split, bounds=bounds,
                            predictors=predictors)
        report['forecast'] = aggregate_forecast(errs, predictors)
        for stratum in ('all', 'group:commercial', 'group:fishing', 'group:gov',
                        'regime:moving', 'cable_zone', 'cable_zone:commercial'):
            md.append(forecast_table(report['forecast'], stratum))
    if args.only in (None, 'sar'):
        print('⏳ sar 回測…')
        rows, totals = run_sar(segments, port_lookup, stride=args.stride,
                               split=args.split, bounds=bounds, estimators=estimators)
        report['sar'] = aggregate_sar(rows, totals)
        md.append(sar_summary_table(report['sar']))
        if args.detail:
            md += [sar_table(report['sar'], name, mode)
                   for name in estimators for mode in ('interp', 'dr')]

    if args.only in (None, 'sim'):
        print('⏳ sar-sim 比對模擬…')
        report['sar_sim'] = run_sar_sim(segments, port_lookup, estimators,
                                        split=args.split, bounds=bounds,
                                        n_times=args.sim_times)
        md.append(sar_sim_table(report['sar_sim']))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, report)
    print('\n\n'.join(t for t in md if t))
    print(f'\n✅ {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

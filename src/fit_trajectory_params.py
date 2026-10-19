#!/usr/bin/env python3
"""
================================================================================
擬合 P1 運動模型參數 — Fit trajectory_model params on the train split
================================================================================

離線工具（不進 CI）。輸入 ais-archive，輸出 data/trajectory_params.json（提交，
match_sar_ais 在 CI 讀它）。只用 train 切分、且排除 holdout MMSI ——
val/test 留給 eval_trajectory.py 驗收。

1. 位置：每個 (群組, 動/靜) 在 τ × 速度來源的網格上找誤差最小者。
   目標函數 = 各時距分箱（≤ OBJECTIVE_MAX_DT_H）的「中位誤差」取平均：
   分箱等權，不被樣本最多的那段時距主導；只看 SAR 比對實際會用到的時距
   （海纜預警要的 6–24h 長時距是第 2 階段另一組參數）。
2. 不確定度：用選定的模型重算誤差，依 (航速分箱, 時距節點) 取 0.9 分位
   → dr_r90；內插（抽掉中間點、兩側內插回來）同法 → interp_r90。
   樣本 < MIN_CELL_SAMPLES 的格子退回全群組合併表，再沿時距強制單調不減；
   開頭仍空的格子以下一個節點的值回填（保守）。

用法：
  python3 src/fit_trajectory_params.py --archive /path/to/ais-archive \\
      [--start] [--end] [-o data/trajectory_params.json]
================================================================================
"""

import argparse
import random
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone

from build_trajectory_dataset import (
    cut_at_bounds, is_holdout_mmsi, load_archive_segments, segment_times,
    split_bounds)
from io_utils import atomic_write_json
from trajectory_core import prediction_error_km
import trajectory_model as tm

TAU_GRID_H = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, None)
SOURCES = ('reported', 'track')
MAX_SAMPLES_PER_CELL = 40000     # 每個 (群組, 動/靜) 擬合平均值時的抽樣上限
MIN_CELL_SAMPLES = 30
OBJECTIVE_MAX_DT_H = 4.0          # 位置模型只對 SAR 會用到的時距最佳化
R90_QUANTILE = 0.9
SEED = 20261019


def _iso(e):
    return datetime.fromtimestamp(e, timezone.utc).isoformat()


def train_segments(segments, bounds):
    """只留 train 段、排除 holdout MMSI。"""
    out = []
    for s in segments:
        if is_holdout_mmsi(s['track']['mmsi']):
            continue
        for label, sub in cut_at_bounds(s['points'], bounds):
            if label == 'train':
                out.append({'track': s['track'], 'group': s['group'], 'points': sub})
    return out


def dt_knot(dt_h, edges=tm.DT_EDGES_H):
    """dt 落在哪個時距節點（第一個 ≥ dt 的節點）；超出範圍回傳 None。"""
    i = bisect_left(edges, dt_h)
    return i if i < len(edges) else None


def collect_dr_samples(segments, port_lookup, max_dt_h=tm.DT_EDGES_H[-1]):
    """單側推算樣本：(group, anchor, neighbor_prev, truth, dt_s)。港內錨點不收。"""
    out = []
    for seg in segments:
        pts = seg['points']
        for i in range(len(pts) - 1):
            a = pts[i]
            if port_lookup(a[1], a[2]):
                continue
            nb = pts[i - 1] if i > 0 else None
            for j in range(i + 1, len(pts)):
                dt = pts[j][0] - a[0]
                if dt > max_dt_h * 3600:
                    break
                out.append((seg['group'], a, nb, pts[j], dt))
    return out


def collect_interp_samples(segments, port_lookup):
    """內插樣本：抽掉 k，用 (k-1, k+1) 以及 (k-1, k+2) 兩種括號內插回 k。"""
    out = []
    for seg in segments:
        pts = seg['points']
        for k in range(1, len(pts) - 1):
            p = pts[k]
            if port_lookup(p[1], p[2]):
                continue
            for a, b in ((pts[k - 1], pts[k + 1]),
                         (pts[k - 1], pts[k + 2]) if k + 2 < len(pts) else (None, None)):
                if a is None:
                    continue
                gap_h = (b[0] - a[0]) / 3600.0
                if gap_h <= 0 or gap_h > tm.INTERP_MAX_GAP_H:
                    continue
                out.append((seg['group'], a, b, p))
    return out


def dr_error(sample, tau_h, source):
    _, a, nb, truth, dt = sample
    return prediction_error_km(tm.extrapolate(a, nb, dt, tau_h, source),
                               (truth[1], truth[2]))


def balanced_median_error(samples, tau_h, source):
    by_knot = defaultdict(list)
    for s in samples:
        dt_h = s[4] / 3600.0
        k = dt_knot(dt_h)
        if k is not None and dt_h <= OBJECTIVE_MAX_DT_H:
            by_knot[k].append(dr_error(s, tau_h, source))
    meds = []
    for errs in by_knot.values():
        errs.sort()
        meds.append(errs[len(errs) // 2])
    return sum(meds) / len(meds) if meds else float('inf')


def fit_dr_means(dr_samples, rng):
    """{group: {regime: {tau_h, source, objective_km, n}}}，含 'all'。"""
    cells = defaultdict(list)
    for s in dr_samples:
        regime = tm.regime_of(s[1][3])
        cells[(s[0], regime)].append(s)
        cells[('all', regime)].append(s)
    out = defaultdict(dict)
    for (group, regime), samples in sorted(cells.items()):
        if len(samples) > MAX_SAMPLES_PER_CELL:
            samples = rng.sample(samples, MAX_SAMPLES_PER_CELL)
        best = None
        for source in SOURCES:
            for tau in TAU_GRID_H:
                obj = balanced_median_error(samples, tau, source)
                if best is None or obj < best[0]:
                    best = (obj, tau, source)
        out[group][regime] = {'tau_h': best[1], 'source': best[2],
                              'objective_km': round(best[0], 3), 'n': len(samples)}
        tau_s = '∞' if best[1] is None else f'{best[1]:g}h'
        print(f'  {group:<10} {regime:<10} τ={tau_s:<5} {best[2]:<8} '
              f'目標 {best[0]:.2f}km（n={len(samples)}）')
    return dict(out)


def quantile(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def build_r90_table(errs_by_cell, pooled_by_cell):
    """errs_by_cell[(knot, speed_bin)] → r90 表（含合併表退回與單調化）。"""
    n_speed = len(tm.SPEED_EDGES_KN) + 1
    rows = []
    for k in range(len(tm.DT_EDGES_H)):
        row = []
        for sb in range(n_speed):
            errs = errs_by_cell.get((k, sb), ())
            if len(errs) < MIN_CELL_SAMPLES:
                errs = pooled_by_cell.get((k, sb), ())
            row.append(round(quantile(errs, R90_QUANTILE), 3)
                       if len(errs) >= MIN_CELL_SAMPLES else None)
        rows.append(row)
    for sb in range(n_speed):  # 沿時距單調不減
        run = None
        for k in range(len(rows)):
            v = rows[k][sb]
            if v is None:
                continue
            run = v if run is None else max(run, v)
            rows[k][sb] = run
        nxt = None               # 前段樣本不足（快照間隔 1–3h，dt<0.25h 很少）→ 用下一節點的值，寧可保守
        for k in range(len(rows) - 1, -1, -1):
            if rows[k][sb] is None:
                rows[k][sb] = nxt
            else:
                nxt = rows[k][sb]
    return {'dt_edges_h': list(tm.DT_EDGES_H),
            'speed_edges_kn': list(tm.SPEED_EDGES_KN), 'r90_km': rows}


def fit_r90_tables(dr_samples, interp_samples, dr_means):
    """回傳 {group: {'dr_r90': table, 'interp_r90': table}}，含 'all'。"""
    dr_cells = defaultdict(lambda: defaultdict(list))
    for s in dr_samples:
        group, a, _, _, dt = s
        k = dt_knot(dt / 3600.0)
        if k is None:
            continue
        mp = dr_means.get(group, dr_means['all'])[tm.regime_of(a[3])]
        e = dr_error(s, mp['tau_h'], mp['source'])
        key = (k, tm.speed_bin(a[3]))
        dr_cells[group][key].append(e)
        dr_cells['all'][key].append(e)

    in_cells = defaultdict(lambda: defaultdict(list))
    for group, a, b, p in interp_samples:
        dt_near = min(p[0] - a[0], b[0] - p[0]) / 3600.0
        k = dt_knot(dt_near)
        if k is None:
            continue
        w = (p[0] - a[0]) / (b[0] - a[0])
        est = (a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]))
        e = prediction_error_km(est, (p[1], p[2]))
        key = (k, tm.speed_bin(max(a[3] or 0.0, b[3] or 0.0)))
        in_cells[group][key].append(e)
        in_cells['all'][key].append(e)

    out = {}
    for group in set(dr_cells) | set(in_cells):
        out[group] = {
            'dr_r90': build_r90_table(dr_cells[group], dr_cells['all']),
            'interp_r90': build_r90_table(in_cells[group], in_cells['all']),
        }
    return out


def fit(segments, bounds, port_lookup, seed=SEED):
    rng = random.Random(seed)
    train = train_segments(segments, bounds)
    print(f'🧪 train 航段 {len(train)} 段')
    dr_samples = collect_dr_samples(train, port_lookup)
    interp_samples = collect_interp_samples(train, port_lookup)
    print(f'   單側推算樣本 {len(dr_samples)}、內插樣本 {len(interp_samples)}')
    print('⏳ 擬合位置模型（τ × 速度來源）…')
    dr_means = fit_dr_means(dr_samples, rng)
    print('⏳ 統計 r90 表…')
    tables = fit_r90_tables(dr_samples, interp_samples, dr_means)
    groups = {}
    for group in sorted(set(dr_means) | set(tables)):
        groups[group] = {'dr': dr_means.get(group) or dr_means['all'],
                         **tables.get(group, tables['all'])}
    return {
        'version': tm.PARAMS_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'train_end': _iso(bounds[0]),
        'samples': {'dr': len(dr_samples), 'interp': len(interp_samples)},
        'admit_max_r90_km': tm.ADMIT_MAX_R90_KM,
        'interp_max_gap_h': tm.INTERP_MAX_GAP_H,
        'r90_quantile': R90_QUANTILE,
        'groups': groups,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description='擬合 P1 運動模型參數')
    ap.add_argument('--archive', required=True)
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('-o', '--out', default=str(tm.PARAMS_PATH))
    args = ap.parse_args(argv)

    import geofence
    segments = load_archive_segments(args.archive, args.start, args.end)
    times = segment_times(segments)
    bounds = split_bounds(times)
    params = fit(segments, bounds, geofence.is_in_port_cached)
    params['data_range'] = {'start': _iso(min(times)), 'end': _iso(max(times))}
    atomic_write_json(args.out, params)
    print(f'✅ {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

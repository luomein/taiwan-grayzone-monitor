#!/usr/bin/env python3
"""
================================================================================
擬合海纜接近預警模型 — Fit the cable-approach warning table
================================================================================

離線工具（不進 CI）。輸入 ais-archive，輸出 data/cable_warning_model.json（提交）。
預測、表鍵、事件定義全部 import 自 cable_warning.py —— 回測量的就是 CI 實際跑的。

樣本 = 每個「在海上、海纜 5km 外、不在港內」的 AIS 點；標籤 = 同一航段內
H 小時內是否出現低速入區事件（cable_warning.is_entry_event）。航段在 >6h 中斷
處切開，所以斷訊後才進區的不算（保守，會低估正例）。

切分沿用 build_trajectory_dataset：train 擬表、val 選警示門檻 alert_p、test 報數。
alert_p = 非漁船在 val 上「精確度 ≥ TARGET_PRECISION」時召回最高的門檻。

用法：
  python3 src/fit_cable_warning.py --archive /path/to/ais-archive [-o data/cable_warning_model.json]
================================================================================
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone

import cable_warning as cw
from build_trajectory_dataset import (
    is_holdout_mmsi, load_archive_segments, segment_times, split_bounds, split_label)
from io_utils import atomic_write_json

MIN_CELL_N = 5              # 格層級（gc/gcs/gcsh）樣本少於此就不存，退回上層
TARGET_PRECISION = 0.20     # 非漁船警示的精確度目標（基準率 ~1%）
THRESHOLD_GRID = tuple(round(0.02 * i, 2) for i in range(1, 41))   # 0.02 … 0.80


def _iso(e):
    return datetime.fromtimestamp(e, timezone.utc).isoformat()


def build_samples(segments, bounds, cable_dist, in_port, horizons=cw.HORIZONS_H,
                  excluded=None):
    """→ [(split, group, lat, lon, cog, sog, d_km, labels_tuple)]。

    excluded(mmsi, name)：與 CI 預警同一套排除規則（漁網信標等不進表）。"""
    out = []
    for seg in segments:
        pts = seg['points']
        group = seg['group']
        mmsi = seg['track']['mmsi']
        if excluded is not None and excluded(mmsi, seg['track'].get('name')):
            continue
        dist = [cable_dist(p[1], p[2]) for p in pts]
        port = [bool(in_port(p[1], p[2])) for p in pts]
        event = [cw.is_entry_event(dist[j], pts[j][3], port[j]) for j in range(len(pts))]
        for i, p in enumerate(pts):
            if port[i] or dist[i] <= cw.ZONE_KM:
                continue
            first = None
            for j in range(i + 1, len(pts)):
                if pts[j][0] - p[0] > max(horizons) * 3600:
                    break
                if event[j]:
                    first = (pts[j][0] - p[0]) / 3600.0
                    break
            labels = tuple(int(first is not None and first <= h) for h in horizons)
            label = split_label(p[0], bounds)
            split = 'test' if (label == 'test' or is_holdout_mmsi(mmsi)) else label
            out.append((split, group, p[1], p[2], p[4], p[3], dist[i], labels))
    return out


def fit_tables(samples, horizons=cw.HORIZONS_H):
    counts = {'g': defaultdict(lambda: [0] * (1 + len(horizons)))}
    for level in cw.LEVELS:
        counts[level] = defaultdict(lambda: [0] * (1 + len(horizons)))
    for split, group, lat, lon, cog, sog, d, labels in samples:
        if split != 'train':
            continue
        rows = [counts['g'][group], counts['g']['all']]
        rows += [counts[lv][k] for lv, k in
                 zip(cw.LEVELS, cw.table_keys(group, lat, lon, cog, sog, d))]
        for row in rows:
            row[0] += 1
            for i, lab in enumerate(labels):
                row[1 + i] += lab
    tables = {'g': dict(counts['g']), 'gds': dict(counts['gds'])}
    for level in ('gc', 'gcs', 'gcsh'):
        tables[level] = {k: v for k, v in counts[level].items() if v[0] >= MIN_CELL_N}
    return tables


def ranking_metrics(scored):
    """scored: [(score, label)] → {n, pos, auc, ap, top1/top5 precision+recall}。"""
    scored = sorted(scored, key=lambda x: -x[0])
    n, pos = len(scored), sum(lab for _, lab in scored)
    if not pos or pos == n:
        return {'n': n, 'pos': pos}
    tp = auc_num = 0
    ap = 0.0
    for idx, (_, lab) in enumerate(scored, 1):
        if lab:
            tp += 1
            ap += tp / idx
        else:
            auc_num += tp
    out = {'n': n, 'pos': pos, 'base_rate': round(pos / n, 4),
           'auc': round(auc_num / (pos * (n - pos)), 3), 'ap': round(ap / pos, 3)}
    for frac in (0.01, 0.05):
        k = max(1, int(n * frac))
        hit = sum(lab for _, lab in scored[:k])
        out[f'top{int(frac * 100)}pct'] = {'precision': round(hit / k, 3),
                                           'recall': round(hit / pos, 3)}
    return out


def threshold_metrics(scored, thr):
    flagged = [lab for s, lab in scored if s >= thr]
    pos = sum(lab for _, lab in scored)
    tp = sum(flagged)
    return {'threshold': thr, 'flagged': len(flagged),
            'precision': round(tp / len(flagged), 3) if flagged else None,
            'recall': round(tp / pos, 3) if pos else None}


def choose_alert_p(scored, target=TARGET_PRECISION, grid=THRESHOLD_GRID):
    """精確度 ≥ target 的門檻中召回最高者（= 最低門檻）；都達不到就取精確度最高者。"""
    best = None
    for thr in grid:
        m = threshold_metrics(scored, thr)
        if m['precision'] is None:
            continue
        if m['precision'] >= target:
            return thr
        if best is None or m['precision'] > best[0]:
            best = (m['precision'], thr)
    return best[1] if best else grid[-1]


def evaluate(samples, model, split, horizon_idx):
    """回傳 {群組: (scored list)}，含 'non_fishing' 與 'all'。"""
    h = model['horizons_h'][horizon_idx]
    out = defaultdict(list)
    for sp, group, lat, lon, cog, sog, d, labels in samples:
        if sp != split:
            continue
        p = cw.predict(model, group, lat, lon, cog, sog, d)[h]
        item = (p, labels[horizon_idx])
        out[group].append(item)
        out['all'].append(item)
        if group != 'fishing':
            out['non_fishing'].append(item)
    return out


def fit(segments, bounds, cable_dist, in_port, excluded=None):
    samples = build_samples(segments, bounds, cable_dist, in_port, excluded=excluded)
    print(f'🧪 樣本 {len(samples)}（train {sum(s[0] == "train" for s in samples)}）')
    model = {
        'version': cw.MODEL_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'train_end': _iso(bounds[0]),
        'horizons_h': list(cw.HORIZONS_H),
        'zone_km': cw.ZONE_KM, 'slow_kn': cw.SLOW_KN,
        'shrink_m': cw.SHRINK_M,
        'tables': fit_tables(samples),
    }
    i12 = model['horizons_h'].index(12)
    val = evaluate(samples, model, 'val', i12)
    model['alert_p'] = choose_alert_p(val['non_fishing'])

    metrics = {}
    for hi, h in enumerate(model['horizons_h']):
        test = evaluate(samples, model, 'test', hi)
        metrics[f'{h}h'] = {g: ranking_metrics(s) for g, s in sorted(test.items())}
    test12 = evaluate(samples, model, 'test', i12)
    metrics['alert_12h_test'] = {g: threshold_metrics(test12[g], model['alert_p'])
                                 for g in ('non_fishing', 'commercial', 'fishing')}
    model['metrics'] = metrics
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description='擬合海纜接近預警表')
    ap.add_argument('--archive', required=True)
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('-o', '--out', default=str(cw.MODEL_PATH))
    args = ap.parse_args(argv)

    import geofence
    segments = load_archive_segments(args.archive, args.start, args.end)
    times = segment_times(segments)
    bounds = split_bounds(times)
    model = fit(segments, bounds, cw.make_cable_dist(), geofence.is_in_port_cached,
                excluded=cw.make_excluder())
    model['data_range'] = {'start': _iso(min(times)), 'end': _iso(max(times))}
    atomic_write_json(args.out, model, compact=True)

    sizes = {k: len(v) for k, v in model['tables'].items()}
    print(f'✅ {args.out}  表格筆數 {sizes}  alert_p={model["alert_p"]}')
    for h in model['horizons_h']:
        for g in ('non_fishing', 'commercial', 'fishing'):
            m = model['metrics'][f'{h}h'].get(g, {})
            if 'auc' in m:
                print(f"  {h:>2}h {g:<12} base {m['base_rate']:.2%} AUC {m['auc']} AP {m['ap']} "
                      f"top1% P {m['top1pct']['precision']:.1%} R {m['top1pct']['recall']:.1%}")
    for g, m in model['metrics']['alert_12h_test'].items():
        print(f'  警示（p12 ≥ {model["alert_p"]}）test {g}: {m}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

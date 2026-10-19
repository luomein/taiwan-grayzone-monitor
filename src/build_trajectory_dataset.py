#!/usr/bin/env python3
"""
================================================================================
航跡預測資料集 — Build the trajectory dataset from the ais-archive branch
================================================================================

離線工具（不進 CI）：把 `ais-archive` 分支的永久 AIS 歸檔重組成逐船航段，
供 eval_trajectory.py 回測與之後的模型訓練使用。

歸檔佈局（兩種並存，見 update-ais.yml）：
  archive/ais_{track,commercial}_<YYYY-MM>.jsonl                 2026-07/08 舊月度檔
  archive/<YYYY-MM>/<DD>/ais_{track,commercial}_<ts>.jsonl.gz     之後每輪一檔
每行一個 track entry：{"timestamp", "vessels": [{mmsi, name, lat, lon, speed,
heading, type_name, anc?, gov?, rt?}]}。

清理（會污染訓練/回測的東西先剔掉）：
  - 非船舶 AIS（浮標/漁網信標/AtoN）：沿用 match_sar_ais.is_nonvessel_ais
  - **共用 MMSI**：以 (mmsi, 船名) 為一條航跡 —— CN 漁船常多船共用一個 MMSI，
    交錯回報會變成來回瞬移（analyze_suspicious 的 mmsi_collision 同一現象）
  - 航段切分：訊號中斷 > MAX_GAP_H，或隱含速度 > MAX_IMPLIED_KMH（瞬移/spoofing
    —— 單一離群點會被切成長度 1 的航段，再被 MIN_SEG_POINTS 丟掉）
  - 定位時刻：有 `rt` 時用 entry 時刻 − rt（trajectory_core.point_epoch）

切分：依時間切 train/val/test（航段在邊界上切開，不跨集），另以 MMSI 雜湊
保留 1/10 的船完全不進訓練（`holdout_mmsi`）—— 隨機切點會讓同一段航程的
前後鄰點分處訓練與測試，嚴重高估準確度。

用法：
  git clone --depth 1 --branch ais-archive <repo> /path/to/ais-archive
  python3 src/build_trajectory_dataset.py --archive /path/to/ais-archive \\
      [--start 2026-08-01] [--end 2026-10-19] [-o data/trajectory]

輸出（data/trajectory/，gitignored）：
  points.parquet   逐點一列（有 pyarrow 時）；否則 points.csv.gz
  meta.json        航段/點數統計、時間範圍、切分邊界
================================================================================
"""

import argparse
import csv
import gzip
import json
import re
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from geo_utils import haversine_km
from io_utils import atomic_write_json
from match_sar_ais import is_nonvessel_ais, parse_ts
from trajectory_core import point_epoch

TIERS = ('track', 'commercial')
MAX_GAP_H = 6.0             # 訊號中斷超過即切段（快照間隔 1–2h，>6h 等於失聯）
MAX_IMPLIED_KMH = 100.0     # 與 analyze_suspicious 的 teleport 門檻一致
MIN_SEG_POINTS = 3
SPLIT_FRACTIONS = (0.70, 0.15)   # train / val；其餘為 test
HOLDOUT_MMSI_MOD = 10            # crc32(mmsi) % 10 == 0 → 保留船

GOV_TYPES = {'coastguard', 'msa', 'rescue', 'research'}
COMMERCIAL_TYPES = {'cargo', 'tanker', 'lng'}

_LEGACY_RE = re.compile(r'^ais_(track|commercial)_(\d{4}-\d{2})\.jsonl$')
_RUN_RE = re.compile(r'^ais_(track|commercial)_(\d{8})T\d{6}Z\.jsonl\.gz$')


# =============================================================================
# 歸檔讀取
# =============================================================================

def discover_archive_files(archive_dir, start=None, end=None, tiers=TIERS):
    """列出歸檔檔案（依檔名時序排序）。start/end 為 'YYYY-MM-DD'（含），
    以檔名日期粗篩；精確的時間過濾在 iter_archive_entries 做。"""
    root = Path(archive_dir)
    if (root / 'archive').is_dir():
        root = root / 'archive'
    files = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        m = _LEGACY_RE.match(path.name)
        if m:
            tier, month = m.group(1), m.group(2)
            if tier not in tiers:
                continue
            if start and month < start[:7]:
                continue
            if end and month > end[:7]:
                continue
            files.append((month + '-00', tier, path))
            continue
        m = _RUN_RE.match(path.name)
        if m:
            tier, ymd = m.group(1), m.group(2)
            day = f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}'
            if tier not in tiers:
                continue
            if (start and day < start) or (end and day > end):
                continue
            files.append((day + path.name, tier, path))
    files.sort(key=lambda f: (f[0], f[2].name))
    return [(tier, path) for _, tier, path in files]


def iter_archive_entries(files, start_epoch=None, end_epoch=None):
    """逐行讀歸檔，yield (tier, entry_epoch, entry)。壞行略過。"""
    for tier, path in files:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(entry, dict):
                    continue
                t = parse_ts(entry.get('timestamp'))
                if t is None:
                    continue
                if start_epoch is not None and t < start_epoch:
                    continue
                if end_epoch is not None and t >= end_epoch:
                    continue
                yield tier, t, entry


def load_tracks(entries):
    """track entries → {(mmsi, name): track}。

    track = {'mmsi', 'name', 'type_name', 'gov', 'tiers', 'points'}，
    points = [(t, lat, lon, sog, cog, anc)]（未排序）。
    以 (mmsi, 船名) 為鍵：共用 MMSI 的不同船分開成不同航跡。
    """
    tracks = {}
    for tier, t, entry in entries:
        for v in entry.get('vessels') or ():
            if not isinstance(v, dict):
                continue
            mmsi = v.get('mmsi')
            lat, lon = v.get('lat'), v.get('lon')
            if mmsi is None or lat is None or lon is None:
                continue
            mmsi = str(mmsi)
            name = (v.get('name') or '').strip()
            if is_nonvessel_ais(mmsi, name):
                continue
            key = (mmsi, name)
            tr = tracks.get(key)
            if tr is None:
                tr = {'mmsi': mmsi, 'name': name, 'type_name': 'unknown',
                      'gov': None, 'tiers': set(), 'points': []}
                tracks[key] = tr
            tn = v.get('type_name')
            if tn and tn != 'unknown':
                tr['type_name'] = tn
            if v.get('gov'):
                tr['gov'] = v['gov']
            tr['tiers'].add(tier)
            heading = v.get('heading')
            tr['points'].append((
                point_epoch(t, v), float(lat), float(lon),
                float(v.get('speed') or 0.0),
                float(heading) if isinstance(heading, (int, float)) else None,
                1 if v.get('anc') else 0,
            ))
    return tracks


# =============================================================================
# 航段切分
# =============================================================================

def segment_track(points, max_gap_h=MAX_GAP_H, max_kmh=MAX_IMPLIED_KMH,
                  min_points=MIN_SEG_POINTS):
    """排序 + 去重（同時刻 <1s）+ 依中斷/瞬移切段，丟掉太短的航段。"""
    pts = sorted(points, key=lambda p: p[0])
    segs, cur = [], []
    for p in pts:
        if cur:
            prev = cur[-1]
            dt = p[0] - prev[0]
            if dt < 1.0:
                continue
            split = dt > max_gap_h * 3600
            if not split:
                d = haversine_km(prev[1], prev[2], p[1], p[2])
                split = d / (dt / 3600.0) > max_kmh
            if split:
                segs.append(cur)
                cur = []
        cur.append(p)
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= min_points]


def type_group(track):
    """回測分層：gov / commercial / fishing / other。

    CN 漁船的 AIS 船種常報 other/unknown/cargo（見 CLAUDE.md 測線誤報），
    所以漁船以船名規則優先於 AIS 船種；公務船分類又優先於船名規則
    （`^XIANG` 會誤中向陽紅）。"""
    from fetch_ais_data import is_cn_fishing_vessel  # 延遲載入：拉 requests
    tn = track.get('type_name')
    if track.get('gov') or tn in GOV_TYPES:
        return 'gov'
    if tn == 'fishing' or is_cn_fishing_vessel(track.get('name')):
        return 'fishing'
    if tn in COMMERCIAL_TYPES:
        return 'commercial'
    return 'other'


def split_bounds(times, fractions=SPLIT_FRACTIONS):
    """依「點的時間分位數」算 train/val 與 val/test 的邊界 epoch。

    不用時間跨度比例：歸檔密度不均（2026-08 每天 ~12 輪，09 月還斷了十天），
    按跨度切會讓 train 吃掉 84% 的點。"""
    ts = sorted(times)
    n = len(ts)
    b1 = ts[min(n - 1, int(n * fractions[0]))]
    b2 = ts[min(n - 1, int(n * (fractions[0] + fractions[1])))]
    return b1, b2


def split_label(t, bounds):
    b1, b2 = bounds
    if t < b1:
        return 'train'
    if t < b2:
        return 'val'
    return 'test'


def cut_at_bounds(segment, bounds, min_points=MIN_SEG_POINTS):
    """把航段在切分邊界上切開，回傳 [(label, sub_segment)]（太短的丟掉）。"""
    out, cur, cur_label = [], [], None
    for p in segment:
        label = split_label(p[0], bounds)
        if cur and label != cur_label:
            out.append((cur_label, cur))
            cur = []
        cur_label = label
        cur.append(p)
    if cur:
        out.append((cur_label, cur))
    return [(lab, s) for lab, s in out if len(s) >= min_points]


def is_holdout_mmsi(mmsi):
    return zlib.crc32(str(mmsi).encode()) % HOLDOUT_MMSI_MOD == 0


def build_segments(tracks, **seg_kwargs):
    """全部航跡 → 航段清單 [{'track': tr, 'group', 'points'}]。"""
    out = []
    for tr in tracks.values():
        group = type_group(tr)
        for seg in segment_track(tr['points'], **seg_kwargs):
            out.append({'track': tr, 'group': group, 'points': seg})
    return out


# =============================================================================
# 輸出
# =============================================================================

COLUMNS = ['seg_id', 'split', 'holdout_mmsi', 'mmsi', 'name', 'group',
           'type_name', 'gov', 't', 'lat', 'lon', 'sog', 'cog', 'anc', 'in_port']


def iter_rows(segments, bounds, port_lookup):
    seg_id = 0
    for s in segments:
        tr = s['track']
        holdout = is_holdout_mmsi(tr['mmsi'])
        for label, sub in cut_at_bounds(s['points'], bounds):
            for p in sub:
                yield {
                    'seg_id': seg_id, 'split': label, 'holdout_mmsi': holdout,
                    'mmsi': tr['mmsi'], 'name': tr['name'], 'group': s['group'],
                    'type_name': tr['type_name'], 'gov': tr['gov'] or '',
                    't': int(p[0]), 'lat': p[1], 'lon': p[2], 'sog': p[3],
                    'cog': p[4], 'anc': p[5],
                    'in_port': bool(port_lookup(p[1], p[2])),
                }
            seg_id += 1


def write_points(rows, out_dir):
    """有 pyarrow 寫 Parquet，否則寫 csv.gz（tests.yml 的 CI 沒裝 pandas）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        path = out_dir / 'points.csv.gz'
        n = 0
        with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for row in rows:
                w.writerow(row)
                n += 1
        return path, n
    cols = {c: [] for c in COLUMNS}
    for row in rows:
        for c in COLUMNS:
            cols[c].append(row[c])
    path = out_dir / 'points.parquet'
    pq.write_table(pa.table(cols), path, compression='zstd')
    return path, len(cols['t'])


def summarize(segments, bounds):
    by_group = Counter()
    pts_by_group = Counter()
    for s in segments:
        by_group[s['group']] += 1
        pts_by_group[s['group']] += len(s['points'])
    iso = lambda e: datetime.fromtimestamp(e, timezone.utc).isoformat()
    return {
        'segments': len(segments),
        'points': sum(pts_by_group.values()),
        'segments_by_group': dict(by_group),
        'points_by_group': dict(pts_by_group),
        'split_bounds': {'train_end': iso(bounds[0]), 'val_end': iso(bounds[1])},
        'params': {'max_gap_h': MAX_GAP_H, 'max_implied_kmh': MAX_IMPLIED_KMH,
                   'min_seg_points': MIN_SEG_POINTS,
                   'split_fractions': list(SPLIT_FRACTIONS),
                   'holdout_mmsi_mod': HOLDOUT_MMSI_MOD},
    }


def day_epoch(day, end=False):
    if not day:
        return None
    dt = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return dt.timestamp() + (86400 if end else 0)


def load_archive_segments(archive_dir, start=None, end=None, tiers=TIERS):
    """歸檔目錄 → 航段清單。eval_trajectory 共用。"""
    files = discover_archive_files(archive_dir, start, end, tiers)
    if not files:
        raise SystemExit(f'❌ {archive_dir} 找不到歸檔檔案')
    print(f'📂 歸檔檔案 {len(files)} 個')
    entries = iter_archive_entries(files, day_epoch(start), day_epoch(end, end=True))
    tracks = load_tracks(entries)
    segments = build_segments(tracks)
    if not segments:
        raise SystemExit('❌ 沒有可用航段')
    print(f'🚢 航跡 {len(tracks)} 條 → 航段 {len(segments)} 段')
    return segments


def segment_times(segments):
    return [p[0] for s in segments for p in s['points']]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--archive', required=True, help='ais-archive 分支的 checkout 目錄')
    ap.add_argument('--start', help='YYYY-MM-DD（含）')
    ap.add_argument('--end', help='YYYY-MM-DD（含）')
    ap.add_argument('-o', '--out', default='data/trajectory')
    args = ap.parse_args(argv)

    import geofence
    segments = load_archive_segments(args.archive, args.start, args.end)
    bounds = split_bounds(segment_times(segments))
    path, n = write_points(iter_rows(segments, bounds, geofence.is_in_port_cached),
                           args.out)
    meta = summarize(segments, bounds)
    meta['rows_written'] = n
    meta['generated_at'] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(Path(args.out) / 'meta.json', meta)
    print(f'✅ {path}（{n} 列）')
    print(json.dumps(meta['segments_by_group'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())

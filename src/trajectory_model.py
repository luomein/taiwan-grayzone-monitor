#!/usr/bin/env python3
"""
================================================================================
航跡預測 P1 — 校準過的運動模型（stdlib only，CI 推論用）
================================================================================

第 0 階段的回測（root CLAUDE.md「Trajectory Prediction」）顯示兩個問題：
  1. match_sar_ais 的固定係數不確定度對漁船太小（單側推算門檻涵蓋率 36–52%），
     真漁船被擋在門檻外 → 誤判暗船
  2. MAX_DT_HOURS=3h 與實際 2–3h 的快照間隔正面衝突，大量點推不出位置

本模組的兩個部件都由 fit_trajectory_params.py 在 ais-archive 的 train 切分上
離線擬合，結果存成 data/trajectory_params.json（提交）：

**位置（點估計）** — 速度衰減推算（integrated Ornstein–Uhlenbeck 的平均值）：
    位移 = v · τ · (1 − e^{−dt/τ})
τ → ∞ 就是等速推算；τ 小（漁船）則幾小時後趨近「原地不動」——正是回測裡
漁船 hold 勝過等速 3–7 倍的現象。每個 (船種群組, 動/靜) 各自擬合 τ 與速度來源
（reported = 回報 SOG/COG；track = 最後兩點實際位移）。

**不確定度** — 經驗 90% 誤差半徑查表 r90(群組, 模式, 航速分箱, 時距)：
不假設分布，直接取 train 上的 0.9 分位；時距方向線性內插並強制單調不減；
樣本不足的格子退回全群組合併表。match_sar_ais 把 r90 當成 `unc_extra_km`，
門檻 = 基礎誤差 + r90（再夾在 [MIN_GATE_KM, MAX_GATE_KM]）。

**收件規則** 取代 MAX_DT_HOURS：時距在表的範圍內、且 r90 ≤ ADMIT_MAX_R90_KM
才給估計。靜止船 6h 外的 r90 只有幾公里，照樣可比；高速船 6h 外 r90 數十公里，
夾在 10km 的門檻裡也沒有意義，不給。

參數檔不存在或格式不對時 load_params() 回傳 None，呼叫端退回舊行為。
================================================================================
"""

import json
import math
from bisect import bisect_left
from pathlib import Path

from trajectory_core import KN_TO_KMH, valid_heading

PARAMS_PATH = Path(__file__).resolve().parent.parent / 'data' / 'trajectory_params.json'
PARAMS_VERSION = 1

GROUPS = ('commercial', 'fishing', 'gov', 'other')
GOV_TYPES = {'coastguard', 'msa', 'rescue', 'research'}
COMMERCIAL_TYPES = {'cargo', 'tanker', 'lng'}

MOVING_KN = 1.0
DT_EDGES_H = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
SPEED_EDGES_KN = (1.0, 5.0, 10.0, 15.0)          # → 5 個分箱：<1, 1–5, 5–10, 10–15, ≥15
INTERP_MAX_GAP_H = 8.0
TRACK_VEL_MAX_BASE_H = 3.0
ADMIT_MAX_R90_KM = 6.5      # eval_trajectory sar-sim 在 val/test 掃出來的取捨點（見 root CLAUDE.md）
STATIONARY_SPEED_KN = 0.5

_KM_PER_DEG_LAT = 110.574


# =============================================================================
# 船種群組
# =============================================================================

def type_group(type_name, gov=None, name=None, fishing_by_name=None):
    """gov / commercial / fishing / other。

    CN 漁船 AIS 船種常報 other/unknown/cargo，所以船名規則優先於 AIS 船種；
    公務船再優先於船名（is_cn_fishing_vessel 的 `^XIANG` 會誤中向陽紅）。
    fishing_by_name 預設延遲載入 fetch_ais_data.is_cn_fishing_vessel。"""
    if gov or type_name in GOV_TYPES:
        return 'gov'
    if fishing_by_name is None:
        from fetch_ais_data import is_cn_fishing_vessel as fishing_by_name
    if type_name == 'fishing' or fishing_by_name(name):
        return 'fishing'
    if type_name in COMMERCIAL_TYPES:
        return 'commercial'
    return 'other'


def regime_of(sog_kn):
    return 'moving' if (sog_kn or 0.0) >= MOVING_KN else 'stationary'


def speed_bin(sog_kn):
    return bisect_left(SPEED_EDGES_KN, (sog_kn or 0.0) + 1e-9) if sog_kn else 0


# =============================================================================
# 平面近似（監測區 19–30°N，數十公里內誤差可忽略）
# =============================================================================

def _km_per_deg_lon(lat):
    return 111.320 * math.cos(math.radians(lat))


def offset(lat, lon, east_km, north_km):
    return (lat + north_km / _KM_PER_DEG_LAT,
            lon + east_km / _km_per_deg_lon(lat))


def displacement_km(a_lat, a_lon, b_lat, b_lon):
    """a → b 的 (東, 北) 位移公里。"""
    return ((b_lon - a_lon) * _km_per_deg_lon((a_lat + b_lat) / 2.0),
            (b_lat - a_lat) * _KM_PER_DEG_LAT)


# =============================================================================
# 位置
# =============================================================================

def velocity_kmh(anchor, neighbor, source):
    """錨點的 (東, 北) 速度 km/h。

    source='track'：錨點與相鄰點（前一點或後一點皆可）的實際位移，相距
    ≤ TRACK_VEL_MAX_BASE_H 才用；否則退回 'reported'（回報 SOG/COG）。"""
    if source == 'track' and neighbor is not None:
        a, b = (neighbor, anchor) if neighbor[0] < anchor[0] else (anchor, neighbor)
        base_h = (b[0] - a[0]) / 3600.0
        if 0 < base_h <= TRACK_VEL_MAX_BASE_H:
            de, dn = displacement_km(a[1], a[2], b[1], b[2])
            return de / base_h, dn / base_h
    sog, cog = anchor[3] or 0.0, anchor[4]
    if sog < STATIONARY_SPEED_KN or not valid_heading(cog):
        return 0.0, 0.0
    v = sog * KN_TO_KMH
    rad = math.radians(cog)
    return v * math.sin(rad), v * math.cos(rad)


def damped_distance_factor(dt_h, tau_h):
    """∫₀^dt e^{−s/τ} ds = τ(1 − e^{−dt/τ})；τ 為 None 時就是 dt。"""
    if tau_h is None:
        return dt_h
    return tau_h * (1.0 - math.exp(-dt_h / tau_h))


def extrapolate(anchor, neighbor, dt_s, tau_h=None, source='reported'):
    """自錨點速度衰減推算 dt_s 秒；dt_s < 0 為往回推（SAR 時刻早於第一個點）。"""
    ve, vn = velocity_kmh(anchor, neighbor, source)
    dt_h = dt_s / 3600.0
    k = math.copysign(damped_distance_factor(abs(dt_h), tau_h), dt_h)
    return offset(anchor[1], anchor[2], ve * k, vn * k)


def predict_damped(history, dt_s, tau_h=None, source='reported'):
    """eval_trajectory 介面：history 到發布時刻為止（依 t 排序）。"""
    neighbor = history[-2] if len(history) >= 2 else None
    return extrapolate(history[-1], neighbor, dt_s, tau_h, source)


def make_group_predictor(group_params_):
    """回傳 eval_trajectory 可直接用的 predict(history, dt_s)：依最後一點動/靜挑參數。"""
    def predict(history, dt_s):
        mp = group_params_['dr'][regime_of(history[-1][3])]
        return predict_damped(history, dt_s, mp['tau_h'], mp['source'])
    return predict


# =============================================================================
# 不確定度查表
# =============================================================================

def r90_lookup(table, sog_kn, dt_h):
    """r90 表 → km；超出時距範圍或格子空著回傳 None。

    table = {'dt_edges_h': [...], 'r90_km': [[每個航速分箱] × 每個時距節點]}
    時距在節點之間線性內插；小於第一個節點用第一個節點的值。"""
    if not table:
        return None
    edges = table['dt_edges_h']
    rows = table['r90_km']
    if dt_h > edges[-1]:
        return None
    sb = speed_bin(sog_kn)
    i = bisect_left(edges, dt_h)
    hi = rows[i][sb]
    if i == 0 or hi is None:
        return hi
    lo = rows[i - 1][sb]
    if lo is None:
        return hi
    w = (dt_h - edges[i - 1]) / (edges[i] - edges[i - 1])
    return lo + w * (hi - lo)


# =============================================================================
# 參數
# =============================================================================

def load_params(path=PARAMS_PATH):
    """讀 data/trajectory_params.json；缺檔/版本不符/格式壞 → None（呼叫端退回舊行為）。"""
    try:
        params = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(params, dict) or params.get('version') != PARAMS_VERSION:
        return None
    if not isinstance(params.get('groups'), dict):
        return None
    return params


def group_params(params, group):
    g = params['groups'].get(group)
    return g if g else params['groups'].get('all')


# =============================================================================
# SAR 時刻推算（match_sar_ais.estimate_position 的 P1 版）
# =============================================================================

def estimate_position(points, t_epoch, group, params):
    """把一艘船推算到 t_epoch。回傳 match_sar_ais 相容的 dict：
    {lat, lon, speed_kn, dt_h, unc_extra_km, method}；不收件回傳 None。

    points: [(t, lat, lon, sog_kn, cog_deg, ...)]，依 t 排序。"""
    if not points:
        return None
    gp = group_params(params, group)
    if not gp:
        return None
    admit = params.get('admit_max_r90_km', ADMIT_MAX_R90_KM)
    times = [p[0] for p in points]
    i = bisect_left(times, t_epoch)
    prev = points[i - 1] if i > 0 else None
    nxt = points[i] if i < len(points) else None

    # 兩側都有點 → 時間加權內插
    if prev is not None and nxt is not None:
        gap_h = (nxt[0] - prev[0]) / 3600.0
        if 0 < gap_h <= params.get('interp_max_gap_h', INTERP_MAX_GAP_H):
            w = (t_epoch - prev[0]) / (nxt[0] - prev[0])
            dt_near_h = min(t_epoch - prev[0], nxt[0] - t_epoch) / 3600.0
            speed = max(prev[3] or 0.0, nxt[3] or 0.0)
            r90 = r90_lookup(gp.get('interp_r90'), speed, dt_near_h)
            if r90 is not None and r90 <= admit:
                return {'lat': prev[1] + w * (nxt[1] - prev[1]),
                        'lon': prev[2] + w * (nxt[2] - prev[2]),
                        'speed_kn': speed, 'dt_h': dt_near_h,
                        'unc_extra_km': r90, 'method': 'interpolated'}

    # 單側 → 取時間上較近的一側做速度衰減推算（後側即往回推）
    if prev is not None and (nxt is None or t_epoch - prev[0] <= nxt[0] - t_epoch):
        last, neighbor = prev, (points[i - 2] if i >= 2 else None)
    elif nxt is not None:
        last, neighbor = nxt, (points[i + 1] if i + 1 < len(points) else None)
    else:
        return None
    dt_s = t_epoch - last[0]
    dt_h = abs(dt_s) / 3600.0
    mp = (gp.get('dr') or {}).get(regime_of(last[3])) or {}
    r90 = r90_lookup(gp.get('dr_r90'), last[3], dt_h)
    if r90 is None or r90 > admit:
        return None
    lat, lon = extrapolate(last, neighbor, dt_s, mp.get('tau_h'),
                           mp.get('source', 'reported'))
    moving = (last[3] or 0.0) >= STATIONARY_SPEED_KN
    return {'lat': lat, 'lon': lon, 'speed_kn': last[3] or 0.0, 'dt_h': dt_h,
            'unc_extra_km': r90,
            'method': 'dead_reckoning' if moving else 'hold'}

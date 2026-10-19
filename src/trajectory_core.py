#!/usr/bin/env python3
"""
================================================================================
航跡預測核心 — Trajectory prediction core (stdlib only)
================================================================================

航跡預測架構（第 0 階段）的共用基礎，供三處使用：
  - fetch_ais_data.py   寫入航跡點時附上 `rt`（定位時刻落後快照幾秒）
  - match_sar_ais.py    讀航跡時用 `rt` 把點還原到真正的定位時刻
  - eval_trajectory.py  離線回測的基準預測器（P0）

只依賴 stdlib + geo_utils：update-ais.yml 只裝 requests + pysocks，
tests.yml 只裝 requests + pytest。

## `rt` 欄位（定位時刻）
航跡 entry 的 `timestamp` 是「我們抓資料」的時刻，不是船的定位時刻。航港局
`Record_Time`（台灣時間 UTC+8，`YYYY/MM/DD HH:MM:SS`）實測落後抓取時刻
2.5–15 分鐘（中位數 3.7 分；超過約 15 分的船航港局就不回傳了）。12 節的
貨輪 15 分鐘走 5.5km，已逼近 SAR 比對的最小門檻 1.5km 的數倍，所以要還原。

`rt` = 抓取時刻 − 定位時刻（整數秒），只在「船在動（≥ RT_MIN_SPEED_KN）且
落後 ≥ RT_MIN_EMIT_S」時寫入：靜止船的定位時刻不影響位置，tier-1 又是給
瀏覽器下載的大檔，能省就省。讀取端沒有 `rt` 一律視為 0（舊資料相容）。

## 基準預測器（P0）
所有預測器同一介面：`predict(history, dt_s) -> (lat, lon)`，history 是到
發布時刻為止的點列 [(t, lat, lon, sog_kn, cog_deg), ...]（依 t 排序）。
第 1 階段之後的模型（Kalman/IMM、歷史行為、深度模型）必須在
eval_trajectory.py 的回測裡贏過這裡的基準才准上線。
================================================================================
"""

import math
from datetime import datetime, timedelta, timezone

from geo_utils import haversine_km

TW_TZ = timezone(timedelta(hours=8))
RECORD_TIME_FMT = '%Y/%m/%d %H:%M:%S'

RT_MIN_EMIT_S = 60          # 落後不到 1 分鐘不寫（1–2h 快照間隔下可忽略）
RT_MIN_SPEED_KN = 1.0       # 靜止/錨泊船定位時刻不影響位置，不寫
RT_MAX_AGE_S = 6 * 3600     # 超過 6h 視為時鐘錯誤，不寫

KN_TO_KMH = 1.852
STATIONARY_SPEED_KN = 0.5   # 與 match_sar_ais.STATIONARY_SPEED_KN 一致
MAX_VELOCITY_BASE_S = 3 * 3600  # cv_track 用的前一點最遠時距


# =============================================================================
# 定位時刻
# =============================================================================

def record_time_epoch(value):
    """航港局 Record_Time（台灣時間）→ epoch 秒。也接受 ISO 8601。無效回傳 None。"""
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    try:
        return datetime.strptime(s, RECORD_TIME_FMT).replace(tzinfo=TW_TZ).timestamp()
    except ValueError:
        pass
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TW_TZ)
    return dt.timestamp()


def compact_rt(record_time, now_epoch, speed_kn):
    """航跡點的 `rt` 值（整數秒）；不該寫入時回傳 None（見模組說明）。"""
    if (speed_kn or 0.0) < RT_MIN_SPEED_KN:
        return None
    t = record_time_epoch(record_time)
    if t is None:
        return None
    age = int(round(now_epoch - t))
    if age < RT_MIN_EMIT_S or age > RT_MAX_AGE_S:
        return None
    return age


def point_epoch(entry_epoch, vessel):
    """航跡點的定位時刻 = entry 時刻 − rt（無 rt 視為 0）。"""
    rt = vessel.get('rt') if isinstance(vessel, dict) else None
    if isinstance(rt, (int, float)) and 0 < rt <= RT_MAX_AGE_S:
        return entry_epoch - rt
    return entry_epoch


# =============================================================================
# 運動學
# =============================================================================

def destination_point(lat, lon, bearing_deg, dist_km):
    """自 (lat, lon) 沿方位角 bearing 前進 dist_km 的目的座標（球面）。"""
    R = 6371.0
    delta = dist_km / R
    theta = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(delta) +
                     math.cos(lat1) * math.sin(delta) * math.cos(theta))
    lon2 = lon1 + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(lat1),
                             math.cos(delta) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def valid_heading(h):
    return isinstance(h, (int, float)) and 0.0 <= h < 360.0


# =============================================================================
# 基準預測器（P0）
# =============================================================================

def predict_hold(history, dt_s):
    """原地不動：最後一點。錨泊/作業中漁船的強基準。"""
    p = history[-1]
    return p[1], p[2]


def predict_cv_reported(history, dt_s):
    """回報 SOG/COG 等速推算（= match_sar_ais 單側 dead reckoning）。"""
    p = history[-1]
    sog, cog = p[3] or 0.0, p[4]
    if sog < STATIONARY_SPEED_KN or not valid_heading(cog):
        return p[1], p[2]
    return destination_point(p[1], p[2], cog, sog * KN_TO_KMH * dt_s / 3600.0)


def predict_cv_track(history, dt_s):
    """以最後兩點的實際位移推等速：不信任回報 COG（CN 漁船常亂報）。
    前一點太遠（> MAX_VELOCITY_BASE_S）時退回 predict_cv_reported。"""
    if len(history) < 2:
        return predict_cv_reported(history, dt_s)
    a, b = history[-2], history[-1]
    base = b[0] - a[0]
    if base <= 0 or base > MAX_VELOCITY_BASE_S:
        return predict_cv_reported(history, dt_s)
    k = dt_s / base
    # 小距離下經緯度線性外插誤差可忽略；經度不跨 ±180（監測區 116–130°E）
    return b[1] + k * (b[1] - a[1]), b[2] + k * (b[2] - a[2])


BASELINE_PREDICTORS = {
    'hold': predict_hold,
    'cv_reported': predict_cv_reported,
    'cv_track': predict_cv_track,
}


def prediction_error_km(pred, truth):
    return haversine_km(pred[0], pred[1], truth[0], truth[1])

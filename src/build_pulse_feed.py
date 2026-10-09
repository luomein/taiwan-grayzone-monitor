#!/usr/bin/env python3
"""
build_pulse_feed.py — 給 Mini Taiwan Pulse 兵棋地圖的精簡情資檔

Outputs: docs/pulse_feed.json（約 1 MB；Pulse 的兵推模式開「灰色地帶情資」圖層時 fetch）

內容：
  - tracks      近 24h tier-1 航跡（ais_track_history.json 只取窗口內快照，壓成每船一條）
  - high_risk   analyze_suspicious 的高風險分類（critical / high）+ 最後位置
  - dark        SAR 暗船偵測（GFW 本身延遲數天，取資料末日往回 7 天，不是近 24h）
  - cables      台灣周邊海纜 + 數位部障礙狀態

原始檔 ais_track_history.json 有 19 MB（14 天），讓瀏覽器自己切 24h 太肥；
這裡在 CI 先切好，Pulse 只抓這一個檔。
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from io_utils import atomic_write_json, load_json

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
OUTPUT_FILE = DOCS_DIR / "pulse_feed.json"

TRACK_WINDOW_HOURS = 24
DARK_WINDOW_DAYS = 7
HIGH_RISK_LEVELS = ("critical", "high")
MAX_FLAGS = 6


def _parse_ts(ts):
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def build_tracks(snapshots, window_hours=TRACK_WINDOW_HOURS, risk_by_mmsi=None):
    """近 window_hours 小時（以最後一個快照為終點）的航跡，每船一條。

    pts: [lon, lat, unix_sec, speed_kn]，依時間排序。
    """
    risk_by_mmsi = risk_by_mmsi or {}
    timed = []
    for snap in snapshots or []:
        dt = _parse_ts(snap.get("timestamp"))
        if dt is not None:
            timed.append((dt, snap))
    if not timed:
        return {"start": None, "end": None, "snapshots": 0, "vessels": []}
    timed.sort(key=lambda x: x[0])
    end = timed[-1][0]
    cutoff = end - timedelta(hours=window_hours)
    window = [(dt, s) for dt, s in timed if dt >= cutoff]

    vessels = {}
    for dt, snap in window:
        t = int(dt.timestamp())
        for v in snap.get("vessels", []):
            mmsi = v.get("mmsi")
            lat, lon = v.get("lat"), v.get("lon")
            if not mmsi or lat is None or lon is None:
                continue
            mmsi = str(mmsi)
            rec = vessels.get(mmsi)
            if rec is None:
                rec = vessels[mmsi] = {
                    "mmsi": mmsi,
                    "name": v.get("name") or "",
                    "type": v.get("gov") or v.get("type_name") or "unknown",
                    "pts": [],
                }
                if mmsi in risk_by_mmsi:
                    rec["risk"] = risk_by_mmsi[mmsi]
            if v.get("name"):
                rec["name"] = v["name"]
            speed = v.get("speed")
            rec["pts"].append([
                round(float(lon), 4), round(float(lat), 4), t,
                round(float(speed), 1) if speed is not None else None,
            ])

    return {
        "start": window[0][0].isoformat(),
        "end": end.isoformat(),
        "snapshots": len(window),
        "vessels": sorted(vessels.values(), key=lambda r: r["mmsi"]),
    }


def build_high_risk(suspicious, end=None, window_hours=TRACK_WINDOW_HOURS):
    """critical / high 分類 → 地圖點。

    評分窗口是 14 天，最後位置可能是幾週前；只留 end 往回 window_hours 內出現過的，
    舊位置上圖會被誤當成現況。沒有最後位置的船無法上圖，略過。
    """
    cutoff = end - timedelta(hours=window_hours) if end else None
    out = []
    for c in (suspicious or {}).get("all_classifications", []):
        if c.get("excluded") or c.get("risk_level") not in HIGH_RISK_LEVELS:
            continue
        lat, lon = c.get("last_lat"), c.get("last_lon")
        if lat is None or lon is None:
            continue
        if cutoff is not None:
            seen = _parse_ts(c.get("last_seen"))
            if seen is None or seen < cutoff:
                continue
        cable = c.get("cable_details") or {}
        out.append({
            "mmsi": str(c.get("mmsi")),
            "name": (c.get("names") or [""])[-1],
            "risk_level": c["risk_level"],
            "risk_score": c.get("risk_score"),
            "vessel_type": c.get("vessel_type"),
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "last_seen": c.get("last_seen"),
            "flags": (c.get("flags") or [])[:MAX_FLAGS],
            "cables_nearby": cable.get("cables_nearby") or [],
            "sanctioned": bool(c.get("sanctioned")),
        })
    out.sort(key=lambda r: -(r["risk_score"] or 0))
    return out


def build_dark(sar_detections, dark_vessels, window_days=DARK_WINDOW_DAYS):
    """SAR 暗船：優先用 sar_detections.json 的全精度清單，缺檔才退回 dark_vessels 的 400 筆抽樣。

    窗口以資料內最新日期為終點（GFW 延遲數天，用「今天」往回算常常是空的）。
    """
    points = (sar_detections or {}).get("dark_detections")
    source = "sar_detections"
    if not points:
        region = ((dark_vessels or {}).get("regions") or {}).get("taiwan_region") or {}
        points = region.get("dark_details") or []
        source = "dark_vessels_sample"
    dates = sorted({p["date"] for p in points if p.get("date")})
    if not dates:
        return {"source": source, "start": None, "end": None, "points": []}
    end = dates[-1]
    start = (datetime.fromisoformat(end) - timedelta(days=window_days - 1)).date().isoformat()
    pts = [
        [round(float(p["lon"]), 3), round(float(p["lat"]), 3), p["date"], int(p.get("detections") or 1)]
        for p in points
        if p.get("date") and start <= p["date"] <= end and p.get("lat") is not None and p.get("lon") is not None
    ]
    return {"source": source, "start": start, "end": end, "points": pts}


def build_cables(cables_geo, cable_status):
    """台灣周邊海纜 GeoJSON，屬性精簡並併入數位部障礙狀態（依 slug）。"""
    faults_by_slug = {}
    for f in (cable_status or {}).get("faults", []):
        if f.get("status") == "fault" and not f.get("repair_date"):
            faults_by_slug.setdefault(f.get("slug"), []).append({
                "segment": f.get("segment"),
                "name_zh": f.get("name_zh"),
                "fault_date": f.get("fault_date"),
                "estimated_repair": f.get("estimated_repair"),
                "location_zh": f.get("location_zh"),
            })
    features = []
    for feat in (cables_geo or {}).get("features", []):
        p = feat.get("properties") or {}
        slug = p.get("slug")
        faults = faults_by_slug.get(slug, [])
        features.append({
            "type": "Feature",
            "geometry": feat.get("geometry"),
            "properties": {
                "slug": slug,
                "name": p.get("name") or slug,
                "cable_type": p.get("cable_type"),
                "status": p.get("status"),
                "fault": bool(faults),
                "faults": faults,
            },
        })
    return {
        "updated_at": (cable_status or {}).get("updated_at"),
        "fault_count": sum(len(v) for v in faults_by_slug.values()),
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def build_feed(snapshots, suspicious, sar_detections, dark_vessels, cables_geo, cable_status, now=None):
    now = now or datetime.now(timezone.utc)
    # 高風險船依航跡終點（最後一個 AIS 快照）往回 24h 篩；沒有航跡時退回現在時間
    snap_times = [t for t in (_parse_ts(s.get("timestamp")) for s in snapshots or []) if t]
    tracks_end = max(snap_times) if snap_times else now
    high_risk = build_high_risk(suspicious, end=tracks_end)
    risk_by_mmsi = {r["mmsi"]: r["risk_level"] for r in high_risk}
    return {
        "version": 1,
        "generated_at": now.isoformat(),
        "source": "https://github.com/s0914712/taiwan-grayzone-monitor",
        "tracks": build_tracks(snapshots, risk_by_mmsi=risk_by_mmsi),
        "high_risk": {
            "updated_at": (suspicious or {}).get("updated_at"),
            "vessels": high_risk,
        },
        "dark": build_dark(sar_detections, dark_vessels),
        "cables": build_cables(cables_geo, cable_status),
    }


def main():
    feed = build_feed(
        snapshots=load_json(DOCS_DIR / "ais_track_history.json", [], expect_type=list),
        suspicious=load_json(DATA_DIR / "suspicious_vessels.json", {}, expect_type=dict),
        sar_detections=load_json(DATA_DIR / "sar_detections.json", {}, expect_type=dict),
        dark_vessels=load_json(DATA_DIR / "dark_vessels.json", {}, expect_type=dict),
        cables_geo=load_json(DOCS_DIR / "taiwan_cables.json", {}, expect_type=dict),
        cable_status=load_json(DOCS_DIR / "cable_status.json", {}, expect_type=dict),
    )
    atomic_write_json(OUTPUT_FILE, feed, compact=True)
    print(
        f"✅ pulse_feed.json：航跡 {len(feed['tracks']['vessels'])} 艘 / "
        f"{feed['tracks']['snapshots']} 快照，高風險 {len(feed['high_risk']['vessels'])} 艘，"
        f"暗船 {len(feed['dark']['points'])} 點（{feed['dark']['start']}～{feed['dark']['end']}），"
        f"海纜 {len(feed['cables']['geojson']['features'])} 條（障礙 {feed['cables']['fault_count']}）"
    )


if __name__ == "__main__":
    main()

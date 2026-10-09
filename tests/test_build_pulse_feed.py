"""Pulse 兵推地圖情資檔 — build_pulse_feed

Pulse 只抓 docs/pulse_feed.json 一個檔；這裡確保 24h 窗口、高風險篩選、
暗船「以資料末日為準」窗口、海纜障礙併入都正確。
"""
from datetime import datetime, timedelta, timezone

import build_pulse_feed as bpf

END = datetime(2026, 10, 8, 18, 0, tzinfo=timezone.utc)


def _snap(dt, vessels):
    return {"timestamp": dt.isoformat(), "vessels": vessels}


def _v(mmsi, lat=24.0, lon=120.0, speed=5.0, name="X", type_name="fishing"):
    return {"mmsi": mmsi, "name": name, "lat": lat, "lon": lon, "speed": speed, "type_name": type_name}


def test_tracks_keep_only_last_24h_relative_to_latest_snapshot():
    snaps = [
        _snap(END - timedelta(hours=30), [_v("1", lon=119.0)]),
        _snap(END - timedelta(hours=10), [_v("1", lon=119.5), _v("2")]),
        _snap(END, [_v("1", lon=120.0)]),
    ]
    out = bpf.build_tracks(snaps)
    assert out["snapshots"] == 2
    assert out["end"] == END.isoformat()
    by = {v["mmsi"]: v for v in out["vessels"]}
    assert [p[0] for p in by["1"]["pts"]] == [119.5, 120.0]
    assert len(by["2"]["pts"]) == 1


def test_tracks_sorted_by_time_even_if_input_is_not():
    snaps = [_snap(END, [_v("1", lon=120.0)]), _snap(END - timedelta(hours=2), [_v("1", lon=119.0)])]
    pts = bpf.build_tracks(snaps)["vessels"][0]["pts"]
    assert pts[0][2] < pts[1][2]


def test_tracks_tag_high_risk_vessels():
    out = bpf.build_tracks([_snap(END, [_v("9")])], risk_by_mmsi={"9": "critical"})
    assert out["vessels"][0]["risk"] == "critical"


def test_empty_track_history():
    out = bpf.build_tracks([])
    assert out["vessels"] == [] and out["end"] is None


def test_high_risk_filters_level_excluded_and_missing_position():
    sus = {"all_classifications": [
        {"mmsi": "1", "names": ["A"], "risk_level": "critical", "risk_score": 20, "last_lat": 24, "last_lon": 120},
        {"mmsi": "2", "names": ["B"], "risk_level": "medium", "risk_score": 6, "last_lat": 24, "last_lon": 120},
        {"mmsi": "3", "names": ["C"], "risk_level": "high", "risk_score": 9, "last_lat": None, "last_lon": None},
        {"mmsi": "4", "names": ["D"], "risk_level": "high", "risk_score": 9, "last_lat": 24, "last_lon": 120,
         "excluded": True},
        {"mmsi": "5", "names": ["E"], "risk_level": "high", "risk_score": 30, "last_lat": 25, "last_lon": 121},
    ]}
    out = bpf.build_high_risk(sus)
    assert [r["mmsi"] for r in out] == ["5", "1"]   # 依分數高→低


def test_high_risk_drops_vessels_not_seen_in_last_24h():
    """評分窗口 14 天；幾週前的最後位置不上圖，以免被當成現況。"""
    sus = {"all_classifications": [
        {"mmsi": "1", "risk_level": "critical", "risk_score": 20, "last_lat": 24, "last_lon": 120,
         "last_seen": (END - timedelta(hours=3)).isoformat()},
        {"mmsi": "2", "risk_level": "critical", "risk_score": 25, "last_lat": 24, "last_lon": 120,
         "last_seen": (END - timedelta(days=9)).isoformat()},
        {"mmsi": "3", "risk_level": "critical", "risk_score": 25, "last_lat": 24, "last_lon": 120},
    ]}
    assert [r["mmsi"] for r in bpf.build_high_risk(sus, end=END)] == ["1"]


def test_feed_anchors_high_risk_window_to_latest_snapshot():
    sus = {"all_classifications": [
        {"mmsi": "1", "risk_level": "high", "risk_score": 9, "last_lat": 24, "last_lon": 120,
         "last_seen": (END - timedelta(hours=20)).isoformat()},
    ]}
    snaps = [_snap(END, [_v("1")])]
    feed = bpf.build_feed(snaps, sus, {}, {}, {}, {}, now=END + timedelta(days=3))
    assert [r["mmsi"] for r in feed["high_risk"]["vessels"]] == ["1"]
    assert feed["tracks"]["vessels"][0]["risk"] == "high"


def test_dark_window_anchored_to_latest_data_date_not_today():
    sar = {"dark_detections": [
        {"lat": 24, "lon": 120, "date": "2026-09-20", "detections": 1},
        {"lat": 24, "lon": 120, "date": "2026-09-29", "detections": 2},
        {"lat": 24, "lon": 120, "date": "2026-10-05", "detections": 1},
    ]}
    out = bpf.build_dark(sar, {})
    assert (out["start"], out["end"]) == ("2026-09-29", "2026-10-05")
    assert [p[2] for p in out["points"]] == ["2026-09-29", "2026-10-05"]


def test_dark_falls_back_to_sample_when_full_list_missing():
    dv = {"regions": {"taiwan_region": {"dark_details": [{"lat": 22, "lon": 119, "date": "2026-10-01"}]}}}
    out = bpf.build_dark({}, dv)
    assert out["source"] == "dark_vessels_sample" and len(out["points"]) == 1


def test_cables_merge_open_faults_only():
    geo = {"features": [
        {"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": []},
         "properties": {"slug": "tpkm3", "name": "TPKM3"}},
        {"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": []},
         "properties": {"slug": "apg", "name": "APG"}},
    ]}
    status = {"faults": [
        {"slug": "tpkm3", "status": "fault", "repair_date": None, "fault_date": "2026-10-06"},
        {"slug": "apg", "status": "fault", "repair_date": "2026-09-01", "fault_date": "2026-05-20"},
    ]}
    out = bpf.build_cables(geo, status)
    props = {f["properties"]["slug"]: f["properties"] for f in out["geojson"]["features"]}
    assert props["tpkm3"]["fault"] is True
    assert props["apg"]["fault"] is False
    assert out["fault_count"] == 1

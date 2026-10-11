"""公務船編隊偵測 — detect_gov_formation.py

覆蓋：分群半徑、持續時間門檻、跨快照串接、護航科考判定，
以及兩道誤報抑制（港內排除 + 泊地抑制）——這兩道是實測必要的：
未加之前，廈門港內 17 艘公務船會被單一鏈結串成一個橫跨 28km 的假編隊。
"""
from datetime import datetime, timedelta, timezone

import detect_gov_formation as gf


def _snap(ts, vessels):
    return {"timestamp": ts.isoformat(), "vessels": vessels}


def _v(mmsi, name, cat, lat, lon, speed=6.0):
    return {"mmsi": mmsi, "name": name, "gov": cat, "type_name": cat,
            "lat": lat, "lon": lon, "speed": speed}


BASE = datetime(2026, 8, 21, 0, 0, tzinfo=timezone.utc)
# 花蓮外海（非任何港口 8km 圈內）
LAT, LON = 23.60, 122.40


def _escort_track(hours, step=2, drift=0.02):
    """科研船 + 兩艘海警同框、整體東移的一串快照。"""
    snaps = []
    for i in range(0, hours + 1, step):
        d = i * drift
        snaps.append(_snap(BASE + timedelta(hours=i), [
            _v("413701510", "XIANG YANG HONG 03", "research", LAT, LON + d),
            _v("413875040", "CHINACOASTGUARD2502", "coastguard",
               LAT + 0.03, LON + d),
            _v("413875017", "CHINACOASTGUARD1306", "coastguard",
               LAT - 0.03, LON + d),
        ]))
    return snaps


def _detect(snaps, now=None):
    formations = gf.track_formations(gf.load_gov_snapshots(snaps))
    if now is not None:
        gf.split_active(formations, now=now)
    return formations


def test_escorted_research_formation_detected():
    formations = _detect(_escort_track(12))
    assert len(formations) == 1
    f = formations[0]
    assert f["escorted_research"] is True
    assert f["vessel_count"] == 3
    assert f["duration_hours"] >= gf.FORMATION_MIN_DURATION_HOURS
    assert f["severity"] == "high"          # 護航科考 ≥12h
    assert set(f["categories"]) == {"research", "coastguard"}


def test_below_duration_threshold_not_reported():
    """同框但只有 4 小時 —— 短暫交會不成案。"""
    assert _detect(_escort_track(4)) == []


def test_vessels_beyond_radius_are_not_a_formation():
    """相距 ~50km（>10km 門檻）→ 不算編隊。"""
    snaps = []
    for i in range(0, 13, 2):
        snaps.append(_snap(BASE + timedelta(hours=i), [
            _v("413701510", "XIANG YANG HONG 03", "research", LAT, LON),
            _v("413875040", "CHINACOASTGUARD2502", "coastguard",
               LAT + 0.45, LON),
        ]))
    assert _detect(snaps) == []


def test_gap_breaks_the_formation():
    """中間斷訊 12h（>FORMATION_MAX_GAP_HOURS）→ 切成兩段，各自不足 6h。"""
    first = _escort_track(4)
    later = [_snap(s["timestamp"] and datetime.fromisoformat(s["timestamp"])
                   + timedelta(hours=16), s["vessels"]) for s in _escort_track(4)]
    assert _detect(first + later) == []


def test_in_port_vessels_excluded():
    """廈門港內同框 24h —— 母港停泊不是編隊作業。"""
    snaps = []
    for i in range(0, 25, 2):
        snaps.append(_snap(BASE + timedelta(hours=i), [
            _v("413875308", "CHINACOASTGUARD14531", "coastguard",
               24.45, 118.07, speed=0.0),
            _v("412461040", "SHIYAN2", "research", 24.46, 118.08, speed=0.0),
        ]))
    assert _detect(snaps) == []


def test_berthed_cluster_suppressed_outside_known_ports():
    """未列入港口清單的沿岸泊地：中心不動 + 成員 0kn → 抑制。"""
    snaps = []
    for i in range(0, 25, 2):
        snaps.append(_snap(BASE + timedelta(hours=i), [
            _v("413225040", "HAIXUN1620", "msa", 26.60, 120.90, speed=0.0),
            _v("413046050", "DONG HAI JIU 115", "rescue",
               26.601, 120.901, speed=0.0),
        ]))
    formations = _detect(snaps)
    assert formations == []


def test_moving_slow_formation_at_sea_not_suppressed():
    """海上低速（0.5kn）但整體位移數十公里 → 仍成案（泊地規則不可誤殺）。"""
    snaps = []
    for i in range(0, 13, 2):
        d = i * 0.05
        snaps.append(_snap(BASE + timedelta(hours=i), [
            _v("413701510", "XIANG YANG HONG 03", "research",
               LAT, LON + d, speed=0.5),
            _v("413875040", "CHINACOASTGUARD2502", "coastguard",
               LAT + 0.02, LON + d, speed=0.5),
        ]))
    formations = _detect(snaps)
    assert len(formations) == 1
    assert formations[0]["escorted_research"] is True


def test_non_gov_vessels_ignored():
    snaps = []
    for i in range(0, 13, 2):
        snaps.append(_snap(BASE + timedelta(hours=i), [
            {"mmsi": "412001", "name": "MINDONGYU1", "type_name": "fishing",
             "lat": LAT, "lon": LON, "speed": 3.0},
            {"mmsi": "412002", "name": "MINDONGYU2", "type_name": "fishing",
             "lat": LAT + 0.01, "lon": LON, "speed": 3.0},
        ]))
    assert _detect(snaps) == []


def test_msa_only_group_is_not_escorted_research():
    snaps = []
    for i in range(0, 13, 2):
        d = i * 0.03
        snaps.append(_snap(BASE + timedelta(hours=i), [
            _v("413225690", "HAIXUN0807", "msa", LAT, LON + d),
            _v("413366000", "HAI XUN 0802", "msa", LAT + 0.02, LON + d),
        ]))
    formations = _detect(snaps)
    assert len(formations) == 1
    assert formations[0]["escorted_research"] is False
    assert formations[0]["severity"] == "low"


def test_vessel_index_maps_every_member():
    formations = _detect(_escort_track(12))
    idx = gf.build_mmsi_index(formations)
    assert set(idx) == {"413701510", "413875040", "413875017"}
    for rec in idx.values():
        assert rec["escorted_research"] is True
        assert rec["max_duration_hours"] >= 12.0


def test_split_active_marks_recent_formations():
    formations = _detect(_escort_track(12))
    now = BASE + timedelta(hours=13)
    active, history = gf.split_active(formations, now=now)
    assert len(active) == 1 and not history
    stale_now = BASE + timedelta(days=5)
    active, history = gf.split_active(formations, now=stale_now)
    assert not active and len(history) == 1


def test_cluster_by_distance_respects_spread_cap():
    """單一鏈結會沿著等距船隊接龍 —— 超過 FORMATION_MAX_SPREAD_KM 不成群。"""
    chain = [
        {"mmsi": str(i), "lat": LAT + i * 0.05, "lon": LON, "speed": 5}
        for i in range(12)   # 跨度 ~66km
    ]
    assert gf.cluster_by_distance(chain) == []


# ── 伴航船（非公務船跟著公務船移動）────────────────────────────
# 2026-08-12~16 閩福鼎漁07902 跟隨向陽紅03 的實例：漁船 3-9km 外同步移動，
# 中間關 AIS ~57h。下面以合成資料重現其幾何，並確認三種常見誤報不成案。

def _fish(mmsi, lat, lon, speed=8.0, name="MINFUDINGYU07902"):
    return {"mmsi": mmsi, "name": name, "type_name": "fishing",
            "lat": lat, "lon": lon, "speed": speed}


def _follow_track(n, step=2, drift=0.1, offset=0.05, start=0,
                  anchor_speed=8.0, fish_speed=8.0, fish=True):
    """科研船每 step 小時東移 drift°（≈10km），漁船在其北方 offset°（≈5.5km）。"""
    snaps = []
    for i in range(n):
        h = start + i * step
        d = (start // step + i) * drift
        vessels = [_v("413701510", "XIANG YANG HONG 03", "research",
                      LAT, LON + d, speed=anchor_speed)]
        if fish:
            vessels.append(_fish("412450186", LAT + offset, LON + d,
                                 speed=fish_speed))
        snaps.append(_snap(BASE + timedelta(hours=h), vessels))
    return snaps


def _companions(snaps):
    return gf.track_companions(gf.load_companion_snapshots(snaps))


def test_companion_following_research_vessel_detected():
    cs = _companions(_follow_track(8))
    assert len(cs) == 1
    c = cs[0]
    assert c["mmsi"] == "412450186"
    assert c["anchor_mmsi"] == "413701510"
    assert c["anchor_category"] == "research"
    assert c["observations"] == 8
    assert c["travel_km"] >= gf.COMPANION_MIN_TRAVEL_KM
    assert c["dark_gaps"] == []


def test_companion_bridges_dark_gap_and_reports_it():
    """兩邊同時關 AIS 40h 後再一起出現 —— 仍是同一段伴航，且記下斷訊。"""
    snaps = _follow_track(4) + _follow_track(4, start=46)
    cs = _companions(snaps)
    assert len(cs) == 1
    assert cs[0]["observations"] == 8
    assert cs[0]["max_dark_gap_hours"] == 40.0


def test_gap_beyond_companion_max_gap_splits_episode():
    snaps = _follow_track(6) + _follow_track(6, start=100)
    cs = _companions(snaps)
    assert len(cs) == 2


def test_fishing_beside_stationary_gov_vessel_not_companion():
    """停在漁場上的救助/海巡船旁圍一圈漁船 —— 同一片漁場，不是伴航。"""
    assert _companions(_follow_track(10, drift=0.0, anchor_speed=0.1,
                                     fish_speed=0.3)) == []


def test_companion_requires_travel():
    """兩船都在動但原地繞圈（位移 <20km）—— 不成案。"""
    assert _companions(_follow_track(10, drift=0.01)) == []


def test_brief_crossing_not_companion():
    assert _companions(_follow_track(3)) == []


def test_companion_ignores_in_port_points():
    snaps = []
    for i in range(8):
        snaps.append(_snap(BASE + timedelta(hours=2 * i), [
            _v("413701510", "XIANG YANG HONG 03", "research",
               22.6153, 120.2664 + 0.001 * i),     # 高雄港
            _fish("412450186", 22.6160, 120.2664 + 0.001 * i),
        ]))
    assert _companions(snaps) == []


def test_companion_index_aggregates_anchors():
    snaps = _follow_track(8)
    for s in snaps:   # 同一艘漁船也跟著第二艘海警
        lon = s["vessels"][0]["lon"]
        s["vessels"].append(_v("413875054", "CHINACOASTGUARD 2504",
                               "coastguard", LAT - 0.03, lon))
    idx = gf.build_companion_index(_companions(snaps))
    rec = idx["412450186"]
    assert rec["count"] == 2
    assert {a["mmsi"] for a in rec["anchors"]} == {"413701510", "413875054"}
    assert rec["research_anchor"] is True
    assert "413875054" not in idx      # 公務船之間交給編隊偵測


def test_snapshot_without_offshore_gov_is_not_counted_as_dark():
    """科研船靠港的那幾個快照，漁船仍被 AIS 看到 —— 不能算伴航中關 AIS。"""
    snaps = _follow_track(4)
    for h in range(8, 40, 2):        # 32h 內只有漁船（科研船不在快照裡）
        snaps.append(_snap(BASE + timedelta(hours=h),
                           [_fish("412450186", LAT + 0.05, LON + 0.4)]))
    snaps += _follow_track(4, start=40)
    cs = _companions(snaps)
    assert len(cs) == 1
    assert cs[0]["dark_gaps"] == []

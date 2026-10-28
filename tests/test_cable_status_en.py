"""fetch_cable_status 的英文障礙描述（英文介面用）。

tests.yml 只裝 requests + pytest；bs4 只在爬取時用到，這裡先放一個空模組讓 import 成功。"""
import json
import sys
import types
from pathlib import Path

sys.modules.setdefault('bs4', types.SimpleNamespace(BeautifulSoup=None))
import fetch_cable_status as fcs  # noqa: E402


def test_describe_fault_en_patterns():
    assert fcs.describe_fault_en('距馬祖東引約45.3公里處斷纜') == \
        'Cable cut about 45.3 km from Dongyin, Matsu'
    assert fcs.describe_fault_en('距淡水沙崙約180公里處發生部分芯線受損障礙') == \
        'Partial fibre damage about 180 km from Shalun, Tamsui'


def test_describe_fault_en_unknown_returns_empty():
    # 地名不在表內、句型不符 → 留空，前端退回中文（不產生半中半英）
    assert fcs.describe_fault_en('距某某港約3公里處斷纜') == ''
    assert fcs.describe_fault_en('海纜維修中') == ''
    assert fcs.describe_fault_en(None) == ''


def test_alt_route_translation():
    seg = {'臺馬三號海纜': 'TM3'}
    assert fcs.translate_alt_route('臺馬三號海纜', seg) == 'TM3'
    assert fcs.translate_alt_route('APG、APCN-2', seg) == 'APG, APCN-2'
    assert fcs.translate_alt_route('未知海纜、APG', seg) == ''
    assert fcs.translate_alt_route('', seg) == ''


def test_committed_cable_status_has_english():
    data = json.loads((Path(__file__).resolve().parent.parent / 'docs' / 'cable_status.json')
                      .read_text(encoding='utf-8'))
    for f in data['faults']:
        assert 'description_en' in f and 'alt_route_en' in f

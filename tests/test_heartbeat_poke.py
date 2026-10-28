"""cloudflare/heartbeat-poke 只戳 pipeline-heartbeat.yml —— 改名或拿掉 workflow_dispatch
會讓外部排程器默默失效（Worker 只在 Cloudflare 的紀錄裡報 404），所以在這裡守住。"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_worker_targets_existing_dispatchable_workflow():
    worker = (ROOT / 'cloudflare' / 'heartbeat-poke' / 'worker.js').read_text(encoding='utf-8')
    name = re.search(r"const WORKFLOW = '([^']+)'", worker).group(1)
    wf = ROOT / '.github' / 'workflows' / name
    assert wf.exists(), name
    assert re.search(r'^\s*workflow_dispatch:', wf.read_text(encoding='utf-8'), re.M)


def test_worker_cron_offset_from_github_schedule():
    toml = (ROOT / 'cloudflare' / 'heartbeat-poke' / 'wrangler.toml').read_text(encoding='utf-8')
    wf = (ROOT / '.github' / 'workflows' / 'pipeline-heartbeat.yml').read_text(encoding='utf-8')
    worker_min = set(re.search(r'crons = \["([\d,]+) ', toml).group(1).split(','))
    gh_min = set(re.search(r"cron: '([\d,]+) ", wf).group(1).split(','))
    # 心跳是 cancel-in-progress：同一分鐘觸發會互相取消
    assert not worker_min & gh_min

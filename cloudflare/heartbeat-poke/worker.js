// 每 20 分鐘戳一次 pipeline-heartbeat.yml（workflow_dispatch）。
//
// GitHub 會丟掉大量 schedule 事件 —— 心跳自己的排程實測一天只觸發約 6 次，
// AIS 快照間隔因此維持在 ~4h。dispatch 不會被丟，所以由外部排程器負責「準時」，
// 「該不該派發 AIS／網路掃描、會不會擠掉排隊中的 update-data」仍全部由 repo 裡的
// src/pipeline_heartbeat.py 判斷。這支 Worker 只做一件事：叫醒心跳。
//
// 需要的 secret：GH_TOKEN —— fine-grained PAT，只授權本 repo，權限只開
// Actions: Read and write。設定方式見同目錄 README.md。

const WORKFLOW = 'pipeline-heartbeat.yml';

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(poke(env));
  },
};

async function poke(env) {
  const repo = env.REPO || 's0914712/taiwan-grayzone-monitor';
  const url = `https://api.github.com/repos/${repo}/actions/workflows/${WORKFLOW}/dispatches`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.GH_TOKEN}`,
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'grayzone-heartbeat-poke',
    },
    body: JSON.stringify({ ref: 'main' }),
  });
  if (res.status !== 204) {
    // 丟出錯誤讓 Cloudflare 的 Cron 事件紀錄顯示失敗（例如 token 過期 → 401）
    throw new Error(`dispatch ${WORKFLOW} failed: HTTP ${res.status} ${await res.text()}`);
  }
}

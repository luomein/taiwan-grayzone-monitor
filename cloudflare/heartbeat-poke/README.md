# heartbeat-poke — 讓 AIS 更新準時

## 為什麼需要

GitHub 在高負載時會丟掉 `schedule` 事件。`pipeline-heartbeat.yml` 設定每 20 分鐘一次，
實測一天只觸發約 6 次，所以 AIS 快照間隔一直在 ~4 小時（2026-10 中位數 3.97h、p90 5.6h）。
這直接造成：

- 海警船約 2 小時的金門限制水域巡弋，只有三到五成機會被快照看到
- SAR 暗船比對變差：比對模擬中，快照間隔約 3h 的期間配錯船 6.4%，約 4h 的期間 12%（兩段期間不完全可比，但方向一致）

`workflow_dispatch` 不會被丟。這支 Cloudflare Worker 每 20 分鐘呼叫一次心跳的
`workflow_dispatch`；**要不要派發 AIS／網路掃描、會不會擠掉排隊中的 `update-data.yml`，
仍由 repo 裡的 `src/pipeline_heartbeat.py` 判斷**，Worker 不含任何排程邏輯。

GitHub 自己的 `schedule` 保留當備援：Worker 停了，一切退回現在的行為。

## 部署（一次，約 10 分鐘）

### 1. 建一把只能做這件事的 GitHub token

GitHub → Settings → Developer settings → **Fine-grained personal access tokens** → Generate new token

- Repository access：**Only select repositories** → `s0914712/taiwan-grayzone-monitor`
- Permissions → Repository permissions → **Actions: Read and write**（其他全部維持 No access）
- Expiration：最長 1 年；到期前要換（過期時 Worker 的 Cron 紀錄會出現 HTTP 401）

### 2. 部署 Worker

用 Cloudflare 網頁（不需安裝任何東西）：

1. Cloudflare dashboard → Workers & Pages → Create → Create Worker，名稱 `grayzone-heartbeat-poke`
2. Edit code：貼上 `worker.js` 全文 → Deploy
3. Settings → Variables and Secrets → Add：
   - `GH_TOKEN`（類型 **Secret**）= 第 1 步的 token
   - `REPO`（類型 Text）= `s0914712/taiwan-grayzone-monitor`
4. Settings → Trigger events → Add → Cron Triggers → `2,22,42 * * * *`
5. Settings → Domains & Routes：關掉 workers.dev（這支不需要對外網址）

或用 CLI：`npx wrangler deploy` 後 `npx wrangler secret put GH_TOKEN`。

免費方案即可（Cron Triggers 免費；每天 72 次呼叫遠低於額度）。

### 3. 確認有效

- Worker → Logs／Cron events：每 20 分鐘一筆、無錯誤
- GitHub → Actions → Pipeline Heartbeat：觸發事件顯示 `workflow_dispatch`，大約每 20 分鐘一次
- 一兩天後，`📡 AIS update` commit 間隔應落在白天約 1 小時、夜間約 2 小時

## Actions 負荷

心跳 job 只 sparse checkout 一個檔、用 runner 內建的 `gh`、跑幾秒：每天 72 個。
真正的資料 run 只在逾期時派發，數量等於原本設計的排程（AIS 約 17 次/天、網路掃描約 12 次/天）。

## 停用

刪掉 Worker 或移除 Cron Trigger 即可，不需要改 repo。

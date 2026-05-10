# SuperTrend Alert

This repository runs `alert_script.py` via GitHub Actions.

## Existing workflow trigger

The workflow file `/home/runner/work/supertrend-alert/supertrend-alert/.github/workflows/alert.yml` already includes:

- `schedule` (every 5 minutes)
- `workflow_dispatch` (manual/API trigger)

No workflow change is required to support external cron dispatch.

## Set up external cron dispatch

### 1) Create a GitHub Personal Access Token (classic)

Create a PAT with:

- `repo` (private repo) or `public_repo` (public repo)
- `workflow`

Store it securely on your external machine, for example in a locked-down env file:

```bash
cat > ~/.supertrend_dispatch_env <<'EOF'
export GITHUB_PAT="YOUR_TOKEN"
EOF
chmod 600 ~/.supertrend_dispatch_env
```

### 2) Dispatch the workflow from your external machine

Use this endpoint:

- `POST /repos/Quek-Xin-Quan/supertrend-alert/actions/workflows/alert.yml/dispatches`

With body:

```json
{"ref":"main"}
```

You can dispatch directly:

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_PAT}" \
  https://api.github.com/repos/Quek-Xin-Quan/supertrend-alert/actions/workflows/alert.yml/dispatches \
  -d '{"ref":"main"}'
```

Or use the helper script in `scripts/dispatch_alert_workflow.sh`.

### 3) Create the cron entry (every 5 minutes)

Example:

```cron
*/5 * * * * . "$HOME/.supertrend_dispatch_env"; /bin/bash /absolute/path/to/supertrend-alert/scripts/dispatch_alert_workflow.sh >> /var/log/supertrend-dispatch.log 2>&1
```

### 4) Verify

Check Actions run history in GitHub and confirm run timestamps match your cron schedule.

## Optional hardening

- Keep retry logic enabled in the helper script
- Keep log output and monitor failures
- Retain GitHub `schedule` as fallback, while using external cron for timing accuracy

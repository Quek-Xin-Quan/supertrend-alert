#!/usr/bin/env bash
set -euo pipefail
# Ensure token is not exposed if caller enabled shell tracing.
set +x

OWNER="${OWNER:-Quek-Xin-Quan}"
REPO="${REPO:-supertrend-alert}"
WORKFLOW_FILE="${WORKFLOW_FILE:-alert.yml}"
REF="${REF:-main}"

if [[ -z "${GITHUB_PAT:-}" ]]; then
  echo "ERROR: GITHUB_PAT is required." >&2
  exit 1
fi

api_url="https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches"
payload="{\"ref\":\"${REF}\"}"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Dispatching ${OWNER}/${REPO}:${WORKFLOW_FILE} on ref=${REF}"

curl --silent --show-error --fail-with-body \
  --retry 3 \
  --retry-delay 2 \
  --retry-all-errors \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_PAT}" \
  "${api_url}" \
  -d "${payload}"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Dispatch request accepted."

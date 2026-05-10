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

safe_pattern='^[A-Za-z0-9._/-]+$'
for v in "$OWNER" "$REPO" "$WORKFLOW_FILE" "$REF"; do
  if [[ ! "$v" =~ $safe_pattern ]]; then
    echo "ERROR: Invalid characters detected in configuration values." >&2
    exit 1
  fi
done

api_url="https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches"
payload="$(printf '{"ref":"%s"}' "$REF")"
printf '[%s] Dispatching %s/%s:%s on ref=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$OWNER" "$REPO" "$WORKFLOW_FILE" "$REF"

curl --silent --show-error --fail-with-body \
  --retry 3 \
  --retry-delay 2 \
  --retry-all-errors \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_PAT}" \
  "${api_url}" \
  -d "${payload}"

printf '[%s] Dispatch request accepted.\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

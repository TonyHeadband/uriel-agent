#!/usr/bin/env bash
# Post-deploy check: one tool-backed chat turn through the public URL.
# Usage: URIEL_URL=https://uriel.example.com URIEL_SMOKE_KEY=... scripts/smoke.sh
set -euo pipefail
: "${URIEL_URL:?set URIEL_URL}" "${URIEL_SMOKE_KEY:?set URIEL_SMOKE_KEY (an admins service key)}"
curl -fsS "$URIEL_URL/livez" >/dev/null
resp=$(curl -fsS -X POST "$URIEL_URL/v1/chat" -H "X-API-Key: $URIEL_SMOKE_KEY" \
  -H 'content-type: application/json' -d '{"message":"What is the status of the homelab?"}')
echo "$resp" | python3 -c '
import json, sys
body = json.load(sys.stdin)
assert "homelab_status" in body["tools_used"], body
print("smoke OK:", body["reply"][:120])
'

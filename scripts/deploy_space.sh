#!/usr/bin/env bash
# Deploy the static Space in space/ to Hugging Face.
#
#   HF_TOKEN=hf_xxx  bash scripts/deploy_space.sh   [space-name]
#
# Requires a WRITE-scoped token (env HF_TOKEN, else ~/.cache/huggingface/token).
# Creates the static Space if it doesn't exist, then pushes space/ contents.
# The token is sent via an HTTP header (never written to git config/URL).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-vibethinker-webgpu}"
TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
[ -z "$TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ] && TOKEN="$(cat "$HOME/.cache/huggingface/token")"
[ -z "$TOKEN" ] && { echo "ERROR: no token. export HF_TOKEN=hf_xxx (write scope)"; exit 1; }

USER="$(curl -s https://huggingface.co/api/whoami-v2 -H "Authorization: Bearer $TOKEN" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("name",""))')"
[ -z "$USER" ] && { echo "ERROR: token invalid (whoami failed)"; exit 1; }
REPO="$USER/$NAME"
echo "deploying space -> $REPO"

# create (idempotent: 409 = already exists, which is fine)
CODE="$(curl -s -o /tmp/hf_create.json -w '%{http_code}' -X POST https://huggingface.co/api/repos/create \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"type\":\"space\",\"name\":\"$NAME\",\"sdk\":\"static\",\"private\":false}")"
if [ "$CODE" = "200" ] || [ "$CODE" = "201" ]; then echo "created space $REPO";
elif [ "$CODE" = "409" ]; then echo "space exists, updating";
else echo "ERROR creating space (HTTP $CODE):"; cat /tmp/hf_create.json; echo; exit 1; fi

# stage a clean copy and push (token via header, not stored)
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cp space/index.html space/bundle.js space/README.md "$TMP/"
git -C "$TMP" init -q
git -C "$TMP" checkout -q -b main
git -C "$TMP" add -A
git -C "$TMP" -c user.email=deploy@local -c user.name=deploy commit -qm "Deploy VibeThinker WebGPU Space"
# credentials provided via an in-memory helper (never written to config/URL/disk)
HF_USER="$USER" HF_PASS="$TOKEN" GIT_TERMINAL_PROMPT=0 \
  git -C "$TMP" -c credential.helper='!f(){ echo "username=$HF_USER"; echo "password=$HF_PASS"; };f' \
  push -f "https://huggingface.co/spaces/$REPO" main

echo
echo "✅ deployed: https://huggingface.co/spaces/$REPO"
echo "   live app:  https://${USER//[._]/-}-${NAME//[._]/-}.static.hf.space"

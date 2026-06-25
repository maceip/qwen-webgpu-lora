#!/usr/bin/env bash
# Rebuild the app bundle and sync it into the Hugging Face Space folder.
# Only the built JS is copied — space/index.html is maintained separately
# (HF-tailored copy + model-source defaults).
set -euo pipefail
cd "$(dirname "$0")/.."
node esbuild.config.mjs
cp docs/bundle.js space/bundle.js
echo "synced -> space/bundle.js ($(wc -c < space/bundle.js | tr -d ' ') bytes)"
echo "deploy: see space/README.md (push space/ to an HF static Space)"

#!/usr/bin/env bash
# Usage: deploy/push_space.sh <hf-username> [space-name]   (run `huggingface-cli login` first)
# Stages a clean copy of exactly what the Space needs and pushes it; creates the Space if missing.
set -euo pipefail
USER_NAME="${1:?hf username}"; SPACE="${2:-crowdlens}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STAGE="$(mktemp -d)"
huggingface-cli repo create "$SPACE" --type space --space_sdk docker -y >/dev/null 2>&1 || true
git clone "https://huggingface.co/spaces/$USER_NAME/$SPACE" "$STAGE/space"
cd "$STAGE/space"
cp "$ROOT"/{Dockerfile,.dockerignore,pyproject.toml} .
cp "$ROOT/deploy/space_README.md" README.md
cp -R "$ROOT/crowdlens" "$ROOT/.streamlit" "$ROOT/space_data" .
find . -name __pycache__ -prune -exec rm -rf {} + ; find . -name .DS_Store -delete
git add -A && git commit -m "Deploy CrowdLens demo" && git push
echo "Space: https://huggingface.co/spaces/$USER_NAME/$SPACE"

#!/usr/bin/env bash
# Usage: deploy/push_space.sh <hf-username> [space-name]
# Needs `huggingface_hub` and a login (`huggingface-cli login` / `hf auth login`, token with WRITE access).
# Stages a clean copy of what the Space needs, creates the Space if missing, and uploads it through the API.
set -euo pipefail
USER_NAME="${1:?hf username}"; SPACE="${2:-crowdlens}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STAGE="$(mktemp -d)/space"; mkdir -p "$STAGE"
cp "$ROOT"/{Dockerfile,.dockerignore,pyproject.toml} "$STAGE/"
cp "$ROOT/deploy/space_README.md" "$STAGE/README.md"
cp -R "$ROOT/crowdlens" "$ROOT/.streamlit" "$ROOT/space_data" "$STAGE/"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} + ; find "$STAGE" -name .DS_Store -delete
PY="${PYTHON:-python3}"
"$PY" - "$USER_NAME/$SPACE" "$STAGE" <<'PYEOF'
import sys
from huggingface_hub import HfApi, whoami
repo, stage = sys.argv[1:]
who = whoami()["name"]
print("logged in as:", who)
if who.lower() != repo.split("/")[0].lower():
    print(f"WARNING: logged in as '{who}' but target namespace is '{repo.split('/')[0]}'")
api = HfApi()
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="space", folder_path=stage, commit_message="Deploy CrowdLens demo")
print("Space: https://huggingface.co/spaces/" + repo)
PYEOF

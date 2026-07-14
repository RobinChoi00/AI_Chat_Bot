#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Usage: $0 <40-character git commit SHA>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_SHA="$1"
cd "$ROOT"

PREVIOUS_SHA="$(git rev-parse HEAD)"
SWITCHED=0

rollback() {
  local exit_code=$?
  if [[ $SWITCHED -eq 1 ]]; then
    echo "Deployment failed; rolling back to $PREVIOUS_SHA" >&2
    git checkout --detach "$PREVIOUS_SHA"
    docker compose build
    docker compose up -d --remove-orphans
  fi
  exit "$exit_code"
}
trap rollback ERR

git fetch --prune origin main
git cat-file -e "${TARGET_SHA}^{commit}"
git checkout --detach "$TARGET_SHA"
SWITCHED=1

mkdir -p db_data faiss_index uploaded_evidence data raw_data rc_audio_cache backups logs
python3 script/preflight.py
if [[ -f db_data/chat_history.db ]]; then
  ./script/backup_db.sh
else
  echo "No existing database yet; skipping pre-deploy backup."
fi

docker compose build --pull
docker compose up -d --remove-orphans

for attempt in $(seq 1 30); do
  if curl --fail --silent --show-error http://127.0.0.1:8000/health/ready >/dev/null \
    && curl --fail --silent --show-error http://127.0.0.1:3000/api/health >/dev/null; then
    trap - ERR
    docker image prune -f --filter "until=168h"
    echo "Deployment succeeded: $TARGET_SHA"
    exit 0
  fi
  sleep 5
done

echo "Readiness checks did not pass within 150 seconds." >&2
false

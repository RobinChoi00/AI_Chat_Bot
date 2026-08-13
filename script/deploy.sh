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

# Live catalog on the host; never let git checkout overwrite these.
HOST_OVERLAYS=(
  data/cleaned_osaki_products.csv
  raw_data/products_export.csv
)

protect_host_overlays() {
  local path
  for path in "${HOST_OVERLAYS[@]}"; do
    if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
      git update-index --skip-worktree "$path"
    fi
  done
}

is_host_overlay() {
  local path="$1" overlay
  for overlay in "${HOST_OVERLAYS[@]}"; do
    if [[ "$path" == "$overlay" ]]; then
      return 0
    fi
  done
  return 1
}

stash_blocking_changes() {
  local -a dirty=() to_stash=()
  local path

  protect_host_overlays

  while IFS= read -r path; do
    [[ -n "$path" ]] && dirty+=("$path")
  done < <({ git diff --name-only; git diff --cached --name-only; } | sort -u)

  for path in "${dirty[@]}"; do
    if ! is_host_overlay "$path"; then
      to_stash+=("$path")
    fi
  done

  if ((${#to_stash[@]} > 0)); then
    echo "Stashing host changes that would block checkout: ${to_stash[*]}"
    git stash push -m "deploy.sh: stash host dirty tree before ${TARGET_SHA}" -- "${to_stash[@]}"
  fi
}

rollback() {
  local exit_code=$?
  if [[ $SWITCHED -eq 1 ]]; then
    echo "Deployment failed; rolling back to $PREVIOUS_SHA" >&2
    stash_blocking_changes
    git checkout --detach "$PREVIOUS_SHA"
    protect_host_overlays
    docker compose build
    docker compose up -d --remove-orphans
  fi
  exit "$exit_code"
}
trap rollback ERR

git fetch --prune origin main
if ! git cat-file -e "${TARGET_SHA}^{commit}" 2>/dev/null; then
  git fetch --all --prune
fi
git cat-file -e "${TARGET_SHA}^{commit}"

stash_blocking_changes
git checkout --detach "$TARGET_SHA"
protect_host_overlays
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

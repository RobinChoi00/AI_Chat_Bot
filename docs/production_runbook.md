# Production runbook

This runbook is the release gate for the single-host Docker Compose deployment. A release is ready only when preflight, CI, health probes, backup verification, and the restore drill all pass.

## 1. Host baseline

- Supported baseline: current LTS Linux, Docker Engine + Compose v2, Git, Python 3.11+, curl, AWS CLI when S3 backups are enabled.
- Put Nginx, Caddy, or an equivalent load balancer in front of the app. Terminate TLS there and proxy customer traffic to `127.0.0.1:3000`; proxy backend API routes to `127.0.0.1:8000` only when they must be public.
- Allow inbound 80/443 only. Do not expose 3000, 8000, SSH, or Docker ports broadly. Restrict SSH to the operator/VPN range.
- Enable automatic security updates, disk encryption, time synchronization, and host-level audit/log shipping.
- Docker JSON logs rotate at 10 MB × 3 files per service in `docker-compose.yml`.

Create the persistent paths and give container UID/GID 1000 access:

```bash
mkdir -p db_data faiss_index uploaded_evidence data raw_data rc_audio_cache backups logs
sudo chown -R 1000:1000 db_data faiss_index uploaded_evidence data rc_audio_cache
sudo chmod 700 db_data uploaded_evidence rc_audio_cache backups
```

`raw_data` is mounted read-only. Do not change ownership if it is maintained by a separate ingestion job.

## 2. Secrets and configuration

```bash
cp .env.example .env
chmod 600 .env
openssl rand -hex 32  # ADMIN_SESSION_SECRET
openssl rand -hex 32  # ADMIN_API_KEY
```

The existing `ADMIN_USERNAME` and `ADMIN_PASSWORD` may be retained. Use different values for the two generated machine secrets. Store the authoritative values in a secrets manager; the host `.env` is a deployment copy only. Rotate admin and integration credentials immediately after staff changes or suspected exposure.

Required policy:

- `PUBLIC_BASE_URL` and `NEXT_PUBLIC_API_BASE_URL` are absolute HTTPS URLs.
- `CORS_ALLOWED_ORIGINS` contains exact storefront origins and never `*`.
- `TRUSTED_HOSTS` contains explicit public/internal hostnames and never `*`.
- `RC_WEBHOOK_VERIFICATION_TOKEN` matches the RingCentral webhook configuration.
- RingCentral credentials, transfer extension/number, and SMS sender are configured. If using `RC_USER_JWT_FILE`, mount that mode-`0600` file into the backend container and set the in-container path.
- `RC_WARRANTY_CLOSED_DATES` contains upcoming holidays and exceptional closures; review it monthly.
- `BACKUP_S3_URI` points to a private, versioned bucket with public access blocked.
- Configure S3 lifecycle retention: at least 35 daily and 12 monthly restore points, adjusted to the business retention policy.

Validate without printing secret values:

```bash
.venv/bin/python script/preflight.py
```

Preflight must return zero errors. A local-only backup warning is not acceptable for production.

Configure the GitHub `production` environment with `EC2_HOST`, `EC2_USERNAME`, `EC2_SSH_KEY`, and `EC2_HOST_FINGERPRINT`. Obtain and verify the host fingerprint through a trusted channel before saving it; do not learn it during the first deployment connection.

## 3. Release and rollback

Every change reaches production through this sequence:

1. Open a pull request and let `.github/workflows/ci.yml` pass.
2. Merge to `main`.
3. The deploy workflow sends that exact successful commit SHA to the host.
4. `script/deploy.sh` fetches and checks out that SHA, runs preflight, takes a verified backup, builds new images, starts the stack, and waits for both health probes.
5. If any command or readiness probe fails, the script checks out the previous SHA and rebuilds/restarts it automatically.

Manual invocation, when approved:

```bash
./script/deploy.sh "$(git rev-parse origin/main)"
```

Never deploy with an abbreviated SHA or uncommitted server edits. Review status after release:

```bash
docker compose ps
curl --fail --silent --show-error https://api.example.com/health/ready
curl --fail --silent --show-error https://api.example.com/rc/health
curl --fail --silent --show-error https://app.example.com/api/health
docker compose logs --since=10m backend frontend
```

Manual rollback:

```bash
git checkout --detach <last-known-good-40-character-sha>
docker compose build
docker compose up -d --remove-orphans
```

## 4. Backup, verification, and restore drill

Install the daily 03:15 host-local-time cron after setting `BACKUP_S3_URI`:

```bash
./script/install_backup_cron.sh
crontab -l
```

The job uses SQLite's online backup API, runs `PRAGMA integrity_check`, archives uploaded evidence, writes SHA-256 checksums, uploads with S3 server-side encryption, and keeps eight local restore points. The S3 lifecycle policy owns off-site retention.

Run and verify on demand:

```bash
./script/backup_db.sh
.venv/bin/python script/verify_backup.py backups/<timestamp>
```

Quarterly restore drill on a non-production copy:

```bash
docker compose stop backend frontend
.venv/bin/python script/restore_backup.py \
  --root "$PWD" \
  --backup "$PWD/backups/<timestamp>" \
  --confirm RESTORE
docker compose up -d
curl --fail http://127.0.0.1:8000/health/ready
curl --fail http://127.0.0.1:3000/api/health
```

The restore tool verifies checksums and SQLite integrity first. It preserves replaced data as `*.pre_restore_<timestamp>`; delete those copies only after business validation. Record backup timestamp, restore duration, ticket spot checks, evidence spot checks, and probe results. A backup is not considered healthy until this drill succeeds.

## 5. Monitoring and alerts

Poll both health endpoints every minute from outside the host. Alert when:

- readiness fails twice consecutively;
- HTTP 5xx exceeds 2% for five minutes;
- p95 request latency exceeds 3 seconds for ten minutes;
- disk use exceeds 70% warning or 85% critical;
- container restarts occur;
- the daily off-site backup object or success log is missing by 05:00;
- OpenAI/integration 401, 403, 429, or sustained timeout rates increase;
- `/rc/health` reports failed callbacks or any dead-letter callback;
- admin login 429s or repeated backend admin 401s spike.

Forward application and reverse-proxy logs to a remote log service. Correlate requests with `X-Request-ID`. Do not log message bodies, customer emails, phone numbers, credentials, uploaded filenames, or evidence paths.

## 6. Incident response

1. Declare severity and assign an incident owner.
2. Preserve logs and the relevant commit/image identifiers.
3. Contain: disable the affected integration, restrict ingress, or roll back. Do not destroy evidence.
4. If a secret may be exposed, rotate it at the provider and on the host, then restart affected services.
5. For data corruption, stop writers, verify a backup, restore, and validate tickets/evidence before reopening traffic.
6. Document timeline, customer impact, root cause, corrective actions, and owners.

Useful commands:

```bash
docker compose ps
docker compose logs --since=30m --timestamps backend frontend
docker inspect --format '{{json .State.Health}}' ai-chat-project-backend-1
df -h
du -sh db_data uploaded_evidence faiss_index backups
```

## 7. Routine operations

- Daily: confirm external probes and off-site backup completion.
- Weekly: review 5xx/429 trends, container restarts, disk growth, and admin authentication failures.
- Weekly: review thumbs-down customer feedback and add confirmed failures to `data/answer_quality_eval_cases.json`; run `script/evaluate_answer_quality.py` before prompt/model changes.
- Monthly: patch the host and dependencies, review `npm audit`/Python advisories, rotate any expiring provider credentials, and validate S3 lifecycle/versioning.
- Before changing `OPENAI_AGENT_MODEL`, run the candidate against representative production-derived cases in staging and compare factual accuracy, correct tool use, latency, and cost. Promote only on a measured gain.
- Quarterly: execute the restore drill and review CORS origins, trusted hosts, operators, retention, and incident contacts.
- Before scaling beyond one backend replica: migrate SQLite to managed PostgreSQL, uploads/audio to object storage, sessions/rate limits to Redis, and async email/sync/rebuild jobs to a durable queue.

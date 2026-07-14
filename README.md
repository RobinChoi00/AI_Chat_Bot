# Titan / Osaki AI Chat

Production-oriented customer chat and warranty intake system for Titan/Osaki storefronts. The backend is FastAPI with OpenAI-assisted routing, Shopify/tracking integrations, FAISS search, warranty workflow persistence, evidence uploads, and RingCentral/Freshdesk integrations. The frontend is a Next.js customer widget plus a protected warranty admin console.

## Services

- `backend` — FastAPI on loopback port `8000`; SQLite, FAISS, uploads, integrations.
- `frontend` — Next.js on loopback port `3000`; customer UI, embed UI, admin UI, server-side backend proxy.
- TLS reverse proxy — required in front of both services in production. Only ports 80/443 should be internet-facing.

## Local setup

Prerequisites: Python 3.11+, Node.js 20+, and Docker with Compose for the production topology.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
cd frontend
npm ci
cp .env.local.example .env.local
```

Fill the local environment values before starting. Never commit `.env` or `.env.local`.

```bash
# backend
PYTHONPATH="$PWD:$PWD/app" .venv/bin/uvicorn app.main:app --reload --port 8000

# frontend, from frontend/
npm run dev
```

## Quality gates

```bash
.venv/bin/pytest -q
.venv/bin/pip-audit -r requirements-prod.txt
.venv/bin/python script/validate_flowchart.py
.venv/bin/python script/smoke_test_warranty_flow.py
.venv/bin/python script/evaluate_answer_quality.py
.venv/bin/python -m compileall -q app config.py tests script

cd frontend
npm test
npm run lint
npm run typecheck
npm run build
npm audit --audit-level=moderate
```

CI runs these gates, builds both container images, and blocks remediable high/critical container CVEs on every pull request and every push to `main`. Production deployment is triggered only after the `main` CI workflow succeeds.

## Production

1. Copy `.env.example` to `.env` on the host and replace every required placeholder.
2. Keep the existing admin username/password, and generate `ADMIN_SESSION_SECRET` and `ADMIN_API_KEY` independently with `openssl rand -hex 32`.
3. Prepare and permission the persistent directories.
4. Run `.venv/bin/python script/preflight.py`.
5. Put a TLS reverse proxy in front of loopback ports 3000/8000.
6. Deploy an exact tested commit with `script/deploy.sh <40-character-commit-sha>`.

The full checklist, health probes, backup/restore drill, rollback, monitoring, and incident procedures are in [docs/production_runbook.md](docs/production_runbook.md).

## Health endpoints

- `GET /health/live` — backend process liveness only.
- `GET /health/ready` — backend database, indexes, and storage readiness; returns 503 when unusable.
- `GET /rc/health` — RingCentral credentials/configuration plus pending, failed, and dead-letter callback counts.
- `GET /api/health` — frontend process health.

## Security model

- Browser admin sessions are signed, HTTP-only, strict-SameSite, and expire after eight hours.
- Every Next.js admin API route verifies the session before adding the backend-only admin key.
- Backend admin APIs fail closed when `ADMIN_API_KEY` is missing and compare it in constant time.
- Production rejects wildcard CORS/trusted hosts and weak admin secrets.
- Uploads are rate-limited, streamed with a 20 MB limit, stored mode `0600`, and checked by file signature.
- Production containers run as non-root with dropped capabilities and bounded resources.
- RingCentral callbacks are authenticated, size/schema checked, deduplicated in a durable inbox, and retried after transient failures; active IVR state survives a backend restart.
- Warranty terminals use a resolution-first gate: customers complete the recommended steps before team-review controls appear, then explicitly report whether the issue was resolved. Self-resolved tickets are closed automatically instead of remaining in a replacement/shipping review queue.
- The admin completion dashboard reports self-service attempts, resolution rate, and escalations after troubleshooting so support savings can be measured.

SQLite and local uploads make this deployment intentionally single-host. Before horizontal scaling, move relational state to a managed database, evidence to object storage, rate-limit/session coordination to Redis, and background work to a durable queue.

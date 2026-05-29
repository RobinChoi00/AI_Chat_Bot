# Legacy Streamlit Frontend

> **Status: ARCHIVED — internal/dev use only**

These files are the original Streamlit-based frontend, preserved for reference.

## Why archived?

The production frontend has been migrated to **Next.js** (`frontend/`).  
Streamlit is no longer part of the production runtime path.

## Files

| File | Description |
|---|---|
| `app.py` | Customer-facing chat UI (Streamlit) |
| `admin_dashboard.py` | Admin analytics dashboard (Streamlit, reads SQLite directly) |

## Running locally (dev/debug only)

```bash
# Customer chat
streamlit run legacy/streamlit/app.py --server.port 8501

# Admin dashboard
streamlit run legacy/streamlit/admin_dashboard.py --server.port 8502
```

Both require a running FastAPI backend at `http://localhost:8000`.

## ⚠️ Do NOT start in production

The production `docker-compose.yml` uses Next.js on port 3000.  
Streamlit services (`frontend`, `admin_dashboard`) are commented out.

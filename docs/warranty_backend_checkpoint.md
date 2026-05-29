# Warranty Backend — Checkpoint

> Last updated: 2026-05-29  
> Status: **Backend complete. Frontend integration safe to begin.**

---

## 1. Completed Phases

| Phase | Description | Status |
|---|---|---|
| A | PDF flowchart → `data/warranty_flowchart.json` + `data/warranty_evidence_specs.json` + `script/validate_flowchart.py` | ✅ Done |
| B | Deterministic `WarrantyEngine` state machine (`app/warranty_workflow.py`) + ORM models (`app/warranty_models.py`) + unit tests | ✅ Done |
| C | Agent tool integration (`start_warranty_workflow`, `answer_warranty_question`, `attach_warranty_evidence`) in `app/agent_tools.py`; warranty-mode locking + structured logging in `app/main.py` | ✅ Done |
| D-lite | Evidence upload endpoint (`POST /api/v1/warranty/{ticket_id}/evidence`) + list endpoint + local file storage + DB metadata | ✅ Done |
| E-lite | Admin API endpoints (`GET/POST /admin/warranty/...`) with `X-Admin-Key` auth; admin is the **only** path that may set `approved`/`rejected` | ✅ Done |

---

## 2. Changed / Created Files

| File | Role |
|---|---|
| `data/warranty_flowchart.json` | Single source of truth for the warranty workflow decision tree |
| `data/warranty_evidence_specs.json` | Evidence type definitions and per-terminal requirements |
| `app/warranty_models.py` | SQLAlchemy ORM: `WarrantyTicket`, `WarrantyTurn`, `WarrantyEvidence` |
| `app/warranty_workflow.py` | `WarrantyEngine` — LLM-free, deterministic state machine |
| `app/warranty_router.py` | FastAPI router: evidence + admin endpoints |
| `app/agent_tools.py` | Three warranty tool functions + `WARRANTY_TOOL_SCHEMAS` subset |
| `app/main.py` | Imports, router registration, tool executor, warranty-mode locking, logging |
| `script/validate_flowchart.py` | Structural + business-rule validation of the flowchart JSON |
| `scripts/smoke_test_warranty_flow.py` | 5 end-to-end engine scenarios without LLM |
| `tests/test_warranty_flow.py` | 43 pytest unit tests |
| `docs/warranty_backend_checkpoint.md` | This file |
| `docs/warranty_api_contract.md` | API contract for frontend integration |

---

## 3. Endpoint List

### Customer-Facing

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/chat` | Main agent chat endpoint (existing) |
| `POST` | `/api/v1/warranty/{ticket_id}/evidence` | Upload evidence file for an open warranty ticket |
| `GET` | `/api/v1/warranty/{ticket_id}/evidence` | List evidence files attached to a ticket |

### Admin-Only (requires `X-Admin-Key` header)

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/warranty/tickets` | List tickets; filterable by `status`, `domain`, `limit`, `offset` |
| `GET` | `/admin/warranty/tickets/{ticket_id}` | Full ticket detail: turns + evidence |
| `POST` | `/admin/warranty/{ticket_id}/decision` | Record admin decision (only path to `approved`/`rejected`) |
| `POST` | `/admin/warranty/{ticket_id}/note` | Append admin note without changing status |

---

## 4. Warranty Tool List

| Schema Name | Python Function | When Called |
|---|---|---|
| `start_warranty_workflow` | `tool_start_warranty_workflow` | Customer reports defect / delivery damage / installation issue |
| `answer_warranty_question` | `tool_answer_warranty_question` | Every subsequent customer answer during an active workflow |
| `attach_warranty_evidence` | `tool_attach_warranty_evidence` | Terminal node requires evidence; customer acknowledges submission |

**`WARRANTY_TOOL_SCHEMAS`** — restricted schema used by the LLM when an active warranty ticket is open. Contains only: `answer_warranty_question`, `attach_warranty_evidence`, `escalate_to_human`, `get_warranty_or_policy`. Prevents the LLM from calling unrelated tools mid-workflow.

---

## 5. Test Results

```
43 passed in 0.26s
```

| Test Group | Count | Coverage |
|---|---|---|
| Original warranty flow scenarios (6) | 6 | Installation, delivery, defect, power/remote paths |
| Edge cases (E1–E5) | 5 | Invalid answer, terminal re-submit, index/label/case matching, admin decision |
| Extended scenarios (7–13) | 7 | Power/remote paths, yes/no branches, evidence lookup, no-LLM assertion |
| Phase C — Agent tools | 11 | Tool functions, schema names, WARRANTY_TOOL_SCHEMAS subset |
| Phase D-lite — Evidence recording | 4 | record_evidence CRUD + nonexistent ticket |
| Phase E-lite — Admin decisions | 10 | approved, rejected, need_more_info, admin_reviewing, closed, invalid, note |

---

## 6. Key Architecture Invariants

1. **`WarrantyEngine` is LLM-free.** Zero OpenAI calls in `warranty_workflow.py` — verified by `test_no_llm_call_in_workflow_engine`.
2. **Only `admin_decision()` may set `approved` or `rejected`.** Customer-facing chat and the workflow engine can never reach these statuses directly — verified by `test_admin_decision_only_approved_path`.
3. **Mid-workflow tool locking.** When `_active_warranty_ticket` is set, `main.py` passes `WARRANTY_TOOL_SCHEMAS` (not `TOOL_SCHEMAS`) to the LLM, blocking unrelated tool calls.
4. **No email sending.** All evidence records have `emailed=0`. A future email sweep is possible by querying `WarrantyEvidence.emailed == 0`.
5. **Path-traversal protected.** Evidence upload endpoint sanitises filenames and resolves the destination path before writing.
6. **Scope guard.** `AGENT_SYSTEM_PROMPT_STATIC` includes a `SCOPE GUARD` section that instructs the LLM to decline non-massage-chair queries.

---

## 7. Remaining Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `ADMIN_API_KEY` not set in production | High | Router returns HTTP 503 with clear error message if unset |
| Admin auth is a static API key, not JWT | Medium | `TODO` in `warranty_router.py` — replace before public launch |
| Evidence files grow unbounded on disk | Medium | No cleanup/rotation; recommend S3 or a cron purge after 90 days |
| Email not sent to `service@osakititan.com` | Low | All evidence rows queryable via `emailed=0`; can add email sweep in Phase F |
| Flowchart has ~4 nodes flagged "needs_review" | Low | `script/validate_flowchart.py` reports them; no blocking bugs found |
| Binary file upload not tested end-to-end via HTTP | Low | Unit tests cover `record_evidence()` engine logic; integration test needed |

---

## 8. Next Steps

### Before Frontend Integration
- [ ] Set `ADMIN_API_KEY` in server `.env` and confirm `/admin/warranty/tickets` returns 200
- [ ] Run `scripts/smoke_test_warranty_flow.py` on the production DB to confirm flowchart loads
- [ ] Review 4 "needs_review" flowchart nodes with Michael and update JSON if needed

### Phase F (Future)
- [ ] Email sweep: send `WarrantyEvidence` records with `emailed=0` to `service@osakititan.com`
- [ ] Replace `X-Admin-Key` with proper JWT authentication
- [ ] S3/cloud storage for uploaded evidence files
- [ ] Streamlit → Next.js frontend migration (backend contract is now stable)
- [ ] Shopify webhook integration for order verification inside warranty flow

# Warranty API Contract

> Version: 1.0 (Phase C + D-lite + E-lite)  
> Base URL: `https://your-server/`  
> All request/response bodies are `application/json` unless noted.

---

## 1. Warranty Tool Responses (Chat Layer)

These are the structured strings returned by the three warranty tools to the LLM. The LLM **must paraphrase** the `PROMPT` field to the customer but **must not** alter the meaning or make promises.

### `start_warranty_workflow` → `tool_start_warranty_workflow`

**Triggered when:** Customer reports a defect, delivery issue, or installation problem.

**Response prefix:** `WARRANTY_TICKET_STARTED`

```
WARRANTY_TICKET_STARTED
TICKET_ID: <uuid>
CURRENT_NODE: issue_type
NODE_TYPE: question
PROMPT: <question text>
OPTIONS (present these to customer):
  - answer_key=installation | Label: Installation Issue
  - answer_key=delivery     | Label: Delivery Issue
  - answer_key=defect       | Label: Product Defect

INSTRUCTION: Present the PROMPT to the customer in a warm, friendly tone.
When the customer responds, map their answer to the closest answer_key and call answer_warranty_question.
DO NOT make any warranty decision yourself. SUPPRESS_LEAD_FOOTER
```

---

### `answer_warranty_question` → `tool_answer_warranty_question`

**Triggered on:** Every customer answer during an active warranty ticket.

**Response (mid-workflow):** `WARRANTY_CONTINUE`

```
WARRANTY_CONTINUE
TICKET_ID: <uuid>
CURRENT_NODE: <node_id>
NODE_TYPE: question | instruction | question_text
PROMPT: <next question>
OPTIONS (present these to customer):
  - answer_key=yes | Label: Yes
  - answer_key=no  | Label: No

INSTRUCTION: Present PROMPT and OPTIONS to customer in a warm, friendly tone.
When they respond, call answer_warranty_question with the matching answer_key.
DO NOT make any warranty decision yourself. SUPPRESS_LEAD_FOOTER
```

**Response (terminal node reached):** `WARRANTY_TERMINAL_REACHED`

```
WARRANTY_TERMINAL_REACHED
TICKET_ID: <uuid>
ACTION: awaiting_admin | send_info | request_evidence | sales_handoff
TERMINAL_CLASS: awaiting_admin_review | send_info | awaiting_evidence | sales_handoff
PROMPT_FOR_CUSTOMER: <message to deliver>
EVIDENCE_REQUIRED: damage_photos, proof_of_purchase   (only if applicable)
EVIDENCE_SEND_TO: service@osakititan.com              (only if applicable)
INTERNAL_NOTE: <admin-only context, not shown to customer>

INSTRUCTION: Deliver the PROMPT_FOR_CUSTOMER verbatim.
DO NOT promise replacement, tech dispatch, compensation, refund, or any approval.
The prompt already says 'our team will review' — keep that language exactly.
AWAITING_ADMIN_REVIEW=TRUE
SUPPRESS_LEAD_FOOTER
```

**Response (answer mismatch):** `WARRANTY_ANSWER_MISMATCH`

```
WARRANTY_ANSWER_MISMATCH: Answer 'xyz' did not match any option at node 'issue_type'.
VALID OPTIONS:
  - answer_key=installation | Installation Issue
  - answer_key=delivery     | Delivery Issue
  - answer_key=defect       | Product Defect
INSTRUCTION: Ask the customer to clarify which option they meant and retry.
```

---

### `attach_warranty_evidence` → `tool_attach_warranty_evidence`

**Triggered when:** Terminal node requires evidence and customer acknowledges submission.

```
EVIDENCE_NOTED
TICKET_ID: <uuid>
EVIDENCE_TYPE: damage_photos
FILENAME: photo.jpg
RECORD_ID: 42
INSTRUCTION: Let the customer know their evidence has been noted.
Ask them to upload the file via the evidence upload link or email it to service@osakititan.com.
```

---

## 2. Ticket Status Values

| Status | Set by | Meaning |
|---|---|---|
| `in_progress` | WarrantyEngine | Workflow is ongoing, waiting for customer answer |
| `awaiting_admin_review` | WarrantyEngine | Terminal reached requiring admin action |
| `awaiting_evidence` | WarrantyEngine | Terminal requesting evidence upload only |
| `send_info` | WarrantyEngine | Self-service terminal, no admin needed |
| `sales_handoff` | WarrantyEngine | Customer routed to sales |
| `admin_reviewing` | Admin API only | Admin has picked up the ticket |
| `need_more_information` | Admin API only | Admin needs more info from customer |
| `resolved` | Admin API only | Admin set decision to approved/rejected/closed |

**IMPORTANT:** `approved` and `rejected` are NEVER set as the ticket status directly. They are stored in `admin_decision` field. When admin sets decision to `approved`, `rejected`, or `closed`, the ticket `status` becomes `resolved`.

---

## 3. Evidence Upload Endpoint

### `POST /api/v1/warranty/{ticket_id}/evidence`

Upload an evidence file for an open warranty ticket.

**Auth:** None (customer-facing)

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `evidence_type` | string (form) | Yes | One of the allowed evidence type keys (see below) |
| `file` | file | Yes | The evidence file |

**Allowed file types:** `.jpg`, `.jpeg`, `.png`, `.pdf`, `.mp4`, `.mov`  
**Max file size:** 20 MB

**Allowed `evidence_type` values:**

| Key | Description |
|---|---|
| `damage_photos` | Photos of visible damage |
| `video_of_issue` | Video demonstrating the fault |
| `proof_of_purchase` | Receipt, invoice, or order confirmation |
| `photo_of_chair` | General photo of the chair |
| `photo_of_defect` | Close-up of specific defect |
| `proof_of_delivery` | Delivery receipt or signature page |
| `assembly_photo` | Photo of assembly state |
| `remote_photo` | Photo of the remote control |
| `other` | Any other supporting document |

**Response `200 OK`:**

```json
{
  "evidence_id": 42,
  "ticket_id": "3f9a1234-...",
  "ticket_status": "awaiting_admin_review",
  "evidence_type": "damage_photos",
  "original_filename": "front_damage.jpg",
  "customer_email": "customer@example.com",
  "mime_type": "image/jpeg",
  "file_size_bytes": 245760
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `404` | Ticket not found |
| `413` | File exceeds 20 MB |
| `422` | Disallowed file extension |

**Guarantees:**
- Email is **NOT** sent to `service@osakititan.com` — `emailed` flag stays `0`.
- Filenames are sanitised before saving (path-traversal protection).
- The saved path is always under `uploaded_evidence/warranty/{ticket_id}/`.

---

### `GET /api/v1/warranty/{ticket_id}/evidence`

List all evidence files attached to a ticket.

**Auth:** None (customer-facing)

**Response `200 OK`:**

```json
{
  "ticket_id": "3f9a1234-...",
  "ticket_status": "awaiting_admin_review",
  "evidence": [
    {
      "id": 42,
      "ticket_id": "3f9a1234-...",
      "evidence_type": "damage_photos",
      "file_path": "/app/uploaded_evidence/...",
      "original_filename": "front_damage.jpg",
      "mime_type": "image/jpeg",
      "file_size_bytes": 245760,
      "emailed": false,
      "created_at": "2026-05-29T12:00:00"
    }
  ]
}
```

---

## 4. Admin Ticket Endpoints

All admin endpoints require the `X-Admin-Key` header set to the `ADMIN_API_KEY` environment variable.

> **TODO:** Replace static API key with JWT / OAuth2 before public launch.

### `GET /admin/warranty/tickets`

List warranty tickets with optional filters.

**Query params:**

| Param | Type | Description |
|---|---|---|
| `status` | string | Filter by ticket status |
| `domain` | string | Filter by domain (e.g. `osaki.com`) |
| `limit` | int | Max results (default 50, max 200) |
| `offset` | int | Pagination offset |

**Response `200 OK`:**

```json
{
  "total": 3,
  "offset": 0,
  "tickets": [ { "ticket_id": "...", "status": "awaiting_admin_review", ... } ]
}
```

---

### `GET /admin/warranty/tickets/{ticket_id}`

Full ticket detail including all Q&A turns and evidence files.

**Response `200 OK`:**

```json
{
  "ticket": { "ticket_id": "...", "status": "awaiting_admin_review", ... },
  "turns":  [ { "node_id": "issue_type", "answer_key": "defect", ... } ],
  "evidence": [ { "evidence_type": "damage_photos", ... } ]
}
```

---

### `POST /admin/warranty/{ticket_id}/decision`

Record an admin decision. **This is the ONLY endpoint that may result in `approved` or `rejected` stored in `admin_decision`.**

**Request body:**

```json
{
  "decision": "approved",
  "note": "Approved for tech visit. Scheduling within 5 business days.",
  "customer_message": "A technician will contact you within 5 business days.",
  "decided_by": "ops_team"
}
```

**Allowed `decision` values:**

| Value | Ticket status after | Meaning |
|---|---|---|
| `admin_reviewing` | `admin_reviewing` | Admin picked up, still reviewing |
| `need_more_information` | `need_more_information` | Needs more info; ticket stays open |
| `approved` | `resolved` | Warranty action approved |
| `rejected` | `resolved` | Warranty claim rejected |
| `closed` | `resolved` | Case closed without action |

**Response `200 OK`:**

```json
{ "ticket": { "ticket_id": "...", "status": "resolved", "admin_decision": "approved", ... } }
```

**Error responses:**

| Status | Condition |
|---|---|
| `401` | Missing or invalid `X-Admin-Key` |
| `404` | Ticket not found |
| `422` | Invalid decision value |
| `503` | `ADMIN_API_KEY` not configured in environment |

---

### `POST /admin/warranty/{ticket_id}/note`

Append a note to a ticket without changing its status.

**Request body:**

```json
{
  "note": "Called customer. Waiting for photos to be emailed.",
  "added_by": "agent_sarah"
}
```

**Response `200 OK`:**

```json
{ "ticket": { "ticket_id": "...", "admin_note": "[agent_sarah] Called customer...", ... } }
```

---

## 5. Forbidden Customer-Facing Promises

The LLM and customer-facing chat layer **MUST NEVER**:

| Forbidden action | Why |
|---|---|
| Approve a warranty claim | Only `POST /admin/warranty/{id}/decision` with `decision=approved` can do this |
| Promise a replacement part | Admin reviews first; outcome is not guaranteed |
| Promise a technician dispatch | Admin decision required; scheduling not automatic |
| Promise a refund or compensation | Requires admin approval and business sign-off |
| Set ticket status to `approved` or `rejected` | These statuses are exclusively set by admin endpoints |
| Skip the `answer_warranty_question` tool | Every step must go through the engine; no shortcuts |
| Make a warranty decision based on RAG | The flowchart JSON is the only source of truth for branching |

**The correct customer-facing language when reaching an admin-review terminal:**

> "I've captured all the details for your case. Our team will review it and follow up with you shortly."

Do NOT say "we will replace it", "a technician will be sent", or "you will receive a refund".

---

## 6. System Prompt Guardrails (for reference)

The `AGENT_SYSTEM_PROMPT_STATIC` in `app/main.py` includes:

- **SCOPE GUARD**: Declines all non-massage-chair topics (cooking, sports, politics, coding, travel, etc.) with a standard message.
- **WARRANTY WORKFLOW RULES (W1–W5)**:
  - W1: Recognise `WARRANTY_TICKET_STARTED`, `WARRANTY_CONTINUE`, `WARRANTY_TERMINAL_REACHED` prefixes.
  - W2: Paraphrase `PROMPT` to the customer without changing the meaning.
  - W3: Map customer answer to `answer_key` and call `answer_warranty_question`.
  - W4: On `AWAITING_ADMIN_REVIEW=TRUE`, use "our team will review" language only.
  - W5: Never skip `answer_warranty_question` to jump to a conclusion.

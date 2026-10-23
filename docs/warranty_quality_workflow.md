# Warranty outcome and knowledge review workflow

The quality workflow measures customer-confirmed results separately from claim
approval and administrative closure. It does not send follow-up emails or
change a warranty decision.

## Customer result updates

After a troubleshooting result is recorded, the existing endpoint returns an
optional `followup_token`. It is a 30-day HMAC capability bound to that ticket,
its session, and the follow-up purpose. `ADMIN_SESSION_SECRET` must already be
configured with at least 32 characters. When signing is unavailable, the base
warranty workflow continues and the optional follow-up control is absent.

The customer can expand **Update your result** after a defect or installation
consultation has resolved. The browser retains only the temporary capability,
ticket ID, issue type and session ID in this tab's session storage so it works
after a reload. No contact details are added to this storage. Clearing browser
storage or starting another session can remove this access; there is no new
automated outreach or cross-device identity system.

`POST /api/v1/warranty/{ticket_id}/follow-up` accepts:

```json
{
  "token": "<signed follow-up capability>",
  "outcome": "still_broken",
  "event_id": "<new UUID for this update>"
}
```

`outcome` is `fixed` or `still_broken`. Reuse the event ID when retrying the same
request. Updates are rate-limited and append observations; they do not reopen a
claim or promise a repair. A customer needing more service must request support
through the usual case or contact flow.

## Reading results

`GET /admin/warranty/quality?all_time=true` requires `X-Admin-Key`. Optional
`days=1..180` and `domain` filters use the ticket-creation cohort, matching the
warranty dashboard. The default is 30 days; `all_time=true` includes all stored
tickets. Fields are:

| Field | Meaning |
|---|---|
| `latest_confirmed_fixed` | Latest customer observation says fixed |
| `latest_still_broken` | Latest result says not fixed or unable to attempt |
| `outcome_unknown` | No customer result was recorded |
| `followed_up` | Tickets with at least one later customer follow-up |
| `reopened_after_fix` | Tickets with a not-fixed observation after a confirmed fix; this is a quality recurrence, not a changed claim status |
| `confirmed_fix_rate_all_pct` | Latest fixed / all selected consultations |
| `confirmed_fix_rate_reported_pct` | Latest fixed / consultations with known results |

Existing explicit self-service outcomes are used as a legacy baseline. An
approval, rejection, restart, or closed status alone is never treated as a fix.
Historical unknown outcomes cannot be reconstructed from closure status.

## Publishing verified troubleshooting knowledge

Raw customer conversations are not automatically added as trusted instructions.
An operator must prepare a contact-free, source-grounded draft and explicitly
approve it using the existing admin authentication.

1. `POST /admin/warranty/knowledge-reviews` creates a **pending** draft:

   ```json
   {
     "title": "Horizon 4D remote response",
     "category": "remote",
     "model_names": ["Horizon 4D"],
     "diagnostic": "Remote does not respond after startup.",
     "customer_steps": ["Turn off the chair and unplug it for one minute."],
     "source_url": "https://support.example.com/manuals/horizon",
     "review_until": "2027-01-31"
   }
   ```

   This is a schema example, not an approved repair recommendation. Use the
   actual current manufacturer documentation and a support specialist's review.
   Customer contact details, unrestricted model applicability, expired reviews,
   internal electrical/disassembly instructions, and source URLs containing
   credentials or query tokens are rejected.

2. `GET /admin/warranty/knowledge-reviews?status=pending` lists submissions.
3. `POST /admin/warranty/knowledge-reviews/{id}/decision` with
   `{"decision":"approved","reviewed_by":"support-reviewer"}` activates a
   reviewed draft. Use `rejected` to reject or revoke it.
4. Approval and revocation invalidate knowledge and embedding caches. Only
   approved, unexpired entries enter retrieval. Their explicit model list is
   enforced; unknown or different models do not receive these steps. Include
   known naming variants explicitly when they are truly applicable.

Every retrieval rechecks the active review records in the shared database.
Other application workers therefore observe a rejection or expiry on their
next retrieval even if their local cache was not explicitly cleared. Semantic
indexes verify the current corpus fingerprint before reusing ready vectors.

Source URL, review date, expiry, and model applicability remain attached to the
knowledge entry. Submit an updated draft when the manual or approved procedure
changes; do not edit historical raw conversations to make an answer appear
correct. Evaluate changed entries against held-out relevant and cross-model
cases before deploying them.

## Validation

```bash
PYTHONPATH="$PWD:$PWD/app" .venv/bin/pytest -q tests/test_warranty_quality.py tests/test_warranty_knowledge.py tests/test_warranty_embeddings.py
cd frontend
npm test -- components/warranty/OutcomeFollowup.test.tsx lib/warrantyFollowup.test.ts
npm run typecheck
```

Tests use in-memory databases and synthetic cases. They establish workflow
correctness, not a measured increase in real customer resolution rate.

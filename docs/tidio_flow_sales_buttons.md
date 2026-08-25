# Tidio Flow — Sales AI buttons

Tidio **cannot** build Decision (quick reply) buttons dynamically from an API
JSON array. Buttons in the Flow editor are static. This project bridges that
with three layers that work together.

## What the API returns

`POST /api/v1/sales/tidio/turn`

| Field | Use in Flow |
|-------|-------------|
| `reply_plain` | **Send a chat message** (includes numbered menu) |
| `quick_replies[]` | Reference / debugging |
| `button_1_label` … `button_5_label` | Mirror labels in a static Decision node |
| `button_1_payload` … `button_5_payload` | Optional: send as next turn `payload` |
| `button_1_url` … `button_5_url` | Link button when payload is `open:https://…` |
| `button_count` | How many choices this turn |
| `next_action` | `reply` / `transfer_operator` / `warranty_redirect` |
| `resolved_from_button` | Visitor typed `1` or tapped a prior label |

Header (prod): `X-Tidio-Turn-Secret: <TIDIO_TURN_SECRET>`

## Recommended Flow (loop)

1. **Trigger** — visitor opens chat / says hi  
2. **Ask a question** — “How can I help?” → save as `visitor_message`  
3. **API call** → `POST …/api/v1/sales/tidio/turn`  
   Body:
   ```json
   {
     "contact_id": "{{contact.id}}",
     "message": "{{visitor_message}}",
     "session_id": "tidio:{{contact.id}}"
   }
   ```
4. Map outputs: `reply_plain`, `next_action`, `button_*`, `session_id`  
5. **Send a chat message** → `{{reply_plain}}`  
6. Branch on `next_action`:
   - `transfer_operator` → Transfer to operator  
   - `warranty_redirect` → End (warranty copy already in reply)  
   - `reply` → **Ask a question** again (“Your choice or message”) → loop to step 3  

Visitors can:

- type free text, or  
- reply `1` / `2` / `3`, or  
- type the exact button label  

The turn endpoint resolves those to the stored payload and re-runs Sales AI.

## Optional: real Tidio buttons (static mirror)

After step 5, add **Decision (quick replies)** with the **same labels** you
usually return (example post-recommend set):

1. Shop this chair  
2. Email me this pick  
3. Prefer stronger  
4. Visit showroom  
5. Talk to a human  

Each branch → API call with:

```json
{
  "contact_id": "{{contact.id}}",
  "session_id": "tidio:{{contact.id}}",
  "message": "Shop this chair",
  "payload": "open:{{button_1_url}}"
}
```

For label-only branches, omit `payload` and set `message` to the button title —
the server matches it against `last_quick_replies`.

**Decision (buttons)** allows max **3** items and can open a URL — use for
Shop / Financing / Showroom when `button_N_url` is set.

## Clarify path (budget / height / …)

Either:

- rely on the numbered menu in `reply_plain`, or  
- add static Decision nodes: Under $3,000 / $3–5k / … that call turn with
  `payload: "recommend:budget:under_3000"` etc.

## Local smoke

```bash
curl -s localhost:8000/api/v1/sales/tidio/turn \
  -H 'Content-Type: application/json' \
  -d '{"contact_id":"demo","message":"recommend a chair"}' | jq '.reply_plain,.button_count,.button_1_label'
```

Then send `"message":"1"` with the same `session_id` / `contact_id`.

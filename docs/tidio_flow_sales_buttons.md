# Tidio Flow — Sales AI (free plan / no webhook upgrade)

Paid **Webhooks** are optional. Live chat works with **Flow + API call** only.

## 1. API Call (already done)

- URL: `https://api.osakichair.com/api/v1/sales/tidio/turn`
- Header: `X-Tidio-Turn-Secret` = same value as server `TIDIO_TURN_SECRET`
- Map outputs: `reply_plain`, `next_action`, `flow_stage`, `sales_session_id`,
  `button_1_label` … `button_5_label`, `button_1_url`

## 2. Core loop

```
Trigger → Ask (visitor_message)
       → API call
       → Send {{sales_ai_reply}}
       → Condition next_action
            transfer_operator → Transfer to operator
            warranty_redirect → End
            reply → Condition flow_stage → (static buttons OR Ask again) → API
```

Visitors can always type `1` / `2` or the button label (server resolves it).

## 3. Free-plan real chips — Condition on `flow_stage`

After Send, add **Condition** on `flow_stage`, then **Decision (quick replies)**
with these **exact** labels (must match API):

### `menu`
- Recommend a chair → API `message=recommend` or `payload=recommend`
- Check a price → `message=price`
- Availability / stock → `message=stock`
- Compare two models → `message=compare`
- Talk to a human → `message=talk to a human`

### `ask_height` (first question — fit before budget)
- Under 5'4" → `payload=recommend:height:petite`
- 5'4"–5'11" → `payload=recommend:height:average`
- 6'0"–6'2" → `payload=recommend:height:tall`
- 6'3"+ → `payload=recommend:height:extra_tall`

### `ask_weight`
- ≤180 lb → `payload=recommend:weight:le180`
- 181–220 lb → `payload=recommend:weight:181_220`
- 221–260 lb → `payload=recommend:weight:221_260`
- 261–300 lb → `payload=recommend:weight:261_300`
- 301+ lb → `payload=recommend:weight:301_plus`

### `ask_space` (doorway / room)
- No space issue → `payload=recommend:space:none`
- Small room → `payload=recommend:space:small_room`
- Narrow doorway → `payload=recommend:space:narrow_door`

### `ask_goal` (max useful set)
- Neck & shoulders → `payload=recommend:goal:neck`
- Lower back → `payload=recommend:goal:lower_back`
- Upper back → `payload=recommend:goal:upper_back`
- Foot & calf → `payload=recommend:goal:feet`
- Full-body relax → `payload=recommend:goal:full_body`

### `ask_budget` (optional refine after the tier list)
- Under $3,000 → `payload=recommend:budget:under_3000`
- $3,000–$4,999 → `payload=recommend:budget:3000_4999`
- $5,000–$6,999 → `payload=recommend:budget:5000_6999`
- $7,000–$9,999 → `payload=recommend:budget:7000_9999`
- $10,000+ → `payload=recommend:budget:10000_plus`

### `recommend` (Value / Mid / Premium list, or focused pick if budget set)
- Specs for … → `payload=specs:…`
- Email me these picks → `message=Email me this pick`
- Under $3,000 / Mid bands → `payload=recommend:budget:…` (narrow)
- Visit showroom → `payload=cta:showroom`
- Talk to a human → `message=talk to a human`

**Decision (buttons)** max 3 + can open URL → use for Shop / Financing / Showroom
when `button_N_url` is set.

Each Decision branch → same API call with `contact_id`, `session_id`,
`domain=osakiusa.com`, plus `message` and/or `payload` as above → Send reply → loop.

## 4. Server `.env` (no extra Tidio fee)

On EC2 `~/AI_Chat_Bot/.env` (then `docker compose up -d backend`):

```bash
TIDIO_ENABLED=1
TIDIO_DOMAIN=osakiusa.com
TIDIO_TURN_SECRET=<same as Flow header>
TIDIO_PUBLIC_KEY=<Project data → Public key>   # free
# OpenAPI id/secret optional until webhooks/tickets
# TIDIO_WEBHOOK_SECRET / OPERATOR_ID — skip until paid webhooks
```

Check: `curl -s https://api.osakichair.com/api/v1/sales/tidio/health`

Expect `turn_secret_set: true` when secret is loaded by the container.

## 5. Skipped (costs money)

- Webhooks stack
- Full OpenAPI CRM features that require Plus (ticket push can wait)

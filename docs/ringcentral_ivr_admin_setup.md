# RingCentral main-line IVR — admin setup checklist

Use this when configuring the **888-848-2630** (ext. 3 warranty) call flow in the RingCentral Admin Portal.  
Our **Automated Voice App** (webhook IVR) only runs when the call is routed to the warranty app extension — not when callers sit in a generic queue that overflows to sales.

---

## Problem we are fixing (team feedback)

- After hours, callers hear **Sales vs Technical** with no **Warranty** option.
- Callers wait on hold, then get **Sales with no warning** that warranty is closed.
- No **warranty hours** or **call-back** guidance.
- Transfers to sales happen **without announcement**.

---

## Target call flow

### When warranty is **OPEN** (Mon–Fri 10:00 AM – 6:00 PM CST)

1. Main greeting → **Press X for Warranty** (separate from Sales and Technical).
2. Route to **warranty queue / extension** (or our Voice App if used during open hours).
3. Our app plays: *“Connecting you to the next available warranty specialist… have invoice/order ready…”* then forwards.

### When warranty is **CLOSED** (evenings, weekends)

1. Main greeting → **Press X for Warranty**.
2. Route **directly** to the **Osaki Warranty Voice App** (webhook URLs on EC2) — **do not** send to a hold queue.
3. **Do not** overflow closed warranty calls to Sales silently.
4. If you must offer Sales after hours, play this **before** transfer:  
   *“Warranty is closed. We are now transferring you to sales for non-warranty questions only.”*

---

## Main menu (recommended)

| Key | Label | Routes to |
|-----|--------|-----------|
| 1 | **Warranty** (installation, delivery, defect) | Warranty Voice App (closed) or warranty queue (open) |
| 2 | Sales | Sales queue |
| 3 | Technical support | Tech queue (if distinct from warranty) |

> Jose’s feedback: customers often pick Sales for warranty because **Warranty was missing**. Add **Warranty** as its own option.

---

## Hours to announce (recorded or TTS on main line)

| Team | Hours (default) |
|------|------------------|
| **Warranty phone** | Mon–Fri, 10:00 AM – 6:00 PM CST |
| **Sales** | Configure in `SALES_BUSINESS_HOURS` in EC2 `.env` (e.g. Sat hours if applicable) |

Weekend note for main-line greeting (optional):

> “Warranty phone support is closed on weekends. Sales may be available Saturday — warranty callbacks are weekdays only.”

---

## Voice App webhook URLs (EC2)

Register on the **Application Extension** (IVR App):

| Event | URL |
|-------|-----|
| Call entered | `POST https://api.osakichair.com/rc/on-call-enter` |
| Command update | `POST https://api.osakichair.com/rc/on-command-update` |
| Call exit | `POST https://api.osakichair.com/rc/on-call-exit` |

TTS audio: `GET https://api.osakichair.com/rc/audio/{key}.wav`

Set in EC2 `.env`:

```bash
RC_WEBHOOK_VERIFICATION_TOKEN=<same token as RingCentral Developer Console>
PUBLIC_BASE_URL=https://api.osakichair.com
RC_WARRANTY_TRANSFER_EXTENSION=3
RC_SMS_FROM_NUMBER=<RingCentral SMS-capable E.164 number>
```

The callback is acknowledged only after it has been saved to the durable inbox.
Exact duplicate callbacks are processed once, transient RingCentral API failures
use bounded backoff, and active call state is restored from SQLite after a
backend restart. Monitor `GET /rc/health`; any `dead_letter` count requires an
operator review before the affected call can be considered complete.

---

## What our app does (after routing is correct)

| When | Behavior |
|------|----------|
| **Closed** | Says department is closed, states hours, next open time, asks for invoice/docs ready, SMS link after call, automated issue menu |
| **Open** | Says connecting to warranty specialist, then forwards |
| **Sales handoff in flowchart** | Open: announces transfer to sales. Closed: **no** silent sales transfer |
| **Call end (closed)** | SMS + email to `service@osakititan.com` |

---

## Software E2E simulation (no live call)

When RC ApplicationExtension is still waiting, verify our IVR logic with:

```bash
# On EC2
cd ~/AI_Chat_Bot
docker compose exec -T backend python script/run_rc_ivr_e2e_sim.py
# or
python3 script/check_rc_ivr_readiness.py --simulate
```

This walks: call-enter → issue menu → digit `3` (defect) and asserts a workflow ticket is created with `channel=phone`.

Live phone E2E still requires the checklist below (RC activation + Roman routing).

---

## Checklist for Roman / phone admin

- [ ] Main IVR has **Warranty** as its own key (not only Sales / Technical).
- [ ] After hours: Warranty key → **Voice App extension** (not warranty hold queue).
- [ ] Remove or disable **overflow to Sales** on closed warranty queue.
- [ ] Any Sales overflow plays **closed + transferring to sales** message first.
- [ ] Main greeting mentions **warranty hours** and **call back next open day**.
- [ ] Keep “have invoice / order / ticket ready” prompt (team liked this).
- [ ] Verify `RC_WEBHOOK_VERIFICATION_TOKEN` on server matches RC app.
- [ ] Verify `/rc/health` is `ok` with zero dead-letter events.
- [ ] Confirm `/rc/health` `last_webhook_received_at` updates after a test call.
- [ ] Simulate one duplicate callback and one backend restart during a test call.
- [ ] Test after close: Cong/Jose/Ryan scenario — should hear **closed + hours**, not 5‑minute hold → Sales.

---

## Test script (after hours)

1. Call warranty line after 6 PM CST (or Saturday).
2. Press **Warranty**.
3. Expect within ~30 seconds:  
   *“Our warranty service department is closed… hours… call back…”*
4. Complete or hang up → SMS with resume link to caller mobile.
5. Confirm **no** unexplained transfer to Sales.

---

## Contact

Backend / Voice App: Robin (`robin.c@osakititan.com`)  
RingCentral admin: Roman Medrano (Ext 17)

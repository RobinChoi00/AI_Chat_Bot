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

1. Main greeting explains extensions: **Press 2 for Sales**, **Press 3 for Warranty**.
2. Route **directly** to the **Osaki Warranty Voice App** (webhook URLs on EC2) — **do not** send to a hold queue.
3. Our app plays the same directory first (`press 2` sales / `press 3` warranty).
4. **Press 3** → after-hours warranty script (closed + hours + docs) then the issue menu: **1 setup**, **2 sales/delivery**, **3 defect**.
5. **Press 2** (department or issue menu) → announced transfer to sales (ext.2). **Do not** overflow closed warranty calls to Sales silently.
6. If the main-line IVR must offer Sales after hours, play this **before** transfer:  
   *“Warranty is closed. We are now transferring you to sales for non-warranty questions only.”*

---

## Main menu (recommended)

Match published extensions on **888-848-2630**:

| Key | Label | Routes to |
|-----|--------|-----------|
| 2 | Sales | Sales queue (ext.2) |
| 3 | Warranty (installation, defect; delivery is sales) | Warranty Voice App (closed) or warranty queue (open) |

> Do **not** use 1=Warranty / 2=Sales / 3=Technical on the main line — callers confuse those keys with ext.2 sales and ext.3 warranty. Our after-hours Voice App uses 2/3 for the department menu. After they press 3, the issue menu keeps **2 = sales/delivery** so it does not collide with the published sales extension.

> Jose’s feedback: customers often pick Sales for warranty because **Warranty was missing**. Keep **Warranty** as its own option (key **3**).

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
RC_SALES_TRANSFER_EXTENSION=2
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
| **Closed** | Department menu first (2=sales, 3=warranty). Press 3: closed + hours + invoice/docs + SMS link + issue menu (1=setup, 2=sales/delivery, 3=defect). Press 2: announced sales transfer (ext.2) |
| **Open** | Says connecting to warranty specialist, then forwards |
| **Sales handoff in flowchart** | Open: announces transfer to sales. Closed: **no** silent sales transfer |
| **Call end (closed)** | SMS + email to `service@osakititan.com` (skipped after a sales forward) |

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

This walks: call-enter → department menu → digit `3` (warranty) → issue menu → digit `3` (defect) and asserts a workflow ticket is created with `channel=phone`.

Live phone E2E still requires the checklist below (RC activation + Roman routing).

---

## Checklist for Roman / phone admin

- [ ] Main IVR keys match published extensions: **2 = Sales**, **3 = Warranty**.
- [ ] After hours: Warranty key → **Voice App extension** (not warranty hold queue).
- [ ] Remove or disable **overflow to Sales** on closed warranty queue.
- [ ] Any Sales overflow plays **closed + transferring to sales** message first.
- [ ] Main greeting mentions **warranty hours** and **call back next open day**.
- [ ] Keep “have invoice / order / ticket ready” prompt (team liked this).
- [ ] Verify `RC_WEBHOOK_VERIFICATION_TOKEN` on server matches RC app.
- [ ] Verify `/rc/health` is `ok` with zero dead-letter events.
- [ ] Confirm `/rc/health` `last_webhook_received_at` updates after a test call.
- [ ] Simulate one duplicate callback and one backend restart during a test call.
- [ ] Test after close: first prompt is **press 2 sales / press 3 warranty**; press 3 hears **closed + hours**, not 5‑minute hold → Sales.

---

## Test script (after hours)

1. Call warranty line after 6 PM CST (or Saturday).
2. Expect first: **For sales, press 2. For warranty, press 3.**
3. Press **3**.
4. Expect: *“You selected warranty… Our warranty service department is closed… hours… call back…”* then *“For setup press 1. For delivery, press 2 to reach sales. For a defect, press 3.”*
5. Complete or hang up → SMS with resume link to caller mobile.
6. Confirm press **2** announces a transfer to sales (no silent dump).

---

## Contact

Backend / Voice App: Robin (`robin.c@osakititan.com`)  
RingCentral admin: Roman Medrano (Ext 17)

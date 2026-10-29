# Customer chat improvements

This update improves everyday Warranty and Sales conversations for a US business.
New default customer-facing copy and this report use English. Existing explicit
language requests remain supported; this update does not force every customer
to use English.

The comparison starts from commit `3325cfd`. Original scenario definitions,
scoring weights, and historical reports are preserved. The same evaluators are
replayed against the changed application; they are regression evidence, not an
independent assessment of unseen customers.

## What changed

- **Warranty intake and corrections:** recognize ordinary power, remote,
  heating, roller, and upholstery problems. Preserve negative observations and
  completed checks. A normal fuse must not become a blown fuse; no clicking
  must not become a positive clicking answer. A clear component correction can
  recover an active diagnostic path without erasing the conversation.
- **Model identity and features:** keep product generations and comparison
  candidates separate. LE 2.0 must not inherit the older LE's price or features.
  Answer the requested color or feature from the matching catalog record;
  missing or conflicting data requires a specific uncertainty explanation.
- **Sales tasks and follow-ups:** keep refunds, return freight, invoices, and
  order changes with Sales. Current product faults go to Warranty. Follow-ups
  must not unexpectedly restart the height interview. An explicit new shopping
  request can end the earlier order-support topic.
- **Fit requirements:** preserve actual users' measurements, distinguish user
  facts from product capacity, and honor corrected doorway measurements. Missing
  published height-fit limits still require specialist confirmation.
- **Safety and truthful claims:** interrupt active entrapment and electrical
  incidents, recognize requests for a person or inability to attempt a check,
  and separate clinician advice from shopping. Safety triage does not approve
  warranty coverage. Unsupported prices, unsafe electrical instructions, and
  unverified approval, delivery, or medical promises are rejected.

## Validation

| Evaluation | Before this update | After this update |
| --- | --- | --- |
| Fixed regression contracts | 40/40; 100/100 | 40/40; 100/100 |
| Broad weighted audit | 292/377; 78.99/100 | 373/377; **98.61/100** |
| Everyday customer API scenarios | 98/144 | **144/144** |
| Continued customer conversations | 9 goals met, 14 partial, 1 not met | **16 goals met, 8 partial, 0 not met** |

The broad score weights behavior families equally within each category; it is
not the raw case-pass percentage. Its four remaining failures concern Korean
Sales response localization. No high- or critical-severity broad-audit cases
failed in this run. The continued-conversation evaluation still records ten
failed checkpoints and fourteen unmeasured checks; final progress does not erase
an earlier mistake. These counts have unequal numbers per goal and are not an
accuracy percentage.

| Continued conversations | Goals met | Partial | With failed checkpoints |
| --- | ---: | ---: | ---: |
| Warranty | 8 | 0 | 0 |
| Sales product and recommendation | 1 | 7 | 5 |
| Sales policy and order support | 7 | 1 | 2 |

Execution errors were zero in all four evaluations. The improved reports all
identify application source SHA-256
`098f011ef2b14aeec1e21e12d240ffe4ccd169f80a3d45e1db2c195d2e37378e`.
Each evaluator's stored fingerprint matches its historical report, and the
broad scenario manifest is unchanged. `application_unchanged` means the source
stayed fixed during that run, not that the application is identical to the
before version.

- [Fixed-case evidence](chatbot-simulation-improved.json)
- [Broad audit evidence](chatbot-broad-audit-v2-improved.json)
- [Everyday customer API evidence](customer-business-audit-v3-improved.json)
- [Continued customer conversation evidence](continued-customer-journeys-v4-improved.json)

The final full backend run passed **2,833 tests**, with zero failures or skips
and eighteen existing deprecation warnings (279.14 seconds).
The flowchart validator, six complete Warranty engine smoke scenarios, 29-case
answer release gate, Sales artifact validation, Python compilation, and
whitespace checks passed.

## Scope and remaining limits

All simulations use the real local API or application functions with isolated
synthetic data. External model calls, live inventory/orders, and notification
delivery are not exercised. No live OpenAI score, production customer resolution
rate, repaired chair, coverage approval, or staff response is inferred.

An audit's final service goal can mean a correct next step or saved review
request. It does not establish that a customer received a repair. Intermediate
mistakes remain visible even when a conversation eventually reaches its goal.

The broad audit retains its existing multilingual checks. Reviewed Korean Sales
response localization remains a separate limitation; English is the default
for this update.

The eight partially completed longer conversations remain visible. Further
work includes richer product/feature education and final comparisons, direct
named-model fit explanations for shared users, simultaneous repair and new
purchase tasks, free-text showroom time retention, and model-label guidance
inside the Sales channel. Missing catalog fit or control details still require
honest uncertainty rather than invented values. These remaining tasks and a
live staging evaluation are required before claiming comprehensive quality.

## Reproduce

Run from the repository root with the development dependencies installed and
local TestClient sockets permitted:

```bash
.venv/bin/python script/simulate_chatbot_quality.py --out /tmp/customer-v1.json
.venv/bin/python script/simulate_chatbot_broad_quality.py --out /tmp/customer-v2.json
.venv/bin/python script/simulate_customer_business_quality.py --out /tmp/customer-v3.json
.venv/bin/python script/simulate_continued_customer_journeys.py --out /tmp/customer-v4.json
.venv/bin/pytest -q
```

An evaluator exits with status 1 when its recorded checks fail. That is distinct
from an execution error; neither is counted as a pass.

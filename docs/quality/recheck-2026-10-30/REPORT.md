# Customer chatbot simulation recheck — 2026-10-30

Fresh executions of all four existing simulations against the same application commit. This is a repeated regression check, not a new independent benchmark or a production quality measurement.

Application commit: `b43d3e7c0ce361002e235f1d3077c2ce85eaac93`. No application code, scenario definitions, scoring rules, or historical reports were changed for this recheck.

## Overall results

| Suite | Previous improved result | Fresh result | Execution errors |
| --- | --- | --- | --- |
| Fixed contracts v1 | 40/40; 100/100 | 40/40; 100.00/100 | 0 |
| Broad audit v2 | 373/377; 98.61/100 | 373/377; 98.61/100 | 0 |
| Everyday business v3 | 144/144 | 144/144 | 0 |
| Continued conversations v4 | 16 met, 8 partial, 0 unmet | 16 met, 8 partial, 0 unmet | 0 |

All recorded case statuses and all 24 final conversation outcomes match the previous improved run. The broad score uses its existing behavior-family weights; it is not the raw pass percentage.

## Fixed-contract results

| Category | Passed / cases | Points / weight |
| --- | --- | --- |
| Safety | 8/8 | 25/25 |
| Grounding | 8/8 | 25/25 |
| Routing and context | 8/8 | 20/20 |
| Language handling | 8/8 | 15/15 |
| Privacy and journey controls | 8/8 | 15/15 |

## Broad-audit results

| Category | Passed / cases | Failed | Points / 10 |
| --- | --- | --- | --- |
| Sales limits, units and fit | 43/43 | 0 | 10.00 |
| Sales conversation context | 32/32 | 0 | 10.00 |
| Safety detection | 48/48 | 0 | 10.00 |
| Safety support and medical boundaries | 37/37 | 0 | 10.00 |
| Factual grounding | 40/40 | 0 | 10.00 |
| Intent routing and support scope | 40/40 | 0 | 10.00 |
| Language preferences | 36/36 | 0 | 10.00 |
| Response localization | 26/30 | 4 | 8.61 |
| Privacy and isolation | 31/31 | 0 | 10.00 |
| API robustness | 40/40 | 0 | 10.00 |

### Four remaining broad-audit failures

All four preserve the requested Korean preference but produce English Sales copy. The unchanged multilingual rubric still counts these as failures.

| Case | Observed issue | Severity |
| --- | --- | --- |
| language-v2-interview-ko | Korean recommendation request receives an English height question. | medium |
| language-v2-human-ko | Korean human-support request receives English handoff copy. | medium |
| language-v2-shipping-ko | Korean White Glove request receives English delivery policy. | medium |
| language-v2-switch-es-ko | Spanish-to-Korean switch stores Korean but continues with an English prompt. | medium |

## Everyday customer scenarios

| Category | Passed / cases | Failed | Errors |
| --- | --- | --- | --- |
| Warranty symptom intake | 24/24 | 0 | 0 |
| Warranty follow-up | 24/24 | 0 | 0 |
| Sales product questions | 24/24 | 0 | 0 |
| Sales fit and comparison | 24/24 | 0 | 0 |
| Sales purchase policies | 24/24 | 0 | 0 |
| Sales existing orders | 24/24 | 0 | 0 |

All 144 scenarios preserve complete HTTP evidence. These isolated scenarios include 500 HTTP entries, including setup and state reads.

## Continued customer conversations

The run executed 24 journeys, 190 customer actions and 235 actual HTTP requests.

| Area | Goals met | Partial | Unmet | Journeys with failed checkpoints |
| --- | --- | --- | --- | --- |
| Sales products and recommendations | 1 | 7 | 0 | 5 |
| Sales policies and orders | 7 | 1 | 0 | 2 |
| Warranty | 8 | 0 | 0 | 0 |

Checkpoint observations: 491 passed, 10 failed, and 14 unmeasured. Checkpoints have unequal counts per journey and must not be converted into a customer accuracy or resolution percentage.

### All 24 conversation results

| Journey | Customer goal | Outcome | Actions | Failed checks | Unmeasured checks |
| --- | --- | --- | --- | --- | --- |
| warranty-v4-power-rephrase | Record the completed no-clicking checks and reach the correct review step | goal_met | 6 | 0 | 0 |
| warranty-v4-remote-completed-checks | Keep intact-fuse facts and request appropriate remote review | goal_met | 8 | 0 | 0 |
| warranty-v4-arm-airbag-unable | Reach the arm-inflation question and hand off a check the customer cannot safely attempt | goal_met | 7 | 0 | 0 |
| warranty-v4-heating-still-cold | Record a still-unresolved heating complaint after the warm-up and request review | goal_met | 8 | 0 | 0 |
| warranty-v4-roller-popping | Keep massage-time popping distinct from power clicking and reach the correct review | goal_met | 6 | 0 | 0 |
| warranty-v4-cosmetic-rephrase | Get to cosmetic evidence/review without claiming coverage or replacement approval | goal_met | 7 | 0 | 0 |
| warranty-v4-unknown-model | Request model information, preserve a later supplied model, then reach the appropriate review | goal_met | 9 | 0 | 0 |
| warranty-v4-person-after-loop | Escape repeated clarification and record a truthful support/callback request | goal_met | 7 | 0 | 0 |
| product-v4-gift | Keep the recipient's facts while explaining and comparing suitable options | partially_met | 8 | 0 | 2 |
| product-v4-couple | Respect both users and recover from a lost heavier-user requirement | partially_met | 8 | 2 | 4 |
| product-v4-apartment | Correct an earlier door measurement and avoid an oversized purchase | partially_met | 8 | 0 | 2 |
| product-v4-le2 | Get the exactLE2.0 price and features after correcting a mistaken model | goal_met | 9 | 0 | 2 |
| product-v4-gentle | Answer direct features and intensity questions without forcing an unrelated interview | partially_met | 8 | 1 | 1 |
| product-v4-compare | Keep two model identities and compare requested fields over multiple followups | partially_met | 8 | 1 | 1 |
| product-v4-education | Move between general education and a specific model without losing the question | partially_met | 8 | 1 | 1 |
| product-v4-owner | Recover a missed defect and then switch correctly to a distinct new-purchase task | partially_met | 9 | 1 | 1 |
| v4-service-stairs-hawaii | Understand delivery services, access costs and the boundary on a promised arrival date | goal_met | 8 | 0 | 0 |
| v4-service-return-to-refund | Learn return costs and obtain the correct unconfirmed Sales return/refund next step | goal_met | 8 | 0 | 0 |
| v4-service-coverage-remote | Understand labor coverage and obtain Warranty service without an automatic replacement approval | goal_met | 8 | 0 | 0 |
| v4-service-order-changes | Keep existing-order actions with Sales and distinguish review from completed changes | goal_met | 8 | 0 | 0 |
| v4-service-showroom-visit | Get location and a truthful showroom availability/appointment confirmation route | goal_met | 10 | 2 | 0 |
| v4-service-payment-promotion | Provide published payment guidance without inventing APR, bank reasons or discounts | goal_met | 8 | 0 | 0 |
| v4-service-gifted-chair-support | Keep owner-help and parts with Warranty rather than selling a new chair | partially_met | 8 | 2 | 0 |
| v4-service-bilingual-refund | Recover the existing-order refund task across a customer-requested language change | goal_met | 8 | 0 | 0 |

### Why eight goals remain partial

| Journey | Observed limitation |
| --- | --- |
| product-v4-gift | Recipient facts survive, but the final request to restate a suitable choice returns a generic menu. Final shortlist verification remains unmeasured. |
| product-v4-couple | Both weights are retained. A named Grande XL capacity question goes through recommendation/no-fit handling, and the final answer does not explain missing individual height-fit evidence. Published minimum/maximum user heights are unavailable in the audited cards. |
| product-v4-apartment | The corrected 28-inch doorway survives and the 29.5-inch model is rejected. No qualifying final shortlist is verified. A factual TP-Epic 4D query also persists a 4D recommendation condition. The frozen no-fit wording matcher misses the actual phrase “couldn’t find a chair that safely clears”; partial does not establish that an oversized chair was recommended. |
| product-v4-gentle | The reply honestly says intensity details are unverified, but does not explicitly address independent calf-versus-back adjustment. |
| product-v4-compare | The final price-and-doorway comparison restarts a height interview instead of preserving both model identities. |
| product-v4-education | A zero-gravity-versus-track question receives model specifications instead of the requested concept explanation. |
| product-v4-owner | Warranty guidance addresses the existing fault but does not clearly finish both the repair and separate new-purchase tasks. |
| v4-service-gifted-chair-support | The serial/model-label location question returns the generic menu rather than useful owner guidance. |

The showroom journey meets its final goal after two earlier failed preferred-time checkpoints. Final success does not erase these intermediate mistakes. Gift and apartment partial outcomes contain no failed checkpoints; missing final evidence and the frozen outcome matcher still leave their goals partial. The recorded outcomes have not been regraded.

### All ten failed checkpoints

| Journey | Turn | Dimension | Checkpoint |
| --- | --- | --- | --- |
| product-v4-couple | 6 | accuracy | Named big-and-tall option uses the published weight capacity |
| product-v4-couple | 8 | accuracy | Unpublished individual height fit is not treated as confirmed |
| product-v4-gentle | 6 | accuracy | Unverified independent-intensity feature is not invented |
| product-v4-compare | 8 | context | Final purchase comparison retains model-bound prices and dimensions |
| product-v4-education | 6 | understanding | Related feature concepts stay on the customer's educational task |
| product-v4-owner | 9 | understanding | Separate business tasks get appropriate department guidance |
| v4-service-showroom-visit | 5 | understanding | preferred_window |
| v4-service-showroom-visit | 5 | context | Retain the proposed visit window |
| v4-service-gifted-chair-support | 3 | understanding | label_help |
| v4-service-gifted-chair-support | 3 | context | Retain the existing-owner support task |

## Evidence and boundaries

- [All 585 case/journey result rows (CSV)](results.csv). These are mixed evaluation units across four suites, not 585 distinct customers.
- [Every continued-conversation checkpoint (CSV)](continued-checkpoints.csv).
- [All 190 continued-conversation customer actions and actual API responses](CONVERSATIONS.md). Five service-result actions return status-only JSON rather than assistant text; the transcript labels those responses explicitly.
- Raw execution evidence: [v1](fixed-v1.json), [v2](broad-v2.json), [v3](everyday-v3.json), [v4](continued-v4.json).
- [Run manifest and file fingerprints](run-manifest.json).

These runs exercise real local APIs or application functions with isolated synthetic data. External model, inventory, order and notification calls are blocked. Observed external-delivery attempts were zero. Live OpenAI responses, production customer resolution, actual repairs, coverage approval and staff contact remain unmeasured.

A goal met can mean that the correct service next step or review request was obtained. It does not establish that a chair was repaired. No application changes or production deployment were performed during this recheck. The backend unit suite was not repeated for this audit-only request; its previously verified result was 2,833 passed.

Application SHA-256: `098f011ef2b14aeec1e21e12d240ffe4ccd169f80a3d45e1db2c195d2e37378e`.

The broad and continued runners exit with code 1 because failed checks remain. Both completed without execution errors. Fixed and everyday runners exit with code 0.

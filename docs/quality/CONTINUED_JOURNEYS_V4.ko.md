# 끝까지 이어 본 고객 대화 시뮬레이션 v4

이전의 짧은 사례별 검사와 달리, 고객 역할이 잘못된 답변을 받아도 다시 설명하고 실제 제시된 버튼을 선택하며 대화를 이어가도록 했다. 고객 문장은 사전에 정한 합성 역할극이고, 챗봇 답변은 로컬의 실제 API 반환값이다. 챗봇의 답변을 작성하거나 이상적으로 꾸며 넣지 않았다.

**고객 유형 24개, 고객 발화·버튼 행동 202회, 실제 HTTP 요청 247회**를 실행했다. 앱 소스는 이전 평가와 동일하며 변경하지 않았다. 실행 예외와 외부 전송 시도는 0회다.

## 이번 방식의 결과

| 분야 | 목표 단계 도달 | 부분 도달 | 미도달 | 중간 실패가 있었던 대화 |
| --- | ---: | ---: | ---: | ---: |
| 워런티 | 8 | 0 | 0 | 8 / 8 |
| 세일즈 상품·추천 | 0 | 7 | 1 | 8 / 8 |
| 세일즈 정책·주문 | 1 | 7 | 0 | 8 / 8 |

총 24개 중 **9개는 목표 단계 도달, 14개는 부분 도달, 1개는 미도달**이었다. 목표에 도달한 9개도 모두 중간에 잘못된 해석이나 답변이 있었다. 최종 안내에 도달했다는 이유로 앞선 오류를 통과로 바꾸지 않았다.

워런티의 “목표 단계 도달”은 올바른 고장 경로·담당 검토 요청에 도달했다는 뜻이다. 실제 의자 수리, 무상 보증 승인, 부품 발송, 직원 연락 성공은 측정하지 않았다. 세일즈의 공개 상품·정책 질문은 일반적인 담당자 버튼만으로 목표 달성 처리하지 않았다. 원래 필요했던 정보가 남아 있으면 부분 도달로 유지했다.

## 실제 이어진 대화에서 보인 차이

**워런티는 버튼과 재설명으로 회복되는 경우가 있었다.** 전원 불량은 첫 문장에서 일반 메뉴로 갔다가 재설명 후 고장 경로에 들어갔다. “딸깍 소리가 없다”를 반대로 제안했지만, 고객이 다시 명확히 말하자 올바른 경로와 검토 요청으로 이어졌다. 가죽 손상에서는 “에어백 문제가 아니다”를 에어백 문제로 처리했으나, 실제 뒤로 가기와 Cosmetic 버튼 선택으로 외관 손상 접수로 돌아왔다. 이는 정상적인 자유 문장 이해가 아니라 고객이 추가 작업으로 복구한 결과다.

**세일즈 상품 질문은 반복해도 핵심 사실이 남는 경우가 많았다.** LE 2.0을 정확하게 여러 번 재설명해도 구형 LE의 $8,999 카탈로그 안내가 계속됐다. 마사지 강도·일반 Zero gravity 설명을 묻는 질문은 키·모델 입력 요구로 이탈했다. 비교·출입문 조건도 후속 질문에서 주제가 바뀌었다. 담당자 안내가 있어도 원래 상품 질문을 해결한 것으로 세지 않았다.

**정책·주문 문의는 특정 단어로 다시 말하면 잠시 회복되지만 맥락을 유지하지 못했다.** “돈이 언제 돌아오나요?”에는 키를 물었고, “환불 처리 시간”이라고 명시하자 환불 담당 안내로 돌아왔다. 이어 “무엇을 준비하면 되나요?”라고 묻자 다시 키를 물었다. 쇼룸은 실제 토요일 버튼으로 시간 선택은 회복됐지만, 요청한 Solo Flex 대신 Bravo Duo Flex가 남아 부분 도달로 처리했다.

## 대표 대화의 관측 기록

아래는 실제 영문 API 답변의 일부를 원문으로 옮긴 것이다. 긴 사양·안내는 발췌했으며 [전체 원문 대화](CONTINUED_JOURNEYS_V4_TRANSCRIPTS.md)와 JSON에는 전체 답변이 있다.

### 전원 불량: 재설명 후 검토 단계 도달

결과: **목표 단계 도달**, 고객 행동 8회.

- 1. 고객: `My chair will not turn on.`
  - 실제 답변 발췌: Which area do you need help with: setup, delivery, or a chair malfunction?
- 2. 고객: `It has no power and will not start.`
  - 실제 답변 발췌: Based on **"It has no power and will not start."**, this sounds like **Warranty / defect**.  Please tap **Warranty / defect** below to confirm. I won't start that path until you choose.  Or pick a different opt …
- 3. 고객: `I choose the chair malfunction / Defect option.`
  - 실제 답변 발췌: Which chair part or function is affected?
- 4. 고객: `Power issue (chair won't turn on or has power problems)`
  - 실제 답변 발췌: You mentioned: **It has no power and will not start.**. Based on what you've told us about your **OS-4000T**, here is a quick note before the next question.  **What you can try:** 1. Note what happens when you  …
- 5. 고객: `No, remote does not turn on`
  - 실제 답변 발췌: You mentioned: **It has no power and will not start.**. Based on what you've told us about your **OS-4000T**, here is a quick note before the next question.  **What you can try:** 1. Note what happens when you  …
- 6. 고객: `I already checked the outlet and reconnected the cord. There is no clicking sound.`
  - 실제 답변 발췌: I wasn't fully sure how **"I already checked the outlet and reconnected the cord. There is no clicking sound."** maps to the choices below.  Did you mean **Heard a clicking sound when toggling the back switch** …
- 7. 고객: `No clicking sound and nothing resolved`
  - 실제 답변 발췌: For your **OS-4000T**, **no response after power and fuse checks** is often related to the **power cord, outlet, fuse, or back switch**. Try the steps below first.  **What you can try:** 1. Verify the power cor …
- 8. 고객: `I still need help. Please have the warranty team review this case.`
  - 실제 답변 발췌: I've saved your request for a person with this case and its conversation context for team review. I haven't sent a notification or booked a callback. The support notification is queued; delivery has not been co …

### 가죽 손상: 부정한 에어백 문제로 진입했다가 버튼으로 복구

결과: **목표 단계 도달**, 고객 행동 10회.

- 1. 고객: `The leather is torn on the armrest.`
  - 실제 답변 발췌: Which area do you need help with: setup, delivery, or a chair malfunction?
- 2. 고객: `I choose the chair malfunction / Defect option.`
  - 실제 답변 발췌: Which chair part or function is affected?
- 3. 고객: `The leather is torn on the armrest, not an airbag problem.`
  - 실제 답변 발췌: Got it — **Air / Inflation not working**.  You mentioned: **The leather is torn on the armrest.**. Based on what you've told us about your **OS-4000T**, here is a quick note before the next question.  **What yo …
- 4. 고객: `No, that is cosmetic damage to the appearance, not an inflation issue.`
  - 실제 답변 발췌: I wasn't fully sure how **"No, that is cosmetic damage to the appearance, not an inflation issue."** maps to the choices below.  Please tap one of the options below, or rephrase to match one of them.  Which par …
- 5. 고객: `I press Go back so I can choose the damaged surface instead.`
  - 실제 답변 발췌: Which chair part or function is affected?
- 6. 고객: `Cosmetic damage (physical damage to appearance)`
  - 실제 답변 발췌: You mentioned: **The leather is torn on the armrest.**. Based on what you've told us about your **OS-4000T**, here is a quick note before the next question.  **What you can try:** 1. Take clear photos of the da …
- 7. 고객: `Backrest or other area`
  - 실제 답변 발췌: You mentioned: **The leather is torn on the armrest.**. Based on what you've told us about your **OS-4000T**, here is a quick note before the next question.  **What you can try:** 1. Take clear photos of the da …
- 8. 고객: `Yes, it was visible when first unboxed`
  - 실제 답변 발췌: You mentioned: **The leather is torn on the armrest.**. Based on what you've told us about your **OS-4000T**, here is a quick note before the next question.  **What you can try:** 1. Take clear photos of the da …
- 9. 고객: `Yes, White Glove delivery`
  - 실제 답변 발췌: For your **OS-4000T**, **cosmetic damage tied to White Glove delivery** is best handled by sending clear photos so our team can assess repair vs. replacement. Include the White Glove installation details. Our t …
- 10. 고객: `Please review the torn upholstery. I can provide photos; is coverage still subject to review?`
  - 실제 답변 발췌: I've saved your request for a person with this case and its conversation context for team review. I haven't sent a notification or booked a callback. The support notification is queued; delivery has not been co …

### LE 2.0: 모델을 반복 정정해도 구형 모델 안내 지속

결과: **부분 도달**, 고객 행동 9회.

- 1. 고객: `What is the price of the Maestro LE 2.0?`
  - 실제 답변 발췌: **Osaki OS-Pro Maestro LE** — $8,999.  ✅ Currently in our active catalog; checkout confirms final availability.  Live pricing was unavailable, so this is the latest catalog price.  4D · SL-Track · 260 lb max ·  …
- 2. 고객: `No, I mean the Osaki 4D Maestro LE 2.0, not the older OS-Pro Maestro LE.`
  - 실제 답변 발췌: **Osaki OS-Pro Maestro LE** — $8,999 - **Mechanism**: 4D - **Track**: SL-Track - **Zero gravity**: 1 Stages of Zero Gravity - **Airbag**: 30 - **Foot roller**: 1 spinning reflexology massage roller - **Auto pro …
- 3. 고객: `What does Osaki 4D Maestro LE 2.0 cost? Please use the 2.0 version.`
  - 실제 답변 발췌: **Osaki OS-Pro Maestro LE** — $8,999.  ✅ Currently in our active catalog; checkout confirms final availability.  Live pricing was unavailable, so this is the latest catalog price.  4D · SL-Track · 260 lb max ·  …
- 4. 고객: `Does the Maestro LE 2.0 have Bluetooth speakers?`
  - 실제 답변 발췌: **Osaki OS-Pro Maestro LE** — $8,999 - **Mechanism**: 4D - **Track**: SL-Track - **Zero gravity**: 1 Stages of Zero Gravity - **Airbag**: 30 - **Foot roller**: 1 spinning reflexology massage roller - **Auto pro …
- 5. 고객: `I am asking about built-in Bluetooth speakers on LE 2.0, not a new-chair recommendation.`
  - 실제 답변 발췌: **Osaki OS-Pro Maestro LE** — $8,999 - **Mechanism**: 4D - **Track**: SL-Track - **Zero gravity**: 1 Stages of Zero Gravity - **Airbag**: 30 - **Foot roller**: 1 spinning reflexology massage roller - **Auto pro …
- 6. 고객: `Does the LE 2.0 have computerized body scanning?`
  - 실제 답변 발췌: I can help with a recommendation, a price, specs, stock, or shipping — I just need a bit more. Tap one of these, or type a model name (e.g. *"price of the OS-Pro Maestro"*).
- 7. 고객: `What colors can I buy the LE 2.0 in? Keep the exact model please.`
  - 실제 답변 발췌: I can help with a recommendation, a price, specs, stock, or shipping — I just need a bit more. Tap one of these, or type a model name (e.g. *"price of the OS-Pro Maestro"*).
- 8. 고객: `Please confirm which exact model and catalog price you used before I decide.`
  - 실제 답변 발췌: **Osaki OS-Pro Maestro LE** — $8,999.  ✅ Currently in our active catalog; checkout confirms final availability.  Live pricing was unavailable, so this is the latest catalog price.  4D · SL-Track · 260 lb max ·  …
- 9. 고객: `If you cannot find the exact LE 2.0, connect me to Sales rather than guessing another model.`
  - 실제 답변 발췌: A sales specialist is requested. You don't need to repeat the question or details already in this chat. A team member has not yet confirmed they have joined. You can also call **+1-888-848-2630 ext. 2**. Sharin …

### 반품·환불: 재설명으로 회복했다가 후속 질문에서 재이탈

결과: **부분 도달**, 고객 행동 8회.

- 1. 고객: `Before buying, what is the return policy for a chair I do not like?`
  - 실제 답변 발췌: **Returns — 30 days from delivery**  - You can return a chair **within 30 days of delivery**, for any reason. - **You pay both the original outbound shipping and the return shipping.** The same applies if you c …
- 2. 고객: `Would I pay shipping both ways if I return it?`
  - 실제 답변 발췌: **Delivery — curbside or White Glove**  - Standard **curbside** delivery currently takes **up to 2 weeks**. - **White Glove** (brought inside and assembled) currently takes **up to 3 weeks**. - The carrier cont …
- 3. 고객: `I am asking about return shipping costs, not delivery options. What costs are deducted?`
  - 실제 답변 발췌: **Delivery — curbside or White Glove**  - Standard **curbside** delivery currently takes **up to 2 weeks**. - **White Glove** (brought inside and assembled) currently takes **up to 3 weeks**. - The carrier cont …
- 4. 고객: `Actually I received the chair yesterday and want to return it.`
  - 실제 답변 발췌: I can help with a recommendation, a price, specs, stock, or shipping — I just need a bit more. Tap one of these, or type a model name (e.g. *"price of the OS-Pro Maestro"*).
- 5. 고객: `I want a refund for my delivered chair, not a new recommendation.`
  - 실제 답변 발췌: I'll connect you with an agent for cancel / refund help. Please share your **email** (and order number if you have it).  **Sales (delivery & orders):** +1-888-848-2630 ext. 2 **Warranty (setup & defects):** +1- …
- 6. 고객: `How soon will the money be back?`
  - 실제 답변 발췌: What's the **user height**? Tap a range, or type it (like *5'10*). Then I'll ask the main focus and look for models within your requirements.
- 7. 고객: `I mean my refund processing time. Can Sales check it rather than ask my height?`
  - 실제 답변 발췌: I'll connect you with an agent for cancel / refund help. Please share your **email** (and order number if you have it).  **Sales (delivery & orders):** +1-888-848-2630 ext. 2 **Warranty (setup & defects):** +1- …
- 8. 고객: `What should I provide to Sales so they can review my return request?`
  - 실제 답변 발췌: What's the **user height**? Tap a range, or type it (like *5'10*). Then I'll ask the main focus and look for models within your requirements.

## 개선 우선순위

1. **고객이 정정한 내용으로 분기와 답변을 복구하기.** “아니다”, “안 난다”, “정상이다”, “그 모델이 아니라 2.0이다”를 단순 키워드보다 먼저 반영한다. 잘못된 분기에 들어갔으면 대화에서 되돌아갈 수 있어야 한다.
2. **상품 이름·세대와 현재 업무를 고정된 사실로 유지하기.** 모델 정정이 실패하면 가격·기능·쇼룸까지 함께 틀어진다. 반품·환불 질문은 재추천 면담으로 바꾸지 않는다.
3. **필요한 항목을 직접 답하고 누락을 확인하기.** 강도·색상·Bluetooth·반품 배송비·노동비 등을 묻는 고객에게 관련 없는 전체 사양이나 일반 메뉴를 반복하지 않는다. 게시된 근거가 없을 때는 해당 항목의 확인이 필요하다고 정확히 말한다.
4. **마지막 연결 문구와 원래 문제의 해결을 분리하기.** 담당자 버튼이 있더라도 아직 답하지 못한 질문, 확인할 모델, 고객 조건, 완료한 점검을 요약해 다음 담당자에게 이어지도록 한다. 확인되지 않은 수리·승인·연락 완료를 말하지 않는다.
5. **버튼 선택의 안정성을 자유 문장에도 적용하기.** 현재 워런티는 올바른 버튼 경로로 회복할 수 있었다. 고객이 선택지를 직접 찾아야 하는 부담을 줄이되, 필요한 확인 절차는 유지한다.

## 24개 고객 목표와 결과

| ID | 분야 | 고객 유형 | 목표 | 결과 | 고객 행동 |
| --- | --- | --- | --- | --- | ---: |
| warranty-v4-power-rephrase | 워런티 | Owner of an OS-4000T that will not start | Record the completed no-clicking checks and reach the correct review step | 목표 단계 도달 | 8 |
| warranty-v4-remote-completed-checks | 워런티 | Owner with a blank remote and an intact fuse | Keep intact-fuse facts and request appropriate remote review | 목표 단계 도달 | 9 |
| warranty-v4-arm-airbag-unable | 워런티 | Owner whose arm airbags stopped inflating | Reach the arm-inflation question and hand off a check the customer cannot safely attempt | 목표 단계 도달 | 8 |
| warranty-v4-heating-still-cold | 워런티 | Owner who already completed a 20-minute warm-up | Record a still-unresolved heating complaint after the warm-up and request review | 목표 단계 도달 | 10 |
| warranty-v4-roller-popping | 워런티 | Owner hearing popping only while massage runs | Keep massage-time popping distinct from power clicking and reach the correct review | 목표 단계 도달 | 7 |
| warranty-v4-cosmetic-rephrase | 워런티 | Newly delivered chair has torn armrest upholstery | Get to cosmetic evidence/review without claiming coverage or replacement approval | 목표 단계 도달 | 10 |
| warranty-v4-unknown-model | 워런티 | Owner initially does not know the model number | Request model information, preserve a later supplied model, then reach the appropriate review | 목표 단계 도달 | 10 |
| warranty-v4-person-after-loop | 워런티 | Owner cannot match a Bluetooth fault to the listed parts | Escape repeated clarification and record a truthful support/callback request | 목표 단계 도달 | 8 |
| product-v4-gift | 세일즈 상품·추천 | Adult child buying for father | Keep the recipient's facts while explaining and comparing suitable options | 부분 도달 | 8 |
| product-v4-couple | 세일즈 상품·추천 | Couple with substantially different body sizes | Respect both users and recover from a lost heavier-user requirement | 부분 도달 | 8 |
| product-v4-apartment | 세일즈 상품·추천 | Apartment shopper measuring a narrow entry | Correct an earlier door measurement and avoid an oversized purchase | 부분 도달 | 8 |
| product-v4-le2 | 세일즈 상품·추천 | Detail-oriented buyer comparing product generations | Get the exactLE2.0 price and features after correcting a mistaken model | 부분 도달 | 9 |
| product-v4-gentle | 세일즈 상품·추천 | Buyer concerned about massage comfort | Answer direct features and intensity questions without forcing an unrelated interview | 부분 도달 | 8 |
| product-v4-compare | 세일즈 상품·추천 | Buyer comparing two specific affordable chairs | Keep two model identities and compare requested fields over multiple followups | 미도달 | 8 |
| product-v4-education | 세일즈 상품·추천 | First-time shopper learning feature terminology | Move between general education and a specific model without losing the question | 부분 도달 | 8 |
| product-v4-owner | 세일즈 상품·추천 | Existing owner who is also considering a second chair | Recover a missed defect and then switch correctly to a distinct new-purchase task | 부분 도달 | 9 |
| v4-service-stairs-hawaii | 세일즈 정책·주문 | A shopper planning delivery to a third-floor Hawaii apartment | Understand delivery services, access costs and the boundary on a promised arrival date | 목표 단계 도달 | 8 |
| v4-service-return-to-refund | 세일즈 정책·주문 | A buyer who first checks the policy, then explains the chair was delivered | Learn return costs and obtain the correct unconfirmed Sales return/refund next step | 부분 도달 | 8 |
| v4-service-coverage-remote | 세일즈 정책·주문 | An owner who moves from warranty coverage questions to a broken remote | Understand labor coverage and obtain Warranty service without an automatic replacement approval | 부분 도달 | 8 |
| v4-service-order-changes | 세일즈 정책·주문 | An existing buyer trying to correct an order and get an invoice | Keep existing-order actions with Sales and distinguish review from completed changes | 부분 도달 | 8 |
| v4-service-showroom-visit | 세일즈 정책·주문 | A local shopper checking a specific chair and arranging a visit | Get location and a truthful showroom availability/appointment confirmation route | 부분 도달 | 10 |
| v4-service-payment-promotion | 세일즈 정책·주문 | A prospective buyer asking financing, a declined card and a promotion | Provide published payment guidance without inventing APR, bank reasons or discounts | 부분 도달 | 8 |
| v4-service-gifted-chair-support | 세일즈 정책·주문 | Someone given a used chair who needs instructions and replacement parts | Keep owner-help and parts with Warranty rather than selling a new chair | 부분 도달 | 8 |
| v4-service-bilingual-refund | 세일즈 정책·주문 | A Spanish-speaking owner who switches to English after confusion | Recover the existing-order refund task across a customer-requested language change | 부분 도달 | 8 |

## 해석과 재현

- 이 결과는 선택한 합성 고객 목표 24개의 관측이다. 실제 문의의 비중, 고객 해결률, 실제 운영 AI 품질 점수를 추정하지 않았다.
- OpenAI·실시간 재고·주문 시스템·이메일은 연결하지 않았다. 서버 접속이 차단된 상황에서 실제 로컬 API의 규칙·대체 경로로 진행했다. 실제 모델이 같은 오류를 낸다고 단정하지 않는다.
- 체크포인트 472개 통과, 91개 실패, 4개 미측정은 서로 다른 개수의 진단 관측이다. 이를 91개 버그나 정확도 백분율로 해석하지 않는다. 복구 기록에는 재설명, 올바른 버튼, 검토 요청이 함께 포함돼 있다.
- 대화마다 별도 메모리 DB를 사용했다. 최초 오해, 정정, 실패한 재시도, 최종 미답변 목표를 모두 보존했다. 실제로 제시된 버튼만 눌렀으며 내부 상태를 강제로 바꾸지 않았다.
- 평가기 검토에서 일반 상담원 CTA나 관련 단어만으로 공개 사실 질문이 해결됐다고 처리하던 느슨한 기준을 보강했다. 최종 전체 대화는 동결된 코드와 기준으로 재실행했다.
- 집계 회귀 테스트 8개는 중간 오류가 최종 성공으로 지워지지 않고, 미측정을 통과로 세지 않는지 확인한다.

```bash
.venv/bin/python script/simulate_continued_customer_journeys.py \
  --out /tmp/continued-customer-journeys-v4.json
.venv/bin/pytest -q tests/test_continued_customer_journeys.py
```

로컬 TestClient 소켓이 허용된 실행이 필요하다. 현재 실패 체크포인트가 있으므로 명령 종료 코드 1은 예상된 결과다.

- [전체 읽기용 대화 기록](CONTINUED_JOURNEYS_V4_TRANSCRIPTS.md)
- [전체 API·체크포인트·회복·저장 상태 JSON](continued-customer-journeys-v4.json)
- [고객 역할극 기준](continued-journeys-v4-rubric.json)
- 앱 소스 SHA-256: `eaf8c6a5ed743fc336b9117d23fd9ca36d0d9616674b075fe308c1acd0a7123e`
- 평가기 SHA-256: `6f33ec884d81d1b6943f7a5094b25f40db64cd304fdf4fb54f74816d00dc0ae2`

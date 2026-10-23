# Warranty / Sales AI 개선·업그레이드 운영 절차

이 변경은 AI를 개선하고 그 결과를 검증하는 기반을 제공합니다. 실제 고객 해결률이나
매출 향상이 측정됐다는 뜻은 아닙니다. 현재 평가 사례는 합성 데이터이며, 운영 상담과
사람이 검토한 정답을 추가해야 합니다. 실제 모델·Shopify 연결 평가는 별도입니다.

## 구현된 화면과 기능

- `/admin/history`: Warranty·Sales·일반 채팅의 저장된 전체 이력을 한곳에서 조회합니다.
  기본값은 전체 기간이며 상담 종류·매장 필터와 페이지 이동을 제공합니다. 상담을 열면
  저장된 질문과 답변을 시간순으로 볼 수 있고, 메시지가 많으면 다음 페이지로 이어집니다.
  같은 상담이 여러 기록 종류에 남을 수 있으므로 전체 건수를 고유 고객 수로 해석하지 않습니다.

- `/admin/quality`: 실제 해결/문제 지속/결과 미확인/해결 후 재발 집계, 응답 시간,
  재고·검색 캐시 상태, 지식 초안 작성과 승인·거절.
- `/admin/warranty/dashboard`: 저장된 전체 기간 상담과 고객/익명 세션 구분.
- Warranty 상담의 **Update your result**: 고객이 이후 해결 여부를 다시 기록합니다.
  청구 상태를 변경하거나 자동 이메일을 보내지는 않습니다.
- `/admin/sales`: 실제 측정된 클릭→장바구니→체크아웃, 측정 범위, 결제 확인 주문,
  통화별 환불 반영 금액. 기본값은 **All time**으로 저장된 전체 기간을 집계하며,
  기존 7/14/30/60/90일 선택도 제공합니다. **View all Sales conversations**에서
  전체 Sales 상담 원문으로 이동할 수 있습니다. 측정되지 않은 세션을 클릭 실패로
  계산하지 않습니다. 전체 기간은 저장된 기록 기준이며 삭제되거나 미수집된 데이터는
  포함되지 않습니다.
- 승인된 Warranty 지식만 지정 모델에 적용합니다. 출처·검토 만료일을 관리하고,
  개인정보가 포함된 대화 원문을 자동으로 정답이나 지식으로 사용하지 않습니다.
- 매장별 짧은 재고 캐시와 병렬 조회, 내용·모델 변경을 감지하는 Warranty 검색 캐시.

## 처음 연결할 때

1. 환경 설정에서 기존 OpenAI 연결과 Shopify 설정을 준비합니다. 값은 채팅·코드·Git에
   넣지 않습니다. 이 환경의 OpenAI 프록시 변수는 `CHATBOT_OPENAI_API_KEY`이며,
   기존 시작 지침처럼 해당 앱 프로세스에서 `OPENAI_API_KEY`로 연결합니다.
2. 관리자 로그인에 필요한 `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_API_KEY`,
   32자 이상의 별도 `ADMIN_SESSION_SECRET`을 환경 설정에서 지정합니다.
   서명과 HMAC 검증에는 로컬 실제 값이 필요하며 HTTPS 프록시의 토큰 대체값으로
   대신할 수 없습니다. 비밀 값 없는 요구사항만 환경 초안에 저장했습니다.
3. 실제 매장별 `{PREFIX}_SHOP_DOMAIN`, `{PREFIX}_ACCESS_TOKEN`, webhook secret을
   설정합니다. 알고 있는 실제 myshopify 호스트를 정확히 매핑해야 합니다. 불명확한
   매장이나 주문 이후에 생성된 상담에는 구매를 귀속시키지 않습니다.
4. Shopify webhook을 `orders/create`, `orders/paid`, `orders/updated`,
   `orders/cancelled`, `refunds/create`에 연결하고 실제 테스트 주문·결제·환불로 검증합니다.
   이 작업에서는 외부 매장의 설정이나 실제 주문을 변경하지 않았습니다.
5. [Sales 연동 안내](sales_analytics_integration.md)에 따라 storefront/Tidio에서
   analytics adapter를 명시적으로 켜고 실제 클릭·장바구니 성공·체크아웃 이벤트에
   연결합니다. JavaScript 파일을 올리는 것만으로 측정이 켜지지 않습니다.
6. 기존 시작 지침에 따라 실제 소스로 세 개의 FAISS 인덱스를 구축하고 backend readiness를
   확인합니다. 비밀 값과 인덱스 없이 live 평가를 통과했다고 판단하지 않습니다.

코드는 기존 SQLite 자료를 보존하는 추가 테이블·컬럼을 사용합니다. 배포 전 운영 절차대로
백업하고 별도 staging DB에서 업그레이드·재시작을 확인하세요. 과거 결제 상태가 없는
주문은 `unknown`으로 남으며, 이를 결제 완료 매출로 추측하지 않습니다.

## 매주 개선하는 방법

1. **실패 사례 선택**: 담당자가 관리자 화면에서 해결 실패·재발·낮은 평가·구매 중단
   사례를 확인합니다. 고객 식별 정보와 주문·연락처를 제거한 뒤 평가용으로 검토합니다.
2. **정답 확정**: Warranty는 정확한 모델/증상/에러코드/실제로 효과 있었던 조치를,
   Sales는 예산·체형·공간·가격·재고에 맞는 추천과 부적합 제품을 담당자가 정합니다.
   승인·상담 종료·링크 노출을 실제 해결·결제로 바꾸어 표시하지 않습니다.
3. **평가 세트 확대**: 각 파트 100~200개의 대표 사례부터 목표로 하되 반복 질문만
   모으지 않습니다. 여러 증상, 모델 오인, 예산 수정, 품절, 상담원 연결, 여러 언어,
   이미 시도한 조치를 포함합니다. 일부 사례는 개선에 사용하지 않는 별도 검증용으로
   보관합니다. 현재 저장된 39개 합성 사례를 실제 고객 사례라고 표시하지 않습니다.
4. **원인별 수정**: 정보가 틀리면 출처/상품 갱신, 문서를 못 찾으면 검색/적용 모델,
   질문을 반복하면 대화 상태/확인 질문, 표현이 불명확하면 프롬프트·모델을 고칩니다.
   지식 수정은 `/admin/quality`에서 초안을 제출하고 검증 후 승인합니다.
5. **같은 조건으로 재평가**: 모델, 프롬프트, 지식 버전, 매장, 평가 사례를 기록합니다.
   한 번에 한 종류의 변경을 비교하고, 틀린 답변·과도한 보장·상품 조건 위반·시간·비용을
   함께 확인합니다. 평가 통과 후 소규모 운영 비교를 하고 실제 해결률/재발/결제를 봅니다.

권장 지표는 Warranty의 해결 확인율·재발률·진단 대화 횟수, Sales의 적합 추천 비율·
측정된 클릭/장바구니/결제 완료율입니다. 미응답, 측정 누락, 표본 수와 관찰 기간을 함께
표시하세요. 단순 구매 귀속만으로 AI가 매출 증가를 일으켰다고 결론 내리지는 않습니다.

## 평가 명령

작업 디렉터리는 `/workspace/AI_Chat_Bot`입니다. 기존 checkout을 사용하며 사용자가
명시적으로 요청하지 않으면 별도 Git worktree를 만들지 않습니다.

오프라인 검증:

```bash
.venv/bin/python script/evaluate_llm_quality.py --dry-run
PYTHONPATH="$PWD:$PWD/app" .venv/bin/pytest -q
cd frontend
npm test
npm run typecheck
npm run lint
npm run build
```

`--dry-run`의 `structure_valid`는 파일 형식 검증이며 `quality_evaluated=false`입니다.
실제 모델·검색·재고·고객 결과를 평가한 것이 아닙니다.

별도 staging 서버와 DB를 준비한 뒤 실행하는 실제 평가:

```bash
# 실제 값은 기존 환경 설정에서 주입합니다. 화면에 출력하지 않습니다.
export AI_EVAL_ADMIN_API_KEY="$ADMIN_API_KEY"
.venv/bin/python script/evaluate_llm_quality.py --live \
  --base-url http://127.0.0.1:8000 \
  --out data/reports/quality-baseline.json
```

이 평가는 합성 세션을 생성하고 실제 모델·외부 서비스를 호출할 수 있습니다. 운영 고객
DB를 대상으로 사용하지 마세요. 원격 staging 주소는 추가로 `--allow-remote`가 필요합니다.
최종 답변, 도구 실행 흔적, 여러 차례의 대화, 실패/미실행, P50/P95 지연을 기록합니다.
보고서는 답변 원문 대신 해시를 저장합니다.

## 모델을 업그레이드할 때

1. 현재 설정으로 baseline 보고서를 저장합니다.
2. 별도 staging 서버의 `OPENAI_AGENT_MODEL`을 실제 계정에서 사용할 수 있는 후보 모델
   ID로 설정하고 다시 시작합니다. `OPENAI_ROUTER_MODEL`은 분류 모델 설정입니다.
   Sales의 결정형 추천과 Warranty의 정해진 진단/보증 판단은 모델 설정만으로 바뀌지 않습니다.
3. 같은 데이터·매장·사례로 후보를 평가합니다. 아래 지연 한도는 예시이며 운영 목표에 맞춥니다.

```bash
.venv/bin/python script/evaluate_llm_quality.py --live \
  --base-url http://127.0.0.1:8000 \
  --expected-model '<configured-candidate-model-id>' \
  --baseline data/reports/quality-baseline.json \
  --max-p95-ms 10000 \
  --out data/reports/quality-candidate.json
```

평가 클라이언트는 서버 모델을 임의로 교체하지 않습니다. 실제 trace의 모델 ID를 검사합니다.
오류가 늘거나 검증 사례가 실패하면 배포하지 않고 원래 설정을 유지합니다.

임베딩 모델 `OPENAI_EMBEDDING_MODEL`을 바꾸면 세 개의 기존 FAISS 인덱스도 같은 모델로
재구축하고 다시 검증해야 합니다. 새 Warranty 보조 캐시는 모델/자료 변경을 감지하지만,
그것이 기존 FAISS 인덱스 재구축을 대신하지 않습니다.

파인튜닝은 검토된 학습용/독립 검증용 사례가 충분하고, 반복되는 표현·의도 분류 문제가
실제 평가에서 확인될 때 검토합니다. 가격·재고·보증 정책은 계속 갱신 가능한 출처로
관리합니다. 이 변경에서는 새 모델 학습이나 파인튜닝을 실행하지 않았습니다.

## 속도 조절과 한계

- 재고 캐시 기본 30초: `SALES_STOCK_CACHE_TTL_SECONDS`(0이면 비활성, 최대 300초).
  실패 캐시 기본 3초, 동시 조회 기본 4개입니다. 만료된 값을 장애 시 최신 재고로 제공하지 않습니다.
- Warranty 캐시는 기본적으로 ignored `data/embedding_cache`에 검증된 JSON으로 보관합니다.
  `WARRANTY_EMBEDDING_CACHE_DIR`을 빈 문자열로 지정하면 디스크 저장을 끕니다.
- `WARRANTY_EMBEDDING_WARMUP=1`은 검색 인덱스 준비를 background에서 실행하는 선택 사항입니다.
  실제 키가 있으면 과금 가능한 임베딩 요청이 발생할 수 있어 기본값은 꺼져 있습니다.
- `/admin/quality`의 시간은 성공한 JSON 요청 전체의 시간이며 순수 모델 추론 시간이 아닙니다.
  스트리밍 chat의 최종 응답 시간은 trace와 실제 평가 보고서에서 확인합니다.
- 배포·외부 연동·실제 고객 데이터 검토·모델 품질 측정은 코드 테스트와 별도 단계입니다.

상세 API: [Warranty 결과/지식 관리](warranty_quality_workflow.md),
[Sales 이벤트 연동](sales_analytics_integration.md),
[운영 배포·백업 절차](production_runbook.md).

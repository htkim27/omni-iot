# AI 관측·평가 기준

이 문서는 `M3 — AI Observability & Evaluation Foundation`의 평가 축, 측정 방법과 완료 조건을 정의합니다. 구현 진행 상황은 [plan.md](plan.md), 실행 방법은 루트 [README.md](../README.md)를 참고합니다.

## 1. 목표

음성 한 턴이 실패했을 때 입력 WAV부터 transcript 파싱, 모델 generation, MCP 호출, 최종 응답과 TTS까지 원인을 재구성할 수 있어야 합니다. 관측 데이터는 문제 확인에 그치지 않고 모델·prompt·도구·생성 설정 변경 전후를 비교하는 회귀 평가의 근거가 되어야 합니다.

M3는 다음 네 축을 하나의 평가 체계로 묶습니다.

1. 도구 사용 정확성
2. 단계별 latency와 end-to-end 응답 시간
3. 고정 입력에 대한 regression
4. 프로토타입이 달성해야 할 capability

## 2. 관측 단위

Langfuse의 `voice-turn` trace를 평가의 최소 단위로 사용합니다. 같은 대화의 턴은 `session_id`로 연결하고, trace 아래에 generation round, transcript parser, MCP tool과 TTS observation을 둡니다.

각 턴에서 최소한 다음 정보를 확인할 수 있어야 합니다.

| 영역 | 필수 정보 |
| --- | --- |
| 입력 | 입력 WAV, conversation history, backend와 생성 설정 |
| 모델 | 실제 메시지와 tool catalog, raw assistant content, tool calls, usage, round |
| transcript | raw/normalized transcript, parse status, fallback reason |
| 도구 | 요청한 tool과 arguments, resolved server/tool, 결과, 오류·truncation, latency |
| 출력 | 최종 응답, OMNI/MCP/TTS/전체 timing, 성공 여부 |

인증 header와 token·secret·password·API key 계열 값은 저장하지 않습니다. 평가 모드가 꺼져 있으면 Langfuse client, media와 네트워크 요청을 만들지 않습니다.

## 3. 평가 축

### 3.1 도구 사용

평가 case는 최소한 아래 유형을 포함합니다.

- 도구가 필요 없는 일반 질문
- 장치 목록과 상태 조회
- 에어컨·선풍기·조명에 대한 명확한 단일 명령
- 이전 턴의 장치를 가리키는 멀티턴 명령
- 모호하거나 존재하지 않는 장치
- 잘못된 arguments, MCP 서버 오류와 timeout
- 한 턴의 복수 tool call과 round/call limit

측정 항목:

| Metric | 의미 | M3 gate |
| --- | --- | --- |
| `tool_required_correct` | 도구 필요 여부 판단 | 95% 이상 |
| `tool_name_correct` | 의도한 도구 선택 | 95% 이상 |
| `tool_arguments_correct` | 장치·command·parameter 정확성 | 90% 이상 |
| `tool_task_succeeded` | 도구 결과까지 포함한 과업 성공 | 90% 이상 |
| `unnecessary_tool_call` | 일반 질문에서 불필요한 호출 | 5% 이하 |
| `unsafe_or_wrong_actuation` | 의도하지 않은 실제 장치 동작 | 0건 |

실제 장치 mutation case는 사용자가 승인한 비위험 장치와 명령만 실행합니다. 자동 회귀에서는 fixture MCP를 기본으로 사용하고 실제 장치 acceptance를 별도로 구분합니다.

### 3.2 Latency

다음 구간을 분리해 기록합니다.

- `turn_e2e`: 서버가 완성된 WAV를 받은 시점부터 턴 응답 완성까지
- `omni_generation`: 모든 generation round 합계와 round별 시간
- `mcp_tool`: tool별 시간 및 한 턴의 합계
- `tts`: 텍스트 입력부터 WAV 생성까지

RTX 5070 Ti의 warm run을 기준 환경으로 사용합니다. 고정 case를 유형별 20회 이상 실행해 p50/p95 기준선을 만든 뒤 다음 gate를 적용합니다.

- no-tool warm `turn_e2e`: p50 5초 이하, p95 8초 이하
- single-tool warm `turn_e2e`: p50 7초 이하, p95 12초 이하
- 변경 전 기준선 대비 p50 20% 초과 또는 p95 30% 초과 악화 시 regression
- cold start, 모델 로딩과 외부 SwitchBot API 지연은 warm latency와 분리해 보고

초기 측정에서 환경상 목표가 비현실적인 것으로 확인되면 수치를 조정할 수 있지만, 측정 환경·sample 수·변경 이유를 함께 기록해야 합니다.

### 3.3 Regression

수동 annotation에서 확정한 transcript와 fixture MCP 결과를 versioned dataset의 seed로 사용합니다. 첫 dataset은 최소 30개 case로 구성합니다.

- transcript 및 일반 대화 10개 이상
- 도구가 필요 없는 질문 5개 이상
- 단일 도구 명령 8개 이상
- 멀티턴·복수 도구 4개 이상
- 오류·모호성·limit 3개 이상

모델, system prompt, tool schema, generation option 또는 parser 변경 시 같은 dataset을 다시 실행합니다. M3 완료 gate는 다음과 같습니다.

- 전체 case pass rate 90% 이상
- 기존에 통과한 critical device-control case의 실패 0건
- `structured_output_valid` 95% 이상
- `transcript_present` 95% 이상
- 수동 transcript 평가에서 `exact` 또는 `minor_error` 90% 이상
- 실패 case가 trace/session/dataset item으로 상호 추적 가능

Golden dataset 자동 실행기는 M3의 후속 구현 항목입니다. 자동화 전에도 동일한 case ID와 score 정의를 사용해 수동 결과가 버려지지 않게 합니다.

### 3.4 Capability

M3가 평가할 프로토타입 capability는 다음과 같습니다.

- 한국어 음성을 transcript와 구조화된 최종 응답으로 생성
- 도구가 필요 없는 질문에 MCP를 호출하지 않고 답변
- 에어컨·선풍기·조명의 조회 및 허용된 명령 수행
- 멀티턴에서 이전 장치와 대화 맥락 유지
- 복수 tool call과 tool result를 사용해 최종 답변 생성
- 존재하지 않는 장치, 잘못된 인자, MCP 장애를 안전하게 설명
- tool loop limit에서 무한 반복 없이 최종 응답 또는 명시적 실패
- transcript/model/tool/TTS 실패를 하나의 trace에서 재구성
- 방1과 거실 입력을 구분하고 요청 위치에 맞는 출력으로 routing

마지막 멀티룸 capability는 오디오 endpoint가 준비된 뒤 acceptance하며, 나머지 capability의 평가 정의와 fixture는 M3 안에서 먼저 확정합니다.

## 4. M3 산출물과 완료 조건

- [x] opt-in Langfuse wrapper와 no-op 경로
- [x] 로컬 self-host Compose와 named volume
- [x] session/root/generation/parser/tool/TTS trace 구조
- [x] 입력 WAV, redaction, error observation과 종료 flush
- [x] transcript 수동 annotation queue와 기초 boolean score
- [ ] 도구 사용 score config와 평가 case 확정
- [ ] warm latency benchmark와 기준선 저장
- [ ] 최소 30개 regression seed dataset 구성
- [ ] capability별 fixture 및 실제 장치 acceptance 결과 기록
- [ ] 변경 전후 regression 실행 절차 자동화

M3는 미완료 항목을 모두 충족하고, self-host 콘솔에서 마지막 턴까지 조회·오디오 재생·score 입력이 가능할 때 완료합니다.

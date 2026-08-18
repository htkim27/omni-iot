# Omni-IoT 세부 개발 계획

이 문서는 구현 단위의 진행 상태, 검증 결과, 알려진 문제와 다음 작업을 관리합니다. 프로젝트의 큰 방향과 시작 방법은 루트 [README.md](../README.md)를 참고하세요.

## 1. 목표와 범위

최종 목표는 사설 로컬 환경에서 항상 대화할 수 있고, MCP 도구를 통해 IoT 기기를 제어하며, 여러 방의 입력 장치와 스피커를 구분해 사용할 수 있는 음성 어시스턴트입니다.

현재 개발 범위는 다음과 같습니다.

```text
브라우저 음성 입력
  → WebSocket 16kHz PCM streaming
  → openWakeWord (sleep → wake)
  → VAD / turn detection
  → Qwen3-Omni 음성 이해 및 텍스트 응답
  → OmniVoice 음성 합성
  → 브라우저 재생
```

MCP 도구 호출 하네스와 SwitchBot MCP 연결까지 구현했으며, 실제 장치 명령 검증과 Android 입력 장치 및 방별 라우팅이 다음 범위입니다.

## 2. 현재 구현 상태

### 2.1 저장소 및 런타임

- [x] `uv` 기반 Python 프로젝트 구성
- [x] `src/omni_iot` 패키지와 CLI entry point 구성
- [x] FastAPI/uvicorn 기반 로컬 서버 구성
- [x] `.env` 기반 런타임 설정 로딩
- [x] 턴별 입력/출력을 `.runtime/<turn-id>/`에 저장
- [x] UUID 형식의 runtime 턴 디렉터리를 최신 20개로 자동 정리
- [x] 핵심 pipeline 및 설정 전달 자동화 테스트 구성

지원 Python 범위는 `>=3.11,<3.13`입니다. 제공되는 명령은 다음과 같습니다.

| 명령 | 역할 |
| --- | --- |
| `omni-iot` | FastAPI 음성 하네스 실행 |
| `omni-iot-omni` | WAV를 Qwen3-Omni에 직접 입력 |
| `omni-iot-tts` | 텍스트를 OmniVoice WAV로 합성 |
| `omni-iot-wakeword-models` | 공식 openWakeWord 모델 준비 |
| `omni-iot-mcp` | MCP stdio/HTTP 서버 등록, allowlist 관리와 진단 |

### 2.2 브라우저 음성 하네스

- [x] 브라우저 마이크 권한 및 mono 입력 수집
- [x] 입력을 16kHz mono Int16 PCM으로 실시간 리샘플링
- [x] `/ws/audio` WebSocket을 통한 연속 PCM 전송
- [x] 서버의 openWakeWord `Hey Jarvis` 감지 및 wake event 전달
- [x] 호출어 종료 후 별도 명령 음성을 기다리는 command-wait 단계
- [x] sleep/recording/processing/speaking/follow-up 상태 전이
- [x] RMS threshold 기반 VAD
- [x] `.env` 기본값 기반 450ms pre-roll buffer
- [x] UI에서 조절 가능한 무음 기준 발화 종료(기본 600ms)
- [x] `.env` 기본값 기반 최대 14초 턴 제한
- [x] threshold UI 및 음량 meter
- [x] 응답 토큰 상한과 TTS 생성 단계 UI
- [x] 응답 WAV 자동 재생
- [x] TTS WAV가 없을 때 브라우저 Speech Synthesis fallback
- [x] 응답 재생 중 사용자 발화 감지 시 barge-in
- [x] 턴 응답과 전체 처리 시간 표시
- [x] 실제 마이크로 반복 대화하며 listening 복귀까지 최종 검증
- [x] 응답 재생 후 8초 후속 대화 window와 자동 sleep 복귀
- [ ] 환경별 echo와 배경 소음에서 VAD 안정성 측정
- [ ] deprecated `ScriptProcessorNode`를 AudioWorklet 기반으로 교체 검토

현재 발화 VAD는 브라우저의 RMS 크기만 사용하는 프로토타입입니다. 서버는 호출어 감지에만 openWakeWord를 사용하며 발화 종료에는 WebRTC VAD 같은 별도 음성 모델을 사용하지 않습니다. 기본 600ms 무음 또는 14초 최대 길이에 도달하면 턴이 끝나므로, 긴 발화는 `VAD_SILENCE_END_MS`와 `VAD_MAX_TURN_MS` 조정이 필요합니다.

UI에서 threshold, 발화 종료 대기, 응답 토큰 상한, TTS 생성 단계를 즉시 조절할 수 있으며 브라우저별 `localStorage`에 저장합니다. `.env`는 UI 최초 기본값과 pre-roll, 최대 턴, threshold 배수처럼 서버 시작 시 읽는 장비별 설정을 관리합니다.

### 2.3 대화 세션과 API

- [x] 세션 ID 발급 및 브라우저 `localStorage` 보관
- [x] `X-Session-Id` 기반 대화 연결
- [x] 사용자 음성 턴과 assistant 텍스트 기록
- [x] 세션 reset API
- [x] OMNI/TTS 설정 상태 health API
- [x] 비밀값을 제외한 UI 기본 설정 API
- [x] 생성된 WAV 전달 API와 runtime 경로 검증
- [x] 실제 대화 history를 다음 OMNI 프롬프트에 반영
- [x] WebSocket 연결별 wake/turn/follow-up 대화 처리
- [ ] 세션 만료 및 메모리 정리
- [ ] 동시 요청과 여러 사용자에 대한 안전성 보강
- [ ] 입력 WAV 형식과 크기 validation

현재 conversation store는 프로세스 메모리에만 존재해 서버 재시작 시 기록이 사라집니다. 같은 세션의 최근 사용자 transcript와 assistant 응답은 다음 OMNI 추론 context에 전달됩니다.

`.runtime` 정리는 서버 시작 및 새 턴 생성 시 실행됩니다. 기본 보존 개수는 `RUNTIME_TURN_LIMIT=20`이며, 자동 생성된 32자리 UUID 디렉터리만 대상으로 하므로 smoke test WAV나 사용자가 만든 다른 경로는 삭제하지 않습니다.

현재 API:

| Method | 경로 | 역할 |
| --- | --- | --- |
| `GET` | `/api/health` | OMNI/TTS 설정 여부 확인 |
| `GET` | `/api/config` | UI용 VAD/생성 기본값 확인 |
| `WS` | `/ws/audio` | 호출어 감지와 대화 턴용 실시간 PCM/event 스트림 |
| `POST` | `/api/turn` | 세션 기반 WAV 턴 처리 |
| `POST` | `/api/demo` | 세션 없는 단일 WAV 처리 |
| `POST` | `/api/session/reset` | 세션 기록 초기화 |
| `GET` | `/api/audio/{turn_id}/{filename}` | 턴별 출력 WAV 반환 |

### 2.4 Qwen3-Omni / llama.cpp

- [x] 로컬 GGUF 및 mmproj 자산 준비
- [x] `llama.cpp`의 CUDA build 생성
- [x] `llama-cli`, `llama-server`, `llama-mtmd-cli` build 확인
- [x] Python CLI wrapper 구현
- [x] WAV 입력, system prompt, user prompt 전달
- [x] system 지시와 audio-only user 메시지 분리
- [x] 텍스트 출력 정리 및 오류 전달
- [x] 실제 WAV → Qwen3-Omni → 텍스트 응답 검증
- [x] CPU 모드의 안정 동작 확인
- [x] CUDA 13.1/Blackwell 전용 build로 RTX 5070 Ti offload 오류 해결
- [x] GPU layer 수에 따른 VRAM/latency 측정
- [x] 관리형 `llama-server` 상시 로딩 및 server mode 전환
- [ ] 스트리밍 응답 검토

기본 로컬 자산:

```text
models/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf
models/mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf
vendor/llama.cpp/build-cuda131-sm120-gcc13/bin/llama-cli
```

현재 안정 기준 설정:

```dotenv
OMNI_COMMAND=uv run omni-iot-omni --audio {audio}
LLAMA_N_GPU_LAYERS=20
LLAMA_DEVICE=
LLAMA_OP_OFFLOAD=true
LLAMA_MMPROJ_OFFLOAD=true
LLAMA_CTX_SIZE=4096
LLAMA_N_PREDICT=192
LLAMA_TEMPERATURE=0.2
LLAMA_PARALLEL=1
LLAMA_THREADS=8
LLAMA_CACHE_PROMPT=true
LLAMA_FLASH_ATTN=off
LLAMA_WARMUP=false
```

기존 오류는 CUDA 12.4에서 Ada용 `sm_89`로 빌드한 바이너리를 Blackwell `sm_120` 장비에서 사용한 것이 원인이었습니다. CUDA Toolkit 13.1과 GCC 13으로 `sm_120a` 전용 빌드를 생성한 뒤 `SOFT_MAX failed / invalid argument`가 재현되지 않았습니다. 운영 안전 기준은 OMNI 20개 layer와 mmproj를 GPU에 올리는 구성으로, OmniVoice를 동시에 실행한 실제 요청에서 약 13.0GB VRAM과 3.84초 end-to-end latency를 확인했습니다. 16GB RTX 5070 Ti에서 26 layers는 OmniVoice 동시 상주 시 OOM이 발생했으므로 `.env.example`은 20 layers를 보수적 기본값으로 유지합니다.

현재 wrapper가 기본으로 참조하는 build 구성:

```bash
cmake -S vendor/llama.cpp \
  -B vendor/llama.cpp/build-cuda131-sm120-gcc13 \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA_NCCL=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCMAKE_BUILD_TYPE=Release

cmake --build vendor/llama.cpp/build-cuda131-sm120-gcc13 \
  --config Release --target llama-server llama-cli -j 8
```

### 2.5 OmniVoice TTS

- [x] `omnivoice` Python dependency 연결
- [x] `k2-fsa/OmniVoice` 로딩 wrapper 구현
- [x] CUDA 사용 가능 시 float16, 그 외 float32 선택
- [x] 한국어 language와 voice design instruction 설정
- [x] reference audio/text 기반 voice cloning 인자 지원
- [x] 24kHz 응답 WAV 생성 확인
- [x] command template 방식의 외부 TTS backend 지원
- [x] `/api/turn`의 `audio_url` 응답 검증
- [x] 서버 프로세스에서 모델을 사전 로드하고 턴 사이에 재사용
- [ ] 긴 응답의 chunk/streaming 합성
- [x] UI에서 턴별 생성 단계 조절
- [ ] TTS latency 및 VRAM 사용량 측정
- [x] OMNI 20-layer offload와 TTS 동시 상주 정책 확정

기본 설정:

```dotenv
TTS_BACKEND=omnivoice
OMNIVOICE_MODEL_ID=k2-fsa/OmniVoice
OMNIVOICE_LANGUAGE=ko
OMNIVOICE_INSTRUCT=male, korean accent, moderate pitch, young adult
OMNIVOICE_SPEED=
OMNIVOICE_NUM_STEPS=32
TTS_TIMEOUT_SECONDS=180
```

현재 FastAPI lifespan에서 OmniVoice를 한 번 로드하고 워밍업하며, 이후 모든 턴이 같은 모델 인스턴스를 재사용합니다. 동시 요청의 모델 상태 충돌을 막기 위해 합성 호출은 직렬화합니다.

Smoke test:

```bash
uv run omni-iot-tts \
  --text "안녕하세요. 로컬 음성 출력 테스트입니다." \
  --language ko \
  --instruct "male, korean accent, moderate pitch, young adult" \
  --output .runtime/omnivoice_smoke.wav
```

### 2.6 MCP 클라이언트와 IoT 도구

대표 milestone은 `M2 — MCP Client & IoT Tool Integration`이며 `Phase 4`에 대응합니다.

- [x] 공식 MCP Python SDK `mcp>=2,<3` 기반 stdio/Streamable HTTP 연결
- [x] `.mcp.json` 설정, exact allowlist, 환경변수 치환과 atomic 관리 CLI
- [x] Qwen3-Omni/llama-server OpenAI tool-call orchestration
- [x] 서버별 fail-closed, 재연결, catalog/result 제한과 최소 trace
- [x] SwitchBot 공식 CLI 인증 및 `switchbot mcp serve` stdio 연결
- [x] `doctor`에서 SwitchBot 연결 `healthy`와 허용 도구 5개 확인 (2026-08-19)
- [ ] 실제 Qwen 음성으로 SwitchBot 장치 조회와 비위험 명령 검증

실제 API 키, 인증 토큰과 로컬 실행 파일 경로는 milestone 문서나 git 추적 파일에 기록하지 않습니다. `.env`와 `.mcp.json`은 모두 gitignore 대상입니다.

## 3. 설정 및 command hook

프로젝트의 `.env.example`을 `.env`로 복사해 사용합니다. `.env` loader는 이미 셸에 존재하는 환경 변수 값을 덮어쓰지 않습니다.

OMNI command에서 사용할 수 있는 placeholder:

| Placeholder | 값 |
| --- | --- |
| `{audio}` | 입력 WAV 절대 경로 |
| `{input}` | `{audio}`와 동일 |
| `{history_file}` | 이전 대화 기록을 담은 UTF-8 JSON 파일 |

TTS command에서 사용할 수 있는 placeholder:

| Placeholder | 값 |
| --- | --- |
| `{text}` | shell 인자용으로 escape된 응답 텍스트 |
| `{text_file}` | 응답 텍스트를 담은 UTF-8 임시 파일 |
| `{output}` | 생성해야 할 WAV 절대 경로 |

외부 backend 예시:

```dotenv
OMNI_COMMAND=/path/to/omni --audio {audio}
TTS_BACKEND=command
TTS_COMMAND=/path/to/tts --text-file {text_file} --output {output}
```

## 4. 검증 현황

완료된 검증:

- [x] Python dependency 설치 및 CLI entry point 실행
- [x] FastAPI 페이지와 API 로컬 실행
- [x] 브라우저 WAV 생성 및 API 업로드
- [x] 브라우저 WebSocket PCM streaming 및 wake event
- [x] model/mmproj 로딩
- [x] 실제 WAV에서 Qwen3-Omni 텍스트 생성
- [x] OmniVoice 단독 smoke WAV 생성
- [x] 실제 OMNI 텍스트로 TTS WAV 생성
- [x] `/api/turn`에서 재생 가능한 `audio_url` 반환
- [x] `Hey Jarvis → 명령 대기 → 응답 → follow-up → sleep` 반복 흐름
- [x] system 지시와 user 오디오 분리
- [x] MCP 설정/CLI/allowlist 회귀 테스트
- [x] MCP pagination, structured/text/error 결과와 catalog 상한 테스트
- [x] 단일·복수 tool call, 순차 실행, 잘못된 인자와 호출 한도 테스트
- [x] llama 요청 tools와 assistant/tool 메시지 재전달 테스트
- [x] SwitchBot MCP 인증, stdio 연결 및 허용 도구 5개 discovery

남은 end-to-end 검증:

- [ ] 다양한 거리·억양에서 `Hey Jarvis` false reject/accept 측정
- [ ] echo와 배경 소음 환경의 VAD 보정
- [ ] 긴 발화의 무음 종료 및 최대 턴 설정 검증
- [ ] 응답 도중 barge-in 장시간 반복
- [ ] 20~24 GPU layers별 peak VRAM과 latency 비교
- [ ] 실제 Qwen 한국어 음성으로 fixture MCP 선택 → 호출 → 최종 TTS
- [ ] 일반 질문에서 불필요한 tool call이 없는지 실제 Qwen 검증
- [ ] SwitchBot 장치 목록/상태/비위험 turnOn·turnOff/오류 acceptance

pipeline과 설정 전달 경로는 자동화된 회귀 테스트로 확인하며, 실제 음질과 브라우저 마이크 동작은 로컬 smoke test로 검증합니다.

## 5. 단계별 로드맵

### Phase 0 — 저장소 기반 구성

상태: 완료

- [x] Python/uv 프로젝트 초기화
- [x] 패키지 구조와 환경 설정
- [x] 모델 및 runtime artifact 경로 분리

### Phase 1 — 로컬 OMNI 추론

상태: GPU partial offload 기준 완료, 추가 최적화 진행 중

- [x] llama.cpp wrapper와 로컬 모델 연결
- [x] 음성 입력 → 텍스트 응답 검증
- [x] RTX 5070 Ti offload 안정화
- [x] latency와 VRAM benchmark

### Phase 2 — 실시간 음성 루프

상태: 호출어 기반 연속 대화 프로토타입 구현, 환경별 튜닝 진행 중

- [x] 브라우저 마이크/VAD/WebSocket PCM streaming
- [x] 로컬 openWakeWord sleep/wake gate
- [x] 호출 후 명령 대기와 follow-up timeout
- [x] 재생 queue의 기본 동작
- [x] barge-in 기본 경로
- [ ] 다양한 소음 환경에서 turn detection 보정
- [x] 반복 대화 기본 흐름 검증

### Phase 3 — OMNI → TTS 통합

상태: 파일 파이프라인 완료, latency 최적화 필요

- [x] OmniVoice 연결 및 한국어 WAV 출력
- [x] OMNI 응답을 TTS로 전달
- [x] OMNI 및 TTS 모델 상시 로딩
- [ ] chunk 또는 streaming TTS
- [x] 응답 길이와 TTS 생성 단계의 턴별 조절
- [x] warm 반복 요청 latency 기준선 측정

### Phase 4 — MCP 클라이언트

상태: 하네스 구현 완료, 실제 Qwen/SwitchBot acceptance 대기

- [x] `.mcp.json` v1 설정, 환경변수 치환과 atomic CLI 관리
- [x] 공식 MCP Python SDK stdio/Streamable HTTP 장기 연결
- [x] pagination, exact allowlist, namespaced OpenAI tool catalog
- [x] structured/text/error 결과 정규화, 크기/시간 제한과 재연결
- [x] Qwen3-Omni/llama-server OpenAI tools agent loop
- [x] 최대 4 round/8 call, 순차 실행과 강제 최종 응답
- [x] 서버별 fail-closed 격리와 `/api/health`, `doctor` 진단
- [x] tool trace 최소 기록 및 최종 대화만 history에 보존
- [ ] 실제 Qwen 음성 tool smoke test
- [x] 공식 SwitchBot CLI 설치·인증 및 5개 도구 discovery
- [ ] 실제 SwitchBot 장치 조회·비위험 명령·오류 acceptance

v1은 tools만 연결하며 resources, prompts, sampling, elicitation, tasks와 웹 관리 UI는 제외합니다. `.mcp.json`의 allowlist를 권한 경계로 사용하고 SwitchBot 자체 안전 정책은 그대로 유지합니다.

### Phase 5 — IoT 및 방별 장치

상태: 미착수

- [ ] Android phone을 always-on 센서/스피커로 사용하는 bridge 설계
- [ ] Bluetooth 스피커와 입력 센서 조합 검토
- [ ] WebSocket, HTTP streaming, MQTT 중 transport 결정
- [ ] device identity와 room identity 정의
- [ ] 방별 응답 출력 routing
- [ ] 네트워크 단절과 재연결 처리

## 6. 우선순위

다음 순서로 진행합니다.

1. 다양한 실제 환경에서 호출어와 VAD 민감도 보정
2. AudioWorklet 전환과 장시간 WebSocket 안정성 검증
3. 16GB VRAM 안에서 OMNI/TTS latency 추가 최적화
4. transcript 오류 방어와 세션 수명 관리 보강
5. 실제 Qwen 음성 및 SwitchBot acceptance
6. Android 및 방별 오디오 장치 연결

## 7. 완료 조건

현재 로컬 음성 하네스 단계는 아래 조건을 만족하면 완료로 봅니다.

- 브라우저에서 별도 파일 조작 없이 여러 턴 연속 대화 가능
- 사용자 발화 종료와 barge-in이 일반적인 실내 소음에서 안정적으로 동작
- 실제 Qwen3-Omni와 OmniVoice가 오류 없이 반복 실행
- 목표 장비에서 허용 가능한 응답 지연과 VRAM 사용량 측정 완료
- 모델/명령 오류가 UI와 로그에 진단 가능한 형태로 노출
- 핵심 pipeline과 API에 자동화된 회귀 테스트 존재

MCP 하네스 완료 조건에는 자동화 테스트 전체 통과와 함께 실제 Qwen 음성 fixture 및 SwitchBot 계정/장치 smoke test가 포함됩니다. 후자는 Node.js 18+, `@switchbot/openapi-cli`, 사용자 인증과 실제 비위험 장치가 준비된 목표 장비에서 수행합니다.

# Omni-IoT 세부 개발 계획

이 문서는 구현 단위의 진행 상태, 검증 결과, 알려진 문제와 다음 작업을 관리합니다. 프로젝트의 큰 방향과 시작 방법은 루트 [README.md](../README.md)를 참고하세요.

## 1. 목표와 범위

최종 목표는 사설 로컬 환경에서 항상 대화할 수 있고, MCP 도구를 통해 IoT 기기를 제어하며, 여러 방의 입력 장치와 스피커를 구분해 사용할 수 있는 음성 어시스턴트입니다.

현재 개발 범위는 다음과 같습니다.

```text
브라우저 음성 입력
  → VAD / turn detection
  → Qwen3-Omni 음성 이해 및 텍스트 응답
  → OmniVoice 음성 합성
  → 브라우저 재생
```

MCP 도구 호출, 실제 IoT 제어, Android 입력 장치 및 방별 라우팅은 로컬 대화 루프가 안정화된 뒤 진행합니다.

## 2. 현재 구현 상태

### 2.1 저장소 및 런타임

- [x] `uv` 기반 Python 프로젝트 구성
- [x] `src/omni_iot` 패키지와 CLI entry point 구성
- [x] FastAPI/uvicorn 기반 로컬 서버 구성
- [x] `.env` 기반 런타임 설정 로딩
- [x] 턴별 입력/출력을 `.runtime/<turn-id>/`에 저장
- [x] UUID 형식의 runtime 턴 디렉터리를 최신 20개로 자동 정리
- [ ] 자동화된 테스트 구성

지원 Python 범위는 `>=3.11,<3.13`입니다. 제공되는 명령은 다음과 같습니다.

| 명령 | 역할 |
| --- | --- |
| `omni-iot` | FastAPI 음성 하네스 실행 |
| `omni-iot-omni` | WAV를 Qwen3-Omni에 직접 입력 |
| `omni-iot-tts` | 텍스트를 OmniVoice WAV로 합성 |

### 2.2 브라우저 음성 하네스

- [x] 브라우저 마이크 권한 및 mono 입력 수집
- [x] 입력을 16-bit PCM WAV로 인코딩
- [x] RMS threshold 기반 VAD
- [x] 450ms pre-roll buffer
- [x] 850ms 무음 기준 발화 종료
- [x] 최대 14초 턴 제한
- [x] threshold UI 및 음량 meter
- [x] 응답 WAV 자동 재생
- [x] TTS WAV가 없을 때 브라우저 Speech Synthesis fallback
- [x] 응답 재생 중 사용자 발화 감지 시 barge-in
- [x] 턴 응답과 전체 처리 시간 표시
- [ ] 실제 마이크로 반복 대화하며 listening 복귀까지 최종 검증
- [ ] 환경별 echo와 배경 소음에서 VAD 안정성 측정
- [ ] deprecated `ScriptProcessorNode`를 AudioWorklet 기반으로 교체 검토

현재 VAD는 브라우저의 RMS 크기만 사용하는 프로토타입입니다. 서버 측 음성 모델이나 WebRTC VAD는 아직 사용하지 않습니다.

### 2.3 대화 세션과 API

- [x] 세션 ID 발급 및 브라우저 `localStorage` 보관
- [x] `X-Session-Id` 기반 대화 연결
- [x] 사용자 음성 턴과 assistant 텍스트 기록
- [x] 세션 reset API
- [x] OMNI/TTS 설정 상태 health API
- [x] 생성된 WAV 전달 API와 runtime 경로 검증
- [ ] 실제 대화 history를 다음 OMNI 프롬프트에 반영
- [ ] 세션 만료 및 메모리 정리
- [ ] 동시 요청과 여러 사용자에 대한 안전성 보강
- [ ] 입력 WAV 형식과 크기 validation

현재 conversation store는 프로세스 메모리에만 존재합니다. 서버 재시작 시 기록이 사라지며, transcript는 UI 표시용 상태일 뿐 OMNI 추론 context에는 아직 전달되지 않습니다.

`.runtime` 정리는 서버 시작 및 새 턴 생성 시 실행됩니다. 기본 보존 개수는 `RUNTIME_TURN_LIMIT=20`이며, 자동 생성된 32자리 UUID 디렉터리만 대상으로 하므로 smoke test WAV나 사용자가 만든 다른 경로는 삭제하지 않습니다.

현재 API:

| Method | 경로 | 역할 |
| --- | --- | --- |
| `GET` | `/api/health` | OMNI/TTS 설정 여부 확인 |
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
- [x] 텍스트 출력 정리 및 오류 전달
- [x] 실제 WAV → Qwen3-Omni → 텍스트 응답 검증
- [x] CPU 모드의 안정 동작 확인
- [ ] RTX 5070 Ti CUDA offload 오류 해결
- [ ] GPU layer 수에 따른 VRAM/latency 측정
- [x] 관리형 `llama-server` 상시 로딩 및 server mode 전환
- [ ] 스트리밍 응답 검토

기본 로컬 자산:

```text
models/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf
models/mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf
vendor/llama.cpp/build-cuda124-sm89/bin/llama-cli
```

현재 안정 기준 설정:

```dotenv
OMNI_COMMAND=uv run omni-iot-omni --audio {audio}
LLAMA_N_GPU_LAYERS=0
LLAMA_DEVICE=none
LLAMA_OP_OFFLOAD=false
LLAMA_MMPROJ_OFFLOAD=false
LLAMA_CTX_SIZE=4096
LLAMA_N_PREDICT=192
LLAMA_FLASH_ATTN=off
LLAMA_WARMUP=false
```

CPU 기준선에서는 전체 파이프라인이 동작합니다. RTX 5070 Ti에서 Qwen3-Omni를 CUDA로 오프로딩하면 현재 `SOFT_MAX failed / invalid argument` 오류가 발생합니다. GPU 문제가 해결되기 전에는 `LLAMA_N_GPU_LAYERS=0`을 유지합니다.

현재 wrapper가 기본으로 참조하는 build 구성:

```bash
cmake -S vendor/llama.cpp \
  -B vendor/llama.cpp/build-cuda124-sm89 \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build vendor/llama.cpp/build-cuda124-sm89 \
  --config Release -j "$(nproc)"
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
- [ ] TTS latency 및 VRAM 사용량 측정
- [ ] OMNI와 TTS의 GPU 메모리 전환 정책 확정

기본 설정:

```dotenv
TTS_BACKEND=omnivoice
OMNIVOICE_MODEL_ID=k2-fsa/OmniVoice
OMNIVOICE_LANGUAGE=ko
OMNIVOICE_INSTRUCT=male, korean accent, moderate pitch, young adult
OMNIVOICE_SPEED=
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

## 3. 설정 및 command hook

프로젝트의 `.env.example`을 `.env`로 복사해 사용합니다. `.env` loader는 이미 셸에 존재하는 환경 변수 값을 덮어쓰지 않습니다.

OMNI command에서 사용할 수 있는 placeholder:

| Placeholder | 값 |
| --- | --- |
| `{audio}` | 입력 WAV 절대 경로 |
| `{input}` | `{audio}`와 동일 |

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
- [x] model/mmproj 로딩
- [x] 실제 WAV에서 Qwen3-Omni 텍스트 생성
- [x] OmniVoice 단독 smoke WAV 생성
- [x] 실제 OMNI 텍스트로 TTS WAV 생성
- [x] `/api/turn`에서 재생 가능한 `audio_url` 반환

남은 end-to-end 검증:

- [ ] 브라우저 Start
- [ ] 실제 사용자 발화 감지
- [ ] 자동 turn 종료와 업로드
- [ ] 실제 OMNI 응답
- [ ] 실제 TTS 재생
- [ ] 재생 종료 후 listening 복귀
- [ ] 다음 발화 반복
- [ ] 응답 도중 barge-in 반복

자동화된 테스트가 아직 없으므로 현재 완료 표시는 로컬 smoke test 기준입니다.

## 5. 단계별 로드맵

### Phase 0 — 저장소 기반 구성

상태: 완료

- [x] Python/uv 프로젝트 초기화
- [x] 패키지 구조와 환경 설정
- [x] 모델 및 runtime artifact 경로 분리

### Phase 1 — 로컬 OMNI 추론

상태: CPU 기준 완료, GPU 최적화 진행 중

- [x] llama.cpp wrapper와 로컬 모델 연결
- [x] 음성 입력 → 텍스트 응답 검증
- [ ] RTX 5070 Ti offload 안정화
- [ ] latency와 VRAM benchmark

### Phase 2 — 실시간 음성 루프

상태: 프로토타입 구현, 실사용 검증 진행 중

- [x] 브라우저 마이크/VAD/WAV 업로드
- [x] 재생 queue의 기본 동작
- [x] barge-in 기본 경로
- [ ] 다양한 소음 환경에서 turn detection 보정
- [ ] 반복 대화 안정성 검증

### Phase 3 — OMNI → TTS 통합

상태: 파일 파이프라인 완료, latency 최적화 필요

- [x] OmniVoice 연결 및 한국어 WAV 출력
- [x] OMNI 응답을 TTS로 전달
- [x] OMNI 및 TTS 모델 상시 로딩
- [ ] chunk 또는 streaming TTS
- [x] warm 반복 요청 latency 기준선 측정

### Phase 4 — MCP 클라이언트

상태: 미착수

- [ ] MCP server 설정 형식 정의
- [ ] server 연결과 tool 목록 조회
- [ ] tool 호출 및 결과 반환
- [ ] OMNI 대화 루프에 tool-call 경계 연결
- [ ] timeout, 실패, 사용자 확인 정책 정의

초기에는 MCP 클라이언트 기능만 구현하며 실제 IoT 도구는 별도 단계로 둡니다.

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

1. 브라우저 기반 전체 대화 반복 루프를 실제 환경에서 검증
2. Qwen3-Omni의 RTX 5070 Ti CUDA 오류 재현 조건과 호환 build 확인
3. OMNI/TTS 모델 상시 로딩으로 턴 지연 단축
4. 세션 history를 실제 추론 context에 반영
5. 자동화된 pipeline/API 테스트 추가
6. MCP 클라이언트 설계와 최소 tool 호출 구현
7. Android 및 방별 오디오 장치 연결

## 7. 완료 조건

현재 로컬 음성 하네스 단계는 아래 조건을 만족하면 완료로 봅니다.

- 브라우저에서 별도 파일 조작 없이 여러 턴 연속 대화 가능
- 사용자 발화 종료와 barge-in이 일반적인 실내 소음에서 안정적으로 동작
- 실제 Qwen3-Omni와 OmniVoice가 오류 없이 반복 실행
- 목표 장비에서 허용 가능한 응답 지연과 VRAM 사용량 측정 완료
- 모델/명령 오류가 UI와 로그에 진단 가능한 형태로 노출
- 핵심 pipeline과 API에 자동화된 회귀 테스트 존재

이 조건을 충족한 뒤 MCP 및 IoT 장치 단계로 넘어갑니다.

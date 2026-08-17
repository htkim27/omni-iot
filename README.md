# omni-iot

로컬 환경에서 동작하는 Jarvis 스타일의 실시간 음성 IoT 어시스턴트 프로젝트입니다.

브라우저 또는 방별 음성 장치에서 사용자의 말을 받고, speech-aware OMNI 모델이 음성을 직접 이해해 응답을 만든 뒤 TTS로 재생하는 구조를 목표로 합니다. 대화 기반이 안정화되면 MCP를 통해 스마트홈과 IoT 도구를 연결합니다.

> 핵심 파이프라인은 일반적인 `STT → LLM → TTS`가 아니라 `Audio → OMNI → Text → TTS`입니다.

## 현재 진행 상황

현재는 로컬 음성 대화 하네스의 첫 번째 end-to-end 프로토타입을 구현한 상태입니다.

- 브라우저 마이크 입력, `Hey Jarvis` 로컬 호출어, WebSocket PCM 스트리밍 및 응답 재생 구현
- `sleeping → recording → processing → speaking → follow_up` 상태 기반 연속 대화 구현
- Qwen3-Omni와 `llama.cpp`를 이용한 음성 입력 → 텍스트 응답 연결
- k2-fsa/OmniVoice를 이용한 텍스트 → 음성 응답 연결
- 대화 세션, 응답 중 끼어들기, 처리 시간 표시 등 기본 대화 기능 구현
- VAD 종료 시간, 응답 길이, TTS 생성 단계를 실행 중 UI에서 조절 가능
- 사용자 발화 transcript와 최근 대화 history를 다음 OMNI 턴에 전달하는 멀티턴 구현
- `llama-server`와 OmniVoice 모델을 서버 수명 동안 유지해 턴별 모델 재로딩 제거
- 파일 단위의 실제 `OMNI → TTS` 전체 파이프라인 검증 완료
- CUDA 13.1/Blackwell 빌드에서 RTX 5070 Ti partial GPU 오프로딩 검증 완료
- MCP 기반 IoT 제어와 방별 입출력 장치 연동은 이후 단계

구현 단위의 상태, 알려진 문제와 다음 작업은 [docs/plan.md](docs/plan.md)를, 장기 설계 원칙은 [ADR 0001](docs/adr/0001-design-philosophy.md)을 참고하세요.

## 아키텍처

```text
마이크 / 방별 입력 장치
  → openWakeWord `Hey Jarvis` 감지 (sleep → wake)
  → VAD 및 발화 구간 감지
  → Qwen3-Omni (llama.cpp)
  → 텍스트 응답 및 대화 상태
  → MCP 도구 호출 (예정)
  → OmniVoice TTS
  → 브라우저 / 방별 스피커
```

현재 구현은 브라우저를 입출력 장치로 사용합니다. 브라우저 입력을 16kHz 16-bit mono PCM으로 리샘플링해 `/ws/audio`로 계속 보내고, FastAPI의 경량 openWakeWord가 `Hey Jarvis`를 감지합니다. 호출어가 끝난 뒤 별도의 명령 음성이 시작될 때까지 기다리므로 호출어 자체가 하나의 질문으로 처리되지 않습니다. 발화가 끝나면 PCM을 WAV로 감싸 상시 실행 중인 `llama-server`에 전달하고, 프로세스 내 OmniVoice가 응답 음성을 생성합니다.

응답 재생이 끝나면 8초 동안 호출어 없이 후속 질문을 받을 수 있습니다. 이 시간 안에 말하지 않으면 자동으로 sleep 상태로 돌아갑니다. OMNI 요청에서는 역할·출력 형식 지시를 system 메시지에 두고 user 메시지에는 오디오만 전달해, 모델이 내부 지시문을 사용자 transcript로 복사할 가능성을 줄였습니다.

턴별 입력과 출력은 `.runtime/`에 저장하며 최신 20개만 유지합니다.

## 빠른 시작

### 요구 사항

- Python 3.11 이상, 3.13 미만
- [uv](https://docs.astral.sh/uv/)
- 마이크를 사용할 수 있는 최신 브라우저
- 실제 추론 시 로컬 Qwen3-Omni GGUF 모델과 `llama.cpp`
- 실제 음성 합성 시 k2-fsa/OmniVoice 모델

### 설치 및 실행

```bash
uv sync
# 공식 Hey Jarvis 모델 준비(최초 한 번)
uv run omni-iot-wakeword-models
cp .env.example .env
uv run omni-iot --host 127.0.0.1 --port 8000
```

모델 다운로드 명령은 최초 준비 단계에서만 인터넷을 사용합니다. 이후 호출어 인식과 음성 대화는 로컬 파일만 사용하며 실행 중 모델을 자동 다운로드하지 않습니다. 기본 모델은 openWakeWord가 제공하는 `hey_jarvis_v0.1.onnx`이고 호출 문구는 영어 `Hey Jarvis`입니다. Air-gapped 환경에서는 모델과 `melspectrogram.onnx`, `embedding_model.onnx`를 `models/openwakeword/`에 미리 복사하면 됩니다.

첫 실행에서는 Qwen3-Omni와 OmniVoice를 메모리에 올린 뒤 서버가 준비되므로 시간이 걸릴 수 있습니다. 종료 시 함께 시작된 `llama-server`도 자동으로 종료됩니다.

브라우저에서 <http://127.0.0.1:8000>을 열고 `Start`를 눌러 마이크 권한을 허용합니다.

기본 사용 흐름은 다음과 같습니다.

1. 화면이 `Say “Hey Jarvis”` 상태인지 확인합니다.
2. “Hey Jarvis”라고 말하고 잠깐 멈춥니다.
3. `Listening for command`가 표시되면 명령을 말합니다.
4. 응답 재생 후 8초 안에는 호출어 없이 후속 질문을 이어갈 수 있습니다.

`.env.example`은 RTX 5070 Ti에서 검증한 CUDA 13.1 기반 `llama-server` 20-layer offload와 OmniVoice 설정을 포함합니다. 해당 장비에서는 OMNI와 TTS를 동시에 실행해 약 13.0GB VRAM, warm end-to-end 3.84초를 확인했습니다. 모델 없이 브라우저 음성 흐름만 시험하려면 `OMNI_BACKEND=command`로 바꾸고 `OMNI_COMMAND`를 비워 mock 응답을 사용할 수 있습니다.

`.env`는 API 키 전용 파일이 아니라 장비·실행 환경별 설정과 비밀값을 함께 두는 파일입니다. 현재는 모델 경로, GPU/서버 설정, 호출어, VAD 기본값과 생성 옵션을 관리합니다. 각 값의 의미·단위·조절 효과는 [.env.example](.env.example)에 한글 주석으로 설명했습니다. 실제 비밀값은 커밋하지 않는 `.env`에만 넣습니다.

Threshold, 발화 종료 대기, 응답 토큰 상한, TTS 단계는 UI에서 즉시 변경할 수 있으며 브라우저별 `localStorage`에 저장됩니다. 이 저장값은 이후 접속에서도 `.env` 기본값보다 우선합니다. 긴 문장이 중간에 잘리면 UI의 발화 종료 대기를 1200~1500ms로 늘리고, 14초 제한 자체를 늘리려면 `VAD_MAX_TURN_MS`를 수정한 뒤 서버를 재시작합니다.

### RTX 5070 Ti용 llama.cpp 빌드

CUDA Toolkit 13.1과 GCC 13을 사용해 Blackwell `sm_120a` 전용 바이너리를 생성합니다.

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

## 주요 명령

서버 실행:

```bash
uv run omni-iot
```

Qwen3-Omni 단독 실행:

```bash
uv run omni-iot-omni --audio path/to/input.wav
```

OmniVoice 단독 실행:

```bash
uv run omni-iot-tts \
  --text "안녕하세요. 음성 출력 테스트입니다." \
  --language ko \
  --output .runtime/tts-smoke.wav
```

## 프로젝트 구조

```text
.
├── docs/plan.md                 # 세부 진행 상황과 단계별 계획
├── docs/adr/                    # 장기 설계 결정 기록
├── models/                      # 로컬 GGUF 모델
├── src/omni_iot/
│   ├── config.py                # 환경 설정
│   ├── conversation.py          # 대화 세션
│   ├── omni_llama.py            # llama.cpp OMNI wrapper
│   ├── pipeline.py              # OMNI → TTS 파이프라인
│   ├── server.py                # FastAPI 서버
│   ├── tts_omnivoice.py         # OmniVoice wrapper
│   ├── wakeword.py              # openWakeWord 스트리밍 감지기
│   └── static/                  # 브라우저 음성 UI
├── .env.example
├── pyproject.toml
└── uv.lock
```

## 로드맵

1. 로컬 OMNI → TTS 대화 루프 안정화
2. GPU 오프로딩과 응답 지연 최적화
3. MCP 클라이언트 및 IoT 도구 연동
4. Android 등 always-on 음성 장치 연결
5. 방별 입력 감지와 스피커 출력 라우팅

현재 개발 범위는 1단계와 2단계입니다.

## 테스트

모델을 실제로 실행하지 않는 자동화 테스트는 다음 명령으로 확인합니다.

```bash
TTS_BACKEND=command OMNI_BACKEND=command \
  uv run python -m unittest discover -s tests -v
```

WebSocket 상태 전이, 호출어 프레임 처리, 대화 history, OMNI 응답 파싱, runtime 정리를 포함합니다. 실제 마이크 음질, 호출어 민감도와 GPU peak VRAM은 목표 장비에서 별도 smoke test가 필요합니다.

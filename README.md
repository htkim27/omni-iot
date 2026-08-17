# omni-iot

로컬 환경에서 동작하는 Jarvis 스타일의 실시간 음성 IoT 어시스턴트 프로젝트입니다.

브라우저 또는 방별 음성 장치에서 사용자의 말을 받고, speech-aware OMNI 모델이 음성을 직접 이해해 응답을 만든 뒤 TTS로 재생하는 구조를 목표로 합니다. 대화 기반이 안정화되면 MCP를 통해 스마트홈과 IoT 도구를 연결합니다.

> 핵심 파이프라인은 일반적인 `STT → LLM → TTS`가 아니라 `Audio → OMNI → Text → TTS`입니다.

## 현재 진행 상황

현재는 로컬 음성 대화 하네스의 첫 번째 end-to-end 프로토타입을 구현한 상태입니다.

- 브라우저 마이크 입력, 발화 감지, WAV 업로드 및 응답 재생 구현
- Qwen3-Omni와 `llama.cpp`를 이용한 음성 입력 → 텍스트 응답 연결
- k2-fsa/OmniVoice를 이용한 텍스트 → 음성 응답 연결
- 대화 세션, 응답 중 끼어들기, 처리 시간 표시 등 기본 대화 기능 구현
- VAD 종료 시간, 응답 길이, TTS 생성 단계를 실행 중 UI에서 조절 가능
- 사용자 발화 transcript와 최근 대화 history를 다음 OMNI 턴에 전달하는 멀티턴 구현
- `llama-server`와 OmniVoice 모델을 서버 수명 동안 유지해 턴별 모델 재로딩 제거
- 파일 단위의 실제 `OMNI → TTS` 전체 파이프라인 검증 완료
- CUDA 13.1/Blackwell 빌드에서 RTX 5070 Ti partial GPU 오프로딩 검증 완료
- MCP 기반 IoT 제어와 방별 입출력 장치 연동은 이후 단계

구현 단위의 상태, 알려진 문제와 다음 작업은 [docs/plan.md](docs/plan.md)를 참고하세요.

## 아키텍처

```text
마이크 / 방별 입력 장치
  → VAD 및 발화 구간 감지
  → Qwen3-Omni (llama.cpp)
  → 텍스트 응답 및 대화 상태
  → MCP 도구 호출 (예정)
  → OmniVoice TTS
  → 브라우저 / 방별 스피커
```

현재 구현은 브라우저를 입출력 장치로 사용합니다. 브라우저에서 16-bit mono WAV를 생성해 FastAPI 서버로 보내고, FastAPI가 상시 실행 중인 `llama-server`에 음성을 전달한 뒤 프로세스 내 OmniVoice 모델로 응답 WAV를 생성합니다.

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
cp .env.example .env
uv run omni-iot --host 127.0.0.1 --port 8000
```

첫 실행에서는 Qwen3-Omni와 OmniVoice를 메모리에 올린 뒤 서버가 준비되므로 시간이 걸릴 수 있습니다. 종료 시 함께 시작된 `llama-server`도 자동으로 종료됩니다.

브라우저에서 <http://127.0.0.1:8000>을 열고 `Start`를 눌러 마이크 권한을 허용합니다.

`.env.example`은 RTX 5070 Ti에서 검증한 CUDA 13.1 기반 `llama-server` 20-layer offload와 OmniVoice 설정을 포함합니다. 해당 장비에서는 OMNI와 TTS를 동시에 실행해 약 13.0GB VRAM, warm end-to-end 3.84초를 확인했습니다. 모델 없이 브라우저 음성 흐름만 시험하려면 `OMNI_BACKEND=command`로 바꾸고 `OMNI_COMMAND`를 비워 mock 응답을 사용할 수 있습니다.

`.env`는 API 키 전용 파일이 아니라 장비·실행 환경별 설정과 비밀값을 함께 두는 파일입니다. 현재는 모델 경로, GPU/서버 설정, VAD 기본값, 생성 기본값을 관리합니다. 실제 비밀값은 커밋하지 않는 `.env`에만 넣고, `.env.example`에는 이름과 안전한 예시값만 기록합니다. Threshold, 발화 종료 대기, 응답 토큰 상한, TTS 단계는 서버 기본값을 UI에 불러온 뒤 실행 중 변경할 수 있으며 브라우저별로 저장됩니다. 나머지 값은 `.env` 변경 후 서버를 재시작해야 적용됩니다.

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
├── models/                      # 로컬 GGUF 모델
├── src/omni_iot/
│   ├── config.py                # 환경 설정
│   ├── conversation.py          # 대화 세션
│   ├── omni_llama.py            # llama.cpp OMNI wrapper
│   ├── pipeline.py              # OMNI → TTS 파이프라인
│   ├── server.py                # FastAPI 서버
│   ├── tts_omnivoice.py         # OmniVoice wrapper
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

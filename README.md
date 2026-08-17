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
- 파일 단위의 실제 `OMNI → TTS` 전체 파이프라인 검증 완료
- RTX 5070 Ti GPU 오프로딩 안정화 및 브라우저 전체 반복 루프 검증 진행 중
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

현재 구현은 브라우저를 입출력 장치로 사용합니다. 브라우저에서 16-bit mono WAV를 생성해 FastAPI 서버로 보내고, 서버가 OMNI 추론과 TTS를 순서대로 실행한 뒤 응답 WAV를 돌려줍니다.

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

브라우저에서 <http://127.0.0.1:8000>을 열고 `Start`를 눌러 마이크 권한을 허용합니다.

`.env.example`은 현재 검증된 CPU 기반 OMNI 설정과 OmniVoice TTS 설정을 포함합니다. `OMNI_COMMAND`가 비어 있으면 실제 모델 대신 mock 응답을 사용하므로 모델 없이도 브라우저 음성 흐름을 시험할 수 있습니다.

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

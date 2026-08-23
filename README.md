# omni-iot

로컬 환경에서 동작하는 Jarvis 스타일의 실시간 음성 IoT 어시스턴트 프로젝트입니다.

## 🎯 프로젝트 목표 (Project Goals)

이 프로젝트는 다음과 같은 궁극적인 목표를 향해 나아갑니다:
- **Always-on Customized Home Chat Bot**: 언제나 사용자의 부름에 응답할 준비가 되어 있는 나만의 맞춤형 홈 비서입니다.
- **Low Latency**: 실제 사람과 대화하듯 빠르고 자연스러운 반응 속도(지연 시간 최소화)를 지향합니다.
- **Bond based on Customized Memory**: 사용자와의 이전 대화와 맥락을 기억하고, 이를 바탕으로 깊은 유대감을 형성하는 반려 AI를 목표로 합니다.

## 🏗 아키텍처 (Architecture)

기존의 `음성 인식(STT) → 텍스트 처리(LLM) → 음성 합성(TTS)` 이라는 지연이 긴 파이프라인에서 벗어나, 사용자의 음성을 직접 이해하고 반응하는 **End-to-End 오디오 파이프라인**으로 구성되어 있습니다.

사용자 입장에서의 대화 흐름은 다음과 같습니다:

1. **Wake Model (Custom Naming)**:
   - 사용자가 원하는 커스텀 호출어(예: "오둥아")를 부르면 챗봇이 이를 감지하고 대화 상태로 깨어납니다. (openWakeWord 기반)
2. **Omni (Qwen3 Omni 30B)**:
   - 깨어난 챗봇은 사용자의 이어지는 음성 명령을 텍스트 변환 없이 바로 듣고 뉘앙스까지 이해합니다.
   - 스스로 판단하여 에어컨을 켜는 등의 스마트홈 기기 조작(MCP 도구 호출)을 수행하거나 답변할 내용을 결정합니다.
3. **TTS (OmniVoice)**:
   - 결정된 텍스트 응답은 빠르고 자연스러운 음성(k2-fsa/OmniVoice)으로 합성되어 브라우저나 방의 스피커를 통해 사용자에게 전달됩니다.

## 🚩 마일스톤 (Milestones)

### ✅ 달성한 마일스톤 (Achieved)
- **로컬 음성 파이프라인 구축**: `Wake Model → OMNI → TTS` 파이프라인 검증 및 연속 대화(멀티턴) 구현 완료.
- **호출어 및 음성 입출력**: 브라우저 마이크 입력, 커스텀 호출어("오둥아") 인식, WebSocket PCM 스트리밍 및 재생 구현.
- **도구 연동 (MCP)**: Qwen3-Omni 모델이 직접 MCP(Model Context Protocol)를 통해 SwitchBot(에어컨, 선풍기) 및 Brave Search와 연동해 실제 행동을 수행하도록 구현.
- **모델 서빙 최적화**: `llama-server`와 OmniVoice 모델을 서버 수명 동안 유지해 턴별 모델 재로딩 시간 제거.

### 📝 TODO 마일스톤 (Next Steps)
- **조명 제어 확장**: SwitchBot 물리 스위치를 이용한 조명 켜기/끄기 기능 연동.
- **다중 입출력 확장**: 현재 방1 환경에 더해, 거실 Bluetooth 스피커 출력 및 여러 방의 입력 Source 동시 수용 구현.
- **신뢰성 및 지연 시간 최적화**: 3턴 이상의 복잡한 대화에서도 도구를 안정적으로 선택하도록 개선하고, 현재 3~7초 수준인 음성 턴 Latency를 더욱 단축.
- **Customized Memory**: 대화 이력과 사용자 취향을 장기 기억으로 구성하여 유대감을 높이는 메모리 시스템 도입.

---

## 💻 사양 및 구축 환경 (Specs)

현재 프로토타입은 쾌적한 로컬 추론을 위해 다음과 같은 환경에서 구축 및 검증되고 있습니다:
- **GPU**: NVIDIA RTX 5070 Ti (Blackwell `sm_120a` 아키텍처)
- **CUDA / 빌드**: CUDA Toolkit 13.1, GCC 13 환경에서 `llama.cpp`를 직접 빌드하여 20-layer Partial GPU 오프로딩을 통해 최적화. (VRAM 약 13.0GB 사용, End-to-end 3.84초 응답 속도 확인)

## 📦 설치 (Install)

1. **사전 요구 사항**: Python 3.11 이상 3.13 미만, [uv](https://docs.astral.sh/uv/) 패키지 매니저, 마이크가 있는 최신 브라우저.
2. **의존성 설치**:
   ```bash
   uv sync
   ```
3. **환경 설정**:
   ```bash
   cp .env.example .env
   # .env 파일을 열어 필요한 API 키 및 모델 경로, 환경 설정을 기입합니다.
   ```
4. **모델 파일 준비**:
   - 로컬 Qwen3-Omni GGUF 모델과 OmniVoice 모델이 필요합니다.
   - 프로젝트에서 학습한 커스텀 Wake 모델(`오둥아.onnx` 등)은 `models/openwakeword/`에 위치시킵니다.

## 🚀 빠른 시작 (Quick Start)

모든 설치가 끝난 후 다음 명령으로 서버를 실행합니다. 첫 실행 시 모델을 메모리에 로드하므로 약간의 시간이 소요될 수 있습니다.

```bash
uv run omni-iot --host 127.0.0.1 --port 8000
```

1. 브라우저에서 `http://127.0.0.1:8000`에 접속하여 **Start** 버튼을 누르고 마이크 권한을 허용합니다.
2. 화면에 `Say "오둥아"` 상태가 표시되면 **"오둥아"**라고 부르고 잠시 멈춥니다.
3. `Listening for command`가 나타나면 **"에어컨 켜줘"** 등 원하는 명령을 말합니다.
4. 응답을 들은 후 8초 이내에는 다시 호출어를 부를 필요 없이 자연스럽게 후속 대화를 이어갈 수 있습니다.

---

### 기타 고급 설정 (MCP, SwitchBot, 빌드 등)

<details>
<summary>MCP와 SwitchBot / Brave Search 연결 방법 보기</summary>

**SwitchBot 연결**
```bash
npm install --global @switchbot/openapi-cli@latest
switchbot auth login

uv run omni-iot-mcp add-stdio switchbot \
  --command "$(command -v switchbot)" \
  --arg mcp --arg serve \
  --allow-tool list_devices \
  --allow-tool get_device_status \
  --allow-tool send_command

uv run omni-iot-mcp doctor --json
```

**Brave Search 연결**
```bash
uv run omni-iot-mcp add-stdio brave-search \
  --command "$(command -v npx)" \
  --arg=-y \
  --arg=@brave/brave-search-mcp-server@2.1.0 \
  --env 'BRAVE_API_KEY=${BRAVE_API_KEY}' \
  --allow-tool brave_web_search \
  --allow-tool brave_news_search

uv run omni-iot-mcp enable brave-search
```

전체 기능(SwitchBot + Brave Search) 통합 실행 스크립트:
```bash
bash run-omni-iot.sh
```
</details>

<details>
<summary>RTX 5070 Ti용 llama.cpp 직접 빌드 방법 보기</summary>

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
</details>

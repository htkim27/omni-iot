# ADR 0004: 단일 Omni 모델의 로컬 Audio-In/Out 도입 보류

## 상태

Accepted

## 배경

프로젝트의 장기 목표는 중간 STT나 별도 TTS 없이 단일 Omni 모델이 실시간으로
음성을 입력받고 음성을 출력하는 양방향 대화입니다. 현재 장비는 RTX 5070 Ti
16GB, 시스템 메모리 32GB이며 로컬 실행 가능성을 다음 두 경로로 검토했습니다.

- `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf`와 mmproj
- `tc-mb/llama.cpp-omni`의 전용 Duplex WebSocket 서버

## 결정

현재 하드웨어와 공개 런타임 조합에서는 단일 Omni 모델 기반 Audio-In/Out을
운영 경로로 채택하지 않습니다. 기존 `origin/main`의 다음 파이프라인을
유지합니다.

```text
브라우저 음성 → Qwen3-Omni GGUF 음성 이해/텍스트 응답 → OmniVoice TTS → 음성 재생
```

GPU offload, Flash Attention, VAD와 TTS streaming 등 기존 latency 실험과 측정
자산은 이후 파이프라인 최적화에 계속 사용합니다.

## 이유

1. 현재 Qwen3-Omni GGUF 배포본은 Thinker와 오디오·비전 입력 projector만
   포함합니다. 네이티브 음성 출력에 필요한 Talker와 Code2Wav 가중치 및
   실행 경로가 없어 Audio-Out을 만들 수 없습니다.
2. 원본 safetensors와 일부 4-bit 체크포인트에는 Talker와 Code2Wav가 있지만,
   전체 체크포인트가 약 25~27GB이고 런타임 메모리도 추가로 필요합니다.
   16GB 단일 GPU에서는 실시간 Duplex에 필요한 형태로 상주시킬 수 없습니다.
   CPU offload는 실행 가능성을 높일 수 있어도 PCIe 전송 지연 때문에 실시간
   대화 목표에 맞지 않습니다.
3. `tc-mb/llama.cpp-omni`의 Duplex 구현은 MiniCPM-o 전용 audio encoder, TTS
   projector, 음성 token과 Token2Wav에 결합돼 있습니다. Qwen3-Omni GGUF를
   지정해도 Qwen Talker/Code2Wav 경로로 바뀌지 않으며 두 모델의 음성 token
   체계는 서로 호환되지 않습니다.

## 결과

- **운영 안정성**: 이미 검증한 llama-server와 OmniVoice 경로로 복귀합니다.
- **지연 시간**: 단일 모델 Duplex의 이론적 이점은 얻지 못하지만, 측정된 GPU
  offload와 TTS/VAD 최적화를 계속 적용할 수 있습니다.
- **재검토 조건**: Qwen Audio-Out을 포함한 llama.cpp 호환 가중치와 런타임이
  공개되거나, Talker·Code2Wav를 포함한 체크포인트를 실시간으로 상주시킬 수
  있는 GPU 메모리를 확보하면 이 결정을 다시 검토합니다.

## 검토한 대안

- **Qwen GGUF에 MiniCPM Token2Wav 연결**: 모델별 hidden state와 음성 token
  contract가 달라 채택하지 않았습니다.
- **Qwen 전체 체크포인트 CPU offload**: 현재 CPU·메모리·PCIe 조건에서 목표
  latency를 충족하기 어려워 채택하지 않았습니다.
- **MiniCPM-o 전체 모델로 교체**: 현재 선택한 Qwen 기반 이해·도구 호출 품질과
  모델 자산을 포기해야 하므로 이번 범위에서는 채택하지 않았습니다.

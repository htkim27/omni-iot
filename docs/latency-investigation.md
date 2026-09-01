# 로컬 Qwen3-Omni 음성 응답은 어디에서 늦어질까

RTX 5070 Ti에서 GPU offloading과 Flash Attention을 360회 측정해 봤다

> 측정일: 2026-09-01~02  
> GPU: NVIDIA GeForce RTX 5070 Ti 16GB  
> 모델: Qwen3-Omni-30B-A3B-Instruct Q4_K_M GGUF + Q8 mmproj  
> 런타임: llama.cpp build 10460 (`373336672`)

## 먼저 결론

이번 실험에서 가장 큰 개선 요인은 Flash Attention이 아니라 GPU offloading이었다.

- Transformer layer 0→24개 GPU offload 시 no-tool latency가 8.011초에서 4.839초로 39.6% 감소했다.
- Tool command latency는 12.104초에서 7.155초로 40.9% 감소했다.
- 24 layers에서 Flash Attention을 켜면 평균 latency가 약 0.24~0.26초 줄었다.
- 다만 Flash ON/OFF는 출력 token 분포와 서버 실행 시점이 달라, 0.24~0.26초 전부를 Flash Attention의 순수 효과라고 단정할 수 없다.
- Token-normalized 처리량은 Flash ON에서 prompt 4.3%, decode 5.8% 개선됐다.
- 현재 구조에는 모델과 무관하게 VAD와 TTS가 만드는 약 1.35초의 고정비가 있다.

## 무엇을 latency라고 불렀나

궁극적으로 알고 싶은 값은 사용자가 말을 끝낸 순간부터 AI 음성이 실제로 재생되기 시작할 때까지의 시간이다.

이번 저장 WAV 기반 실험에서는 다음 값으로 근사했다.

```text
사용자의 마지막 유성음
  → VAD 무음 대기 600ms
  → Qwen3-Omni inference
  → OmniVoice가 reply.wav 전체 생성
  → 음성 파일 준비 완료
```

측정 필드명은 `speech_end_to_audio_ready_seconds`다. 브라우저 다운로드, 오디오 버퍼링, OS가 스피커 재생을 시작하는 짧은 시간은 포함하지 않는다. 같은 로컬 환경에서 GPU 옵션 간 상대 차이를 비교하기에는 충분한 근사치다.

실제 서비스에는 브라우저가 감지한 마지막 유성음부터 HTML audio의 `playing` 이벤트까지 측정하는 `browser_speech_end_to_audio_start_seconds`도 추가했다. 이후 실사용 로그에서는 이 값으로 근사를 검증할 수 있다.

## 실험 설계

초기 실험에서는 temperature 0과 고정 seed를 사용했다. 같은 조건의 반복 출력은 재현됐지만, GPU layer나 Flash Attention 설정이 바뀌면 부동소수점 계산 경로가 달라져 출력 문장과 token 수가 달라졌다. 그 상태에서는 latency 차이와 출력 길이 차이를 구분하기 어려웠다.

최종 실험은 실제 사용 환경의 평균 분포를 보기 위해 다음과 같이 설계했다.

- Temperature 0.2
- Random seed (`-1`)
- 최대 출력 64 tokens
- 각 조건과 입력 조합을 30회 반복
- 모든 trial은 conversation history가 없는 single turn
- 매 trial 직전 llama.cpp slot/KV/prefix cache 삭제
- 서버와 TTS warmup 결과 제외
- 복잡도가 높고 tool 횟수가 불안정한 입력 제외
- 실제 IoT 장치 대신 동일 tool schema와 고정 replay 결과 사용
- 각 trial의 latency, prompt tokens, completion tokens, LLM round 수를 함께 기록

두 개의 저장 WAV를 사용했다.

| 입력 | 경로 특성 |
|---|---|
| No tool | 일반 대답, LLM 1 round |
| Tool command | Tool 선택과 결과 반영, LLM 3 rounds |

GPU offloading 실험은 5 layer 조건×2 inputs×30 trials로 300개 sample을 수집했다. Flash Attention 실험에서는 24-layer Flash ON 60개 sample을 추가했고, GPU 실험의 동일한 Flash OFF 60개와 비교했다. 고유 sample은 총 360개다.

## 그릇 무게: 모델 밖에서 발생하는 시간

모델이 아무리 빨라도 현재 파이프라인에는 다음 비용이 남는다.

| 구간 | 측정값 |
|---|---:|
| VAD silence wait | 0.600초 |
| OmniVoice TTS | 평균 약 0.74~0.78초 |
| WAV 저장 및 Python overhead | 수 ms |
| 합계 | 약 1.35초 |

현재는 모델 답변 전체가 끝난 뒤 TTS 전체 WAV를 생성한다. 따라서 모델 inference가 0초가 되더라도 사용자 체감 latency를 약 1.35초 아래로 내리기 어렵다.

## 실험 1: CPU offloading을 GPU offloading으로 바꾸면

Flash Attention은 끄고 GPU에 올리는 transformer layer 수만 변경했다. 표의 값은 30회 `평균 (표준편차)`이며, 단위는 초다.

| GPU layers | No tool | Tool command |
|---:|---:|---:|
| 0 | 8.011 (0.850) | 12.104 (0.476) |
| 8 | 6.799 (0.718) | 9.934 (0.178) |
| 16 | 5.774 (0.607) | 8.616 (0.186) |
| 20 | 5.512 (0.357) | 7.679 (0.295) |
| 24 | **4.839 (0.394)** | **7.155 (0.222)** |

0→24 layers의 개선 폭은 다음과 같다.

| 입력 | 단축 시간 | 단축률 | 0~24 평균 layer당 gain |
|---|---:|---:|---:|
| No tool | 3.172초 | 39.6% | 약 0.132초/layer |
| Tool command | 4.949초 | 40.9% | 약 0.206초/layer |

Layer당 gain은 관측 구간의 평균 기울기일 뿐이다. 모든 layer가 동일한 크기나 계산 비용을 갖는다고 가정할 수 없고 CPU↔GPU 경계 비용도 달라지므로, 이 값을 48 layers까지 선형으로 곱하면 안 된다.

### 모델 inference만 보면

VAD와 TTS를 제외한 LLM 시간도 layer 증가에 따라 꾸준히 줄었다.

| GPU layers | No-tool LLM | Tool-command LLM |
|---:|---:|---:|
| 0 | 6.647초 | 10.746초 |
| 8 | 5.416초 | 8.589초 |
| 16 | 4.414초 | 7.259초 |
| 20 | 4.135초 | 6.335초 |
| 24 | **3.467초** | **5.785초** |

Tool command는 LLM을 세 번 실행하므로 한 번의 GPU 최적화 이득도 round마다 누적된다. 실제 device API 시간은 replay에서 제거했는데도 no-tool보다 약 2.3초 느렸다. Tool 자체보다 여러 번의 순차 LLM inference가 큰 비용이라는 뜻이다.

### 출력 길이 영향을 정규화하면

Random sampling에서는 no-tool completion tokens가 조건별 평균 43.3~48.0, 표준편차 7.8~12.4로 달랐다. 따라서 raw latency만 보면 출력 길이 차이가 섞인다.

Layer별 llama-server 요청 121개의 token 수와 시간을 합쳐 처리량을 계산하면 GPU offloading 효과가 더 선명하다.

| GPU layers | Prompt throughput | Decode throughput |
|---:|---:|---:|
| 0 | 122.2 tok/s | 13.1 tok/s |
| 8 | 144.8 tok/s | 17.4 tok/s |
| 16 | 174.1 tok/s | 20.8 tok/s |
| 20 | 197.9 tok/s | 23.6 tok/s |
| 24 | **221.4 tok/s** | **26.0 tok/s** |

0→24 layers에서 prompt throughput은 81%, decode throughput은 98% 증가했다. CPU offloading이 주요 병목이라는 가설은 end-to-end와 token-normalized 지표 양쪽에서 확인됐다.

## VRAM 한계와 full GPU 추정

GGUF metadata에서 transformer block은 총 48개다. 하지만 RTX 5070 Ti 16GB에서는 Qwen3-Omni와 OmniVoice를 함께 둔 full GPU가 불가능했다.

- 24 layers + mmproj + TTS warmup: 약 14.3GB, 정상 동작
- 28 layers: llama-server 시작 후 TTS warmup에서 OOM
- 32 layers: mmproj 로드 중 llama-server OOM
- Layer별 VRAM 증가 추세로 계산한 48-layer 요구량: 약 22.5~22.9GB

따라서 현재 하드웨어에서는 24 layers가 안정적인 상한에 가깝다. 기존 처리량 추세로 full GPU를 거칠게 외삽하면 24-layer 대비 no-tool 약 1초, tool command 약 2초의 추가 단축 가능성이 있다. 이는 실제 full-GPU 측정값이 아니므로 방향성 추정으로만 사용해야 한다.

## 실험 2: Flash Attention을 켜면

GPU offloading 실험의 최고 안정 설정인 24 layers에서 Flash Attention OFF/ON을 비교했다. 나머지 조건은 동일하다.

| 입력 | Flash OFF | Flash ON | Raw 평균 차이 |
|---|---:|---:|---:|
| No tool | 4.839 (0.394)초 | **4.596 (0.371)초** | -0.243초, -5.0% |
| Tool command | 7.155 (0.222)초 | **6.892 (0.248)초** | -0.263초, -3.7% |

Raw 결과만 보면 Flash ON이 두 입력에서 모두 약 0.25초 빨랐다. 하지만 두 가지 주의점이 있다.

첫째, 출력 token 분포가 완전히 같지 않았다.

| 입력 | Flash OFF completion tokens | Flash ON completion tokens |
|---|---:|---:|
| No tool | 43.7 (9.1) | 42.3 (9.3) |
| Tool command | 65.0 (0.0) | 65.9 (2.7) |

No-tool ON은 평균 1.4 tokens 짧아 raw latency에 유리했다. 반대로 tool ON은 평균 0.9 tokens 길었는데도 빨랐으므로 Flash Attention의 개선 방향과는 일치한다.

둘째, OFF와 ON을 trial 단위로 번갈아 실행하지 않고 별도의 서버 실행 묶음으로 측정했다. 시간 경과에 따른 GPU clock, 온도, 백그라운드 부하가 조건 효과에 섞였을 수 있다. 따라서 0.243~0.263초를 순수한 Flash Attention 효과로 확정할 수는 없다.

### Token-normalized 처리량

서버 로그 전체를 token 수로 정규화하면 다음과 같다.

| 설정 | Prompt throughput | Decode throughput |
|---|---:|---:|
| Flash OFF | 221.4 tok/s | 26.0 tok/s |
| Flash ON | **230.9 tok/s** | **27.5 tok/s** |
| 개선 | **4.3%** | **5.8%** |

Flash Attention은 실제로 동작했고 처리량을 몇 퍼센트 개선했다. 다만 수 초 단위 latency의 주된 해결책은 아니다. 이 workload는 context가 짧고 batch가 1이며, 전체 48 layers 중 절반만 GPU에 있다. Flash Attention이 최적화하는 GPU attention 외에도 CPU layer, MoE/FFN, audio projection, CPU↔GPU 이동, VAD, TTS 비용이 그대로 남는다.

현재 근거로는 Flash Attention을 켜는 것이 합리적이다. 정확한 순수 효과를 구하려면 OFF/ON 서버를 trial마다 교차하거나 실행 묶음을 여러 번 반복하는 randomized paired benchmark가 필요하다.

## 최종 판단

이번 실험에서 확인한 우선순위는 다음과 같다.

1. CPU offloading은 큰 병목이다. 24 layers offload로 실제 평균 latency가 약 40% 감소했다.
2. Tool turn은 tool API보다 여러 번의 순차 LLM round가 비싸다.
3. Flash Attention은 켜는 편이 좋지만 개선 규모는 GPU offloading보다 작다.
4. VAD 600ms와 비스트리밍 TTS 약 0.75초는 모델 최적화로 없어지지 않는 고정비다.
5. 16GB에서는 24 layers가 안정적인 상한이며 full GPU에는 약 23GB VRAM이 필요할 것으로 보인다.

다음 실험은 VAD silence 설정과 TTS streaming을 각각 분리해 측정하는 것이 좋다. 두 항목은 현재 약 1.35초의 고정비를 직접 줄일 수 있다. Tool path에서는 LLM round를 줄이거나 thinker와 talker를 병렬화하는 구조도 유력한 개선 후보다.

## 재현 자료

- Benchmark harness: `scripts/benchmark_gpu_offload.py`
- GPU offloading manifest: `.runtime/benchmarks/natural-latency-20260901/experiment-1-gpu-offload/manifest.json`
- GPU offloading raw samples: 같은 디렉터리의 `samples.jsonl`
- GPU offloading summary: 같은 디렉터리의 `summary.json`
- Flash ON manifest: `.runtime/benchmarks/natural-latency-20260901/experiment-2-flash-on/manifest.json`
- Flash ON raw samples: 같은 디렉터리의 `samples.jsonl`
- Flash ON summary: 같은 디렉터리의 `summary.json`
- Layer별 llama-server log: 각 결과 디렉터리의 `gpu-layers-*/turns/llama-server.log`
- 28/32-layer OOM 기록: `.runtime/benchmarks/gpu-offload-20260901-225007-boundary/failures.jsonl`

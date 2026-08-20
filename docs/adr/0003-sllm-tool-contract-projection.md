# ADR 0003: sLLM용 MCP tool contract projection

## 상태

Accepted

## 배경

MCP 서버의 `tools/list`는 서버가 제공하는 전체 도구 설명과 JSON Schema를
반환합니다. 범용 API의 schema는 다양한 클라이언트와 사용 상황을 위해 길고
상세할 수 있지만, 4K 수준의 로컬 sLLM context에서는 tool catalog만으로도
유의미한 비중을 차지합니다. 이는 대화 history, 음성 입력과 최종 응답에
사용할 context를 줄이고 tool selection 품질을 떨어뜨릴 수 있습니다.

MCP 서버가 제공한 schema는 서버의 전체 capability contract이지, 특정 sLLM에
그대로 전달해야 하는 prompt contract는 아닙니다.

## 결정

MCP 클라이언트에 tool contract projection 단계를 둡니다.

```text
MCP server tools/list
  → exact allowlist
  → sLLM profile: 짧은 설명 + 최소 입력 schema
  → OpenAI tool catalog
  → sLLM tool call
  → 인자 filter / default / upper bound
  → 원본 MCP tool call
```

1. 서버가 제공한 tool 이름과 전체 schema는 discovery와 호환성 확인에 사용합니다.
2. 모델에게는 현재 제품 시나리오에 필요한 의도, 인자와 제한만 담은
   짧은 description과 JSON Schema를 전달합니다.
3. schema 축약은 prompt 최적화에 그치지 않습니다. 실제 호출 경계에서도
   허용 인자만 전달하고, 기본값·상한·고정 옵션을 강제해 모델에 보인
   contract과 runtime contract를 일치시킵니다.
4. 현재 구현은 `(server_name, tool_name)`을 직접 검사하는 명시적 projection을
   사용합니다. Brave Search와 SwitchBot을 제외한 tool은 catalog 예산 안에서
   원본 description과 schema를 전달합니다.
5. catalog 문자 수 상한은 fail-closed 방어선으로 유지합니다. 상한을 높이기 전에
   allowlist와 projection profile을 먼저 검토합니다.

Brave Search projection은 `query`, `count`, `freshness`, `country`, `search_lang`만
노출하고 검색 종류별 짧은 설명을 사용합니다. `count`는 5로 제한하며
나머지 모델 인자는 실제 MCP 호출 전에 제거합니다.

SwitchBot projection은 현재 allowlist의 `list_devices`, `get_device_status`, `send_command`에만
적용합니다. `send_command`는 `deviceId`, `command`, `parameter`만 모델에 노출하고
`commandType=command`, `confirm=false`를 호출 경계에서 고정합니다.

## 구현 경계

projection은 기존 프로세스의 `McpManager`에서 수행합니다. 별도 MCP 서버,
프록시 프로세스나 네트워크 hop을 추가하지 않습니다. MCP 서버와 protocol은
수정하지 않고, 모델에 보여줄 catalog와 서버에 전달할 arguments만 하네스가
제품 정책에 맞게 투영합니다.

범용 registry, 외부 profile 설정, 자동 schema 축약은 v1에 포함하지 않습니다.
명시적 projection이 늘어나 중복이 확인될 때 [GitHub 이슈 #17](https://github.com/htkim27/omni-iot/issues/17)에서
registry 범용화를 다시 검토합니다.

## 결과

- **장점**: 작은 context에서 tool catalog 비용을 줄이고, 모델이 선택해야 할
  인자와 의도를 명확하게 만듭니다.
- **안전성**: 숨긴 인자를 모델이 생성해도 서버에 전달되지 않습니다.
- **비용**: upstream tool schema가 바뀐 때 profile과 호환되는지 테스트해야 하며,
  축약에서 제외한 기능은 모델이 사용할 수 없습니다.
- **운영 기준**: 명시적 projection마다 catalog 크기, 노출 인자, runtime filtering과
  대표 tool call을 회귀 테스트합니다.

## 검토한 대안

- **원본 schema를 그대로 사용**: 개발 비용은 낮지만 sLLM context와 tool selection
  품질에 불리해 채택하지 않았습니다.
- **context 크기만 확장**: GPU 메모리와 prompt processing 비용을 늘리고 불필요한
  옵션을 계속 노출하므로 채택하지 않았습니다.
- **별도 MCP wrapper/proxy 서버**: 서버 경계가 명확하지만 현재 규모에서는
  프로세스, 설정과 장애 지점을 늘리므로 채택하지 않았습니다.

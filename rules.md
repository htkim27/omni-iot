# Git Rules

## Branch

- `main`에 직접 커밋하지 않는다.
- 브랜치명은 `<type>/<short-description>` 형식을 사용한다.
- type은 `feat`, `fix`, `docs`, `refactor`, `test`, `chore` 중에서 선택한다.

예: `feat/voice-harness-prototype`

## Commit

- 하나의 커밋에는 하나의 논리적 변경만 포함한다.
- 메시지는 `<type>: <imperative summary>` 형식의 영어로 작성한다.
- 모델, `.env`, `.runtime`, build 결과물과 비밀값은 커밋하지 않는다.

예: `feat: retain latest runtime turns`

## Pull Request

- 제목은 짧은 영어 요약 뒤에 한국어 설명을 병기한다.
- 본문은 `Summary`, `Changes`, `Milestone alignment`, `Verification`, `Known limitations`, `Next steps` 순서로 작성한다.
- 주요 설명은 영어와 한국어를 함께 제공하되 같은 내용을 불필요하게 반복하지 않는다.
- 변경 범위 밖의 작업과 알려진 제약을 명확히 적는다.
- 검증한 항목만 체크하며, 미검증 항목은 완료로 표시하지 않는다.

## Milestone

- PR에는 결과물 기준의 대표 milestone 하나만 할당한다.
- `docs/plan.md`의 Phase 대응 상태를 PR 본문의 표로 기록한다.
- 후속 작업은 Issue로 분리해 milestone에 연결하고 PR에서 참조한다.
- Issue를 실제로 완료하는 경우에만 `Closes #<issue>`를 사용한다. 그 외에는 `Related to #<issue>`를 사용한다.

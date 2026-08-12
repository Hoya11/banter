# server — banter 백엔드

파이썬 단독 (FastAPI + LangGraph). 패키지 관리는 `uv`.

## 구조
- `api/` — 세션·대화기록·설정 REST/WS (FastAPI)
- `engine/` — 3자 오케스트레이터
  - `graph/` — LangGraph 상태·노드 (엔진설계 §1.1~1.2), 화자 선정 v1 + 규칙 가드 (§1.3)
  - `personas/` — 페르소나 설정, 데이터 주도 (D-004/D-007)
  - `eval/` — judge 루브릭·시나리오·채점 하네스 (D-005, §4.1)

## 경계 원칙
`api`는 `engine`을 경계 인터페이스로 호출한다. Phase 3에서 `engine`이 LiveKit agent worker로 물리 분리되므로(PRD §6.2), 지금부터 결합을 최소화한다.

## 개발 (Phase 1)
```bash
uv sync        # 의존성 설치 (A단계에서 fastapi·langgraph 등 추가 후)
uv run ...     # 실행 커맨드는 착수 후 확정
```

설계 문서: [PRD](../docs/prd-v0.1.md) · [엔진 설계](../docs/engine-design-v0.1.md) · [Phase 1 계획](../docs/phase-1-plan.md)

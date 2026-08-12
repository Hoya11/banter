# engine — 3자 오케스트레이터

유저 1명 + AI 2명의 발화권 중재. 상세 설계는 [엔진 설계 v0.1](../../docs/engine-design-v0.1.md).

- `graph/` — LangGraph 상태 스키마·노드 그래프 (§1.1~1.2). 화자 선정 v1(supervisor) + 규칙 가드 (§1.3)
- `personas/` — 대비형 듀오 1쌍 설정. 코드가 아닌 데이터(프롬프트·스탠스·화제성향)로 분리 (D-004/D-007)
- `eval/` — 평가 하네스 (하위 README 참고)

Phase 3에서 이 패키지가 LiveKit agent worker로 분리된다. `api`와의 결합을 인터페이스로 제한할 것.

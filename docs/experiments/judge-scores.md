# judge 점수 이력

시나리오 셋 채점 결과를 버전별로 누적한다. 각 행은 프롬프트/페르소나 변경 전후 비교의 근거.

## 2026-08-13 — 시나리오 셋 v1 / 페르소나 v1 (대비 강화 + 유저 소환)

| 시나리오 | 유저발화 | 판정 | 평균 | turn_taking | persona | context | liveliness | user_incl |
|---|---|---|---|---|---|---|---|---|
| user_participates | 2 | GO | 7.8 | 8 | 9 | 7 | 8 | 7 |
| topic_switch | 1 | GO | 7.6 | 8 | 9 | 7 | 8 | 6 |
| radio_silence | 0 | NO-GO | 7.4 | 9 | 8 | 7 | 8 | 5 |
| **전체** | | | **7.60** | | | | | |

judge 모델: gpt-4o-mini / 발화: gpt-4o-mini(temp 0.9) / go 컷: 평균 ≥7.0 & 최저 ≥6

### 관찰
- **user_inclusion이 유저 발화 수에 비례**(2→7, 1→6, 0→5) — 항목이 실제 참여도를 측정함을 확인.
- 유저 참여 시나리오는 GO. 엔진·페르소나 품질은 유저가 있을 때 컷 통과.
- `radio_silence` NO-GO는 유저 부재 특성 — "실패"가 아니라 시나리오 성격. (정교화 대상: 라디오 모드는 실제 포함이 아니라 '유저 소환 시도'로 평가할지)
- persona_consistency 9 유지 — 도현 냉소 강화(페르소나 v1) 효과.

### 이전 대비
- 초기 라디오 파일럿(페르소나 v0, 유저 소환 없음): user_inclusion 5 / liveliness 6 → NO-GO.
- v1(대비 강화 + 유저 소환 + 유저 참여 시나리오): user_participates GO 7.8, liveliness 8.

### 다음 후보
- 끼어들기(§2.1) 케이스 추가 — context_on_interrupt를 실제 개입으로 측정.
- 라디오 모드 user_inclusion 평가 정교화.
- 소은 캐릭터 깊이(단순 긍정 → 위트) 튜닝 후 재측정.

## 2026-08-13 — supervisor v1 도입 (기계적 교대 → 맥락 기반 화자 선정, D-003)

| 시나리오 | 판정 | 평균 | 이전(기계적) | user_incl | liveliness |
|---|---|---|---|---|---|
| user_participates | GO | 8.2 | 7.8 ↑ | 9 (was 7) | 8 |
| topic_switch | GO | 7.6 | 7.6 = | 6 | 8 |
| radio_silence | NO-GO | 7.0 | 7.4 ↓ | 5 | 6 (was 8) |
| **전체** | | **7.60** | 7.60 = | | |

### 관찰
- supervisor는 **유저 참여 시 값을 한다** — user_participates user_inclusion 7→9, 평균 7.8→8.2. 유저 발화 맥락을 반영한 화자 선정.
- **라디오 모드는 하락** — supervisor가 도현 냉소·화제 정체를 심화(liveliness 8→6). AI끼리 무한 대화엔 화제 관리(topic_stack)가 필요.
- **전체 평균은 동일(7.60)이지만 시나리오별로 갈림** — 단일 지표만 보면 "효과 없음"이나, 세분하면 "유저 상호작용 개선 + 라디오 부작용" 트레이드오프. 평균의 함정을 시나리오 셋이 드러냄.
- 비용: supervisor는 AI 턴마다 LLM 호출 +1 (D-003 v2 통합 생성으로 최적화 대상).

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

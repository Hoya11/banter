# banter

유저 1명 + AI 페르소나 2명이 실시간 음성으로 나누는 퇴근 후 수다.
유저가 침묵하면 AI 둘이 대화를 이어가고, 끼어들면 자연스럽게 화자가 양보한다.

> 데모 영상: (Phase 3 완료 후 추가)
> 핵심 지표: (지연 p95 / barge-in 반응 / judge 점수 / 분당 비용 — 측정 후 갱신)

## 문서
- [PRD v0.1](docs/prd-v0.1.md) — 제품 정의, 3단계 범위, 성공 지표, 아키텍처
- [엔진 설계 v0.1](docs/engine-design-v0.1.md) — turn-taking, 끼어들기, 지연 설계
- [Phase 1 실행 계획](docs/phase-1-plan.md) — 착수 체크리스트 (A 평가기준 → B 엔진 → C 페르소나 → D 판정)
- [결정 기록](docs/decisions.md) / [실패 기록](docs/failures.md) / [실험](docs/experiments/)

## 단계
- [ ] Phase 1 — 텍스트 3자 엔진 + 평가 체계
- [ ] Phase 2 — 반이중 음성 (push-to-talk) + TTS 청음 비교
- [ ] Phase 3 — 전이중 barge-in (LiveKit) + 데모 영상

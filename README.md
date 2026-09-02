# banter

**유저 1명 + AI 페르소나 2명이 실시간 음성으로 나누는 퇴근 후 수다.**

음성 AI는 대부분 1:1 대화입니다. 여기에 AI를 한 명 더 앉히면 어떻게 될까 —
누가 언제 말할지(발화권), 유저가 끼어들면 누가 양보할지, 침묵하면 누가 이어갈지를
전부 시스템이 중재해야 합니다. 이 "3자 발화권 중재"가 궁금해서 시작한 프로젝트입니다.

> 데모 영상: Phase 3 완료 후 추가 예정

## 어떤 경험인가

- `대화 시작`을 누르면 **도현**(냉소적 현실주의자)과 **소은**(낙천적 응원러)이 수다를 시작한다
- 가만히 들으면 둘이 계속 티키타카한다 — **라디오 모드**
- 마이크 버튼을 누르고 말하면(또는 타이핑하면) 말하던 AI가 **발화를 멈추고 양보**한다 (barge-in)
- 목소리가 대화의 주인공: 자막은 소리에 맞춰 뜨고, 다음 발화는 재생이 끝나야 나온다

실제 대화 한 토막:

> **나**: 퇴근!!
> **소은**: 퇴근 축하해! 오늘 할 일 다 해낸 거 진짜 멋지다, 이제부터는 아무 생각 말고 푹 쉬자!
> **도현**: 축하는 내일 출근 취소됐을 때 하는 거지. 일단 씻고 뭐라도 먹어, 피곤하다고 누우면 그대로 굶고 자게 된다.
> **소은**: 도현 말도 맞긴 한데, 오늘만큼은 대충 먹고 일찍 누워도 괜찮아! 오늘 하루는 어땠어?

## 숫자로 보는 현재 상태

| 항목                                   | 수치                                                | 비고                                          |
| -------------------------------------- | --------------------------------------------------- | --------------------------------------------- |
| AI 발화 간 체감 지연                   | **평균 2.1s** (로컬 n=3, 도입 전 약 8s)             | [실측 기록](docs/experiments/latency.md)      |
| 대화 품질 (LLM-as-judge, 5항목 루브릭) | 유저 참여 시나리오 **GO** (컷: 평균 ≥7.0 & 최저 ≥6) | [점수 추이](docs/experiments/judge-scores.md) |
| judge 신뢰성                           | 고정 대화 반복 채점 표준편차 **0**                  | 재현 가능한 평가                              |
| 테스트                                 | 112 passed (웹 상태 15개 포함)                      | 엔진, 평가, WS 프로토콜, 재시작 경계          |

## 아키텍처

```
[웹 (바닐라 JS)] ── WebSocket ──> [FastAPI]
   자막 동기 재생 · played ack        │
   push-to-talk                       ├─ 3자 오케스트레이터 (LangGraph)
                                      │    supervisor v2: 화자, 의도, 대사 통합 생성 + 규칙 가드
                                      │    barge-in: 실시간 생성은 취소, 준비된 음성은 브라우저에서 중단
                                      │    prefetch: 재생 중 다음 턴 미리 생성·합성
                                      ├─ STT (gpt-4o-mini-transcribe)
                                      ├─ LLM (스트리밍) / 문장 단위 TTS (ElevenLabs v3)
                                      └─ 평가: LLM-as-judge + 시나리오 셋 + Langfuse
```

### 설계에서 중요했던 결정 몇 가지

- **조립형 파이프라인 (STT+LLM+TTS)** — speech-to-speech 일체형은 페르소나 2개와 발화권 중재 로직을
  끼워 넣을 지점이 없다. 조립형의 대가인 지연은 prefetch로 상쇄했다 ([D-002](docs/decisions.md))
- **prefetch** — AI끼리의 다음 턴은 유저 입력에 의존하지 않으므로, 현재 발화가 재생되는 동안
  미리 생성·합성해 둘 수 있다. 발화 간 지연 8s → 2.1s. 유저가 끼어들면 폐기한다.
  1:1 구조에서는 불가능한, 3자 구조라서 생기는 이점
- **소리가 페이스메이커** — 자막·다음 턴 진행을 오디오 재생 완료(seq 기반 ack)에 동기화.
  텍스트가 소리보다 앞서 달리는 어긋남을 프로토콜 레벨에서 제거
- **평가를 제품보다 먼저** — "수다가 재밌는가"는 주관적이라, judge 루브릭·go/no-go 컷을
  엔진 코드보다 먼저 만들었다. 이후 모든 튜닝은 감이 아니라 점수 비교로 판단
- **페르소나는 데이터** — 성격·말투·목소리(보이스 쌍 프리셋)를 코드가 아닌 설정으로 분리.
  케미 튜닝은 쌍 단위로만 ([D-004](docs/decisions.md))

## 품질을 어떻게 검증하나

"재밌다"를 계량화하기 위해 5항목 루브릭(turn-taking·페르소나 일관성·개입 대응·생동감·유저 포함도)의
LLM-as-judge를 두고, 끼어들기·화제전환·침묵·소외 시나리오 셋을 채점합니다.
측정 도구 자체도 검증합니다 — 반복 채점으로 judge의 재현성을 확인했고,
끊긴 발화가 judge에 전달되지 않아 barge-in 점수가 무효였던 것을 발견해
정정·재구현·재측정한 기록을 [experiments](docs/experiments/)에 남겼습니다.

## 끼어들기 지연 기록

`server/.env`에 `BANTER_EVENT_LOG=artifacts/runs/dev.jsonl`을 설정하면 대화 원문 없이
hold 수신, 브라우저 재생 중단, 생성 취소 완료, 다음 AI 응답 시작 이벤트가 기록됩니다.
말하기를 취소하거나 STT가 실패한 경우도 별도 종료 결과로 남겨 정상 누락과 구분합니다.
세션이 끝난 뒤 아래 명령으로 hold별 지연과 누락 이벤트를 확인할 수 있습니다.

```bash
cd server
PYTHONPATH=. uv run python scripts/summarize_barge_in.py artifacts/runs/dev.jsonl
```

브라우저는 마이크 버튼을 누른 시점부터 현재 Audio의 pause 적용까지 걸린 시간을 함께 보냅니다.
`paused` 결과만 재생 중단 지연 분포에 포함하고, 대기열만 비운 경우와 재생할 항목이 없던 경우는
건수로 분리합니다. 이 값은 실제 스피커 출력이 멎은 시점이 아니라 브라우저에서 pause가 적용된
시점입니다. 고정 시나리오와 해석 기준은 [push-to-talk 기준 측정](docs/experiments/push-to-talk-baseline.md)에
정리했습니다. 원본 실행 로그는 Git에서 제외됩니다.

## 실행

```bash
cd server
cp .env.example .env   # OPENAI_API_KEY 필수, ELEVENLABS_API_KEY는 음성용(선택)
uv sync
PYTHONPATH=. uv run uvicorn api.main:app --reload
# http://localhost:8000
```

`ELEVENLABS_API_KEY`를 비우면 AI 음성 출력 없이 실행됩니다. OpenAI 키 없이 구조만 보려면
`api.app:app`(스텁 발화)으로 띄우면 됩니다. 화면의 `대화 시작`을 누른 뒤 세션이 열리며,
12턴 종료 후 같은 화면에서 다시 시작할 수 있습니다. 테스트는 `uv run pytest`로 실행하며,
Node.js가 설치돼 있으면 재생 중단, 빠른 마이크 해제, 재시작 경계 테스트 15개도 함께 확인합니다.

## 문서

- [PRD](docs/prd-v0.1.md) — 제품 정의, 단계별 범위, 성공 지표
- [엔진 설계](docs/engine-design-v0.1.md) — turn-taking·끼어들기·지연 설계
- [결정 기록](docs/decisions.md) — 아키텍처 선택의 이유들 (D-001~008)
- [트러블슈팅](docs/troubleshooting.md) — 실사용 문제 → 진단 → 개선 히스토리
- [실험 기록](docs/experiments/) — 지연 실측, judge 점수 추이, TTS 청음 비교
- [Push-to-talk 기준 측정](docs/experiments/push-to-talk-baseline.md): VAD 전환 전 비교 기준
- [Phase 1 계획](docs/phase-1-plan.md) / [실패 기록](docs/failures.md)

## 로드맵

- [x] Phase 1 — 텍스트 3자 엔진 + 평가 체계
- [x] Phase 2 — 음성: push-to-talk STT · 2보이스 TTS · 음성 동기 · prefetch
- [x] Phase 3a: 문장 단위 TTS 스트리밍, hold 기반 빠른 barge-in
- [ ] Phase 3b: 스트리밍 STT + VAD로 push-to-talk 제거
- [ ] Phase 3c: WebRTC 전이중 오디오, 실제 재생 중단 측정, 데모 영상

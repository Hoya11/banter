# Push-to-talk 기준 측정

## 상태

측정 도구와 고정 시나리오를 준비한 상태다. 실제 provider를 사용한 수치는 아직 기록하지 않았다.
수치가 없는 칸을 추정값으로 채우지 않고, 같은 조건으로 직접 실행한 결과만 이후에 추가한다.

## 목적

VAD와 스트리밍 STT를 붙이기 전에 현재 push-to-talk 방식의 끼어들기 성능을 보존한다.
이 결과는 새 방식이 실제로 개선됐는지 비교하는 기준점으로 사용한다.

한 실행의 JSONL에는 다음 정보가 함께 남는다.

- baseline ID와 코드 버전
- interaction, STT, WebSocket, 오디오 전송 방식
- LLM, STT, TTS 모델
- 세션별 max turns, radio timeout, played ack timeout
- hold와 재생 중단 결과, generation 취소, 다음 응답 시작 시각

API key, 대화 원문, 음성 데이터는 기록하지 않는다.

## 지표 정의

| 지표 | 시작 | 종료 | 해석 |
| --- | --- | --- | --- |
| client stop latency | 마이크 버튼 pointerdown | 브라우저에서 현재 Audio의 pause 적용 확인 | 현재 UI가 재생을 멈추도록 지시한 시간 |
| generation invalidation latency | 서버가 hold 수신 | 진행 중 generation 무효화 기록 | 늦은 생성 결과가 폐기되기까지 걸린 시간 |
| next turn latency | 서버가 hold 수신 | 다음 AI 응답 시작 | 유저 개입 뒤 대화가 다시 이어진 시간 |
| hold cancellation latency | 서버가 hold 수신 | 짧은 입력, 권한 실패, STT 실패 등으로 hold 반납 | 정상 취소와 측정 누락을 구분하는 시간 |

client stop latency는 `paused` 결과만 p50, p95, max 계산에 포함한다.

| 결과 | 의미 | 지연 분포 포함 |
| --- | --- | --- |
| paused | 재생 중인 Audio에 pause를 적용하고 정지 상태를 확인함 | 포함 |
| pause_failed | 재생 중이었지만 pause 적용 또는 확인에 실패함 | 제외 |
| queued_only | 들리는 Audio는 없고 대기 중인 항목만 폐기함 | 제외 |
| idle | 재생 중이거나 대기 중인 항목이 없음 | 제외 |

이 값은 브라우저 호출 지연이다. 운영체제 오디오 버퍼와 실제 스피커 출력이 멎는 시점까지
측정하는 값은 아니다. WebRTC 전환 뒤에는 오디오 프레임과 재생 장치 기준 지표를 별도로 둔다.

## 고정 시나리오

각 시나리오는 같은 브라우저, 같은 출력 장치, 같은 네트워크 조건에서 먼저 5회 실행해
측정 계약을 확인한다. 이력서에 p95를 쓸 때는 latency 시나리오별 유효 표본을 최소 20개까지
모은다. 그전에는 p50, max, 개별값을 중심으로 해석하고 p95는 참고값으로만 둔다.
비교 대상인 VAD 버전도 같은 대사 구간과 반복 횟수를 사용한다.

| ID | 실행 방법 | 확인 항목 |
| --- | --- | --- |
| PTT-01 | 첫 번째 AI 문장이 실제 재생 중일 때 버튼을 누르고 1초 이상 말한다 | paused, 다음 응답 시작. generation이 이미 끝났다면 invalidation 없음 |
| PTT-02 | 첫 문장이 재생 중이고 다음 문장이 큐에 들어온 시점에 끼어든다 | 현재 재생 정지, 남은 큐 폐기, 가장 높은 seq ack |
| PTT-03 | 사용자 입력 직후 AI가 답을 생성하는 동안 버튼을 누른다 | generation 취소. 재생 전이면 queued_only 또는 idle |
| PTT-04 | 버튼을 짧게 눌렀다가 음성 없이 바로 놓는다 | hold_off 1회, 정상 cancellation, 새 AI 턴 재개 |
| PTT-05 | 한 세션 동안 끼어들지 않고 정상 재생한다 | 잘못된 hold와 audio stop 이벤트가 생기지 않음 |
| PTT-06 | 세션 종료 뒤 같은 화면에서 다시 시작하고 끼어든다 | 새 session ID, client event 번호 초기화, 이전 세션과 로그 분리 |

PTT-01, PTT-02, PTT-03, PTT-06에서 말할 문장은 하나로 고정한다. hold부터 다음 응답까지의
시간에는 사용자가 말한 길이와 STT 시간이 포함되므로, 문장과 말하는 속도가 달라지면 비교할 수 없다.
시나리오별로 로그 파일을 분리해 서로 다른 조건의 표본이 한 분포에 섞이지 않게 한다.

## 실행 방법

실제 음성 기준 측정은 OpenAI와 ElevenLabs key가 준비된 상태에서만 실행한다.
아래 명령은 비용이 발생할 수 있으므로 자동 테스트나 이번 기준 버전 준비 과정에서는 실행하지 않는다.

```bash
cd server
cp -n .env.example .env
# .env에 provider key와 아래 두 값을 설정한다.
# 예시: PTT-01은 BANTER_EVENT_LOG=artifacts/runs/push-to-talk-v1-ptt-01.jsonl
# BANTER_BUILD_VERSION은 비워 두면 현재 Git 버전을 자동 기록한다.
PYTHONPATH=. uv run uvicorn api.main:app
```

기존 `.env`가 있으면 `cp -n`은 덮어쓰지 않는다. `BANTER_BUILD_VERSION`을 직접 지정해야 한다면
설명용 별칭 대신 실행한 코드의 전체 commit hash를 사용한다.

고정 시나리오를 마친 뒤 결과를 요약한다.

```bash
cd server
PYTHONPATH=. uv run python scripts/summarize_barge_in.py \
  artifacts/runs/push-to-talk-v1-ptt-01.jsonl
```

출력의 `summary`는 결과별 건수와 각 지연의 p50, p95, max를 보여준다. `samples`에는 hold별
상관관계와 누락 항목이 남는다. 원본 JSONL은 Git에 올리지 않는다.

## 결과 기록

| 항목 | 값 |
| --- | --- |
| 실행일 | 측정 전 |
| 브라우저와 버전 | 측정 전 |
| 운영체제와 출력 장치 | 측정 전 |
| 유효 paused 표본 수 | 측정 전 |
| client stop p50 / p95 / max | 측정 전 |
| generation invalidation p50 / p95 / max | 측정 전 |
| next turn p50 / p95 / max | 측정 전 |
| cancellation 및 제외 결과 | 측정 전 |

결과를 채울 때는 요약 JSON과 원본 로그 경로, 실행 환경을 같이 남긴다. 이후 VAD 실험 문서에서
이 표를 그대로 복사해 전후 차이와 실패 사례를 함께 기록한다.

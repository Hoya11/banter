# 스트리밍 STT 비교 계획

## 현재 상태

- 기존 방식: push-to-talk 녹음이 끝난 뒤 WebM 파일 전체를 전사한다.
- 새 방식: push-to-talk를 유지하면서 말하는 동안 24 kHz PCM을 100 ms 단위로 전송한다.
- 모델: 기존 방식은 `gpt-4o-mini-transcribe`, 새 방식은 `gpt-live-transcribe`를 사용한다.
- A/B 성능 측정은 아직 하지 않았다. VAD 실사용에서 발견한 연결 오류와 수정은
  [VAD 실험 기록](vad-interruption.md#연결-생성-단계-재검증)에 남긴다.
- 코드 경로는 가짜 공급자로 검증하며, 이 결과를 실제 성능 수치로 쓰지 않는다.

이번 단계에서는 VAD와 WebRTC를 넣지 않는다. 버튼 조작, 대화 엔진, LLM, TTS는 그대로 둔다. 다만 두 모드는 전송 형식, API, STT 모델이 함께 달라진다. 따라서 결과는 파일 전사와 Realtime 전사의 전체 경로 비교로 해석하고, 차이를 스트리밍 하나의 효과라고 단정하지 않는다.

## 구현 방식

브라우저는 서버가 스트리밍 기능을 알려준 경우에만 AudioWorklet을 사용한다. 입력은 mono PCM16, 24 kHz이며 한 조각은 2,400 samples다. 모든 메시지는 기존 `hold`와 같은 `client_event_id`를 쓴다. AudioContext와 worklet 모듈은 세션을 시작할 때 준비하고, 각 입력에서는 마이크 트랙만 연결한다.

```text
hold
voice_stream_start
voice_stream_chunk 1..N
voice_stream_commit
you
stt_observed
```

버튼을 놓으면 마지막 PCM 조각을 먼저 보낸 뒤 commit한다. 서버는 중복 조각은 무시하고 순번 누락, 잘못된 base64, 홀수 길이 PCM, 크기 초과를 해당 입력의 오류로 처리한다. 부분 전사는 지연 측정에만 사용하며 대화 이력과 LLM prompt에는 최종 전사 한 건만 들어간다.

마이크 권한 승인이나 입력 준비가 오래 걸려 hold 제한 시간을 넘기면 서버가 `hold_expired`를 보낸다. 브라우저는 진행 중인 캡처와 요청 잠금을 함께 정리하므로 다음 입력을 바로 다시 시도할 수 있다.

스트리밍 모드에서 AudioWorklet을 시작하지 못하면 파일 전사로 조용히 바꾸지 않는다. 해당 입력을 취소하고 오류를 표시한다. 그래야 한 실행 안에 두 방식의 표본이 섞이지 않는다. 필요한 경우 서버를 파일 전사 모드로 다시 실행한다.

## 전송 지연과 연결 종료 복구

2026-09-18 검수에서 음성 청크 전송을 기다리는 동안 같은 수신 경로의 텍스트 입력,
취소와 다음 발화가 지연될 수 있음을 확인했다. 연결 준비 중 쌓인 청크도 다음 입력이
들어와야 전송을 시작하는 경우가 있었다. 이는 모의 공급자로 재현한 제어 흐름 문제이며,
실사용에서 관측한 `4000` 연결 종료의 원인이라고 단정하지 않는다.

청크별 제한 시간을 줄이는 방법은 전송 대기 중 입력을 처리하지 못하는 구조를 그대로 둔다.
청크마다 독립 작업을 만드는 방법은 음성 순서와 마지막 청크 이후 commit을 보장하기 어렵다.
따라서 발화마다 하나의 순차 전송 작업을 두고, 입력 수신은 검증과 대기열 추가만 처리하도록
분리했다. 기존 WebSocket 전송과 push-to-talk 기준선은 유지하며 VAD도 같은 경로를 사용한다.

- 연결 준비가 끝나면 새 입력을 기다리지 않고 대기 중인 청크를 전송한다.
- 음성 대기열은 최대 150개 청크이며 commit을 위한 한 칸을 따로 둔다.
  정상적인 100 ms 청크 기준으로 15초 분량이다. 전체 음성 크기 제한도 별도로 유지한다.
- 청크 전송 제한은 2초, 각 청크의 대기 제한과 commit 이후 완료 제한은 각각 15초다.
  대기열 초과나 기한 만료는 오류로 알리고, 음성을 조용히 버리지 않는다.
- 취소 시 현재 요청을 즉시 분리하고 연결 정리는 별도로 진행한다. 늦게 준비된 연결도
  정리하며 이전 요청의 준비 이벤트, 부분 전사와 최종 결과가 다음 입력에 섞이지 않게 한다.
- 마지막으로 수락한 PCM까지 순서대로 전송한 뒤에만 commit하고, 최종 전사만 대화에 반영한다.

연결 종료도 별도로 확인했다. 바깥쪽 정리 제한 2초가 WebSocket의 기본 종료 대기 10초보다
짧아, 종료 응답이 없는 경우 정리 작업만 취소되고 실제 연결이 남을 수 있었다. 내부 종료
제한을 기본 1초로 줄이고, 종료 프레임 전송 자체가 막히는 경우에도 남은 전송 연결을 강제로
닫도록 보완했다. 이 마지막 정리에는 설치된 OpenAI SDK의 내부 연결 핸들을 사용하므로,
SDK나 WebSocket 라이브러리 버전을 바꿀 때 관련 회귀 테스트를 다시 확인해야 한다.

기존 코드에서 실패한 입력 처리 사례 3개와 연결 종료 사례 8개가 수정 후 통과했다.
전송 순서, 대기열 초과, 만료된 청크, 늦은 연결 준비, 오류 정보 보존까지 포함한 전체 Python
테스트는 217개가 통과했고, 이 실행에서 호출하는 웹 테스트는 122개가 통과했다. 두 건수를
별도 독립 실행 건수처럼 합산하지 않는다. Ruff, Python 컴파일, AudioWorklet 문법과 오프라인
잠금 파일 검사도 통과했다.

연결 종료 검증은 실제 SDK와 WebSocket 프로토콜을 메모리 안에서 실행하고, 네트워크 대신
가짜 전송 계층으로 종료 응답 누락과 쓰기 지연을 만든다. 외부 LLM, STT, TTS 호출은 0회다.
대기 기한 테스트의 가상 시간과 이 테스트 결과는 실제 지연 개선 수치가 아니다. 실제 마이크
품질, `4000` 오류 재발 여부와 정상 전사 성공은 이후 별도 확인해야 한다.

## 실행 방법

기본값은 기존 방식이다.

```bash
cd server
BANTER_STT_MODE=record_then_transcribe PYTHONPATH=. uv run uvicorn api.main:app --reload
```

스트리밍 방식은 명시적으로 켠다.

```bash
cd server
BANTER_STT_MODE=streaming_push_to_talk PYTHONPATH=. uv run uvicorn api.main:app --reload
```

측정할 때는 방식마다 새 로그 파일을 사용한다. 로그는 기존 파일에 이어 쓰므로, 다시 비교할 때는 날짜나 실행 ID를 붙인 사용하지 않은 경로로 바꾼다.

```bash
BANTER_EVENT_LOG=artifacts/runs/stt-file.jsonl BANTER_STT_MODE=record_then_transcribe PYTHONPATH=. uv run uvicorn api.main:app
BANTER_EVENT_LOG=artifacts/runs/stt-stream.jsonl BANTER_STT_MODE=streaming_push_to_talk PYTHONPATH=. uv run uvicorn api.main:app
PYTHONPATH=. uv run python scripts/summarize_stt.py artifacts/runs/stt-file.jsonl --skip-first 3
PYTHONPATH=. uv run python scripts/summarize_stt.py artifacts/runs/stt-stream.jsonl --skip-first 3
```

각 서버를 한 번만 실행한 채 준비 3회와 측정 20회를 연속으로 진행한다. 같은 프로세스를 유지해야 준비 실행에서 만든 연결 상태가 측정에도 이어진다. 로그에는 방식별로 총 23개 표본이 남고, 요약할 때 `--skip-first 3`으로 앞의 준비 표본을 제외한다. 결과의 `total_count`, `skipped_warmup_count`, `count`로 제외 범위를 확인한다.

`baseline-push-to-talk-v1` 태그는 이전 코드 상태를 보존하는 기준점이다. 실제 A/B 측정은 현재 코드에서 환경 변수만 바꾸는 편이 좋다. 이렇게 하면 브라우저 관측 방식과 로그 형식이 두 방식에서 같아진다.

## 기록하는 지표

- `release_to_final_ms`: 버튼을 놓은 뒤 브라우저가 최종 전사를 받은 시간
- `input_to_completed_ms`: 서버가 입력을 확정한 뒤 공급자가 최종 전사를 준 시간
- `stream_to_provider_ready_ms`: 스트리밍 시작 뒤 공급자 연결 준비 시간
- `stream_to_first_delta_ms`: 스트리밍 시작 뒤 첫 부분 전사 시간
- 완료 결과: success, empty, error, cancelled, too_short
- 누락 이벤트 수

로그에는 음성, base64, 부분 전사, 최종 전사, API key, 장치 정보가 들어가지 않는다.
지연 분포에는 완료 결과가 success인 표본만 포함한다. error나 cancelled가 빨리 끝났다는 이유로 지연 개선처럼 보이지 않게 하기 위한 기준이다.

지연 로그와 정확도 표본은 분리한다. 글자 오류율을 확인할 때만 `artifacts/runs/stt-accuracy.tsv` 같은 Git 제외 파일에 `case_id`, `mode`, `reference`, `hypothesis`, `success`를 기록한다. 비교 전에는 공백과 문장부호 처리 규칙을 하나로 고정하고, 두 모드에 같은 정답 문장을 사용한다. 이 정확도 표는 현재 자동 요약기의 입력이 아니므로 글자 오류율은 별도로 계산한다.

## 실제 비교 조건

- 같은 기기와 네트워크에서 같은 날 측정한다.
- 첫 비교는 한 데스크톱 브라우저와 정확한 버전을 고정하고 localhost 또는 HTTPS에서 실행한다.
- 한국어 문장 20개 이상을 두 방식에 같은 순서로 입력한다.
- 각 방식의 최초 3회는 같은 프로세스에서 준비 실행으로 진행하고 `--skip-first 3`으로 제외한다.
- LLM, TTS, 대화 설정은 바꾸지 않는다.
- 지연뿐 아니라 전사 성공률과 한국어 글자 오류율도 함께 본다.

목표는 `release_to_final_ms` 중앙값 30% 이상 감소, p95 비악화, 최종 전사 성공률 95% 이상, 한국어 글자 오류율 차이 2%p 이내다. 이 목표는 Realtime 전사 모드 전체의 채택 기준이다. 목표를 충족하지 못하면 연결 준비 시간과 첫 부분 전사 시간을 보고 병목을 나눈다.

## 실제 호출 전 확인

`gpt-live-transcribe`는 현재 Free tier를 지원하지 않는다. 실제 측정 전 유료 API 프로젝트, 결제 상태, 지출 상한을 먼저 확인한다. API key는 `server/.env`에만 두고 Git에 넣지 않는다.

- [OpenAI Realtime transcription 가이드](https://developers.openai.com/api/docs/guides/realtime-transcription)
- [gpt-live-transcribe 모델 정보](https://developers.openai.com/api/docs/models/gpt-live-transcribe)

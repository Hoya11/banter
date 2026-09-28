# Banter 서버

FastAPI가 HTTP와 WebSocket 연결을 제공하고, asyncio 작업이 입력 수신, 응답 생성과 음성
합성을 제어한다. LangGraph는 공통 대화 로직을 사용하는 평가 시나리오에 쓰인다.
현재 구조와 검증 범위는 [프로젝트 README](../README.md)에 있다.

## 코드 탐색

| 경로 | 현재 역할 |
| --- | --- |
| [api/app.py](api/app.py) | 앱 생성, WebSocket 세션, 발화권, 생성 취소, 문장별 합성과 재생 확인 |
| [api/main.py](api/main.py) | 실제 공급자 주입과 STT 및 VAD 설정을 사용하는 실행 진입점 |
| [api/stt_stream.py](api/stt_stream.py) | 발화별 순차 음성 전송, 제한된 대기열과 취소 |
| [api/playback_history.py](api/playback_history.py) | 음성 순번과 발화 연결, 재생 중단 표시와 이력 버전 관리 |
| [api/event_log.py](api/event_log.py) | 끼어들기, 전사와 복구 시점의 JSONL 기록 |
| [engine/graph/](engine/graph) | 대화 상태, 화자 규칙, 프롬프트, 후처리와 평가용 그래프 |
| [engine/personas/](engine/personas) | 두 페르소나의 말투, 성향과 목소리 설정 |
| [engine/eval/](engine/eval) | 시나리오 평가, judge, 추적과 공급자 어댑터 |
| [tests/](tests) | 가짜 공급자와 제어된 이벤트를 사용하는 회귀 테스트 |

실시간 API는 평가 그래프를 실행하지 않고 공통 함수를 직접 조합한다. `create_app`에 공급자를
주입하므로 외부 호출 없이 완료 순서, 오류와 늦은 응답을 제어할 수 있다. 공급자 어댑터는
현재 `engine/eval` 아래에 있지만 API에서도 사용한다.

## 실행과 확인

Python 3.13 이상과 uv를 사용한다. 아래 명령은 `server` 디렉터리에서 실행한다.

```bash
uv sync --locked
PYTHONPATH=. uv run uvicorn api.app:app --reload
```

`http://localhost:8000`에서 고정 스텁 대사로 화면과 연결을 확인한다. 이 진입점은 실제
LLM, STT와 TTS를 호출하지 않는다. 실제 음성 실행과 자동 테스트 명령은
[프로젝트 실행 안내](../README.md#실행)를 따른다. 실제 공급자를 사용하는 `api.main:app`은
키, 결제 상태, 지출 상한과 호출 범위를 먼저 확인해야 한다.

## 전사 연결 점검

AI 대화 전체를 실행하지 않고 `gpt-live-transcribe` 연결과 세션 설정만 확인하는 명령.
기본 실행은 안내만 출력하며 키 파일 접근과 API 호출 없음.

```bash
PYTHONPATH=. uv run python scripts/check_stt_connection.py
```

결제 상태와 지출 상한 확인 후 같은 명령에 `--live`를 붙여 실제 연결 1회 실행.
기존 환경변수 또는 `server/.env`의 OpenAI 키 사용. 음성 전송, commit, LLM, TTS 호출 없음.
연결 대기 10초, 세션 및 SDK 정리 5초 예산. 키와 원문 오류를 제외한 진단 정보만 출력.
연결 성공은 실제 음성 전사와 대화 복구 검증을 대체하지 않음. 현재 원격 실행은 미실시.

## 구조의 한계

현재 `api/app.py`에 연결 처리와 여러 대화 상태의 제어가 집중돼 있다. 미완료 사전 생성
대기의 입력 지연은 수정했으며, [별도 회귀](tests/test_prefetch.py)로 입력과 완료의 경합 및
취소를 확인한다. 세션 제어의 책임 분리는 실환경 검증 이후의 개선 후보.
[재현과 후속 계획](../docs/phase-3-plan.md#다음-개발-범위)

대화 상태는 메모리에 보관하며 서버 재시작 후 세션 복원을 제공하지 않는다. WebRTC나
LiveKit 전환은 필수 단계가 아니며, 전송 지연, 지터나 에코가 실제 병목일 때 검토한다.
선택 이유와 대안은 [설계 결정](../docs/decisions.md)에 있다.

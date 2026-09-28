"""음성 전송 없이 Realtime 전사 세션의 연결과 설정을 한 번 확인한다."""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from engine.eval.providers import OpenAIRealtimeSTTClient, realtime_stt_diagnostics

CONNECT_SECONDS = 10.0
CLEANUP_SECONDS = 3.0
SDK_CLOSE_SECONDS = 2.0
SERVER_DIR = Path(__file__).resolve().parents[1]


async def _session(client) -> None:
    turn = None
    try:
        turn = await client.start()
    finally:
        if turn is not None:
            await turn.cancel()


async def check_connection(api_key: str) -> dict:
    """연결 대기 10초, 세션 정리 3초, SDK 정리 2초의 예산으로 확인한다."""
    sdk = None
    session_task = None
    failures = []

    def failed(error: BaseException, phase: str) -> None:
        failures.append({'phase': phase, **realtime_stt_diagnostics(error)})

    try:
        sdk = AsyncOpenAI(api_key=api_key, timeout=CONNECT_SECONDS, max_retries=0)
        client = OpenAIRealtimeSTTClient(
            async_client=sdk,
            model='gpt-live-transcribe',
            timeout=CONNECT_SECONDS,
            close_timeout=1.0,
        )
        session_task = asyncio.create_task(_session(client))
        done, _ = await asyncio.wait({session_task}, timeout=CONNECT_SECONDS)
        if not done:
            failed(TimeoutError(), 'connect')
        else:
            session_task.result()
    except Exception as exc:  # noqa: BLE001 - 공급자 원문 대신 안전한 진단만 출력
        failed(exc, 'connect')
    finally:
        if session_task is not None and not session_task.done():
            session_task.cancel()
            done, _ = await asyncio.wait({session_task}, timeout=CLEANUP_SECONDS)
            if not done:
                failed(TimeoutError(), 'session_cleanup')
        if (
            session_task is not None and session_task.done()
            and not session_task.cancelled()
        ):
            # 이미 처리한 예외도 회수해 종료 시 미회수 경고를 방지한다.
            error = session_task.exception()
            if error is not None and not failures:
                failed(error, 'session_cleanup')
        if sdk is not None:
            close_task = asyncio.create_task(sdk.close())
            done, _ = await asyncio.wait({close_task}, timeout=SDK_CLOSE_SECONDS)
            if not done:
                close_task.cancel()
                failed(TimeoutError(), 'sdk_cleanup')
            else:
                try:
                    close_task.result()
                except Exception as exc:  # noqa: BLE001 - 종료 오류 원문 노출 방지
                    failed(exc, 'sdk_cleanup')
    return {'status': 'failed' if failures else 'configured', 'diagnostics': failures}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='외부 전사 연결을 한 번 실제 확인')
    args = parser.parse_args(argv)
    print('범위: gpt-live-transcribe 전사 세션 1회 연결 및 24 kHz PCM 설정 확인.')
    print('음성 전송·commit·LLM·TTS 호출 없음. 연결 대기와 정리 예산 합계 15초.')
    print('실행 전 계정의 결제 상태와 지출 상한 확인. 이 명령은 비용이나 전사 품질을 검증하지 않음.')
    if not args.live:
        print('안내만 출력. 실제 확인은 --live 지정 시 실행.')
        return 0

    logging_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        load_dotenv(SERVER_DIR / '.env', override=False)
        api_key = os.environ.get('OPENAI_API_KEY', '').strip()
        if not api_key:
            print(json.dumps({'status': 'failed', 'code': 'missing_api_key'}))
            return 1
        result = asyncio.run(check_connection(api_key))
    except KeyboardInterrupt:
        result = {'status': 'failed', 'code': 'interrupted'}
    except Exception as exc:  # noqa: BLE001 - 예상 밖 오류도 키나 원문 없이 종료
        result = {'status': 'failed', 'diagnostics': [realtime_stt_diagnostics(exc)]}
    finally:
        logging.disable(logging_level)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'configured' else 1


if __name__ == '__main__':
    raise SystemExit(main())

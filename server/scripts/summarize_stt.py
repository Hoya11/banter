"""STT 방식별 지연 JSONL을 요약한다."""

import argparse
import json

from api.event_log import build_stt_report, summarize_stt_latencies


def main() -> None:
    parser = argparse.ArgumentParser(description='STT 지연 JSONL 요약')
    parser.add_argument('path', help='BANTER_EVENT_LOG로 기록한 JSONL 경로')
    parser.add_argument(
        '--skip-first',
        type=int,
        default=0,
        help='각 STT 방식에서 앞쪽 준비 표본을 제외할 개수',
    )
    args = parser.parse_args()
    if args.skip_first < 0:
        parser.error('--skip-first는 0 이상이어야 합니다')
    report = build_stt_report(
        summarize_stt_latencies(args.path),
        skip_first=args.skip_first,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

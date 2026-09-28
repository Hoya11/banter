"""VAD 감지와 최종 전사 관측의 로컬 JSONL을 요약한다."""

import argparse
import json

from api.event_log import build_vad_report


def main() -> None:
    parser = argparse.ArgumentParser(description='VAD 지연 JSONL 요약')
    parser.add_argument('path', help='BANTER_EVENT_LOG로 기록한 JSONL 경로')
    args = parser.parse_args()
    print(json.dumps(build_vad_report(args.path), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

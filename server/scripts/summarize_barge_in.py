"""한 실행의 hold 지연 샘플을 JSON으로 출력한다."""

import argparse
import json
from dataclasses import asdict

from api.event_log import build_hold_report, summarize_hold_latencies


def main() -> None:
    parser = argparse.ArgumentParser(description='banter 끼어들기 이벤트 로그 요약')
    parser.add_argument('path', help='BANTER_EVENT_LOG로 기록한 JSONL 파일')
    args = parser.parse_args()

    samples = summarize_hold_latencies(args.path)
    result = {
        'summary': build_hold_report(samples),
        'samples': [asdict(sample) for sample in samples],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

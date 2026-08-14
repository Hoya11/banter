"""단일 judge (D-005).

LLM provider를 주입받아 대화를 루브릭으로 채점한다.
단일→앙상블 전환 시 judge() 시그니처와 rubric/schema 계약은 불변 —
바뀌는 건 주입되는 LLMClient 구현뿐이다.
"""

import json
from pathlib import Path
from typing import Protocol

import yaml

from .schema import JudgeResult, Turn

RUBRIC_PATH = Path(__file__).parent / 'rubric.yaml'


def load_rubric(version: str = 'v1') -> dict:
    """rubric.yaml 로드. 필수 키 부재·버전 불일치 시 에러."""
    data = yaml.safe_load(RUBRIC_PATH.read_text(encoding='utf-8'))
    if not data or not {'version', 'scale', 'items'} <= set(data):
        raise ValueError('rubric.yaml에 필수 키(version/scale/items)가 없다')
    if data['version'] != version:
        raise ValueError(f'루브릭 버전 불일치: 요청={version}, 파일={data["version"]}')
    return data


class LLMClient(Protocol):
    """judge가 쓰는 LLM 호출 계약. 실제 구현(OpenAI/Anthropic 등)은 LLM 선택 후 주입."""

    def complete(self, system: str, user: str) -> str:
        """system·user 프롬프트로 채점 결과 JSON 문자열을 반환한다."""
        ...


def build_prompt(rubric: dict, transcript: list[Turn]) -> tuple[str, str]:
    """루브릭과 대화로 (system, user) 프롬프트를 구성한다."""
    items = '\n'.join(
        f'- {it["key"]}: {it["label"]} — {it["description"]}' for it in rubric['items']
    )
    lo, hi = rubric['scale']
    system = (
        '너는 3자 수다(유저 1명 + AI 2명) 대화의 품질을 채점하는 평가자다.\n'
        f'각 항목을 {lo}~{hi} 정수로 채점하고 근거를 한국어로 짧게 남겨라.\n'
        f'채점 항목:\n{items}\n'
        'JSON만 반환한다. overall_comment는 scores 바깥의 최상위 필드다 (scores 안에 넣지 마라):\n'
        '{"scores": {"<key>": {"score": int, "reason": str}, ... (위 항목 전부)}, "overall_comment": "총평"}'
    )
    # 끊긴 발화는 표시해서 judge가 '개입 후 수습'을 실제로 평가할 수 있게 한다
    convo = '\n'.join(
        f'{t.speaker}: {t.text}' + (' [말하다 끊김]' if t.interrupted else '')
        for t in transcript
    )
    user = f'다음 대화를 채점하라:\n{convo}'
    return system, user


def judge(
    transcript: list[Turn], client: LLMClient, rubric_version: str = 'v1'
) -> JudgeResult:
    """대화를 루브릭으로 채점한다 (단일 심판).

    채점 결과가 루브릭 항목과 정확히 일치하지 않으면(누락·오타 key) 예외 —
    is_go가 잘못된 항목 집합으로 판정하는 것을 막는다.
    """
    rubric = load_rubric(rubric_version)
    system, user = build_prompt(rubric, transcript)
    raw = client.complete(system, user)
    result = JudgeResult.model_validate(_parse_judge_json(raw))
    expected = {it['key'] for it in rubric['items']}
    if set(result.scores) != expected:
        raise ValueError(
            f'채점 항목 불일치: 기대={sorted(expected)}, 실제={sorted(result.scores)}'
        )
    return result


def _parse_judge_json(raw: str) -> dict:
    """judge 출력 JSON을 파싱하고 흔한 구조 오류를 복구한다.

    LLM이 overall_comment를 scores 안에 잘못 넣는 경우를 최상위로 끄집어낸다.
    """
    data = json.loads(raw)
    scores = data.get('scores')
    if (
        isinstance(scores, dict)
        and 'overall_comment' in scores
        and 'overall_comment' not in data
    ):
        data['overall_comment'] = scores.pop('overall_comment')
    return data

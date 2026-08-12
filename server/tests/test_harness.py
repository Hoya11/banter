"""end-to-end 파일럿 배관 테스트.

fake client로 '대화 생성 → 채점 → go 판정' 흐름이 이어지는지 검증한다
(LLM 답변 품질이 아니라 파이프 연결을 본다).
"""

from engine.eval.harness import load_scenarios, run_pilot, run_scenario
from engine.eval.judge import load_rubric


class FakeUtterance:
    def complete(self, system: str, user: str) -> str:
        return '오늘 피곤하다'


class FakeJudge:
    """루브릭 5항목을 모두 8점으로 채점하는 fake 심판."""

    def complete(self, system: str, user: str) -> str:
        keys = [it['key'] for it in load_rubric()['items']]
        items = ', '.join(f'"{k}": {{"score": 8, "reason": "ok"}}' for k in keys)
        return '{"scores": {' + items + '}, "overall_comment": "good"}'


def test_run_pilot_produces_scored_transcript():
    transcript, result, go = run_pilot(FakeUtterance(), FakeJudge(), turns=4)
    # 4턴 → 발화 4개
    assert len(transcript) == 4
    # 채점 결과가 루브릭 5항목을 정확히 담는다
    assert set(result.scores) == {it['key'] for it in load_rubric()['items']}
    # 8점 균일 → go 컷 통과
    assert go is True


def test_run_scenario_includes_user_utterance():
    # ai → user(주입) → ai 순서면 대화에 유저 발화가 실제로 들어간다
    steps = [{'type': 'ai'}, {'type': 'user', 'text': '안녕'}, {'type': 'ai'}]
    transcript, result, go = run_scenario(steps, FakeUtterance(), FakeJudge())
    assert any(t.speaker == 'user' and t.text == '안녕' for t in transcript)
    assert set(result.scores) == {it['key'] for it in load_rubric()['items']}


def test_scenarios_yaml_loads_and_is_wellformed():
    # 실제 scenarios.yaml을 파싱·검증한다 (파일 문법 오류를 잡는 관문)
    scenarios = load_scenarios()
    assert len(scenarios) >= 1
    for sc in scenarios:
        assert sc['steps']
        for step in sc['steps']:
            assert step['type'] in ('ai', 'user')
            if step['type'] == 'user':
                assert step.get('text')  # 유저 step엔 발화 텍스트 필수

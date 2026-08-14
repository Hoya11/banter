"""end-to-end 파일럿 배관 테스트.

fake client로 '대화 생성 → 채점 → go 판정' 흐름이 이어지는지 검증한다
(LLM 답변 품질이 아니라 파이프 연결을 본다).
"""

from engine.eval.harness import (
    _interrupt,
    load_scenarios,
    run_pilot,
    run_scenario,
    score_stats,
    score_transcript_repeated,
)
from engine.eval.judge import load_rubric
from engine.eval.schema import ItemScore, JudgeResult, Turn


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
            assert step['type'] in ('ai', 'user', 'interrupt')
            if step['type'] in ('user', 'interrupt'):
                assert step.get('text')  # 유저/개입 step엔 발화 텍스트 필수


def test_interrupt_truncates_and_marks_previous_ai():
    # 끊긴 발화는 플래그만이 아니라 텍스트도 실제로 잘려야 한다 (검수 정정 반영)
    state = {
        'messages': [
            {'speaker': 'ai_a', 'text': '오늘 회사에서 진짜 별일 다 있었거든', 'ts': 0.0, 'interrupted': False}
        ]
    }
    out = _interrupt(state, '잠깐만')
    cut = out['messages'][0]
    assert cut['interrupted'] is True
    assert len(cut['text']) < len('오늘 회사에서 진짜 별일 다 있었거든')  # 실제로 잘림
    assert out['messages'][-1]['speaker'] == 'user'
    assert out['current_speaker'] == 'user'
    assert out['consecutive_ai_turns'] == 0


def test_transcript_carries_interrupted_to_judge():
    # interrupted 플래그가 judge 입력(Turn)과 프롬프트까지 전달돼야 수습 평가가 성립한다
    from engine.eval.harness import _to_transcript
    from engine.eval.judge import build_prompt, load_rubric

    messages = [
        {'speaker': 'ai_a', 'text': '오늘 회사에서', 'ts': 0.0, 'interrupted': True},
        {'speaker': 'user', 'text': '잠깐만', 'ts': 0.0, 'interrupted': False},
    ]
    transcript = _to_transcript(messages)
    assert transcript[0].interrupted is True
    _, user_prompt = build_prompt(load_rubric(), transcript)
    assert '[말하다 끊김]' in user_prompt


def test_run_scenario_handles_interrupt():
    steps = [{'type': 'ai'}, {'type': 'interrupt', 'text': '잠깐만'}, {'type': 'ai'}]
    transcript, result, _ = run_scenario(steps, FakeUtterance(), FakeJudge())
    assert any(t.speaker == 'user' and t.text == '잠깐만' for t in transcript)
    assert set(result.scores) == {it['key'] for it in load_rubric()['items']}


def test_score_stats_mean_and_std():
    def _r(value):
        return JudgeResult(scores={'k': ItemScore(score=value, reason='')}, overall_comment='')

    stats = score_stats([_r(6), _r(8)])
    assert stats['k']['mean'] == 7.0
    assert stats['k']['std'] == 1.0  # pstdev([6, 8]) = 1.0
    assert stats['k']['values'] == [6, 8]


def test_score_transcript_repeated_batches_all_runs():
    # FakeJudge는 항상 8점 균일 → 평균 8, 표준편차 0, n회 수집
    stats = score_transcript_repeated([Turn(speaker='ai_a', text='x')], FakeJudge(), n=3)
    for s in stats.values():
        assert s['mean'] == 8.0
        assert s['std'] == 0.0
        assert len(s['values']) == 3

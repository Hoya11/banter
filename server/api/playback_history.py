"""전송한 음성과 대화 이력을 연결해 재생 중단 표시를 반영한다."""


class PlaybackHistory:
    def __init__(self) -> None:
        self._next_id = 0
        self._segments: dict[str, tuple[int, str]] = {}
        self._interrupted: set[str] = set()
        self.revision = 0

    def begin(self) -> str:
        self._next_id += 1
        return f'utterance-{self._next_id}'

    def sent(self, seq: int, utterance_id: str) -> None:
        self._segments[f'audio-{seq}'] = (seq, utterance_id)

    def acknowledge(self, seq: int) -> None:
        self._segments = {
            segment: value
            for segment, value in self._segments.items()
            if value[0] > seq
        }

    def interrupt(self, audio_stop: dict | None) -> None:
        if not audio_stop or audio_stop.get('outcome') != 'paused':
            return
        segment = self._segments.get(audio_stop.get('segment_id'))
        if segment is None:
            return
        _, utterance_id = segment
        if utterance_id not in self._interrupted:
            self._interrupted.add(utterance_id)
            self.revision += 1

    def apply(self, state: dict) -> dict:
        """생성된 전문은 보존하고 중단 여부만 표시한다. 청취 위치는 추정하지 않는다."""
        messages = []
        changed = False
        for message in state['messages']:
            if (
                message.get('utterance_id') in self._interrupted
                and not message.get('interrupted')
            ):
                message = {**message, 'interrupted': True}
                changed = True
            messages.append(message)
        return {**state, 'messages': messages} if changed else state

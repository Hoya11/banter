"""발화별 순차 전송을 입력 수신과 분리한다."""

import asyncio
from collections.abc import Callable


class StreamingSTTRequest:
    def __init__(
        self, provider, *, on_delta: Callable, on_ready: Callable,
        track_task: Callable, max_pending_chunks: int,
        append_timeout: float, queue_timeout: float,
    ) -> None:
        self._provider = provider
        self._on_delta = on_delta
        self._on_ready = on_ready
        self._track_task = track_task
        self._max_pending_chunks = max_pending_chunks
        # Reserve one slot for commit, even when the audio queue is full.
        self._queue = asyncio.Queue(maxsize=max_pending_chunks + 1)
        self._append_timeout = append_timeout
        self._queue_timeout = queue_timeout
        self._turn = None
        self._start_task = None
        self._close_task = None
        self._cancelled = False
        self._finished = False
        self._commit_deadline = None
        self.committed = False
        self.task = asyncio.create_task(self._run())
        track_task(self.task)

    def append(self, audio: bytes) -> None:
        if self.committed or self._cancelled:
            raise RuntimeError('종료된 STT 요청입니다')
        if self.task.done():
            if not self.task.cancelled():
                self.task.result()
            raise RuntimeError('종료된 STT 요청입니다')
        if self._queue.qsize() >= self._max_pending_chunks:
            raise asyncio.QueueFull
        deadline = asyncio.get_running_loop().time() + self._queue_timeout
        self._queue.put_nowait((audio, deadline))

    def commit(self) -> None:
        if self.committed or self._cancelled:
            return
        self.committed = True
        self._commit_deadline = asyncio.get_running_loop().time() + self._queue_timeout
        self._queue.put_nowait(None)

    def _clear_queue(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    def _schedule_close(self, *, success: bool = False) -> None:
        if self._turn is None or self._close_task is not None:
            return
        # finish() owns normal provider cleanup; close() lets us track it when exposed.
        close = getattr(self._turn, 'close', None) if success else self._turn.cancel
        if close is not None:
            self._close_task = asyncio.create_task(close())
            self._track_task(self._close_task)

    def _collect_started_turn(self, task: asyncio.Task) -> None:
        try:
            turn = task.result()
        except (asyncio.CancelledError, Exception):
            return
        if self._cancelled or self._finished:
            self._turn = turn
            self._schedule_close()

    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._clear_queue()
        if not self.task.done():
            self.task.cancel()
        self._schedule_close()

    def _check_current(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError

    def _remaining(self, deadline: float) -> float:
        if self._commit_deadline is not None:
            deadline = min(deadline, self._commit_deadline)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError('STT 전송 대기 시간이 초과됐습니다')
        return remaining

    async def _run(self) -> str:
        success = False
        try:
            self._start_task = asyncio.create_task(
                self._provider.start(on_delta=self._on_delta)
            )
            self._track_task(self._start_task)
            self._start_task.add_done_callback(self._collect_started_turn)
            self._turn = await asyncio.wait_for(
                asyncio.shield(self._start_task), timeout=self._queue_timeout,
            )
            self._check_current()
            self._on_ready()
            while True:
                item = await self._queue.get()
                self._check_current()
                if item is None:
                    break
                audio, deadline = item
                timeout = min(self._append_timeout, self._remaining(deadline))
                await asyncio.wait_for(
                    self._turn.append(audio), timeout=timeout,
                )
                self._check_current()
            timeout = self._remaining(self._commit_deadline)
            text = await asyncio.wait_for(
                self._turn.finish(), timeout=timeout,
            )
            self._check_current()
            success = True
            return text
        finally:
            self._finished = True
            if self._start_task is not None:
                if not self._start_task.done():
                    self._start_task.cancel()
                elif self._turn is None:
                    self._collect_started_turn(self._start_task)
            self._clear_queue()
            self._schedule_close(success=success)

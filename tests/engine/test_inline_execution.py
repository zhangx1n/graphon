"""Single-threaded inline execution selected with ``workers=0``.

Runtimes such as Pyodide cannot start threads, so the engine must run every
node on the thread that consumes ``Engine.run()`` and still surface each node
event as soon as the node produces it.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import yaml

import graphon.dsl.node_factory as node_factory_module
import graphon.engine.event.node_failure as node_failure_module
from graphon.dsl import loads
from graphon.engine_events.base import EngineEvent
from graphon.engine_events.graph import (
    GraphRunAbortedEvent,
    GraphRunPartialSucceededEvent,
)
from graphon.engine_events.node import (
    NodeRunRetryEvent,
    NodeRunStartedEvent,
    NodeRunStreamChunkEvent,
)
from graphon.model_runtime.entities.llm_entities import (
    LLMResultChunk,
    LLMResultChunkDelta,
)
from graphon.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
)
from tests.helpers.workflow_events import (
    FakeSlimLLM,
    final_outputs,
    threads_forbidden,
    threads_forbidden_if_inline,
)

_OPENAI_PLUGIN_ID = "langgenius/openai:0.3.8@test"


class _StreamingSlimLLM(FakeSlimLLM):
    """Stream fixed text pieces and record how far the model has produced."""

    pieces: ClassVar[list[str]] = []
    failures_before_success: ClassVar[int] = 0
    produced: ClassVar[list[str]] = []
    stream_closed: ClassVar[list[bool]] = []

    def invoke_llm(
        self,
        *,
        prompt_messages: Sequence[PromptMessage],
        model_parameters: Mapping[str, Any],
        tools: Sequence[PromptMessageTool] | None,
        stop: Sequence[str] | None,
        stream: bool,
    ) -> Generator[LLMResultChunk, None, None]:
        _ = prompt_messages, model_parameters, tools, stop
        assert stream
        if _StreamingSlimLLM.failures_before_success > 0:
            _StreamingSlimLLM.failures_before_success -= 1
            msg = "model temporarily unavailable"
            raise RuntimeError(msg)
        return self._stream()

    def _stream(self) -> Generator[LLMResultChunk, None, None]:
        try:
            for piece in self.pieces:
                self.produced.append(piece)
                yield LLMResultChunk(
                    model=self.model_name,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=piece),
                    ),
                )
        finally:
            self.stream_closed.append(True)


@pytest.fixture
def streaming_llm(monkeypatch: pytest.MonkeyPatch) -> type[_StreamingSlimLLM]:
    _StreamingSlimLLM.instances = []
    _StreamingSlimLLM.pieces = ["Hel", "lo"]
    _StreamingSlimLLM.failures_before_success = 0
    _StreamingSlimLLM.produced = []
    _StreamingSlimLLM.stream_closed = []
    monkeypatch.setitem(node_factory_module.__dict__, "SlimLLM", _StreamingSlimLLM)
    return _StreamingSlimLLM


def _llm_dsl(*, retry_config: Mapping[str, Any] | None = None) -> str:
    llm_data: dict[str, Any] = {
        "type": "llm",
        "title": "LLM",
        "model": {
            "provider": "langgenius/openai/openai",
            "name": "gpt-test",
            "mode": "chat",
            "completion_params": {},
        },
        "prompt_template": [{"role": "user", "text": "{{#sys.query#}}"}],
        "context": {"enabled": False},
    }
    if retry_config is not None:
        llm_data["retry_config"] = dict(retry_config)
    return yaml.safe_dump({
        "kind": "graph",
        "dependencies": [
            {
                "type": "marketplace",
                "value": {"marketplace_plugin_unique_identifier": _OPENAI_PLUGIN_ID},
            }
        ],
        "graph": {
            "nodes": [
                {"id": "start", "data": {"type": "start", "variables": []}},
                {"id": "llm", "data": llm_data},
                {
                    "id": "end",
                    "data": {
                        "type": "end",
                        "outputs": [
                            {"variable": "text", "value_selector": ["llm", "text"]}
                        ],
                    },
                },
            ],
            "edges": [
                {"source": "start", "target": "llm"},
                {"source": "llm", "target": "end"},
            ],
        },
    })


def _load(dsl: str, *, workers: int = 0) -> Any:
    return loads(
        dsl,
        workers=workers,
        start_inputs={"query": "hi"},
        credentials={
            "model_credentials": [
                {"vendor": "openai", "values": {"api_key": "secret-key"}},
            ],
        },
    )


def _is_llm_chunk(event: EngineEvent) -> bool:
    return isinstance(event, NodeRunStreamChunkEvent) and event.node_id == "llm"


def _started_node_ids(events: Sequence[EngineEvent]) -> list[str]:
    return [event.node_id for event in events if isinstance(event, NodeRunStartedEvent)]


def test_inline_engine_delivers_each_llm_chunk_before_the_model_continues(
    streaming_llm: type[_StreamingSlimLLM],
) -> None:
    produced_at_each_chunk: list[list[str]] = []
    events: list[EngineEvent] = []

    with threads_forbidden():
        engine = _load(_llm_dsl())
        for event in engine.run():
            events.append(event)
            if _is_llm_chunk(event) and event.chunk:
                produced_at_each_chunk.append(list(streaming_llm.produced))

    assert produced_at_each_chunk == [["Hel"], ["Hel", "lo"]]
    assert final_outputs(events) == {"text": "Hello"}


def test_inline_engine_closes_the_model_stream_when_the_consumer_stops(
    streaming_llm: type[_StreamingSlimLLM],
) -> None:
    with threads_forbidden():
        engine = _load(_llm_dsl())
        run = engine.run()
        for event in run:
            if _is_llm_chunk(event) and event.chunk:
                break
        run.close()
        # Checked while the engine is still referenced, so a node generator
        # the engine kept open would show up here.
        assert streaming_llm.stream_closed == [True]

    assert streaming_llm.produced == ["Hel"]


@pytest.mark.usefixtures("streaming_llm")
def test_inline_engine_aborts_before_scheduling_the_next_node() -> None:
    events: list[EngineEvent] = []

    with threads_forbidden():
        engine = _load(_llm_dsl())
        for event in engine.run():
            events.append(event)
            if _is_llm_chunk(event) and event.chunk == "Hel":
                engine.request_abort("user stopped the run")

    assert isinstance(events[-1], GraphRunAbortedEvent)
    assert "end" not in _started_node_ids(events)


@pytest.mark.parametrize("workers", [2, 0], ids=["threads", "inline"])
def test_zero_interval_retry_does_not_sleep(
    streaming_llm: type[_StreamingSlimLLM],
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
) -> None:
    def fail_sleep(seconds: float) -> None:
        msg = f"zero-interval retry must not sleep (slept {seconds}s)"
        raise AssertionError(msg)

    monkeypatch.setattr(node_failure_module, "time", SimpleNamespace(sleep=fail_sleep))
    streaming_llm.failures_before_success = 1

    with threads_forbidden_if_inline(workers):
        engine = _load(
            _llm_dsl(
                retry_config={
                    "retry_enabled": True,
                    "max_retries": 1,
                    "retry_interval": 0,
                }
            ),
            workers=workers,
        )
        events = list(engine.run())

    terminal = events[-1]
    assert [type(event) for event in events].count(NodeRunRetryEvent) == 1
    # The failed attempt counts as a handled exception, so the run ends as
    # partially succeeded with the retried output.
    assert isinstance(terminal, GraphRunPartialSucceededEvent)
    assert terminal.outputs == {"text": "Hello"}

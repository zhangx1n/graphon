from __future__ import annotations

import json
from collections.abc import Generator, Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

import graphon.dsl.slim.client as slim_client_module
from graphon.dsl.slim.client import (
    SlimChunkEvent,
    SlimClient,
    SlimClientConfig,
    SlimClientError,
    SlimDoneEvent,
    SlimEvent,
    parse_slim_event,
)
from graphon.model_runtime.entities.model_entities import ModelType, ParameterType


class _FakeSlimClient(SlimClient):
    def __init__(
        self,
        *,
        config: SlimClientConfig,
        events: list[SlimEvent],
    ) -> None:
        super().__init__(config=config, binary_path="slim")
        self.events = events

    def invoke_events(
        self,
        *,
        plugin_id: str,
        action: str,
        data: Mapping[str, Any],
    ) -> Generator[SlimEvent, None, None]:
        _ = plugin_id, action, data
        yield from self.events


def test_remote_slim_client_unwraps_daemon_success_payload(tmp_path: Path) -> None:
    client = _FakeSlimClient(
        config=SlimClientConfig(
            folder=tmp_path,
            mode="remote",
            daemon_addr="http://daemon.test",
            daemon_key="secret",
        ),
        events=[
            SlimChunkEvent(
                data={
                    "code": 0,
                    "message": "success",
                    "data": {"num_tokens": 42},
                },
            ),
            SlimDoneEvent(),
        ],
    )

    chunks = list(client.invoke_chunks(plugin_id="plugin", action="action", data={}))

    assert chunks == [{"num_tokens": 42}]


def test_remote_slim_client_raises_daemon_error(tmp_path: Path) -> None:
    client = _FakeSlimClient(
        config=SlimClientConfig(
            folder=tmp_path,
            mode="remote",
            daemon_addr="http://daemon.test",
            daemon_key="secret",
        ),
        events=[
            SlimChunkEvent(
                data={
                    "code": 500,
                    "message": "bad credentials",
                    "data": None,
                },
            ),
            SlimDoneEvent(),
        ],
    )

    with pytest.raises(SlimClientError, match="bad credentials"):
        list(client.invoke_chunks(plugin_id="plugin", action="action", data={}))


def test_remote_slim_client_preserves_daemon_error_metadata(tmp_path: Path) -> None:
    client = _FakeSlimClient(
        config=SlimClientConfig(
            folder=tmp_path,
            mode="remote",
            daemon_addr="http://daemon.test",
            daemon_key="secret",
        ),
        events=[
            SlimChunkEvent(
                data={
                    "code": "killed_by_timeout",
                    "message": "killed by timeout",
                    "data": {"stage": "token_count"},
                },
            ),
            SlimDoneEvent(),
        ],
    )

    with pytest.raises(SlimClientError) as exc_info:
        list(client.invoke_chunks(plugin_id="plugin", action="action", data={}))

    assert exc_info.value.code == "killed_by_timeout"
    assert exc_info.value.stage == "token_count"
    assert exc_info.value.data["data"] == {"stage": "token_count"}


def test_remote_slim_client_preserves_top_level_daemon_stage(
    tmp_path: Path,
) -> None:
    client = _FakeSlimClient(
        config=SlimClientConfig(
            folder=tmp_path,
            mode="remote",
            daemon_addr="http://daemon.test",
            daemon_key="secret",
        ),
        events=[
            SlimChunkEvent(
                data={
                    "code": "killed_by_timeout",
                    "stage": "token_count",
                    "message": "killed by timeout",
                    "data": "not-a-mapping",
                },
            ),
            SlimDoneEvent(),
        ],
    )

    with pytest.raises(SlimClientError) as exc_info:
        list(client.invoke_chunks(plugin_id="plugin", action="action", data={}))

    assert exc_info.value.code == "killed_by_timeout"
    assert exc_info.value.stage == "token_count"


def test_slim_error_event_preserves_code_and_stage() -> None:
    with pytest.raises(SlimClientError) as exc_info:
        parse_slim_event(
            '{"event":"error","data":{'
            '"code":"killed_by_timeout",'
            '"stage":"get_llm_num_tokens",'
            '"message":"killed by timeout"'
            "}}",
        )

    assert exc_info.value.code == "killed_by_timeout"
    assert exc_info.value.stage == "get_llm_num_tokens"


def test_slim_error_event_accepts_non_mapping_data() -> None:
    with pytest.raises(SlimClientError) as exc_info:
        parse_slim_event('{"event":"error","data":"plain failure"}')

    assert str(exc_info.value) == "plain failure"
    assert exc_info.value.code is None
    assert exc_info.value.stage is None


def test_slim_extract_preserves_process_error_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def run_stub(*args: Any, **kwargs: Any) -> object:
        _ = args, kwargs

        class ProcessResult:
            returncode = 1
            stdout = ""
            stderr = (
                '{"error_code":"killed_by_timeout",'
                '"stage":"get_llm_num_tokens",'
                '"message":"killed by timeout"}'
            )

        return ProcessResult()

    monkeypatch.setattr(slim_client_module.subprocess, "run", run_stub)
    client = SlimClient(config=SlimClientConfig(folder=tmp_path), binary_path="slim")

    with pytest.raises(SlimClientError) as exc_info:
        client.extract(plugin_id="plugin")

    assert str(exc_info.value) == "killed by timeout"
    assert exc_info.value.code == "killed_by_timeout"
    assert exc_info.value.stage == "get_llm_num_tokens"
    assert exc_info.value.return_code == 1


@pytest.mark.parametrize(
    ("stderr", "expected_message"),
    [
        ("", "Slim process exited with code 1"),
        ("not json", "Slim process exited with code 1: not json"),
        ('["not", "a", "mapping"]', "Slim process exited with code 1: "),
    ],
)
def test_slim_extract_preserves_unstructured_process_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stderr: str,
    expected_message: str,
) -> None:
    def run_stub(*args: Any, **kwargs: Any) -> object:
        _ = args, kwargs

        class ProcessResult:
            returncode = 1
            stdout = ""
            stderr = stderr_value

        return ProcessResult()

    stderr_value = stderr
    monkeypatch.setattr(slim_client_module.subprocess, "run", run_stub)
    client = SlimClient(config=SlimClientConfig(folder=tmp_path), binary_path="slim")

    with pytest.raises(SlimClientError) as exc_info:
        client.extract(plugin_id="plugin")

    assert str(exc_info.value).startswith(expected_message)
    assert exc_info.value.code is None
    assert exc_info.value.stage is None
    assert exc_info.value.return_code == 1


def test_local_slim_client_keeps_raw_payloads(tmp_path: Path) -> None:
    payload = {"code": 0, "message": "raw", "data": {"kept": True}}
    client = _FakeSlimClient(
        config=SlimClientConfig(folder=tmp_path, mode="local"),
        events=[
            SlimChunkEvent(data=payload),
            SlimDoneEvent(),
        ],
    )

    chunks = list(client.invoke_chunks(plugin_id="plugin", action="action", data={}))

    assert chunks == [payload]


def test_slim_client_converts_dify_model_schema_parameter_templates(
    tmp_path: Path,
) -> None:
    client = _FakeSlimClient(
        config=SlimClientConfig(folder=tmp_path),
        events=[
            SlimChunkEvent(
                data={
                    "model_schema": {
                        "model": "fake-chat",
                        "label": {"en_US": "Fake Chat"},
                        "model_type": "llm",
                        "fetch_from": "predefined-model",
                        "model_properties": {
                            "mode": "chat",
                            "context_size": 8192,
                        },
                        "parameter_rules": [
                            {
                                "name": "temperature",
                                "use_template": "temperature",
                            }
                        ],
                    }
                },
            ),
            SlimDoneEvent(),
        ],
    )

    schema = client.get_ai_model_schema(
        plugin_id="author/fake:0.0.1@test",
        provider="fake-provider",
        model_type="llm",
        model="fake-chat",
        credentials={"api_key": "secret"},
    )

    assert schema is not None
    assert schema.model == "fake-chat"
    assert schema.model_type == ModelType.LLM
    assert schema.parameter_rules[0].name == "temperature"
    assert schema.parameter_rules[0].use_template == "temperature"
    assert schema.parameter_rules[0].label.en_us == "temperature"
    assert schema.parameter_rules[0].type == ParameterType.STRING


_NIL_TENANT_ID = "00000000-0000-0000-0000-000000000000"


def _remote_config(tmp_path: Path) -> SlimClientConfig:
    return SlimClientConfig(
        folder=tmp_path,
        mode="remote",
        daemon_addr="http://daemon.test/",
        daemon_key="secret",
    )


def _remote_client(
    tmp_path: Path,
    handler: Any,
) -> tuple[SlimClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = SlimClient(config=_remote_config(tmp_path), http_client=http_client)
    return client, http_client


@pytest.fixture
def no_local_slim(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        msg = "remote mode must not use the Slim binary"
        raise AssertionError(msg)

    monkeypatch.setattr(slim_client_module, "resolve_slim_binary_path", fail)
    monkeypatch.setattr(slim_client_module.subprocess, "Popen", fail)
    monkeypatch.setattr(slim_client_module.subprocess, "run", fail)


@pytest.mark.usefixtures("no_local_slim")
def test_remote_slim_client_dispatches_over_http(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    greeting = '{"code":0,"message":"","data":{"text":"你好"}}'.encode()
    split_at = greeting.index("你".encode()) + 1

    def body() -> Iterator[bytes]:
        yield b"data: " + greeting[:split_at]
        yield greeting[split_at:] + b"\n\n: keep-alive\n\n"
        yield b'data: {"code":0,"message":"","data":{"text":"!"}}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=body())

    client, http_client = _remote_client(tmp_path, handler)

    chunks = list(
        client.invoke_chunks(
            plugin_id="langgenius/openai:0.0.1@abc",
            action="invoke_llm",
            data={"provider": "openai", "model": "gpt-4o-mini"},
        )
    )

    assert chunks == [{"text": "你好"}, {"text": "!"}]
    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == "http://daemon.test/v2/invoke/dispatch/llm/invoke"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["X-Api-Key"] == "secret"
    assert request.headers["X-Plugin-Unique-Identifier"] == (
        "langgenius/openai:0.0.1@abc"
    )
    assert json.loads(request.content) == {
        "tenant_id": _NIL_TENANT_ID,
        "user_id": "",
        "data": {"provider": "openai", "model": "gpt-4o-mini"},
    }
    assert not http_client.is_closed


@pytest.mark.usefixtures("no_local_slim")
def test_remote_slim_client_yields_chunks_before_the_response_ends(
    tmp_path: Path,
) -> None:
    sent: list[str] = []

    def body() -> Iterator[bytes]:
        sent.append("first")
        yield b'data: {"code":0,"message":"","data":"first"}\n\n'
        sent.append("second")
        yield b'data: {"code":0,"message":"","data":"second"}\n\n'

    client, _ = _remote_client(
        tmp_path,
        lambda _request: httpx.Response(200, content=body()),
    )

    chunks = client.invoke_chunks(plugin_id="plugin", action="invoke_llm", data={})

    assert next(chunks) == "first"
    assert sent == ["first"]
    assert list(chunks) == ["second"]


@pytest.mark.usefixtures("no_local_slim")
def test_remote_slim_client_closes_the_response_when_the_reader_stops(
    tmp_path: Path,
) -> None:
    responses: list[httpx.Response] = []
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=iter([
                    b'data: {"code":0,"message":"","data":"first"}\n\n',
                    b'data: {"code":0,"message":"","data":"second"}\n\n',
                ]),
            )
        ),
        event_hooks={"response": [responses.append]},
    )
    client = SlimClient(config=_remote_config(tmp_path), http_client=http_client)
    chunks = client.invoke_chunks(plugin_id="plugin", action="invoke_llm", data={})

    assert next(chunks) == "first"
    (response,) = responses
    assert not response.is_closed
    chunks.close()

    assert response.is_closed
    assert not http_client.is_closed


@pytest.mark.usefixtures("no_local_slim")
@pytest.mark.parametrize(
    ("status", "body"),
    [(401, "invalid api key"), (202, "accepted")],
)
def test_remote_slim_client_reports_non_200_daemon_responses(
    tmp_path: Path,
    status: int,
    body: str,
) -> None:
    client, _ = _remote_client(
        tmp_path,
        lambda _request: httpx.Response(status, text=body),
    )

    with pytest.raises(SlimClientError) as exc_info:
        list(client.invoke_chunks(plugin_id="plugin", action="invoke_llm", data={}))

    assert str(status) in str(exc_info.value)
    assert body in str(exc_info.value)


def _refuse_connection(request: httpx.Request) -> httpx.Response:
    msg = "connection refused"
    raise httpx.ConnectError(msg, request=request)


@pytest.mark.usefixtures("no_local_slim")
@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(
            lambda _request: httpx.Response(200, content=b"data: {not json}\n\n"),
            id="malformed-event",
        ),
        pytest.param(_refuse_connection, id="connection-refused"),
    ],
)
def test_remote_slim_client_raises_client_error_when_dispatch_breaks(
    tmp_path: Path,
    handler: Any,
) -> None:
    client, _ = _remote_client(tmp_path, handler)

    with pytest.raises(SlimClientError):
        list(client.invoke_chunks(plugin_id="plugin", action="invoke_llm", data={}))


@pytest.mark.usefixtures("no_local_slim")
def test_remote_slim_client_rejects_unknown_actions_without_a_request(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    client, _ = _remote_client(tmp_path, handler)

    with pytest.raises(SlimClientError):
        list(client.invoke_chunks(plugin_id="plugin", action="not_an_action", data={}))

    assert requests == []


@pytest.mark.usefixtures("no_local_slim")
def test_remote_slim_extract_returns_daemon_manifest(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    manifest = {"tool": {"identity": {"name": "search"}, "tools": []}}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "message": "success",
                "data": {"unique_identifier": "plugin", "manifest": manifest},
            },
        )

    client, http_client = _remote_client(tmp_path, handler)

    payload = client.extract(plugin_id="plugin")

    assert payload["data"]["manifest"] == manifest
    assert not http_client.is_closed
    (request,) = requests
    assert request.method == "GET"
    assert str(request.url) == "http://daemon.test/v2/invoke/extract"
    assert request.headers["X-Api-Key"] == "secret"
    assert request.headers["X-Plugin-Unique-Identifier"] == "plugin"


@pytest.mark.usefixtures("no_local_slim")
@pytest.mark.parametrize(
    ("response", "message"),
    [
        pytest.param(
            httpx.Response(
                200,
                json={"code": -500, "message": "plugin not found", "data": None},
            ),
            "plugin not found",
            id="daemon-error-code",
        ),
        pytest.param(httpx.Response(500, text="boom"), "boom", id="http-500"),
    ],
)
def test_remote_slim_extract_raises_daemon_errors(
    tmp_path: Path,
    response: httpx.Response,
    message: str,
) -> None:
    client, _ = _remote_client(tmp_path, lambda _request: response)

    with pytest.raises(SlimClientError, match=message):
        client.extract(plugin_id="plugin")

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]
import sys
import tempfile
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any

import httpx

from graphon.model_runtime.entities.model_entities import AIModelEntity
from graphon.model_runtime.utils.encoders import jsonable_encoder

from .config import (
    SlimConfig,
    SlimLocalSettings,
    SlimProviderBinding,
)
from .package_loader import SlimPackageLoader

logger = logging.getLogger(__name__)

_SLIM_BINARY_NAME = "dify-plugin-daemon-slim"
_SLIM_BINARY_PATH_ENV = "SLIM_BINARY_PATH"

# Remote mode speaks the plugin daemon HTTP API directly, mirroring the Slim
# binary's remote mode (dify-plugin-daemon pkg/slim/remote.go).
_DAEMON_DISPATCH_PATH = "/v2/invoke/dispatch"
_DAEMON_EXTRACT_PATH = "/v2/invoke/extract"
_DAEMON_EVENT_PREFIX = "data: "
_NIL_TENANT_ID = "00000000-0000-0000-0000-000000000000"
_DAEMON_CONNECT_TIMEOUT_SECONDS = 10.0
_DAEMON_ACTION_PATHS: Mapping[str, str] = MappingProxyType({
    # tool
    "invoke_tool": "/tool/invoke",
    "validate_tool_credentials": "/tool/validate_credentials",
    "get_tool_runtime_parameters": "/tool/get_runtime_parameters",
    # model
    "invoke_llm": "/llm/invoke",
    "get_llm_num_tokens": "/llm/num_tokens",
    "invoke_text_embedding": "/text_embedding/invoke",
    "invoke_multimodal_embedding": "/multimodal_embedding/invoke",
    "get_text_embedding_num_tokens": "/text_embedding/num_tokens",
    "invoke_rerank": "/rerank/invoke",
    "invoke_multimodal_rerank": "/multimodal_rerank/invoke",
    "invoke_tts": "/tts/invoke",
    "get_tts_model_voices": "/tts/model/voices",
    "invoke_speech2text": "/speech2text/invoke",
    "invoke_moderation": "/moderation/invoke",
    "validate_provider_credentials": "/model/validate_provider_credentials",
    "validate_model_credentials": "/model/validate_model_credentials",
    "get_ai_model_schemas": "/model/schema",
    # agent strategy
    "invoke_agent_strategy": "/agent_strategy/invoke",
    # endpoint
    "invoke_endpoint": "/endpoint/invoke",
    # oauth
    "get_authorization_url": "/oauth/get_authorization_url",
    "get_credentials": "/oauth/get_credentials",
    "refresh_credentials": "/oauth/refresh_credentials",
    "revoke_credentials": "/oauth/revoke_credentials",
    # datasource
    "validate_datasource_credentials": "/datasource/validate_credentials",
    "invoke_website_datasource_get_crawl": "/datasource/get_website_crawl",
    "invoke_online_document_datasource_get_pages": (
        "/datasource/get_online_document_pages"
    ),
    "invoke_online_document_datasource_get_page_content": (
        "/datasource/get_online_document_page_content"
    ),
    "invoke_online_drive_browse_files": "/datasource/online_drive_browse_files",
    "invoke_online_drive_download_file": "/datasource/online_drive_download_file",
    # dynamic parameter
    "fetch_parameter_options": "/dynamic_select/fetch_parameter_options",
    # trigger
    "invoke_trigger_event": "/trigger/invoke_event",
    "dispatch_trigger_event": "/trigger/dispatch_event",
    "subscribe_trigger": "/trigger/subscribe",
    "unsubscribe_trigger": "/trigger/unsubscribe",
    "refresh_trigger": "/trigger/refresh",
    "validate_trigger_credentials": "/trigger/validate_credentials",
})


class SlimClientError(RuntimeError):
    """Raised when the DSL slim client cannot run or parse Slim output."""

    def __init__(
        self,
        message: str,
        *,
        code: str | int | None = None,
        stage: str | None = None,
        data: Mapping[str, Any] | None = None,
        return_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.stage = stage
        self.data = dict(data or {})
        self.return_code = return_code


@dataclass(frozen=True, slots=True)
class SlimMessageEvent:
    stage: str
    message: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SlimChunkEvent:
    data: Any


@dataclass(frozen=True, slots=True)
class SlimDoneEvent:
    pass


type SlimEvent = SlimMessageEvent | SlimChunkEvent | SlimDoneEvent


@dataclass(slots=True, frozen=True)
class SlimClientConfig:
    folder: Path
    mode: str = "local"
    daemon_addr: str = ""
    daemon_key: str = ""
    python_path: str = "python3"
    uv_path: str = ""
    python_env_init_timeout: int = 120
    max_execution_timeout: int = 600
    pip_mirror_url: str = ""
    pip_extra_args: str = ""
    marketplace_url: str = "https://marketplace.dify.ai"
    ignore_uv_lock: bool = False

    def __post_init__(self) -> None:
        python_path = self.python_path
        if python_path == "python3":
            python_path = sys.executable

        object.__setattr__(self, "folder", self.folder.expanduser().resolve())
        object.__setattr__(self, "mode", self.mode or "local")
        object.__setattr__(self, "python_path", python_path)
        object.__setattr__(self, "uv_path", self.uv_path or (shutil.which("uv") or ""))

    def build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["SLIM_MODE"] = self.mode
        if self.mode == "remote":
            env["SLIM_DAEMON_ADDR"] = self.daemon_addr
            env["SLIM_DAEMON_KEY"] = self.daemon_key
        else:
            env["SLIM_FOLDER"] = str(self.folder)
            env["SLIM_PYTHON_PATH"] = self.python_path
            env["SLIM_PYTHON_ENV_INIT_TIMEOUT"] = str(self.python_env_init_timeout)
            env["SLIM_MAX_EXECUTION_TIMEOUT"] = str(self.max_execution_timeout)
            env["SLIM_MARKETPLACE_URL"] = self.marketplace_url
            if self.uv_path:
                env["SLIM_UV_PATH"] = self.uv_path
            if self.pip_mirror_url:
                env["SLIM_PIP_MIRROR_URL"] = self.pip_mirror_url
            if self.pip_extra_args:
                env["SLIM_PIP_EXTRA_ARGS"] = self.pip_extra_args
            if self.ignore_uv_lock:
                env["SLIM_IGNORE_UV_LOCK"] = "true"

        return env


@dataclass(slots=True)
class SlimClient:
    config: SlimClientConfig
    binary_path: str | None = None
    http_client: httpx.Client | None = None
    _binary_path: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.mode == "remote":
            self._binary_path = self.binary_path or ""
            return
        self._binary_path = self.binary_path or resolve_slim_binary_path()

    def cached_plugin_root(self, plugin_id: str) -> Path | None:
        return cached_slim_plugin_root(config=self.config, plugin_id=plugin_id)

    def invoke_events(
        self,
        *,
        plugin_id: str,
        action: str,
        data: Mapping[str, Any],
    ) -> Generator[SlimEvent, None, None]:
        if self.config.mode == "remote":
            yield from self._invoke_daemon_events(
                plugin_id=plugin_id,
                action=action,
                data=data,
            )
            return
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]
                [
                    self._binary_path,
                    "-id",
                    plugin_id,
                    "-action",
                    action,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                env=self.config.build_env(),
            )
            request_payload = {"data": jsonable_encoder(dict(data))}
            if process.stdin is None:
                msg = "Slim subprocess did not expose stdin."
                raise SlimClientError(msg)
            process.stdin.write(json.dumps(request_payload))
            process.stdin.close()

            if process.stdout is None:
                msg = "Slim subprocess did not expose stdout."
                raise SlimClientError(msg)

            pending_error: Exception | None = None
            try:
                yield from _iter_slim_events(process.stdout)
            except Exception as error:
                pending_error = error
                raise
            finally:
                try:
                    _check_slim_process_exit(process=process, stderr_file=stderr_file)
                except SlimClientError:
                    if pending_error is None:
                        raise

    def invoke_chunks(
        self,
        *,
        plugin_id: str,
        action: str,
        data: Mapping[str, Any],
    ) -> Generator[Any, None, None]:
        for event in self.invoke_events(plugin_id=plugin_id, action=action, data=data):
            match event:
                case SlimMessageEvent():
                    logger.debug(
                        "slim[%s] %s: %s",
                        action,
                        event.stage,
                        event.message,
                    )
                case SlimChunkEvent():
                    yield (
                        _unwrap_remote_daemon_payload(event.data)
                        if self.config.mode == "remote"
                        else event.data
                    )
                case SlimDoneEvent():
                    return

    def invoke_unary(
        self,
        *,
        plugin_id: str,
        action: str,
        data: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        chunks = list(
            self.invoke_chunks(plugin_id=plugin_id, action=action, data=data),
        )
        if not chunks:
            return {}
        payload = chunks[-1]
        if not isinstance(payload, Mapping):
            msg = f"Expected dict payload for Slim action {action}, got {type(payload)}"
            raise SlimClientError(msg)
        return payload

    def get_ai_model_schema(
        self,
        *,
        plugin_id: str,
        provider: str,
        model_type: str,
        model: str,
        credentials: Mapping[str, Any],
    ) -> AIModelEntity | None:
        result = self.invoke_unary(
            plugin_id=plugin_id,
            action="get_ai_model_schemas",
            data={
                "provider": provider,
                "model_type": model_type,
                "model": model,
                "credentials": dict(credentials),
            },
        )
        raw_schema = result.get("model_schema")
        if raw_schema is None:
            return None
        if not isinstance(raw_schema, Mapping):
            msg = f"Unexpected model schema payload: {raw_schema!r}"
            raise SlimClientError(msg)
        return _convert_ai_model_entity(
            raw_schema,
            config=self.config,
            plugin_id=plugin_id,
            provider=provider,
        )

    def extract(
        self,
        *,
        plugin_id: str | None = None,
        path: str | Path | None = None,
        action: str | None = None,
    ) -> Mapping[str, Any]:
        if self.config.mode == "remote":
            return self._extract_from_daemon(
                plugin_id=plugin_id,
                path=path,
                action=action,
            )
        args = [self._binary_path, "extract", "-output", "json"]
        if plugin_id is not None:
            args.extend(("-id", plugin_id))
        if path is not None:
            args.extend(("-path", str(path)))
        if action is not None:
            args.extend(("-action", action))

        process = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
            args,
            text=True,
            encoding="utf-8",
            capture_output=True,
            env=self.config.build_env(),
            check=False,
        )
        if process.returncode != 0:
            raise _slim_process_error(process.stderr, process.returncode)

        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            msg = f"Slim extract returned invalid JSON: {error}"
            raise SlimClientError(msg) from error
        if not isinstance(payload, Mapping):
            msg = f"Slim extract returned {type(payload).__name__}, expected object."
            raise SlimClientError(msg)
        return payload

    def _invoke_daemon_events(
        self,
        *,
        plugin_id: str,
        action: str,
        data: Mapping[str, Any],
    ) -> Generator[SlimEvent, None, None]:
        route = _DAEMON_ACTION_PATHS.get(action)
        if route is None:
            msg = f"Unknown Slim action: {action}"
            raise SlimClientError(msg, code="UNKNOWN_ACTION")
        request_payload = {
            "tenant_id": _NIL_TENANT_ID,
            "user_id": "",
            "data": jsonable_encoder(dict(data)),
        }
        with self._daemon_http_client() as client:
            try:
                with client.stream(
                    "POST",
                    self._daemon_url(_DAEMON_DISPATCH_PATH + route),
                    json=request_payload,
                    headers=self._daemon_headers(plugin_id),
                ) as response:
                    yield from _read_daemon_events(response)
            except httpx.HTTPError as error:
                msg = f"Slim daemon request failed: {error}"
                raise SlimClientError(msg, code="NETWORK_ERROR") from error
        yield SlimDoneEvent()

    def _extract_from_daemon(
        self,
        *,
        plugin_id: str | None,
        path: str | Path | None,
        action: str | None,
    ) -> Mapping[str, Any]:
        if path is not None or action is not None or not plugin_id:
            msg = "Remote Slim extract supports only plugin_id."
            raise SlimClientError(msg, code="INVALID_INPUT")
        with self._daemon_http_client() as client:
            try:
                response = client.get(
                    self._daemon_url(_DAEMON_EXTRACT_PATH),
                    headers=self._daemon_headers(plugin_id),
                )
            except httpx.HTTPError as error:
                msg = f"Slim daemon request failed: {error}"
                raise SlimClientError(msg, code="NETWORK_ERROR") from error
        if response.status_code != httpx.codes.OK:
            raise _daemon_status_error(response)
        try:
            payload = response.json()
        except json.JSONDecodeError as error:
            msg = f"Slim daemon returned invalid extract JSON: {error}"
            raise SlimClientError(msg, code="STREAM_PARSE_ERROR") from error
        extract_data = _unwrap_remote_daemon_payload(payload)
        if not isinstance(extract_data, Mapping):
            msg = f"Slim daemon returned extract data {extract_data!r}."
            raise SlimClientError(msg, code="STREAM_PARSE_ERROR")
        return {"tenant_id": _NIL_TENANT_ID, "data": dict(extract_data)}

    @contextmanager
    def _daemon_http_client(self) -> Generator[httpx.Client]:
        # An injected client belongs to the caller; only close clients made here.
        if self.http_client is not None:
            yield self.http_client
            return
        timeout = httpx.Timeout(
            float(self.config.max_execution_timeout),
            connect=_DAEMON_CONNECT_TIMEOUT_SECONDS,
        )
        with httpx.Client(timeout=timeout) as client:
            yield client

    def _daemon_url(self, path: str) -> str:
        return self.config.daemon_addr.rstrip("/") + path

    def _daemon_headers(self, plugin_id: str) -> dict[str, str]:
        return {
            "X-Api-Key": self.config.daemon_key,
            "X-Plugin-Unique-Identifier": plugin_id,
        }


def resolve_slim_binary_path() -> str:
    configured_path = os.environ.get(_SLIM_BINARY_PATH_ENV, "").strip()
    if configured_path:
        binary_path = Path(configured_path).expanduser().resolve()
        if not binary_path.is_file():
            msg = f"{_SLIM_BINARY_PATH_ENV} points to a missing file: {binary_path}"
            raise SlimClientError(msg)
        if not os.access(binary_path, os.X_OK):
            msg = (
                f"{_SLIM_BINARY_PATH_ENV} points to a non-executable file: "
                f"{binary_path}"
            )
            raise SlimClientError(msg)
        return str(binary_path)

    binary_path = shutil.which(_SLIM_BINARY_NAME)
    if binary_path is None:
        msg = (
            f"{_SLIM_BINARY_NAME} is not available in PATH. "
            f"Set {_SLIM_BINARY_PATH_ENV} to override it."
        )
        raise SlimClientError(msg)
    return binary_path


def slim_plugin_cache_path(*, folder: Path, plugin_id: str) -> Path:
    return folder / plugin_id.replace(":", "-")


def cached_slim_plugin_root(
    *,
    config: SlimClientConfig,
    plugin_id: str,
) -> Path | None:
    candidate = slim_plugin_cache_path(folder=config.folder, plugin_id=plugin_id)
    return candidate if candidate.exists() else None


def _unwrap_remote_daemon_payload(payload: Any) -> Any:
    if not isinstance(payload, Mapping) or "code" not in payload:
        return payload

    code = payload.get("code")
    if code != 0:
        msg = str(payload.get("message") or "Slim daemon returned an error.")
        data = payload.get("data")
        stage = data.get("stage") if isinstance(data, Mapping) else payload.get("stage")
        raise SlimClientError(
            msg,
            code=code if isinstance(code, str | int) else None,
            stage=stage if isinstance(stage, str) else None,
            data=payload,
        )
    return payload.get("data")


def _convert_ai_model_entity(
    raw_schema: Mapping[str, Any],
    *,
    config: SlimClientConfig,
    plugin_id: str,
    provider: str,
) -> AIModelEntity:
    try:
        loader = SlimPackageLoader(
            SlimConfig(
                bindings=[
                    SlimProviderBinding(plugin_id=plugin_id, provider=provider),
                ],
                local=SlimLocalSettings(
                    folder=config.folder,
                    python_path=config.python_path,
                    uv_path=config.uv_path,
                    python_env_init_timeout=config.python_env_init_timeout,
                    max_execution_timeout=config.max_execution_timeout,
                    pip_mirror_url=config.pip_mirror_url,
                    pip_extra_args=config.pip_extra_args,
                    marketplace_url=config.marketplace_url,
                ),
            )
        )
        converted = loader.convert_model_entity(dict(raw_schema))
    except Exception as error:
        msg = f"Failed to convert Slim model schema: {error}"
        raise SlimClientError(msg) from error

    if converted is None:
        msg = f"Unsupported Slim model schema: {raw_schema!r}"
        raise SlimClientError(msg)
    return converted


def _read_daemon_events(
    response: httpx.Response,
) -> Generator[SlimChunkEvent, None, None]:
    if response.status_code != httpx.codes.OK:
        response.read()
        raise _daemon_status_error(response)
    for line in response.iter_lines():
        if line.startswith(_DAEMON_EVENT_PREFIX):
            yield SlimChunkEvent(
                data=_parse_daemon_event_data(line.removeprefix(_DAEMON_EVENT_PREFIX)),
            )


def _parse_daemon_event_data(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"Slim daemon emitted invalid JSON event: {text}"
        raise SlimClientError(msg, code="STREAM_PARSE_ERROR") from error


def _daemon_status_error(response: httpx.Response) -> SlimClientError:
    msg = f"Slim daemon returned HTTP {response.status_code}: {response.text}"
    return SlimClientError(msg, code="DAEMON_ERROR")


def _iter_slim_events(stdout: Iterable[str]) -> Generator[SlimEvent, None, None]:
    for line in stdout:
        if not line.strip():
            continue
        event = parse_slim_event(line)
        yield event
        if isinstance(event, SlimDoneEvent):
            return


def parse_slim_event(line: str) -> SlimEvent:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        msg = f"Slim emitted invalid JSON event: {line.strip()}"
        raise SlimClientError(msg) from error
    if not isinstance(event, Mapping):
        msg = f"Slim emitted {type(event).__name__}, expected event object."
        raise SlimClientError(msg)

    event_type = event.get("event")
    match event_type:
        case "message":
            return _parse_message_event(event)
        case "chunk":
            return SlimChunkEvent(data=event.get("data"))
        case "done":
            return SlimDoneEvent()
        case "error":
            error = event.get("data") or {}
            if isinstance(error, Mapping):
                message = str(error.get("message") or "Slim error.")
                code = error.get("code") or error.get("error_code")
                stage = error.get("stage")
                raise SlimClientError(
                    message,
                    code=code if isinstance(code, str | int) else None,
                    stage=stage if isinstance(stage, str) else None,
                    data=error,
                )
            message = str(error or "Slim error.")
            raise SlimClientError(message)
        case _:
            msg = f"Unknown Slim event type: {event_type}"
            raise SlimClientError(msg)


def _parse_message_event(event: Mapping[str, Any]) -> SlimMessageEvent:
    message = event.get("data") or {}
    if not isinstance(message, Mapping):
        msg = f"Unexpected Slim message payload: {message!r}"
        raise SlimClientError(msg)
    return SlimMessageEvent(
        stage=str(message.get("stage") or ""),
        message=str(message.get("message") or ""),
        data=message,
    )


def _check_slim_process_exit(
    *,
    process: subprocess.Popen[str],
    stderr_file: IO[str],
) -> None:
    return_code = process.wait()
    stderr_file.seek(0)
    stderr_text = stderr_file.read().strip()
    if return_code == 0:
        return
    raise _slim_process_error(stderr_text, return_code)


def _slim_process_error(stderr_text: str, return_code: int) -> SlimClientError:
    if not stderr_text:
        return SlimClientError(
            f"Slim process exited with code {return_code}",
            return_code=return_code,
        )
    try:
        stderr_payload = json.loads(stderr_text.splitlines()[-1])
    except json.JSONDecodeError:
        return SlimClientError(
            f"Slim process exited with code {return_code}: {stderr_text}",
            return_code=return_code,
        )
    if isinstance(stderr_payload, Mapping):
        message = str(
            stderr_payload.get("message")
            or f"Slim process exited with code {return_code}",
        )
        code = stderr_payload.get("code") or stderr_payload.get("error_code")
        stage = stderr_payload.get("stage")
        return SlimClientError(
            message,
            code=code if isinstance(code, str | int) else None,
            stage=stage if isinstance(stage, str) else None,
            data=stderr_payload,
            return_code=return_code,
        )
    return SlimClientError(
        f"Slim process exited with code {return_code}: {stderr_text}",
        return_code=return_code,
    )

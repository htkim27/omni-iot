from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from .mcp_config import McpConfigError, McpServerConfig, load_mcp_config

OPENAI_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]")
OPENAI_TOOL_NAME_MAX_LENGTH = 64


@dataclass(frozen=True)
class McpToolBinding:
    model_name: str
    server_name: str
    tool_name: str
    definition: dict[str, object]


@dataclass(frozen=True)
class McpCallResult:
    content: str
    is_error: bool
    server_name: str
    tool_name: str
    seconds: float
    truncated: bool = False

    def trace(self) -> dict[str, object]:
        return {
            "server": self.server_name,
            "tool": self.tool_name,
            "seconds": round(self.seconds, 3),
            "ok": not self.is_error,
            "truncated": self.truncated,
        }


@dataclass
class _ServerRuntime:
    config: McpServerConfig
    status: str = "pending"
    error: str | None = None
    client: Client | None = None
    stack: AsyncExitStack | None = None
    tools: list[Any] = field(default_factory=list)


class McpManager:
    """Own MCP client connections and expose an allowlisted OpenAI tool catalog."""

    def __init__(
        self,
        config_path: Path,
        catalog_max_chars: int = 12_000,
        result_max_chars: int = 12_000,
        brave_search_max_results: int = 5,
    ) -> None:
        self.config_path = config_path
        self.catalog_max_chars = catalog_max_chars
        self.result_max_chars = result_max_chars
        self.brave_search_max_results = brave_search_max_results
        self._servers: dict[str, _ServerRuntime] = {}
        self._bindings: dict[str, McpToolBinding] = {}
        self._reverse_bindings: dict[tuple[str, str], str] = {}
        self._catalog: list[dict[str, object]] = []
        self._config_error: str | None = None
        self._catalog_error: str | None = None
        self._started = False
        self._lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._servers) or self.config_path.exists()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._config_error = None
        self._catalog_error = None
        self._servers = {}
        self._bindings = {}
        self._reverse_bindings = {}
        self._catalog = []
        try:
            config = load_mcp_config(self.config_path)
        except McpConfigError as exc:
            self._config_error = str(exc)
            return

        self._servers = {
            name: _ServerRuntime(
                config=server,
                status="disabled" if not server.enabled else "pending",
            )
            for name, server in config.servers.items()
        }
        for runtime in self._servers.values():
            if runtime.config.enabled:
                await self._connect(runtime)
        self._rebuild_catalog()

    async def stop(self) -> None:
        for runtime in reversed(list(self._servers.values())):
            await self._disconnect(runtime)
        self._started = False

    async def prepare_turn(self) -> list[dict[str, object]]:
        if not self._started:
            await self.start()
        async with self._lock:
            changed = False
            for runtime in self._servers.values():
                if runtime.config.enabled and runtime.status == "unhealthy":
                    await self._connect(runtime)
                    changed = True
            if changed:
                self._rebuild_catalog()
            return list(self._catalog)

    def openai_tools(self) -> list[dict[str, object]]:
        return list(self._catalog)

    def model_name_for(self, server_name: str, tool_name: str) -> str | None:
        return self._reverse_bindings.get((server_name, tool_name))

    async def call_tool(
        self,
        model_name: str,
        arguments: dict[str, object],
    ) -> McpCallResult:
        started_at = time.perf_counter()
        binding = self._bindings.get(model_name)
        if binding is None:
            return McpCallResult(
                content=json.dumps(
                    {"isError": True, "error": "Tool is not registered or allowed."},
                    ensure_ascii=False,
                ),
                is_error=True,
                server_name="unknown",
                tool_name=model_name,
                seconds=time.perf_counter() - started_at,
            )
        runtime = self._servers[binding.server_name]
        if runtime.client is None:
            await self._connect(runtime)
        if runtime.client is None:
            return self._error_result(binding, "MCP server is unavailable.", started_at)

        try:
            bounded_arguments = _project_tool_arguments(
                binding.server_name,
                binding.tool_name,
                arguments,
                self.brave_search_max_results,
            )
            result = await runtime.client.call_tool(
                binding.tool_name,
                bounded_arguments,
                read_timeout_seconds=runtime.config.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - transport errors become model-visible results
            await self._disconnect(runtime)
            runtime.status = "unhealthy"
            runtime.error = f"Tool call failed: {type(exc).__name__}."
            return self._error_result(binding, runtime.error, started_at)

        content = _tool_result_text(result)
        truncated = len(content) > self.result_max_chars
        content = _truncate_result(content, self.result_max_chars)
        return McpCallResult(
            content=content,
            is_error=bool(getattr(result, "is_error", False)),
            server_name=binding.server_name,
            tool_name=binding.tool_name,
            seconds=time.perf_counter() - started_at,
            truncated=truncated,
        )

    def health(self) -> dict[str, object]:
        servers = []
        for name, runtime in sorted(self._servers.items()):
            exposed = sum(
                binding.server_name == name for binding in self._bindings.values()
            )
            servers.append(
                {
                    "name": name,
                    "transport": runtime.config.transport,
                    "status": runtime.status,
                    "tool_count": exposed,
                    "error": runtime.error,
                }
            )
        return {
            "enabled": bool(self._servers),
            "configured": self.configured,
            "config": str(self.config_path),
            "ok": self._config_error is None
            and self._catalog_error is None
            and all(item["status"] in {"healthy", "disabled"} for item in servers),
            "error": self._config_error or self._catalog_error,
            "servers": servers,
        }

    async def _connect(self, runtime: _ServerRuntime) -> None:
        await self._disconnect(runtime)
        config = runtime.config
        stack = AsyncExitStack()
        try:
            if config.command:
                executable = _resolve_executable(config.command)
                if config.cwd and not config.cwd.is_dir():
                    raise McpConfigError(
                        f"Working directory for MCP server {config.name} does not exist."
                    )
                transport = stdio_client(
                    StdioServerParameters(
                        command=executable,
                        args=list(config.args),
                        env=_child_environment(config),
                        cwd=config.cwd,
                    )
                )
            else:
                http_client = httpx2.AsyncClient(
                    headers=config.resolved_headers(),
                    follow_redirects=True,
                    timeout=httpx2.Timeout(
                        connect=30,
                        write=30,
                        pool=30,
                        read=config.timeout_seconds,
                    ),
                )
                await stack.enter_async_context(http_client)
                transport = streamable_http_client(
                    config.url or "",
                    http_client=http_client,
                )

            client = Client(
                transport,
                read_timeout_seconds=config.timeout_seconds,
            )
            async with asyncio.timeout(config.timeout_seconds):
                await stack.enter_async_context(client)
                tools = await _list_all_tools(client)
        except Exception as exc:  # noqa: BLE001 - isolate individual server failures
            try:
                await stack.aclose()
            except Exception:
                pass
            runtime.status = "unhealthy"
            runtime.error = _safe_connection_error(config, exc)
            runtime.client = None
            runtime.stack = None
            runtime.tools = []
            return

        runtime.client = client
        runtime.stack = stack
        runtime.tools = tools
        runtime.status = "healthy"
        runtime.error = None

    async def _disconnect(self, runtime: _ServerRuntime) -> None:
        stack = runtime.stack
        runtime.client = None
        runtime.stack = None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                pass

    def _rebuild_catalog(self) -> None:
        bindings: dict[str, McpToolBinding] = {}
        reverse_bindings: dict[tuple[str, str], str] = {}
        catalog: list[dict[str, object]] = []
        for server_name, runtime in sorted(self._servers.items()):
            if runtime.status != "healthy":
                continue
            for tool in sorted(runtime.tools, key=lambda item: item.name):
                if not runtime.config.allows(tool.name):
                    continue
                model_name = _model_tool_name(server_name, tool.name)
                if model_name in bindings:
                    runtime.status = "unhealthy"
                    runtime.error = f"Tool name collision: {model_name}."
                    self._catalog_error = runtime.error
                    self._bindings = {}
                    self._reverse_bindings = {}
                    self._catalog = []
                    return
                definition: dict[str, object] = {
                    "type": "function",
                    "function": {
                        "name": model_name,
                        "description": _project_tool_description(
                            server_name,
                            tool.name,
                            tool.description or tool.name,
                            self.brave_search_max_results,
                        ),
                        "parameters": _project_tool_schema(
                            server_name,
                            tool.name,
                            tool.input_schema,
                            self.brave_search_max_results,
                        ),
                    },
                }
                binding = McpToolBinding(
                    model_name=model_name,
                    server_name=server_name,
                    tool_name=tool.name,
                    definition=definition,
                )
                bindings[model_name] = binding
                reverse_bindings[(server_name, tool.name)] = model_name
                catalog.append(definition)

        size = len(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")))
        if size > self.catalog_max_chars:
            self._catalog_error = (
                f"Allowed MCP tool catalog is {size} characters; "
                f"limit is {self.catalog_max_chars}. Reduce allowedTools."
            )
            self._bindings = {}
            self._reverse_bindings = {}
            self._catalog = []
            return
        self._catalog_error = None
        self._bindings = bindings
        self._reverse_bindings = reverse_bindings
        self._catalog = catalog

    @staticmethod
    def _error_result(
        binding: McpToolBinding,
        message: str,
        started_at: float,
    ) -> McpCallResult:
        return McpCallResult(
            content=json.dumps(
                {"isError": True, "error": message},
                ensure_ascii=False,
            ),
            is_error=True,
            server_name=binding.server_name,
            tool_name=binding.tool_name,
            seconds=time.perf_counter() - started_at,
        )


async def _list_all_tools(client: Client) -> list[Any]:
    tools: list[Any] = []
    cursor: str | None = None
    while True:
        result = await client.list_tools(cursor=cursor)
        tools.extend(result.tools)
        cursor = result.next_cursor
        if cursor is None:
            return tools


def _resolve_executable(command: str) -> str:
    path = Path(command).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        absolute = path.absolute()
        if not absolute.is_file() or not os.access(absolute, os.X_OK):
            raise McpConfigError(
                f"MCP executable does not exist or is not executable: {absolute}."
            )
        return str(absolute)
    resolved_command = shutil.which(command)
    if resolved_command is None:
        raise McpConfigError(f"MCP executable is not on PATH: {command}.")
    return resolved_command


def _project_tool_arguments(
    server_name: str,
    tool_name: str,
    arguments: dict[str, object],
    maximum: int,
) -> dict[str, object]:
    if server_name == "brave-search" and tool_name in {
        "brave_web_search",
        "brave_news_search",
    }:
        bounded = {
            key: value
            for key, value in arguments.items()
            if key in {"query", "count", "freshness", "country", "search_lang"}
        }
        requested = bounded.get("count")
        if requested is None:
            bounded["count"] = maximum
        elif isinstance(requested, int) and not isinstance(requested, bool):
            bounded["count"] = min(requested, maximum)
        bounded["extra_snippets"] = False
        if tool_name == "brave_web_search":
            bounded["summary"] = False
        return bounded
    if server_name == "switchbot" and tool_name == "list_devices":
        return {}
    if server_name == "switchbot" and tool_name == "get_device_status":
        return {
            key: value for key, value in arguments.items() if key == "deviceId"
        }
    if server_name == "switchbot" and tool_name == "send_command":
        bounded = {
            key: value
            for key, value in arguments.items()
            if key in {"deviceId", "command", "parameter"}
        }
        bounded.update({"commandType": "command", "confirm": False})
        return bounded
    return arguments


def _project_tool_description(
    server_name: str,
    tool_name: str,
    original: str,
    maximum: int,
) -> str:
    if server_name == "brave-search" and tool_name == "brave_web_search":
        return (
            "Search the current web and return at most "
            f"{maximum} compact title, URL, and snippet results. "
            "If evidence is insufficient, use a different focused query."
        )
    if server_name == "brave-search" and tool_name == "brave_news_search":
        return (
            "Search recent news and return at most "
            f"{maximum} compact source, title, URL, and snippet results. "
            "If evidence is insufficient, use a different focused query."
        )
    if server_name == "switchbot" and tool_name == "list_devices":
        return (
            "List SwitchBot devices and IR remotes with names, types, IDs, and "
            "command capabilities. Use before device status or control calls."
        )
    if server_name == "switchbot" and tool_name == "get_device_status":
        return (
            "Get the current status of a physical SwitchBot device. IR remotes "
            "do not expose status."
        )
    if server_name == "switchbot" and tool_name == "send_command":
        return (
            "Send a safe command to a SwitchBot device. Use list_devices first; "
            "server-side validation and safety policy still apply."
        )
    return original


def _project_tool_schema(
    server_name: str,
    tool_name: str,
    input_schema: dict[str, object],
    maximum: int,
) -> dict[str, object]:
    if server_name == "brave-search" and tool_name in {
        "brave_web_search",
        "brave_news_search",
    }:
        return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Focused search query (max 400 characters).",
                "maxLength": 400,
            },
            "count": {
                "type": "integer",
                "description": "Number of compact results.",
                "minimum": 1,
                "maximum": maximum,
                "default": maximum,
            },
            "freshness": {
                "type": "string",
                "description": "Optional pd, pw, pm, py, or date range filter.",
            },
            "country": {
                "type": "string",
                "description": "Optional two-letter country code such as KR.",
            },
            "search_lang": {
                "type": "string",
                "description": "Optional search language code such as ko.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
        }
    if server_name == "switchbot" and tool_name == "list_devices":
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    if server_name == "switchbot" and tool_name == "get_device_status":
        return {
            "type": "object",
            "properties": {
                "deviceId": {
                    "type": "string",
                    "description": "Device ID returned by list_devices.",
                }
            },
            "required": ["deviceId"],
            "additionalProperties": False,
        }
    if server_name == "switchbot" and tool_name == "send_command":
        return {
            "type": "object",
            "properties": {
                "deviceId": {
                    "type": "string",
                    "description": "Device ID returned by list_devices.",
                },
                "command": {
                    "type": "string",
                    "description": "Case-sensitive device command such as turnOn.",
                },
                "parameter": {
                    "description": "Optional device-specific command parameter.",
                },
            },
            "required": ["deviceId", "command"],
            "additionalProperties": False,
        }
    return input_schema


def _model_tool_name(server_name: str, tool_name: str) -> str:
    raw = OPENAI_TOOL_NAME_PATTERN.sub("_", f"{server_name}__{tool_name}")
    if len(raw) <= OPENAI_TOOL_NAME_MAX_LENGTH:
        return raw
    digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
    prefix_length = OPENAI_TOOL_NAME_MAX_LENGTH - len(digest) - 2
    return f"{raw[:prefix_length]}__{digest}"


def _tool_result_text(result: Any) -> str:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        payload = {
            "isError": bool(getattr(result, "is_error", False)),
            "structuredContent": structured,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    text_parts: list[str] = []
    other_parts: list[object] = []
    for block in getattr(result, "content", []):
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", ""))
        elif hasattr(block, "model_dump"):
            dumped = block.model_dump(by_alias=True, exclude_none=True)
            if isinstance(dumped, dict):
                dumped.pop("data", None)
                dumped.pop("blob", None)
            other_parts.append(dumped)
        else:
            other_parts.append({"type": getattr(block, "type", "unsupported")})
    payload: dict[str, object] = {
        "isError": bool(getattr(result, "is_error", False)),
        "content": "\n".join(text_parts),
    }
    if other_parts:
        payload["otherContent"] = other_parts
    return json.dumps(payload, ensure_ascii=False, default=str)


def _truncate_result(content: str, maximum: int) -> str:
    if len(content) <= maximum:
        return content
    envelope = {
        "truncated": True,
        "originalCharacters": len(content),
        "content": "",
    }
    empty_size = len(json.dumps(envelope, ensure_ascii=False))
    envelope["content"] = content[: max(0, maximum - empty_size - 8)]
    serialized = json.dumps(envelope, ensure_ascii=False)
    return serialized[:maximum]


def _child_environment(config: McpServerConfig) -> dict[str, str] | None:
    configured = config.resolved_env()
    if configured is None:
        return None
    environment = dict(os.environ)
    environment.update(configured)
    return environment


def _safe_connection_error(config: McpServerConfig, exc: Exception) -> str:
    if isinstance(exc, McpConfigError):
        return str(exc)
    return f"Could not connect to MCP server {config.name}: {type(exc).__name__}."

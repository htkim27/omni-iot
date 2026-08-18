from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SERVER_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
ENV_REFERENCE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class McpConfigError(ValueError):
    """A safe-to-display MCP configuration error."""


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    enabled: bool
    allowed_tools: tuple[str, ...]
    timeout_seconds: float
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: Path | None = None
    url: str | None = None
    headers: dict[str, str] | None = None

    @property
    def transport(self) -> str:
        return "stdio" if self.command else "streamable-http"

    def allows(self, tool_name: str) -> bool:
        return "*" in self.allowed_tools or tool_name in self.allowed_tools

    def resolved_env(self) -> dict[str, str] | None:
        if self.env is None:
            return None
        return {
            key: expand_env(value, f"server {self.name} env {key}")
            for key, value in self.env.items()
        }

    def resolved_headers(self) -> dict[str, str] | None:
        if self.headers is None:
            return None
        return {
            key: expand_env(value, f"server {self.name} header {key}")
            for key, value in self.headers.items()
        }

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "enabled": self.enabled,
            "allowedTools": list(self.allowed_tools),
            "timeoutSeconds": self.timeout_seconds,
        }
        if self.command:
            payload["command"] = self.command
            payload["args"] = list(self.args)
            if self.env:
                payload["env"] = dict(self.env)
            if self.cwd:
                payload["cwd"] = str(self.cwd)
        else:
            payload["url"] = self.url or ""
            if self.headers:
                payload["headers"] = dict(self.headers)
        return payload


@dataclass(frozen=True)
class McpConfig:
    servers: dict[str, McpServerConfig]
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mcpServers": {
                name: server.to_dict()
                for name, server in sorted(self.servers.items())
            },
        }


def default_config() -> McpConfig:
    return McpConfig(servers={})


def load_mcp_config(path: Path) -> McpConfig:
    if not path.exists():
        return default_config()
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError, McpConfigError) as exc:
        raise McpConfigError(f"Invalid MCP config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise McpConfigError("MCP config root must be a JSON object.")
    if payload.get("version") != 1:
        raise McpConfigError("MCP config version must be 1.")
    raw_servers = payload.get("mcpServers")
    if not isinstance(raw_servers, dict):
        raise McpConfigError("mcpServers must be a JSON object.")
    servers = {
        name: _parse_server(name, raw)
        for name, raw in raw_servers.items()
    }
    return McpConfig(servers=servers)


def save_mcp_config(path: Path, config: McpConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n"
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.chmod(temp_path, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def expand_env(value: str, context: str) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.getenv(name)
        if resolved is None:
            missing.append(name)
            return ""
        return resolved

    expanded = ENV_REFERENCE_PATTERN.sub(replace, value)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise McpConfigError(f"Missing environment variable(s) for {context}: {names}.")
    if "${" in expanded:
        raise McpConfigError(f"Invalid environment reference in {context}.")
    return expanded


def _parse_server(name: object, raw: object) -> McpServerConfig:
    if not isinstance(name, str) or not SERVER_ID_PATTERN.fullmatch(name):
        raise McpConfigError(
            f"Invalid MCP server id {name!r}; use letters, numbers, '_' or '-'."
        )
    if not isinstance(raw, dict):
        raise McpConfigError(f"MCP server {name} must be a JSON object.")
    command = _optional_string(raw.get("command"), f"server {name} command")
    url = _optional_string(raw.get("url"), f"server {name} url")
    if bool(command) == bool(url):
        raise McpConfigError(
            f"MCP server {name} must define exactly one of command or url."
        )
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise McpConfigError(f"MCP server {name} enabled must be boolean.")
    allowed = raw.get("allowedTools")
    if not isinstance(allowed, list) or not all(
        isinstance(item, str) and item for item in allowed
    ):
        raise McpConfigError(
            f"MCP server {name} allowedTools must be an array of tool names."
        )
    if "*" in allowed and len(allowed) != 1:
        raise McpConfigError(
            f"MCP server {name} allowedTools '*' must be used alone."
        )
    timeout = raw.get("timeoutSeconds", 30)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or timeout <= 0
    ):
        raise McpConfigError(f"MCP server {name} timeoutSeconds must be positive.")

    args = _string_list(raw.get("args", []), f"server {name} args")
    env = _string_map(raw.get("env"), f"server {name} env")
    headers = _string_map(raw.get("headers"), f"server {name} headers")
    cwd_text = _optional_string(raw.get("cwd"), f"server {name} cwd")

    if command:
        if headers is not None:
            raise McpConfigError(f"MCP stdio server {name} cannot define headers.")
        cwd = Path(cwd_text).expanduser().resolve() if cwd_text else None
        return McpServerConfig(
            name=name,
            enabled=enabled,
            allowed_tools=tuple(dict.fromkeys(allowed)),
            timeout_seconds=float(timeout),
            command=command,
            args=tuple(args),
            env=env,
            cwd=cwd,
        )

    if args or env is not None or cwd_text is not None:
        raise McpConfigError(
            f"MCP HTTP server {name} cannot define args, env, or cwd."
        )
    parsed_url = urlparse(url or "")
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise McpConfigError(f"MCP server {name} url must be HTTP or HTTPS.")
    return McpServerConfig(
        name=name,
        enabled=enabled,
        allowed_tools=tuple(dict.fromkeys(allowed)),
        timeout_seconds=float(timeout),
        url=url,
        headers=headers,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise McpConfigError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise McpConfigError(f"{context} must be a non-empty string.")
    return value


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise McpConfigError(f"{context} must be an array of strings.")
    return value


def _string_map(value: object, context: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise McpConfigError(f"{context} must be an object of string values.")
    return dict(value)

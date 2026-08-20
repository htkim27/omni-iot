from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .config import PROJECT_ROOT, Settings
from .mcp_client import McpManager
from .mcp_config import (
    McpConfig,
    McpConfigError,
    McpServerConfig,
    load_mcp_config,
    save_mcp_config,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / ".mcp.json"


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    path = args.config.expanduser().resolve()
    try:
        exit_code = _run(args, path)
    except (McpConfigError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(exit_code)


def _run(args: argparse.Namespace, path: Path) -> int:
    if args.action == "doctor":
        return asyncio.run(_doctor(path, args.json))

    config = load_mcp_config(path)
    servers = dict(config.servers)
    if args.action == "add-stdio":
        _ensure_new_server(servers, args.name)
        executable = _resolve_cli_executable(args.command)
        cwd = args.cwd.expanduser().resolve() if args.cwd else None
        if cwd is not None and not cwd.is_dir():
            raise McpConfigError(f"MCP working directory does not exist: {cwd}.")
        servers[args.name] = McpServerConfig(
            name=args.name,
            enabled=not args.disabled,
            allowed_tools=tuple(dict.fromkeys(args.allow_tool)),
            timeout_seconds=args.timeout,
            command=executable,
            args=tuple(args.arg),
            env=_key_values(args.env, "env"),
            cwd=cwd,
        )
        _save(path, config, servers)
        print(f"Registered stdio MCP server {args.name}. Restart omni-iot to load it.")
        return 0
    if args.action == "add-http":
        _ensure_new_server(servers, args.name)
        servers[args.name] = McpServerConfig(
            name=args.name,
            enabled=not args.disabled,
            allowed_tools=tuple(dict.fromkeys(args.allow_tool)),
            timeout_seconds=args.timeout,
            url=args.url,
            headers=_key_values(args.header, "header"),
        )
        # Round-trip through the parser to validate URL and transport fields.
        candidate = McpConfig(servers=servers)
        _validate_candidate(path, candidate)
        save_mcp_config(path, candidate)
        print(f"Registered HTTP MCP server {args.name}. Restart omni-iot to load it.")
        return 0
    if args.action == "remove":
        _require_server(servers, args.name)
        del servers[args.name]
        _save(path, config, servers)
        print(f"Removed MCP server {args.name}.")
        return 0
    if args.action in {"enable", "disable"}:
        server = _require_server(servers, args.name)
        servers[args.name] = replace(server, enabled=args.action == "enable")
        _save(path, config, servers)
        print(f"{args.action.title()}d MCP server {args.name}. Restart omni-iot to apply.")
        return 0
    if args.action in {"allow", "disallow"}:
        server = _require_server(servers, args.name)
        current = list(server.allowed_tools)
        if args.action == "allow":
            if "*" in args.tools:
                current = ["*"]
            elif "*" in current:
                current = ["*"]
            else:
                current.extend(args.tools)
                current = list(dict.fromkeys(current))
        else:
            if "*" in current and "*" not in args.tools:
                raise McpConfigError(
                    "Cannot disallow individual tools while '*' is active; "
                    "disallow '*' and add an exact allowlist."
                )
            current = [tool for tool in current if tool not in set(args.tools)]
        servers[args.name] = replace(server, allowed_tools=tuple(current))
        _save(path, config, servers)
        print(f"Updated allowed tools for {args.name}. Restart omni-iot to apply.")
        return 0
    if args.action == "list":
        payload = [
            {
                "name": name,
                "transport": server.transport,
                "enabled": server.enabled,
                "allowedTools": list(server.allowed_tools),
            }
            for name, server in sorted(servers.items())
        ]
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif not payload:
            print("No MCP servers are registered.")
        else:
            for item in payload:
                state = "enabled" if item["enabled"] else "disabled"
                tools = ",".join(item["allowedTools"]) or "(none)"
                print(f"{item['name']}\t{item['transport']}\t{state}\t{tools}")
        return 0
    raise McpConfigError("Unknown MCP CLI action.")


async def _doctor(path: Path, as_json: bool) -> int:
    settings = Settings()
    manager = McpManager(
        path,
        catalog_max_chars=settings.mcp_tool_catalog_max_chars,
        result_max_chars=settings.mcp_tool_result_max_chars,
        brave_search_max_results=settings.mcp_brave_search_max_results,
    )
    await manager.start()
    try:
        health = manager.health()
    finally:
        await manager.stop()
    if as_json:
        print(json.dumps(health, ensure_ascii=False, indent=2))
    else:
        print(f"config: {path}")
        if health["error"]:
            print(f"error: {health['error']}")
        for server in health["servers"]:
            line = (
                f"{server['name']}\t{server['transport']}\t{server['status']}"
                f"\ttools={server['tool_count']}"
            )
            if server["error"]:
                line += f"\t{server['error']}"
            print(line)
    return 0 if health["ok"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Omni-IoT MCP servers.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="action", required=True)

    stdio = subparsers.add_parser("add-stdio")
    stdio.add_argument("name")
    stdio.add_argument("--command", required=True)
    stdio.add_argument("--arg", action="append", default=[])
    stdio.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    stdio.add_argument("--cwd", type=Path)
    _add_registration_options(stdio)

    http = subparsers.add_parser("add-http")
    http.add_argument("name")
    http.add_argument("--url", required=True)
    http.add_argument("--header", action="append", default=[], metavar="KEY=VALUE")
    _add_registration_options(http)

    for action in ("remove", "enable", "disable"):
        command = subparsers.add_parser(action)
        command.add_argument("name")
    for action in ("allow", "disallow"):
        command = subparsers.add_parser(action)
        command.add_argument("name")
        command.add_argument("tools", nargs="+")
    listing = subparsers.add_parser("list")
    listing.add_argument("--json", action="store_true")
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")
    return parser


def _add_registration_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-tool", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--disabled", action="store_true")


def _save(path: Path, original: McpConfig, servers: dict[str, McpServerConfig]) -> None:
    candidate = McpConfig(version=original.version, servers=servers)
    _validate_candidate(path, candidate)
    save_mcp_config(path, candidate)


def _validate_candidate(path: Path, config: McpConfig) -> None:
    # Validation uses the same parser as runtime without touching the target file.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mcp-validate-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        save_mcp_config(temporary, config)
        load_mcp_config(temporary)


def _resolve_cli_executable(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        absolute = candidate.absolute()
        if not absolute.is_file() or not os.access(absolute, os.X_OK):
            raise McpConfigError(
                f"MCP executable does not exist or is not executable: {absolute}."
            )
        return str(absolute)
    resolved_command = shutil.which(command)
    if resolved_command is None:
        raise McpConfigError(f"MCP executable is not on PATH: {command}.")
    return str(Path(resolved_command).absolute())


def _key_values(values: list[str], label: str) -> dict[str, str] | None:
    if not values:
        return None
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise McpConfigError(f"Invalid {label} value; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            raise McpConfigError(f"Invalid {label} value; key must not be empty.")
        if key in result:
            raise McpConfigError(f"Duplicate {label} key: {key}.")
        result[key] = value
    return result


def _ensure_new_server(
    servers: dict[str, McpServerConfig],
    name: str,
) -> None:
    if name in servers:
        raise McpConfigError(f"MCP server already exists: {name}.")


def _require_server(
    servers: dict[str, McpServerConfig],
    name: str,
) -> McpServerConfig:
    try:
        return servers[name]
    except KeyError as exc:
        raise McpConfigError(f"Unknown MCP server: {name}.") from exc


if __name__ == "__main__":
    main()

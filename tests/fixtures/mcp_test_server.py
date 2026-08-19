from __future__ import annotations

import argparse
import asyncio

from mcp.server.mcpserver import MCPServer


server = MCPServer("omni-iot-test")


@server.tool(structured_output=True)
def add(left: int, right: int) -> dict[str, int]:
    """Add two integers."""
    return {"sum": left + right}


@server.tool(structured_output=False)
def echo(text: str) -> str:
    """Return the supplied text."""
    return text


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.http:
        await server.run_streamable_http_async(
            host="127.0.0.1",
            port=args.port,
            json_response=True,
        )
    else:
        await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env bash

set -euo pipefail

# Fail closed if the compact SwitchBot/Brave catalog exceeds its context budget.
export MCP_TOOL_CATALOG_MAX_CHARS="${MCP_TOOL_CATALOG_MAX_CHARS:-12000}"

mcp_server_registered() {
  local server_name="$1"
  uv run omni-iot-mcp list | awk -F '\t' -v expected="$server_name" '
    $1 == expected { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

if ! mcp_server_registered switchbot; then
  switchbot_command="$(command -v switchbot)"
  uv run omni-iot-mcp add-stdio switchbot \
    --command "$switchbot_command" \
    --arg mcp --arg serve \
    --allow-tool list_devices \
    --allow-tool get_device_status \
    --allow-tool send_command
fi

if ! uv run python -c '
import os
from omni_iot import config  # noqa: F401 - importing loads .env
raise SystemExit(0 if os.getenv("BRAVE_API_KEY") else 1)
'; then
  echo "error: BRAVE_API_KEY is missing or empty in .env" >&2
  exit 2
fi

if ! mcp_server_registered brave-search; then
  npx_command="$(command -v npx)"
  uv run omni-iot-mcp add-stdio brave-search \
    --command "$npx_command" \
    --arg=-y \
    --arg=@brave/brave-search-mcp-server@2.1.0 \
    --env 'BRAVE_API_KEY=${BRAVE_API_KEY}' \
    --allow-tool brave_web_search \
    --allow-tool brave_news_search
fi

uv run omni-iot-mcp enable switchbot
uv run omni-iot-mcp enable brave-search
uv run omni-iot-mcp doctor --json
exec uv run omni-iot

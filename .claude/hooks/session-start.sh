#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Static HTML/CSS site — no dependencies to install.
# Verify Python is available for local serving.
python3 --version
echo "Environment ready. Run ./serve.sh to start the local server."

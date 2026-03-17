#!/bin/bash
set -euo pipefail

PORT=${PORT:-8080}
echo "Serving Nordic-nexus (olluri) on http://localhost:$PORT"
cd olluri
python3 -m http.server "$PORT"

#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?model path or name}"
if [[ -f "$MODEL" ]]; then
  echo "$MODEL"; exit 0
fi
if [[ -f "models/$MODEL" ]]; then
  echo "models/$MODEL"; exit 0
fi
echo "Error: model not found: $MODEL" >&2
exit 2

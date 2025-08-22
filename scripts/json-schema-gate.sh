#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <one-or-more-json-files>" >&2
  exit 2
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1" >&2; exit 2; }; }
need jq

ok=1

for f in "$@"; do
  if ! [[ -s "$f" ]]; then
    echo "✗ $f: file missing or empty" >&2
    ok=0; continue
  fi

  if ! jq -e '.schema_version == "1"' "$f" >/dev/null; then
    echo "✗ $f: schema_version != \"1\"" >&2
    ok=0; continue
  fi

  t=$(jq -r '.type // empty' "$f")

  case "$t" in
    run)
      if ! jq -e '.gen_policy.bos != null' "$f" >/dev/null; then
        echo "✗ $f(run): .gen_policy.bos missing" >&2; ok=0
      fi
      if ! jq -e '(.throughput.tokens_per_second|numbers)' "$f" >/dev/null; then
        echo "✗ $f(run): .throughput.tokens_per_second not numeric" >&2; ok=0
      fi
      ;;
    score)
      if ! jq -e '(.mean_nll|numbers) and (.ppl|numbers)' "$f" >/dev/null; then
        echo "✗ $f(score): mean_nll/ppl not numeric" >&2; ok=0
      fi
      ;;
    tokenize)
      if ! jq -e '(.tokens.ids|type) == "array"' "$f" >/dev/null; then
        echo "✗ $f(tokenize): .tokens.ids not an array" >&2; ok=0
      fi
      ;;
    *)
      echo "⚠︎ $f: unknown or missing .type ($t). Skipping type-specific checks." >&2
      ;;
  esac
done

[[ $ok -eq 1 ]] && echo "✓ JSON schema checks passed" || { echo "✗ JSON schema checks failed"; exit 1; }
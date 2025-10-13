#!/usr/bin/env bash
set -euo pipefail

echo "Checking for duplicate env lock definitions..."
ALLOW='tests/common/env\.rs'

if command -v rg >/dev/null 2>&1; then
  if rg -n --pcre2 'OnceLock<\s*Mutex<\s*\(\s*\)\s*>\s*>' tests | rg -v "$ALLOW" ; then
    echo "❌ Found duplicate env lock definitions; use common::env_guard()" >&2
    exit 1
  fi
else
  # BSD/GNU grep; use POSIX classes for portability.
  if grep -RIn --include='*.rs' -E 'OnceLock<[[:space:]]*Mutex<[[:space:]]*\([[:space:]]*\)[[:space:]]*>[[:space:]]*>' tests | grep -vE "$ALLOW" ; then
    echo "❌ Found duplicate env lock definitions; use common::env_guard()" >&2
    exit 1
  fi
fi

echo "✅ No duplicate env locks found"

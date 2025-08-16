#!/usr/bin/env bash
set -euo pipefail

# Ensure no duplicate per-file env locks slipped in
# The only env lock should be in tests/common/env.rs

echo "Checking for duplicate env lock definitions..."

if rg -n 'OnceLock<\s*Mutex<\s*\(\s*\)\s*>\s*>' tests | rg -v 'tests/common/env\.rs' ; then
  echo "❌ Found duplicate env lock definitions; use common::env_guard()" >&2
  exit 1
fi

echo "✅ No duplicate env locks found"
#!/usr/bin/env bash
set -euo pipefail

# Check for raw MB conversions that don't use BYTES_PER_MB or BYTES_PER_GB
# Allow units.rs itself since it defines the constants

echo "Checking for raw MB conversions..."

if rg -nE '(?<!BYTES_PER_(MB|GB))\b(1024\s*\*\s*1024|1_048_576)\b' tests | rg -v 'tests/common/units\.rs' ; then
  echo "❌ Raw MB conversions found. Use BYTES_PER_MB or BYTES_PER_GB." >&2
  exit 1
fi

echo "✅ No raw MB conversions found"

# Optional: Check for raw KB conversions (commented out by default - enable once ready to enforce)
# echo "Checking for raw KB conversions..."
# if rg -nE '(?<!BYTES_PER_KB)\b1024\b' tests | rg -v 'tests/common/units\.rs' ; then
#   echo "❌ Raw KB conversions found. Use BYTES_PER_KB." >&2
#   exit 1
# fi
# echo "✅ No raw KB conversions found"
#!/usr/bin/env bash
set -euo pipefail

echo "Checking for raw MB/GB conversions..."
# Allow units.rs itself since it defines the constants
ALLOW='tests/common/units\.rs'

# 1) 1024 * 1024 and 1024 * 1024 * 1024 (with spaces/underscores)
if grep -rE '(1024\s*\*\s*1024(\s*\*\s*1024)?)\b' tests --include="*.rs" \
  | grep -v "$ALLOW" | grep -v 'BYTES_PER_' \
  | grep -vE '^[^:]+:\s*//' ; then
  echo "❌ Raw 1024*1024 or 1024*1024*1024 found. Use BYTES_PER_MB or BYTES_PER_GB." >&2
  exit 1
fi

# 2) Decimal/underscore constants (1_048_576 / 1_073_741_824)
if grep -rE '\b(1_048_576|1048576|1_073_741_824|1073741824)\b' tests --include="*.rs" \
  | grep -v "$ALLOW" | grep -v 'BYTES_PER_' \
  | grep -vE '^[^:]+:\s*//' ; then
  echo "❌ Raw MB/GB decimal constants found. Use BYTES_PER_MB or BYTES_PER_GB." >&2
  exit 1
fi

# 3) Bit-shifts: 1<<20 (MB) or 1<<30 (GB) in various integer types
if grep -rE '\b(1[uiz]?[0-9_]*\s*<<\s*20|1[uiz]?[0-9_]*\s*<<\s*30)\b' tests --include="*.rs" | grep -v "$ALLOW" ; then
  echo "❌ Bit-shift MB/GB constants found (1<<20 or 1<<30). Use BYTES_PER_MB or BYTES_PER_GB." >&2
  exit 1
fi

echo "✅ No raw MB/GB conversions detected."

# Optional: Check for raw KB conversions (commented out by default - enable once ready to enforce)
# echo "Checking for raw KB conversions..."
# if grep -rE '\b(\d+\s*\*\s*)?1024\b' tests --include="*.rs" | grep -v "$ALLOW" | grep -v 'BYTES_PER_KB' ; then
#   echo "❌ Raw KB conversions found. Use BYTES_PER_KB." >&2
#   exit 1
# fi
# echo "✅ No raw KB conversions found"

#!/usr/bin/env bash
set -euo pipefail

# Disallow the old nested path in tests; we want bitnet_tests::units::* instead.
# Ignore single-line // comments to reduce false positives.
if grep -RIn --exclude-dir=target --include='*.rs' 'bitnet_tests::common::units::' tests \
  | grep -vE '^[^:]+:\s*//' ; then
  echo "❌ Use bitnet_tests::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB}" >&2
  exit 1
fi

echo "✅ Units imported via bitnet_tests::units::*"

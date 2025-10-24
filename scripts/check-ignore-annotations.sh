#!/usr/bin/env bash
#
# CI wrapper for unified #[ignore] annotation checking
#
# This script wraps scripts/lib/ignore_check.sh for CI/CD integration.
# It adds GitHub Actions annotations and uses the unified policy.

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call unified ignore check
if ! "${SCRIPT_DIR}/lib/ignore_check.sh" crates tests tests-new xtask; then
  # Add GitHub Actions error annotation
  echo "::error::Found #[ignore] tests without justification. See output above for details."
  exit 1
fi

exit 0

#!/usr/bin/env bash
set -euo pipefail

# BitNet-rs E2E Gate - Control concurrent test execution
# Prevents multiple heavy test suites from overwhelming system resources
# by allowing only MAX_E2E concurrent cross-validation/integration runs

MAX_E2E="${MAX_E2E:-2}"
LOCK="/tmp/bitnet-e2e-suite.lock"

# Source common utilities if available
if [[ -f "$(dirname "$0")/common.sh" ]]; then
    source "$(dirname "$0")/common.sh"
fi

echo "🔒 BitNet-rs E2E Gate: Acquiring test slot (max $MAX_E2E concurrent)"

# Acquire a shared lock with a small queue (emulates -j MAX_E2E)
exec 200>"$LOCK"

# Try immediate lock; if busy, wait (keeps logs cleaner)
if ! flock -n 200; then
    echo "⏳ E2E slot busy → waiting for available slot..."
    flock 200
fi

echo "✅ E2E slot acquired, running: $*"

# Source preflight caps before running tests
if [[ -f "$(dirname "$0")/preflight.sh" ]]; then
    source "$(dirname "$0")/preflight.sh"
fi

# Execute the requested command with capped resources
RUST_TEST_THREADS="${RUST_TEST_THREADS:-2}" \
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}" \
CROSSVAL_WORKERS="${CROSSVAL_WORKERS:-2}" \
    "$@"

exit_code=$?

echo "🏁 E2E slot released (exit code: $exit_code)"
exit $exit_code
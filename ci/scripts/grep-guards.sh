#!/usr/bin/env bash
# Evidence hygiene guard: prevent absolute paths and cuda-only feature gates
set -euo pipefail

fail=0

# Check for absolute paths in source code (excluding docs, tests, and build artifacts)
echo "Checking for absolute paths in source code..."
if rg -n '/home/steven' \
    -g '!target/**' \
    -g '!.git/**' \
    -g '!Cargo.lock' \
    -g '!docs/**' \
    -g '!*.md' \
    -g '!ci/receipts/**' \
    -g '!**/tests/**' \
    -g '!tests/**' \
    crates/*/src/; then
    echo "ERROR: Found absolute paths in production source code"
    fail=1
fi

# Check for cuda-only feature gates in source code (excluding docs, examples, and test fixtures)
echo "Checking for cuda-only feature gates in source code..."
if rg -n '#\[cfg\(feature\s*=\s*"cuda"\)\]' \
    -g '!Cargo.lock' \
    -g '!target/**' \
    -g '!.git/**' \
    -g '!docs/**' \
    -g '!*.md' \
    -g '!tests/fixtures/**' \
    -g '!examples/**' \
    crates/; then
    echo "ERROR: Found cuda-only feature gates in source code (should use 'any(feature = \"gpu\", feature = \"cuda\")')"
    fail=1
fi

# Check for runtime cuda-only cfg! checks in source code
if rg -n 'cfg!\(feature\s*=\s*"cuda"\)' \
    -g '!Cargo.lock' \
    -g '!target/**' \
    -g '!.git/**' \
    -g '!docs/**' \
    -g '!*.md' \
    -g '!tests/fixtures/**' \
    -g '!examples/**' \
    crates/; then
    echo "ERROR: Found runtime cuda-only cfg! checks in source code (should use 'any(feature = \"gpu\", feature = \"cuda\")')"
    fail=1
fi

if [ $fail -eq 0 ]; then
    echo "✅ All evidence hygiene checks passed"
fi

exit $fail

#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking for unannotated #[ignore] tests..."

# Find all #[ignore] attributes
IGNORE_TESTS=$(rg -n '#\[ignore\]' crates tests --type rust 2>/dev/null || true)

if [ -z "$IGNORE_TESTS" ]; then
  echo "✅ No ignored tests found"
  exit 0
fi

# Check each #[ignore] has a comment with justification
# Valid patterns: "Blocked by Issue #NNN", "Slow: <reason>", "TODO: <reason>"
UNANNOTATED=""

while IFS= read -r line; do
  FILE=$(echo "$line" | cut -d':' -f1)
  LINE_NUM=$(echo "$line" | cut -d':' -f2)

  # Extract 2 lines before #[ignore] for comment check
  CONTEXT=$(sed -n "$((LINE_NUM-2)),$((LINE_NUM))p" "$FILE")

  if ! echo "$CONTEXT" | grep -qE '(Blocked by Issue #[0-9]+|Slow:|TODO:)'; then
    UNANNOTATED="${UNANNOTATED}\n${FILE}:${LINE_NUM}"
  fi
done <<< "$IGNORE_TESTS"

if [ -n "$UNANNOTATED" ]; then
  echo "::error::Found #[ignore] tests without issue reference or justification:"
  echo -e "$UNANNOTATED"
  echo ""
  echo "Valid annotation patterns:"
  echo "  // Blocked by Issue #254 - shape mismatch in layer-norm"
  echo "  #[ignore]"
  echo ""
  echo "  // Slow: QK256 scalar kernels (~0.1 tok/s). Run with --ignored."
  echo "  #[ignore]"
  echo ""
  echo "  // TODO: Implement GPU mixed-precision tests after #439 resolution"
  echo "  #[ignore]"
  echo ""
  echo "See: https://github.com/microsoft/BitNet/blob/main/CLAUDE.md#test-status"
  exit 1
fi

echo "✅ All #[ignore] tests properly annotated"

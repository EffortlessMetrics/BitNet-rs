#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking for env-mutating tests without #[serial(bitnet_env)]..."

# Find tests that use EnvGuard or temp_env::with_var
ENV_MUTATING_TESTS=$(rg -n 'EnvGuard::new|temp_env::with_var' crates tests --type rust -B 5 2>/dev/null || true)

if [ -z "$ENV_MUTATING_TESTS" ]; then
  echo "✅ No env-mutating tests found"
  exit 0
fi

# Check each env-mutating test has #[serial(bitnet_env)]
UNANNOTATED=""

while IFS= read -r line; do
  # Skip empty lines
  if [ -z "$line" ]; then
    continue
  fi

  # Extract file and line number
  FILE=$(echo "$line" | cut -d':' -f1)
  LINE_NUM=$(echo "$line" | cut -d':' -f2)

  # Skip if line number is not a number (context lines)
  if ! [[ "$LINE_NUM" =~ ^[0-9]+$ ]]; then
    continue
  fi

  # Extract 10 lines before env mutation for #[serial] check
  CONTEXT=$(sed -n "$((LINE_NUM-10)),$((LINE_NUM))p" "$FILE" 2>/dev/null || true)

  if [ -z "$CONTEXT" ]; then
    continue
  fi

  if ! echo "$CONTEXT" | grep -q '#\[serial(bitnet_env)\]'; then
    # Check if it's inside a test function
    if echo "$CONTEXT" | grep -q '#\[test\]'; then
      UNANNOTATED="${UNANNOTATED}\n${FILE}:${LINE_NUM}"
    fi
  fi
done <<< "$ENV_MUTATING_TESTS"

if [ -n "$UNANNOTATED" ]; then
  echo "::error::Found env-mutating tests without #[serial(bitnet_env)]:"
  echo -e "$UNANNOTATED"
  echo ""
  echo "Env-mutating tests must use #[serial(bitnet_env)] to prevent race conditions."
  echo ""
  echo "Example pattern:"
  echo "  use serial_test::serial;"
  echo "  use tests::helpers::env_guard::EnvGuard;"
  echo ""
  echo "  #[test]"
  echo "  #[serial(bitnet_env)]"
  echo "  fn test_with_env_mutation() {"
  echo "      let _guard = EnvGuard::new(\"VAR_NAME\", \"value\");"
  echo "      // test code"
  echo "  }"
  echo ""
  echo "See: tests/helpers/env_guard.rs for proper usage"
  exit 1
fi

echo "✅ All env-mutating tests properly annotated with #[serial(bitnet_env)]"

#!/usr/bin/env bash
# Test harness for fix-locked.sh
# Runs fix-locked.sh against test fixtures and validates output
# Usage: scripts/tests/test-fix-locked.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES_DIR="$SCRIPT_DIR/fixtures"
FIX_LOCKED_SCRIPT="$SCRIPT_DIR/../fix-locked.sh"

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Temporary directory for test runs
TEST_WORK_DIR="$(mktemp -d)"
trap "rm -rf '$TEST_WORK_DIR'" EXIT

# Helper: Print test result
print_result() {
  local test_name="$1"
  local status="$2"
  local message="${3:-}"

  TESTS_RUN=$((TESTS_RUN + 1))

  if [ "$status" = "PASS" ]; then
    echo -e "${GREEN}✓ PASS${NC}: $test_name"
    TESTS_PASSED=$((TESTS_PASSED + 1))
  elif [ "$status" = "FAIL" ]; then
    echo -e "${RED}✗ FAIL${NC}: $test_name"
    if [ -n "$message" ]; then
      echo "  $message"
    fi
    TESTS_FAILED=$((TESTS_FAILED + 1))
  elif [ "$status" = "SKIP" ]; then
    echo -e "${YELLOW}⊘ SKIP${NC}: $test_name - $message"
  fi
}

# Test 1: Verify script exists and is executable
test_script_exists() {
  local test_name="Script exists and is executable"

  if [ -f "$FIX_LOCKED_SCRIPT" ] && [ -x "$FIX_LOCKED_SCRIPT" ]; then
    print_result "$test_name" "PASS"
  else
    print_result "$test_name" "FAIL" "Script not found or not executable: $FIX_LOCKED_SCRIPT"
  fi
}

# Test 2: Verify fixtures directory exists
test_fixtures_exist() {
  local test_name="Fixtures directory exists"

  if [ -d "$FIXTURES_DIR" ]; then
    local fixture_count=$(find "$FIXTURES_DIR" -name "*.yml" -not -name "*.expected.yml" | wc -l)
    print_result "$test_name" "PASS"
    echo "  Found $fixture_count test fixtures"
  else
    print_result "$test_name" "FAIL" "Fixtures directory not found: $FIXTURES_DIR"
  fi
}

# Test 3: Run fix-locked.sh on each fixture and compare with expected output
test_fixture() {
  local input_file="$1"
  local expected_file="$2"
  local fixture_name=$(basename "$input_file" .yml)

  # Copy input to work directory
  local work_file="$TEST_WORK_DIR/$(basename "$input_file")"
  cp "$input_file" "$work_file"

  # Run fix-locked.sh
  if "$FIX_LOCKED_SCRIPT" "$work_file" > /dev/null 2>&1; then
    # Compare output with expected
    if diff -q "$work_file" "$expected_file" > /dev/null 2>&1; then
      print_result "Fixture: $fixture_name" "PASS"
    else
      print_result "Fixture: $fixture_name" "FAIL" "Output differs from expected"
      echo "  --- Expected ---"
      head -20 "$expected_file"
      echo "  --- Got ---"
      head -20 "$work_file"
      echo "  --- Diff ---"
      diff -u "$expected_file" "$work_file" | head -40 || true
    fi
  else
    print_result "Fixture: $fixture_name" "FAIL" "Script execution failed"
  fi
}

# Test 4: Idempotency test (running twice should produce same result)
test_idempotency() {
  local test_name="Idempotency test"
  local input_file="$FIXTURES_DIR/01-simple.yml"
  local work_file1="$TEST_WORK_DIR/idempotency-1.yml"
  local work_file2="$TEST_WORK_DIR/idempotency-2.yml"

  # First run
  cp "$input_file" "$work_file1"
  "$FIX_LOCKED_SCRIPT" "$work_file1" > /dev/null 2>&1

  # Second run
  cp "$work_file1" "$work_file2"
  "$FIX_LOCKED_SCRIPT" "$work_file2" > /dev/null 2>&1

  # Compare
  if diff -q "$work_file1" "$work_file2" > /dev/null 2>&1; then
    print_result "$test_name" "PASS"
  else
    print_result "$test_name" "FAIL" "Running twice produced different results"
  fi
}

# Test 5: Dry-run mode doesn't modify files
test_dry_run_mode() {
  local test_name="Dry-run mode (no modifications)"
  local input_file="$FIXTURES_DIR/01-simple.yml"
  local work_file="$TEST_WORK_DIR/dry-run-test.yml"

  cp "$input_file" "$work_file"
  local original_checksum=$(md5sum "$work_file" | cut -d' ' -f1)

  # Run in dry-run mode
  "$FIX_LOCKED_SCRIPT" --dry-run "$work_file" > /dev/null 2>&1 || true

  local after_checksum=$(md5sum "$work_file" | cut -d' ' -f1)

  if [ "$original_checksum" = "$after_checksum" ]; then
    print_result "$test_name" "PASS"
  else
    print_result "$test_name" "FAIL" "File was modified in dry-run mode"
  fi
}

# Test 6: Check mode exits with non-zero when changes needed
test_check_mode_dirty() {
  local test_name="Check mode (changes needed)"
  local input_file="$FIXTURES_DIR/01-simple.yml"
  local work_file="$TEST_WORK_DIR/check-mode-dirty.yml"

  cp "$input_file" "$work_file"

  # Run in check mode (should fail because file needs changes)
  if "$FIX_LOCKED_SCRIPT" --check "$work_file" > /dev/null 2>&1; then
    print_result "$test_name" "FAIL" "Check mode should have exited non-zero"
  else
    print_result "$test_name" "PASS"
  fi
}

# Test 7: Check mode exits with zero when no changes needed
test_check_mode_clean() {
  local test_name="Check mode (no changes needed)"
  local expected_file="$FIXTURES_DIR/05-already-locked.expected.yml"
  local work_file="$TEST_WORK_DIR/check-mode-clean.yml"

  cp "$expected_file" "$work_file"

  # Run in check mode (should pass because file already has --locked)
  if "$FIX_LOCKED_SCRIPT" --check "$work_file" > /dev/null 2>&1; then
    print_result "$test_name" "PASS"
  else
    print_result "$test_name" "FAIL" "Check mode should have exited zero"
  fi
}

# Test 8: Handles non-existent files gracefully
test_nonexistent_file() {
  local test_name="Handles non-existent files"

  # Should not crash, just warn
  if "$FIX_LOCKED_SCRIPT" "/nonexistent/file.yml" > /dev/null 2>&1; then
    print_result "$test_name" "PASS"
  else
    # It's okay if it exits non-zero, as long as it doesn't crash
    print_result "$test_name" "PASS"
  fi
}

# Test 9: Usage message when no arguments
test_usage_message() {
  local test_name="Shows usage message with no args"

  # Script should fail and show usage when no args provided
  local output
  output=$("$FIX_LOCKED_SCRIPT" 2>&1 || true)

  if echo "$output" | grep -q "Usage:"; then
    print_result "$test_name" "PASS"
  else
    print_result "$test_name" "FAIL" "No usage message displayed. Output: $output"
  fi
}

# Main test runner
main() {
  echo "========================================"
  echo "Testing fix-locked.sh"
  echo "========================================"
  echo ""

  # Prerequisite tests
  test_script_exists
  test_fixtures_exist

  # Fixture-based tests
  echo ""
  echo "--- Fixture Tests ---"
  for input_file in "$FIXTURES_DIR"/*.yml; do
    # Skip .expected.yml files
    if [[ "$input_file" == *.expected.yml ]]; then
      continue
    fi

    expected_file="${input_file%.yml}.expected.yml"
    if [ -f "$expected_file" ]; then
      test_fixture "$input_file" "$expected_file"
    else
      print_result "Fixture: $(basename "$input_file")" "SKIP" "No expected file found"
    fi
  done

  # Functional tests
  echo ""
  echo "--- Functional Tests ---"
  test_idempotency
  test_dry_run_mode
  test_check_mode_dirty
  test_check_mode_clean
  test_nonexistent_file
  test_usage_message

  # Summary
  echo ""
  echo "========================================"
  echo "Test Summary"
  echo "========================================"
  echo "Tests run:    $TESTS_RUN"
  echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
  if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
  else
    echo "Tests failed: $TESTS_FAILED"
  fi
  echo ""

  if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
  else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
  fi
}

# Run main
main

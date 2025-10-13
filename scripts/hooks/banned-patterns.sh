#!/usr/bin/env bash
set -euo pipefail

fail=0

echo "🔍 Checking for banned patterns in test API..."

# 1) Old fallible type: forbid generic TestResult<T> in tests/common
if rg -n --glob 'tests/common/**' --pcre2 '->\s*TestResult<' 2>/dev/null; then
  echo "❌ Found legacy 'TestResult<T>' – use 'TestResultCompat<T>'."
  fail=1
fi

# 2) Old record alias and ctors
if rg -n --glob 'tests/common/**' --pcre2 '\bTestResultData\b' 2>/dev/null; then
  echo "❌ Found legacy 'TestResultData' – use 'TestRecord'."
  fail=1
fi

if rg -n --glob 'tests/common/**' --pcre2 '\bTestResult::(passed|failed|timeout)\b' 2>/dev/null; then
  echo "❌ Found legacy 'TestResult::...' – use 'TestRecord::...'."
  fail=1
fi

# 3) Split doc comments (caught a bunch): `}    ///`
if rg -n --glob 'tests/**' --pcre2 '^\s*}\s*///' 2>/dev/null; then
  echo "❌ Found split doc-comment after a closing brace. Move '///' to next line."
  fail=1
fi

# 4) No &Vec<T> in public signatures (clippy::ptr_arg will also fail)
if rg -n --glob '**/*.rs' --pcre2 '&\s*Vec<[^>]+>' 2>/dev/null | rg -v target 2>/dev/null; then
  echo "❌ Prefer slices: '&[T]' instead of '&Vec<T>'."
  fail=1
fi

# 5) Check for direct TestResult usage outside of tests/common
if rg -n --glob 'tests/*.rs' --pcre2 '\bTestResult\b' 2>/dev/null | rg -v 'TestResultCompat' 2>/dev/null; then
  echo "⚠️  Found 'TestResult' usage outside tests/common - verify it's using the correct import."
  # Don't fail for this, just warn
fi

if [ $fail -eq 0 ]; then
  echo "✅ All API contract checks passed!"
else
  echo ""
  echo "❌ API contract violations found. Please fix before committing."
fi

exit $fail

#!/usr/bin/env bash
set -euo pipefail

# Codemod raw MB/GB constants in tests/ to BYTES_PER_MB / BYTES_PER_GB, one class at a time.
# Usage:
#   scripts/codemod-units.sh --preview   # show impact
#   scripts/codemod-units.sh --apply     # perform replacements + commit after each class
#
# Notes:
# - Skips tests/common/units.rs
# - Leaves bit-shifts (1<<20/30) as "review-only" by default
# - Runs `cargo check` after each committed class (abort+revert if it fails)

MODE="${1:---preview}"
ALLOW='tests/common/units\.rs'

has_cmd() { command -v "$1" >/dev/null 2>&1; }

rg_files() {
  # $1 = PCRE2 pattern (for rg) or $2 = simpler grep pattern
  local pcre="$1"
  local simple="${2:-$1}"
  if has_cmd rg; then
    rg -l --pcre2 "$pcre" tests -g '!tests/common/units.rs' -g '!**/target/**' --iglob '*.rs' || true
  else
    # Fallback: grep with simpler pattern
    grep -RIl --include='*.rs' -E "$simple" tests | grep -vE "$ALLOW" || true
  fi
}

perl_inplace() {
  # $1 = perl substitution expr, $@ = files
  [ $# -ge 2 ] || return 0
  local expr="$1"; shift
  perl -0777 -pe "$expr" -i -- "$@"
}

commit_or_skip() {
  local msg="$1"
  if git diff --quiet -- tests; then
    echo "No changes for: $msg"
  else
    git add -A
    git commit -m "$msg"
    echo "Committed: $msg"
    echo "→ Running cargo check..."
    if ! cargo check -q ; then
      echo "❌ cargo check failed; reverting last commit."
      git reset --hard HEAD~1
      exit 1
    fi
  fi
}

do_class() {
  local name="$1" pcre_pat="$2" simple_pat="$3" perl_expr="$4" commit_msg="$5"
  echo "==> $name"
  mapfile -t files < <(rg_files "$pcre_pat" "$simple_pat")
  echo "   files: ${#files[@]}"
  if [ "$MODE" = "--preview" ]; then
    printf '%s\n' "${files[@]}" | sed 's/^/   - /'
    return 0
  fi
  if [ "${#files[@]}" -gt 0 ]; then
    perl_inplace "$perl_expr" "${files[@]}"
  fi
  commit_or_skip "$commit_msg"
}

echo "codemod-units.sh mode: $MODE"
echo "Will skip: tests/common/units.rs"
echo

# 1) 1024 * 1024 → BYTES_PER_MB
do_class "MB literal (1024*1024)" \
  '\b(?<!BYTES_PER_MB)1024\s*\*\s*1024\b' \
  '1024[[:space:]]*\*[[:space:]]*1024' \
  's/\b1024\s*\*\s*1024\b/BYTES_PER_MB/g' \
  "codemod(units): 1024*1024 → BYTES_PER_MB (tests)"

# 2) N * 1024 * 1024 → N * BYTES_PER_MB (allow underscores in N)
do_class "MB scaled (N*1024*1024)" \
  '\b(\d(?:_?\d)*)\s*\*\s*1024\s*\*\s*1024\b' \
  '[0-9_]+[[:space:]]*\*[[:space:]]*1024[[:space:]]*\*[[:space:]]*1024' \
  's/\b(\d(?:_?\d)*)\s*\*\s*1024\s*\*\s*1024\b/$1 * BYTES_PER_MB/g' \
  "codemod(units): N*1024*1024 → N*BYTES_PER_MB (tests)"

# 3) MB decimal constants → BYTES_PER_MB
do_class "MB decimal (1048576 / 1_048_576)" \
  '\b(?<!BYTES_PER_MB)(1_048_576|1048576)\b' \
  '(1_048_576|1048576)' \
  's/\b(1_048_576|1048576)\b/BYTES_PER_MB/g' \
  "codemod(units): MB decimal → BYTES_PER_MB (tests)"

# 4) 1024*1024*1024 → BYTES_PER_GB
do_class "GB literal (1024*1024*1024)" \
  '\b(?<!BYTES_PER_GB)1024\s*\*\s*1024\s*\*\s*1024\b' \
  '1024[[:space:]]*\*[[:space:]]*1024[[:space:]]*\*[[:space:]]*1024' \
  's/\b1024\s*\*\s*1024\s*\*\s*1024\b/BYTES_PER_GB/g' \
  "codemod(units): 1024*1024*1024 → BYTES_PER_GB (tests)"

# 5) N*1024*1024*1024 → N*BYTES_PER_GB (allow underscores in N; unwrapped pattern)
do_class "GB scaled (N*1024*1024*1024)" \
  '\b(\d(?:_?\d)*)\s*\*\s*1024\s*\*\s*1024\s*\*\s*1024\b' \
  '[0-9_]+[[:space:]]*\*[[:space:]]*1024[[:space:]]*\*[[:space:]]*1024[[:space:]]*\*[[:space:]]*1024' \
  's/\b(\d(?:_?\d)*)\s*\*\s*1024\s*\*\s*1024\s*\*\s*1024\b/$1 * BYTES_PER_GB/g' \
  "codemod(units): N*1024*1024*1024 → N*BYTES_PER_GB (tests)"

# 6) GB decimal constants → BYTES_PER_GB
do_class "GB decimal (1073741824 / 1_073_741_824)" \
  '\b(?<!BYTES_PER_GB)(1_073_741_824|1073741824)\b' \
  '(1_073_741_824|1073741824)' \
  's/\b(1_073_741_824|1073741824)\b/BYTES_PER_GB/g' \
  "codemod(units): GB decimal → BYTES_PER_GB (tests)"

# 7) KB scaled (conservative; run after MB/GB steps)
# Replace: N * 1024   (NOT followed by another * 1024)
# Rationale: Avoid touching MB/GB chains; Perl does the negative-lookahead filter.
do_class "KB scaled (N*1024, not followed by another *1024)" \
  '\b(\d(?:_?\d)*)\s*\*\s*1024\b(?!\s*\*)' \
  '[0-9_]+[[:space:]]*\*[[:space:]]*1024' \
  's/\b(\d(?:_?\d)*)\s*\*\s*1024\b(?!\s*\*)/$1 * BYTES_PER_KB/g' \
  "codemod(units): N*1024 → N*BYTES_PER_KB (tests, conservative)"

# 8) Bit-shifts (review-only)
echo
echo "Candidates for manual review (bit-shifts):"
if has_cmd rg; then
  rg -n --pcre2 '\b1(?:_u(?:8|16|32|64|128)|_i(?:8|16|32|64|128)|_usize|_isize)?\s*<<\s*(20|30)\b' tests -g '!tests/common/units.rs' || true
else
  # Simpler grep pattern for bit shifts
  grep -RIn --include='*.rs' -E '1[[:space:]]*<<[[:space:]]*(20|30)' tests | grep -vE "$ALLOW" || true
fi

echo
echo "Done. Mode: $MODE"

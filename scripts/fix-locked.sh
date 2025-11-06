#!/usr/bin/env bash
# Inserts --locked into cargo|cross (build|test|run|bench|clippy) commands.
# - If a standalone " -- " exists, inserts before it
# - Skips lines already containing --locked
# Usage:
#   scripts/fix-locked.sh .github/workflows/*.yml          # Apply changes
#   scripts/fix-locked.sh --dry-run .github/workflows/*.yml # Preview changes
#   scripts/fix-locked.sh --check .github/workflows/*.yml   # Check mode (CI)

set -euo pipefail

# Parse mode from arguments
mode="apply"
files=()

for arg in "$@"; do
  case "$arg" in
    --dry-run|--preview)
      mode="dry-run"
      ;;
    --check)
      mode="check"
      ;;
    *)
      files+=("$arg")
      ;;
  esac
done

if [ ${#files[@]} -eq 0 ]; then
  echo "Usage: $0 [--dry-run|--check] <file1> [file2 ...]" >&2
  echo "" >&2
  echo "Modes:" >&2
  echo "  (default)  Apply changes in-place" >&2
  echo "  --dry-run  Show what would be changed (no modifications)" >&2
  echo "  --check    Exit with non-zero if changes would be made (CI mode)" >&2
  exit 1
fi

# Track if any changes would be made (for --check mode)
changes_detected=0

for f in "${files[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Warning: File not found: $f" >&2
    continue
  fi

  tmp="$(mktemp)"

  # Process the file with awk
  awk '
    function has_locked(s)  { return match(s, /(^|[[:space:]])--locked([[:space:]]|$)/) }
    function is_target(s)   { return match(s, /(^|[[:space:]])(cargo|cross)[[:space:]]+(build|test|run|bench|clippy)([[:space:]]|$)/) }
    function is_comment(s)  { return match(s, /^[[:space:]]*#/) }
    {
      if (is_target($0) && !has_locked($0) && !is_comment($0)) {
        line = $0

        # Extract trailing comment if present
        comment = ""
        if (match(line, /[[:space:]]*#/)) {
          comment = substr(line, RSTART)
          line = substr(line, 1, RSTART-1)
        }

        # Extract trailing backslash if present
        backslash = ""
        if (match(line, /\\[[:space:]]*$/)) {
          backslash = " \\"
          sub(/\\[[:space:]]*$/, "", line)
        }

        # If there is a standalone " -- " token, inject before it; else append at end
        if (match(line, /[[:space:]]--[[:space:]]/)) {
          prefix = substr(line, 1, RSTART-1)
          suffix = substr(line, RSTART)
          print prefix " --locked" suffix backslash comment
        } else {
          # Remove trailing whitespace before appending
          sub(/[[:space:]]+$/, "", line)
          print line " --locked" backslash comment
        }
      } else {
        print
      }
    }
  ' "$f" > "$tmp"

  # Check if file would change
  if ! diff -q "$f" "$tmp" > /dev/null 2>&1; then
    changes_detected=1

    case "$mode" in
      apply)
        mv "$tmp" "$f"
        echo "✓ Updated: $f"
        ;;
      dry-run)
        echo "Would update: $f"
        echo "--- Diff ---"
        diff -u "$f" "$tmp" || true
        echo ""
        rm "$tmp"
        ;;
      check)
        echo "Changes needed in: $f" >&2
        rm "$tmp"
        ;;
    esac
  else
    rm "$tmp"
    if [ "$mode" = "dry-run" ]; then
      echo "No changes: $f"
    fi
  fi
done

# Final status reporting
case "$mode" in
  apply)
    echo "✓ Applied --locked where missing"
    ;;
  dry-run)
    if [ $changes_detected -eq 0 ]; then
      echo "✓ No changes needed (all files already have --locked)"
    else
      echo "⚠ Changes would be made (see diffs above)"
    fi
    ;;
  check)
    if [ $changes_detected -ne 0 ]; then
      echo "❌ Some files are missing --locked flags" >&2
      echo "Run: scripts/fix-locked.sh .github/workflows/*.yml" >&2
      exit 1
    else
      echo "✓ All cargo commands have --locked flags"
    fi
    ;;
esac

exit 0

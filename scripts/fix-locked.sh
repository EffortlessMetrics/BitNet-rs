#!/usr/bin/env bash
# Inserts --locked into cargo|cross (build|test|run|bench|clippy) commands.
# - If a standalone " -- " exists, inserts before it
# - Skips lines already containing --locked
# Usage: scripts/fix-locked.sh .github/workflows/*.yml

set -euo pipefail
files=("$@")
for f in "${files[@]}"; do
  tmp="$(mktemp)"
  awk '
    function has_locked(s)  { return match(s, /(^|[[:space:]])--locked([[:space:]]|$)/) }
    function is_target(s)   { return match(s, /(^|[[:space:]])(cargo|cross)[[:space:]]+(build|test|run|bench|clippy)([[:space:]]|$)/) }
    {
      if (is_target($0) && !has_locked($0)) {
        # If there is a standalone " -- " token, inject before it; else append
        if (match($0, /[[:space:]]--[[:space:]]/)) {
          prefix = substr($0, 1, RSTART-1)
          suffix = substr($0, RSTART)
          print prefix " --locked" suffix
        } else {
          print $0 " --locked"
        }
      } else {
        print
      }
    }
  ' "$f" > "$tmp"
  mv "$tmp" "$f"
done
echo "✓ Applied --locked where missing"

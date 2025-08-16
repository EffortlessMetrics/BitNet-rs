#!/bin/bash
set -euo pipefail

# Add imports for BYTES_PER_* constants to files that need them

for file in tests/common/*.rs tests/examples/*.rs tests/*.rs; do
  if [[ ! -f "$file" ]]; then
    continue
  fi
  
  # Skip units.rs itself
  if [[ "$file" == "tests/common/units.rs" ]]; then
    continue
  fi
  
  # Check if file uses BYTES_PER_* constants
  if ! grep -q "BYTES_PER_" "$file" 2>/dev/null; then
    continue
  fi
  
  # Check if already has the import
  if grep -q "use.*units::" "$file" 2>/dev/null; then
    continue
  fi
  
  # Determine the correct import path based on file location
  if [[ "$file" == tests/common/*.rs ]]; then
    # Files in tests/common/ use super::units
    import_line="use super::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};"
  elif [[ "$file" == tests/examples/*.rs ]] || [[ "$file" == tests/*.rs ]]; then
    # Files in tests/ or tests/examples/ use common::units or bitnet_tests::common::units
    if grep -q "use common::" "$file" 2>/dev/null; then
      import_line="use common::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};"
    else
      import_line="use bitnet_tests::common::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};"
    fi
  fi
  
  # Add the import after the first use statement or at the beginning
  if grep -q "^use " "$file"; then
    # Find the last use statement and add after it
    awk -v import="$import_line" '
      /^use / { last_use = NR }
      { lines[NR] = $0 }
      END {
        for (i = 1; i <= NR; i++) {
          print lines[i]
          if (i == last_use) {
            print import
          }
        }
      }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
  else
    # Add at the beginning after any module docs
    awk -v import="$import_line" '
      !added && !/^\/\/\// && !/^#\[/ {
        print import
        print ""
        added = 1
      }
      { print }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
  fi
  
  echo "Added import to $file"
done

echo "Done fixing imports"
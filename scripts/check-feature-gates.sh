#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking feature gate consistency..."

# Extract all defined features from root Cargo.toml
DEFINED_FEATURES=$(grep -A 100 '^\[features\]' Cargo.toml | grep '^[a-z0-9_-]* =' | cut -d' ' -f1 | sort | uniq)

echo "Defined features ($(echo "$DEFINED_FEATURES" | wc -l)):"
echo "$DEFINED_FEATURES" | sed 's/^/  /'

# Find all #[cfg(feature = "...")] usage in code
USED_FEATURES=$(rg -oI '#\[cfg.*feature\s*=\s*"([^"]+)"' --replace '$1' crates --type rust 2>/dev/null | sort | uniq)

echo ""
echo "Used features in #[cfg] ($(echo "$USED_FEATURES" | wc -l)):"
echo "$USED_FEATURES" | sed 's/^/  /'

# Check for undefined features
UNDEFINED=""

for feature in $USED_FEATURES; do
  if ! echo "$DEFINED_FEATURES" | grep -qx "$feature"; then
    UNDEFINED="${UNDEFINED}\n  - $feature"
  fi
done

if [ -n "$UNDEFINED" ]; then
  echo "::error::Found #[cfg(feature = ...)] using undefined features:"
  echo -e "$UNDEFINED"
  echo ""
  echo "These features are referenced in code but not defined in Cargo.toml [features] section."
  echo "Either define the feature or remove the #[cfg] annotation."
  exit 1
fi

# Check for common patterns that suggest feature gate bugs
echo ""
echo "🔍 Checking for feature gate antipatterns..."

# Pattern 1: #[cfg(feature = "gpu")] without #[cfg(any(feature = "gpu", feature = "cuda"))]
GPU_WITHOUT_CUDA=$(rg -n '#\[cfg\(feature = "gpu"\)\]' crates --type rust 2>/dev/null | grep -v 'any(' || true)

if [ -n "$GPU_WITHOUT_CUDA" ]; then
  echo "::warning::Found #[cfg(feature = \"gpu\")] without fallback to \"cuda\":"
  echo "$GPU_WITHOUT_CUDA"
  echo ""
  echo "Recommended pattern:"
  echo "  #[cfg(any(feature = \"gpu\", feature = \"cuda\"))]"
  echo ""
  echo "This ensures backward compatibility with legacy 'cuda' feature."
fi

echo "✅ Feature gate consistency check passed"

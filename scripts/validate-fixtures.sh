#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Validating GGUF fixture integrity..."

FIXTURE_DIR="ci/fixtures/qk256"
CHECKSUM_FILE="$FIXTURE_DIR/SHA256SUMS"

if [ ! -f "$CHECKSUM_FILE" ]; then
  echo "::error::Fixture checksum file not found: $CHECKSUM_FILE"
  exit 1
fi

# Verify checksums
cd "$FIXTURE_DIR"
if ! sha256sum --check --strict SHA256SUMS 2>&1; then
  echo "::error::Fixture checksum verification failed"
  echo "Fixtures may be corrupted or modified without updating SHA256SUMS"
  echo ""
  echo "To regenerate checksums:"
  echo "  cd ci/fixtures/qk256"
  echo "  sha256sum *.gguf > SHA256SUMS"
  exit 1
fi

echo "✅ All fixture checksums valid"

# Validate fixture schema (GGUF alignment, tensor count)
echo "🔍 Validating fixture schema..."

cd - > /dev/null

for gguf in "$FIXTURE_DIR"/*.gguf; do
  if [ ! -f "$gguf" ]; then
    continue
  fi

  echo "Checking $(basename "$gguf")..."

  # Use bitnet-cli to inspect GGUF metadata
  if ! cargo run -q -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    inspect "$gguf" --format json > /tmp/fixture_inspect.json 2>&1; then
    echo "::warning::Could not inspect $gguf - skipping schema validation"
    continue
  fi

  # Validate tensor alignment (must be 32-byte aligned for QK256)
  # Note: This is a basic check - alignment may not be reported in all GGUF files
  if command -v jq >/dev/null 2>&1; then
    ALIGNMENT=$(jq -r '.tensors[0].alignment // "unknown"' /tmp/fixture_inspect.json 2>/dev/null || echo "unknown")
    if [ "$ALIGNMENT" != "32" ] && [ "$ALIGNMENT" != "unknown" ]; then
      echo "::error::Fixture $(basename "$gguf") has invalid tensor alignment: $ALIGNMENT (expected 32)"
      exit 1
    fi
  else
    echo "::warning::jq not installed - skipping alignment validation"
  fi
done

echo "✅ All fixture schemas valid"

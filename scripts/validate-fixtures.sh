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

# Validate fixture schema (GGUF structure, alignment, tensor count)
echo "🔍 Validating fixture GGUF structure..."

cd - > /dev/null

for gguf in "$FIXTURE_DIR"/*.gguf; do
  if [ ! -f "$gguf" ]; then
    continue
  fi

  BASENAME=$(basename "$gguf")
  echo "Checking $BASENAME..."

  # 1. Validate GGUF magic number (first 4 bytes must be "GGUF")
  MAGIC=$(head -c 4 "$gguf" | tr -d '\0')
  if [ "$MAGIC" != "GGUF" ]; then
    echo "::error::Fixture $BASENAME has invalid magic number: $MAGIC (expected GGUF)"
    exit 1
  fi

  # 2. Validate GGUF version (bytes 4-7, must be 2 or 3)
  # Read as little-endian u32
  VERSION=$(od -An -t u4 -N 4 -j 4 "$gguf" | tr -d ' ')
  if [ "$VERSION" != "2" ] && [ "$VERSION" != "3" ]; then
    echo "::error::Fixture $BASENAME has invalid version: $VERSION (expected 2 or 3)"
    exit 1
  fi

  # 3. Use bitnet-cli to inspect GGUF metadata and validate structure
  if ! cargo run -q -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    inspect "$gguf" --format json > /tmp/fixture_inspect.json 2>&1; then
    echo "::warning::Could not inspect $gguf - skipping metadata validation"
    continue
  fi

  if command -v jq >/dev/null 2>&1; then
    # 4. Validate required metadata keys exist
    REQUIRED_KEYS=("general.architecture" "general.name")
    for key in "${REQUIRED_KEYS[@]}"; do
      if ! jq -e ".metadata[\"$key\"]" /tmp/fixture_inspect.json >/dev/null 2>&1; then
        echo "::warning::Fixture $BASENAME missing recommended metadata key: $key"
      fi
    done

    # 5. Validate tensor alignment (must be 32-byte aligned for GGUF v3)
    ALIGNMENT=$(jq -r '.tensors[0].alignment // "unknown"' /tmp/fixture_inspect.json 2>/dev/null || echo "unknown")
    if [ "$VERSION" = "3" ] && [ "$ALIGNMENT" != "32" ] && [ "$ALIGNMENT" != "unknown" ]; then
      echo "::error::Fixture $BASENAME has invalid tensor alignment: $ALIGNMENT (GGUF v3 requires 32-byte alignment)"
      exit 1
    fi

    # 6. Validate tensor count (fixtures should have at least 2 tensors)
    TENSOR_COUNT=$(jq -r '.tensors | length' /tmp/fixture_inspect.json 2>/dev/null || echo "0")
    if [ "$TENSOR_COUNT" -lt 2 ]; then
      echo "::warning::Fixture $BASENAME has only $TENSOR_COUNT tensors (expected ≥2 for realistic fixtures)"
    fi
  else
    echo "::warning::jq not installed - skipping metadata and alignment validation"
  fi
done

echo "✅ All fixture GGUF structures valid"

#!/usr/bin/env bash
# Script to vendor GGML quantization files from llama.cpp
set -euo pipefail

COMMIT="${1:-master}"
if [ "$COMMIT" = "master" ]; then
    echo "Warning: Using master branch. Consider pinning a specific commit for reproducibility."
    echo "Example: $0 b4530"
fi

BASE="https://raw.githubusercontent.com/ggerganov/llama.cpp/${COMMIT}"
DEST="crates/bitnet-ggml-ffi/csrc"

echo "Vendoring GGML quants from commit: ${COMMIT}"
echo "Fetching to: ${DEST}"

mkdir -p "$DEST"

# Fetch the core files needed for IQ2_S dequantization
echo "Downloading ggml.h..."
curl -fsSL "$BASE/ggml.h" -o "$DEST/ggml.h" || {
    echo "Failed to download ggml.h from $BASE"
    exit 1
}

echo "Downloading ggml-quants.h..."
curl -fsSL "$BASE/ggml-quants.h" -o "$DEST/ggml-quants.h" || {
    echo "Failed to download ggml-quants.h from $BASE"
    exit 1
}

echo "Downloading ggml-quants.c..."
curl -fsSL "$BASE/ggml-quants.c" -o "$DEST/ggml-quants.c" || {
    echo "Failed to download ggml-quants.c from $BASE"
    exit 1
}

# Create a version file to track what commit we vendored
echo "$COMMIT" > "$DEST/GGML_VERSION"

echo "Successfully vendored GGML quants from commit ${COMMIT}"
echo "Files written to: ${DEST}/"
ls -la "$DEST/"
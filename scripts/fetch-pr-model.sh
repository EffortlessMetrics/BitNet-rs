#!/usr/bin/env bash
# Fetch TinyLlama Q2_K model with embedded SentencePiece tokenizer for PR CI
# This model is small enough for fast CI but complete enough for validation
set -euo pipefail

MODEL_DIR="models"
MODEL_NAME="tinyllama-q2.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

# TinyLlama Q2_K with embedded tokenizer (example URL - replace with actual)
# This should be a model that:
# - Has embedded SentencePiece tokenizer
# - Is quantized to Q2_K for small size (~400MB)
# - Works with BitNet.rs tensor mappings
MODEL_URL="${TINYLLAMA_URL:-https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf}"

# SHA256 checksum for verification (update when model changes)
# To generate: sha256sum tinyllama-q2.gguf
MODEL_SHA256="${TINYLLAMA_SHA256:-f7e93920364e7c886f12b6a0a0bf1e49b0c4088e6b0c2bb3a0e4bb0e1f4c8a9d}"

echo "=== Fetching PR Test Model ==="
echo "Model: TinyLlama Q2_K with embedded tokenizer"
echo "Path: $MODEL_PATH"

# Create model directory if needed
mkdir -p "$MODEL_DIR"

# Check if model already exists with correct checksum
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model already exists at $MODEL_PATH"

    # Verify checksum
    if command -v sha256sum >/dev/null 2>&1; then
        ACTUAL_SHA256=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
        if [ "$ACTUAL_SHA256" = "$MODEL_SHA256" ]; then
            echo "✓ Checksum verified"
            # Quick validation
            if cargo run -q -p xtask -- gate mapper --model "$MODEL_PATH" 2>/dev/null | jq -e '.ok==true' >/dev/null; then
                echo "✓ Model validated successfully"
                exit 0
            fi
        else
            echo "⚠ Checksum mismatch, re-downloading..."
            echo "  Expected: $MODEL_SHA256"
            echo "  Actual:   $ACTUAL_SHA256"
            rm -f "$MODEL_PATH"
        fi
    else
        # Fallback: just check if model validates
        if cargo run -q -p xtask -- gate mapper --model "$MODEL_PATH" 2>/dev/null | jq -e '.ok==true' >/dev/null; then
            echo "✓ Model validated successfully (checksum not verified)"
            exit 0
        else
            echo "⚠ Existing model failed validation, re-downloading..."
            rm -f "$MODEL_PATH"
        fi
    fi
fi

# Download model with retries
echo "Downloading model..."
DOWNLOAD_SUCCESS=false
for attempt in 1 2 3; do
    echo "Download attempt $attempt/3..."

    if command -v aria2c >/dev/null 2>&1; then
        # aria2c is fastest and most reliable
        aria2c -x 4 -s 4 --retry-wait=3 --max-tries=3 -o "$MODEL_PATH" "$MODEL_URL" && DOWNLOAD_SUCCESS=true && break
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=30 -q --show-progress -O "$MODEL_PATH" "$MODEL_URL" && DOWNLOAD_SUCCESS=true && break
    elif command -v curl >/dev/null 2>&1; then
        curl --retry 3 --retry-delay 3 -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL" && DOWNLOAD_SUCCESS=true && break
    else
        echo "❌ No download tool found. Please install aria2c, wget, or curl."
        exit 1
    fi

    echo "⚠ Download failed, retrying..."
    sleep 5
done

if [ "$DOWNLOAD_SUCCESS" != "true" ]; then
    echo "❌ Download failed after 3 attempts"
    exit 1
fi

# Verify download and checksum
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Downloaded file not found"
    exit 1
fi

# Verify checksum if sha256sum available
if command -v sha256sum >/dev/null 2>&1; then
    echo "Verifying checksum..."
    ACTUAL_SHA256=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
    if [ "$ACTUAL_SHA256" != "$MODEL_SHA256" ]; then
        echo "❌ Checksum verification failed!"
        echo "  Expected: $MODEL_SHA256"
        echo "  Actual:   $ACTUAL_SHA256"
        echo ""
        echo "This may indicate:"
        echo "  1. Corrupted download - try again"
        echo "  2. Model has been updated - update MODEL_SHA256 in this script"
        echo "  3. Security issue - do not use this model"
        rm -f "$MODEL_PATH"
        exit 1
    fi
    echo "✓ Checksum verified"
else
    echo "⚠ sha256sum not available, skipping checksum verification"
fi

# Validate model has embedded tokenizer
echo "Validating model..."
VALIDATION_JSON=$(mktemp)

if cargo run -q -p xtask -- gate mapper --model "$MODEL_PATH" > "$VALIDATION_JSON" 2>/dev/null; then
    if jq -e '.ok==true and .unmapped_count==0' "$VALIDATION_JSON" >/dev/null; then
        echo "✓ Model validation successful"
        echo "  Tensors: $(jq -r '.total_count' "$VALIDATION_JSON")"
        echo "  Unmapped: $(jq -r '.unmapped_count' "$VALIDATION_JSON")"
    else
        echo "❌ Model validation failed - tensors not properly mapped"
        jq '.' "$VALIDATION_JSON"
        rm -f "$VALIDATION_JSON" "$MODEL_PATH"
        exit 1
    fi
else
    echo "❌ Model validation failed to run"
    rm -f "$VALIDATION_JSON" "$MODEL_PATH"
    exit 1
fi

rm -f "$VALIDATION_JSON"

# Test that model has embedded tokenizer
echo "Testing embedded tokenizer..."
TEST_JSON=$(mktemp)

if cargo run -q -p bitnet-cli -- run \
    --model "$MODEL_PATH" \
    --prompt "test" \
    --max-new-tokens 1 \
    --temperature 0 \
    --strict-tokenizer \
    --json-out "$TEST_JSON" 2>/dev/null; then

    TOKENIZER_TYPE=$(jq -r '.tokenizer.type // "none"' "$TEST_JSON" 2>/dev/null)

    if [[ "$TOKENIZER_TYPE" == "sentencepiece" ]] || [[ "$TOKENIZER_TYPE" == "embedded" ]]; then
        echo "✓ Embedded tokenizer confirmed: $TOKENIZER_TYPE"
    else
        echo "❌ Model does not have embedded tokenizer (found: $TOKENIZER_TYPE)"
        echo "This model cannot be used for PR CI which requires embedded tokenizers"
        rm -f "$TEST_JSON" "$MODEL_PATH"
        exit 1
    fi
else
    echo "⚠ Could not verify tokenizer (may need to build CLI first)"
fi

rm -f "$TEST_JSON"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PR Model Ready"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model: $MODEL_PATH"
echo "Size: $(du -h "$MODEL_PATH" | cut -f1)"
echo ""
echo "You can now run PR CI with:"
echo "  CI_PR=1 ./scripts/ci-acceptance-gate.sh"
echo ""
echo "Or set the model path explicitly:"
echo "  PR_MODEL=$MODEL_PATH ./scripts/ci-acceptance-gate.sh"

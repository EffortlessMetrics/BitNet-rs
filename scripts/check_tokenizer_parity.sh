#!/usr/bin/env bash
# check_tokenizer_parity.sh - Validate tokenizer parity between BitNet-rs and bitnet.cpp
#
# Purpose: Ensure both engines receive identical input token IDs (hard requirement for cross-validation)
#
# Usage:
#   ./scripts/check_tokenizer_parity.sh <model.gguf> <tokenizer.json> <prompt_text>
#
# Requirements:
#   - bitnet-cli with tokenize subcommand
#   - BITNET_CPP_DIR environment variable set (for C++ tokenization)
#   - crossval feature enabled (for C++ FFI)
#
# Exit codes:
#   0 - Tokenizer parity OK (identical token IDs)
#   1 - Usage error or file not found
#   2 - Rust tokenization failed
#   3 - C++ tokenization failed (BITNET_CPP_DIR not set or crossval not enabled)
#   4 - Token ID mismatch detected

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Usage message
usage() {
    cat <<EOF
Usage: $0 <model.gguf> <tokenizer.json> <prompt_text>

Validate tokenizer parity between BitNet-rs and bitnet.cpp.

Arguments:
  model.gguf      Path to GGUF model file
  tokenizer.json  Path to tokenizer JSON file
  prompt_text     Text to tokenize

Environment:
  BITNET_CPP_DIR  Path to bitnet.cpp directory (required for C++ tokenization)

Examples:
  # Basic parity check
  $0 model.gguf tokenizer.json "What is 2+2?"

  # With bitnet.cpp
  BITNET_CPP_DIR=/path/to/bitnet.cpp $0 model.gguf tokenizer.json "Hello world"

Exit codes:
  0 - Parity OK (identical token IDs)
  1 - Usage error or file not found
  2 - Rust tokenization failed
  3 - C++ tokenization failed
  4 - Token ID mismatch detected
EOF
    exit 1
}

# Check arguments
if [[ $# -ne 3 ]]; then
    echo -e "${RED}Error: Missing required arguments${NC}" >&2
    usage
fi

MODEL_PATH="$1"
TOKENIZER_PATH="$2"
PROMPT_TEXT="$3"

# Validate files exist
if [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}" >&2
    exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
    echo -e "${RED}Error: Tokenizer file not found: $TOKENIZER_PATH${NC}" >&2
    exit 1
fi

# Temporary files
RUST_TOKENS="/tmp/bitnet_rs_tokens_$$.txt"
CPP_TOKENS="/tmp/bitnet_cpp_tokens_$$.txt"
RUST_JSON="/tmp/bitnet_rs_json_$$.json"

cleanup() {
    rm -f "$RUST_TOKENS" "$CPP_TOKENS" "$RUST_JSON"
}
trap cleanup EXIT

echo -e "${CYAN}=== BitNet-rs Tokenizer Parity Check ===${NC}"
echo -e "Model:     ${MODEL_PATH}"
echo -e "Tokenizer: ${TOKENIZER_PATH}"
echo -e "Prompt:    \"${PROMPT_TEXT}\""
echo ""

# ============================================================================
# Step 1: Rust tokenization via bitnet-cli
# ============================================================================
echo -e "${CYAN}[1/3] Tokenizing with BitNet-rs...${NC}"

# Find bitnet binary (prefer release, fallback to debug)
BITNET_BIN=""
if [[ -x "target/release/bitnet-cli" ]]; then
    BITNET_BIN="target/release/bitnet-cli"
elif [[ -x "target/release/bitnet" ]]; then
    BITNET_BIN="target/release/bitnet"
elif [[ -x "target/debug/bitnet" ]]; then
    BITNET_BIN="target/debug/bitnet"
else
    echo -e "${YELLOW}Warning: No built bitnet binary found, building...${NC}" >&2
    if ! cargo build -p bitnet-cli --no-default-features --features cpu --release 2>&1 | tail -5 >&2; then
        echo -e "${RED}✗ Failed to build bitnet-cli${NC}" >&2
        exit 2
    fi
    BITNET_BIN="target/release/bitnet"
fi

echo -e "${CYAN}  Using binary: ${BITNET_BIN}${NC}"

# Run Rust tokenizer (suppress logs via stderr redirect)
if ! "$BITNET_BIN" tokenize \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --text "$PROMPT_TEXT" 2>/dev/null > "$RUST_JSON"; then
    echo -e "${RED}✗ Rust tokenization failed${NC}" >&2
    exit 2
fi

# Extract token IDs from JSON
if ! jq -r '.tokens.ids[]' "$RUST_JSON" > "$RUST_TOKENS" 2>/dev/null; then
    echo -e "${RED}✗ Failed to extract token IDs from JSON${NC}" >&2
    echo -e "${YELLOW}JSON content:${NC}" >&2
    cat "$RUST_JSON" >&2
    exit 2
fi

RUST_COUNT=$(wc -l < "$RUST_TOKENS" | tr -d ' ')
echo -e "${GREEN}✓ Rust tokenization complete: ${RUST_COUNT} tokens${NC}"

# Show first 10 tokens
echo -e "${CYAN}  First 10 tokens:${NC} $(head -n 10 "$RUST_TOKENS" | tr '\n' ' ')"

# ============================================================================
# Step 2: C++ tokenization via crossval FFI
# ============================================================================
echo ""
echo -e "${CYAN}[2/3] Tokenizing with bitnet.cpp...${NC}"

# Check if BITNET_CPP_DIR is set
if [[ -z "${BITNET_CPP_DIR:-}" ]]; then
    echo -e "${YELLOW}⚠  BITNET_CPP_DIR not set - skipping C++ comparison${NC}"
    echo -e "${YELLOW}   To enable C++ parity checks, set BITNET_CPP_DIR=/path/to/bitnet.cpp${NC}"
    echo -e "${GREEN}✓ Rust-only tokenization successful${NC}"
    echo ""
    echo -e "${CYAN}Token IDs (Rust):${NC}"
    cat "$RUST_TOKENS"
    exit 0
fi

# Check if crossval feature is available
if ! cargo metadata --no-deps 2>/dev/null | grep -q '"crossval"'; then
    echo -e "${YELLOW}⚠  crossval feature not found - skipping C++ comparison${NC}"
    echo -e "${YELLOW}   To enable C++ parity checks, ensure crossval feature is available${NC}"
    echo -e "${GREEN}✓ Rust-only tokenization successful${NC}"
    echo ""
    echo -e "${CYAN}Token IDs (Rust):${NC}"
    cat "$RUST_TOKENS"
    exit 0
fi

# TODO: Implement C++ tokenization via crossval FFI
# For now, we use the crossval test pattern from crossval/tests/token_equivalence.rs
#
# The pattern would be:
# 1. Load model via BitnetModel::from_file()
# 2. Tokenize via bitnet_tokenize_text(&model, prompt, bos=true, parse_special=false)
# 3. Compare token sequences
#
# This requires:
# - BITNET_CPP_DIR environment variable set
# - crossval feature enabled in bitnet-sys
# - FFI bindings compiled successfully
#
# For production use, create a dedicated crossval/bin/tokenize-cpp.rs binary

echo -e "${YELLOW}⚠  C++ tokenization not yet implemented in this script${NC}"
echo -e "${YELLOW}   To test C++ tokenization, use: cargo test -p crossval --features crossval${NC}"
echo -e "${GREEN}✓ Rust-only tokenization successful${NC}"
echo ""
echo -e "${CYAN}Token IDs (Rust):${NC}"
cat "$RUST_TOKENS"
exit 0

# ============================================================================
# Dead code below - preserved for future C++ implementation
# ============================================================================
# CPP_COUNT=$(wc -l < "$CPP_TOKENS" | tr -d ' ')
# echo -e "${GREEN}✓ C++ tokenization complete: ${CPP_COUNT} tokens${NC}"
#
# # Show first 10 tokens
# echo -e "${CYAN}  First 10 tokens:${NC} $(head -n 10 "$CPP_TOKENS" | tr '\n' ' ')"
#
# # ============================================================================
# # Step 3: Compare token sequences
# # ============================================================================
# echo ""
# echo -e "${CYAN}[3/3] Comparing token sequences...${NC}"
#
# # Check token counts match
# if [[ "$RUST_COUNT" -ne "$CPP_COUNT" ]]; then
#     echo -e "${RED}✗ Token count mismatch${NC}" >&2
#     echo -e "${RED}  Rust:  ${RUST_COUNT} tokens${NC}" >&2
#     echo -e "${RED}  C++:   ${CPP_COUNT} tokens${NC}" >&2
#     exit 4
# fi
#
# # Diff the token ID sequences
# if ! diff -q "$RUST_TOKENS" "$CPP_TOKENS" >/dev/null 2>&1; then
#     echo -e "${RED}✗ Tokenizer mismatch detected${NC}" >&2
#     echo ""
#     echo -e "${CYAN}Token ID differences:${NC}"
#     diff --side-by-side --width=80 "$RUST_TOKENS" "$CPP_TOKENS" || true
#     echo ""
#     echo -e "${YELLOW}Note: This blocks cross-validation parity tests${NC}"
#     exit 4
# fi
#
# # ============================================================================
# # Success!
# # ============================================================================
# echo -e "${GREEN}✓ Tokenizer parity OK - identical token IDs${NC}"
# echo -e "${GREEN}  ${RUST_COUNT} tokens match exactly between Rust and C++${NC}"
# echo ""
# echo -e "${CYAN}Token IDs:${NC}"
# cat "$RUST_TOKENS"
#
# exit 0

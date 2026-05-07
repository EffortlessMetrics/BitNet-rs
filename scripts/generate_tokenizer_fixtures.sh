#!/usr/bin/env bash
# Tokenizer Test Fixture Generator
#
# **Purpose**: Generate test fixtures for TokenizerAuthority E2E tests
#
# **Specification**: docs/specs/tokenizer-authority-validation-tests.md#9
#
# **Fixtures Generated**:
#   - valid_tokenizer_a.json       # Reference tokenizer
#   - valid_tokenizer_b.json       # Byte-identical clone of A
#   - different_vocab_size.json    # Modified tokenizer config
#   - corrupted.json               # Malformed JSON (error handling tests)
#   - README.md                    # Fixture documentation
#
# **Requirements**:
#   - Model downloaded: cargo run -p xtask -- download-model
#   - jq installed (for JSON manipulation)
#
# **Usage**:
#   ./scripts/generate_tokenizer_fixtures.sh

set -euo pipefail

FIXTURES_DIR="${FIXTURES_DIR:-tests/fixtures/tokenizers}"
MODELS_DIR="${MODELS_DIR:-models/microsoft-bitnet-b1.58-2B-4T-gguf}"
SOURCE_TOKENIZER="${SOURCE_TOKENIZER:-$MODELS_DIR/tokenizer.json}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_non_empty_file() {
    local path="$1"
    [[ -s "$path" ]] || fail "Expected non-empty file: $path"
}

echo "==================================================================="
echo "Tokenizer Test Fixture Generator"
echo "==================================================================="
echo ""

if [[ ! -f "$SOURCE_TOKENIZER" ]]; then
    fail "Missing source tokenizer: $SOURCE_TOKENIZER
Run: cargo run -p xtask -- download-model"
fi

if ! command -v jq >/dev/null 2>&1; then
    fail "jq is required to generate different_vocab_size.json"
fi

mkdir -p "$FIXTURES_DIR"

echo "1. Copying reference tokenizer..."
cp "$SOURCE_TOKENIZER" "$FIXTURES_DIR/valid_tokenizer_a.json"

echo "2. Creating byte-identical clone..."
cp "$FIXTURES_DIR/valid_tokenizer_a.json" "$FIXTURES_DIR/valid_tokenizer_b.json"

echo "3. Creating different tokenizer config variant..."
jq '
    if (.added_tokens? | type) == "array" then
        .
    else
        .added_tokens = []
    end
    | .added_tokens += [{
        "id": 999999,
        "content": "<|fake_token_for_test|>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true
    }]
' "$FIXTURES_DIR/valid_tokenizer_a.json" > "$FIXTURES_DIR/different_vocab_size.json"

echo "4. Creating corrupted JSON fixture..."
source_bytes="$(wc -c < "$FIXTURES_DIR/valid_tokenizer_a.json" | tr -d '[:space:]')"
corrupt_bytes=500
if (( source_bytes <= corrupt_bytes )); then
    corrupt_bytes=$((source_bytes / 2))
fi
if (( corrupt_bytes < 1 )); then
    fail "Source tokenizer is too small to create a corrupted fixture"
fi
head -c "$corrupt_bytes" "$FIXTURES_DIR/valid_tokenizer_a.json" > "$FIXTURES_DIR/corrupted.json"

echo "5. Verifying fixtures..."
cat > "$FIXTURES_DIR/README.md" <<'README'
# Tokenizer Test Fixtures

This directory contains tokenizer fixtures for TokenizerAuthority E2E integration tests.

## Fixtures

- `valid_tokenizer_a.json`: Reference tokenizer copied from `models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json`.
- `valid_tokenizer_b.json`: Byte-identical clone of `valid_tokenizer_a.json`.
- `different_vocab_size.json`: Reference tokenizer with one extra special token in `added_tokens`, used to verify config-hash divergence.
- `corrupted.json`: Malformed JSON produced by truncating `valid_tokenizer_a.json`, used for error handling tests.

## Regeneration

```bash
cargo run -p xtask -- download-model
./scripts/generate_tokenizer_fixtures.sh
```

The script requires `jq`. Tests copy these fixtures to temporary directories before use.
README

require_non_empty_file "$FIXTURES_DIR/README.md"
require_non_empty_file "$FIXTURES_DIR/valid_tokenizer_a.json"
require_non_empty_file "$FIXTURES_DIR/valid_tokenizer_b.json"
require_non_empty_file "$FIXTURES_DIR/different_vocab_size.json"
require_non_empty_file "$FIXTURES_DIR/corrupted.json"

cmp -s "$FIXTURES_DIR/valid_tokenizer_a.json" "$FIXTURES_DIR/valid_tokenizer_b.json" \
    || fail "valid_tokenizer_a.json and valid_tokenizer_b.json must be byte-identical"

jq -e . "$FIXTURES_DIR/valid_tokenizer_a.json" >/dev/null
jq -e . "$FIXTURES_DIR/valid_tokenizer_b.json" >/dev/null
jq -e . "$FIXTURES_DIR/different_vocab_size.json" >/dev/null
jq -e --slurpfile reference "$FIXTURES_DIR/valid_tokenizer_a.json" '
    .model == $reference[0].model
    and ((.added_tokens | length) == (($reference[0].added_tokens // []) | length + 1))
    and (.added_tokens[-1].content == "<|fake_token_for_test|>")
' "$FIXTURES_DIR/different_vocab_size.json" >/dev/null \
    || fail "different_vocab_size.json should preserve model data and add one fake token"

if cmp -s "$FIXTURES_DIR/valid_tokenizer_a.json" "$FIXTURES_DIR/different_vocab_size.json"; then
    fail "different_vocab_size.json must differ from valid_tokenizer_a.json"
fi

if jq -e . "$FIXTURES_DIR/corrupted.json" >/dev/null 2>&1; then
    fail "corrupted.json should be malformed JSON"
fi

echo ""
echo "==================================================================="
echo "Fixture generation complete: $FIXTURES_DIR"
echo "Generated 4 tokenizer fixtures + README"
echo "==================================================================="
ls -lh "$FIXTURES_DIR"

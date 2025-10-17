#!/usr/bin/env bash
set -euo pipefail

model="${1:?usage: $0 /path/to/model.gguf}"

export CROSSVAL_GGUF="$(realpath "$model")"
export RAYON_NUM_THREADS=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42
: "${BITNET_CPP_DIR:=$HOME/.cache/bitnet_cpp}"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src:$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src:${LD_LIBRARY_PATH:-}"

echo "== link check =="
ldd target/release/deps/parity_bitnetcpp-* | egrep 'llama|ggml' || true

echo "== run parity (release) =="
cargo test -p bitnet-crossval --release --features crossval,integration-tests \
  --test parity_bitnetcpp -- --nocapture

echo "== latest receipt =="
latest="$(find docs/baselines -name parity-bitnetcpp.json -type f | sort | tail -1)"
jq '{status: .parity.status,
     cpp_available: .parity.cpp_available,
     tokenizer_source: .tokenizer.source,
     cosine: .parity.cosine_similarity,
     exact: .parity.exact_match_rate,
     div: .parity.first_divergence_step}' "$latest"

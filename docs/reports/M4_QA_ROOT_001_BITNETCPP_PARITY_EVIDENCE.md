# M4-QA-ROOT-001 BitNet.cpp Parity Evidence

**Date:** 2026-05-07
**Status:** Local model artifact also garbles under reference execution
**Campaign:** `apple-m4-local-answer`

## Executive Summary

`M4-QA-001` should remain blocked. The strict Apple M4 CPU/NEON route is not silently falling back, and the prompt is tokenized with the real GGUF tokenizer path, but the tested local GGUF does not produce an intelligible answer. The same local model artifact also produces non-coherent output under the local BitNet.cpp/llama.cpp reference runner, which logs a missing GGUF pre-tokenizer metadata warning that generation quality will be degraded.

This means the next user-facing answer work should not weaken output-quality gates or claim Apple CPU/NEON local answers are coherent. The next step is to validate or replace the supported model artifact/tokenizer metadata, then rerun `M4-QA-001`.

## Scope

Tested model:

```text
models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
sha256=4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
repo=microsoft/bitnet-b1.58-2B-4T-gguf
format=gguf
kernel_family=i2_s
```

Tested prompt/settings:

```text
prompt="What is 2+2? Answer briefly."
prompt_template=llama3-chat
temperature=0.0
greedy=true
max_tokens=1 for the strict Rust receipt
max_tokens=16 for the reference output smoke
```

## Tokenizer Path Finding

The top-level `bitnet run` path loads the tokenizer through `bitnet_tokenizers::auto::resolve_tokenizer`, which prefers GGUF metadata and calls `gguf_loader::RustTokenizer::from_gguf`. That is the real GGUF BPE/SPM tokenizer path, not the legacy byte-level `gguf_tokenizer::GgufTokenizer`.

This branch also fixes the generic `.gguf` `load_tokenizer(path)` path to use the same `RustTokenizer`. That matters for cross-validation tooling and explicit tokenizer-path users, because the old generic `.gguf` loader could tokenize LLaMA-3 chat markers as ordinary bytes.

## Rust Evidence

Strict Apple M4 CPU/NEON receipt:

```bash
BITNET_DISABLE_MINIMAL_LOADER=1 \
BITNET_STRICT_MODE=1 \
RUST_LOG=warn \
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- run \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "What is 2+2? Answer briefly." \
  --max-tokens 1 \
  --temperature 0.0 \
  --greedy \
  --device apple-m4-cpu-neon \
  --prompt-template llama3-chat \
  --strict-loader \
  --strict-tokenizer \
  --json-out target/apple-m4-local-answer/M4-QA-ROOT-001/strict-one-token-after-tokenizer-fix.json
```

Relevant receipt fields:

```json
{
  "text": "'E",
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "fallback_used": false,
  "tokens": {
    "prompt": 21,
    "generated": 1,
    "generated_ids": [89048],
    "prompt_ids": [
      128000, 128006, 882, 128007, 198, 198, 3923, 374, 220, 17, 10,
      17, 30, 22559, 27851, 13, 128009, 128006, 78191, 128007, 271
    ]
  },
  "tokenizer": {
    "source": "gguf_metadata",
    "strict": true,
    "type": "llama3"
  },
  "model": {
    "loader_mode": "real_gguf",
    "fallback_loader_used": false
  }
}
```

The prompt token IDs match the LLaMA-3 chat-tokenized prompt, so the first visible failure is not hidden fallback or prompt byte-tokenization.

## Cross-Validation Token Evidence

After fixing the LLaMA fallback wrapper to honor `parse_special=true`, `xtask crossval-per-token` tokenizes the LLaMA-3 chat prompt identically in Rust and llama.cpp:

```bash
BITNET_CPP_DIR=/Users/steven/.cache/bitnet_cpp \
cargo run --locked -p xtask --features crossval-all -- \
  crossval-per-token \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "What is 2+2? Answer briefly." \
  --prompt-template llama3-chat \
  --cpp-backend llama \
  --positions 1 \
  --dump-ids \
  --dump-cpp-ids
```

Observed token parity:

```text
Rust tokens (21 total):
[128000, 128006, 882, 128007, 198, 198, 3923, 374, 220, 17, 10,
 17, 30, 22559, 27851, 13, 128009, 128006, 78191, 128007, 271]

C++ tokens (21 total):
[128000, 128006, 882, 128007, 198, 198, 3923, 374, 220, 17, 10,
 17, 30, 22559, 27851, 13, 128009, 128006, 78191, 128007, 271]
```

The full logits stage is still too slow in the current Rust scalar path for this branch to use as the first blocking signal. The work-item acceptance allows the other branch of evidence: proving the local artifact itself also garbles under reference execution.

## Reference Execution Evidence

Reference run from the local BitNet.cpp checkout:

```bash
cd /Users/steven/.cache/bitnet_cpp
python3 run_inference.py \
  -m /Users/steven/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "What is 2+2? Answer briefly." \
  -n 16 \
  -temp 0
```

Observed output:

```text
What is 2+2? Answer briefly. above thereof scroll in issued@s done foreground departure duplicate set gluterı gradu Pregn
```

The reference loader logs this tokenizer metadata warning for the same file:

```text
llm_load_vocab: missing pre-tokenizer type, using: 'default'
llm_load_vocab: GENERATION QUALITY WILL BE DEGRADED!
llm_load_vocab: CONSIDER REGENERATING THE MODEL
```

The same metadata dump shows tokenizer keys for `tokenizer.ggml.model`, `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.bos_token_id`, and `tokenizer.ggml.eos_token_id`, but no `tokenizer.ggml.pre`.

## Conclusion

For this local artifact, the current local-answer blocker is not Apple M4 backend routing, strict-mode fallback, or prompt byte-tokenization. The tested GGUF also fails the reference path and is missing tokenizer pre-tokenizer metadata that the reference runtime explicitly warns will degrade generation quality.

`M4-QA-001` should stay blocked until the campaign has one of:

1. a known-good supported GGUF/tokenizer artifact that produces coherent output under the reference runner;
2. a regenerated or repaired GGUF with correct tokenizer pre-tokenizer metadata and coherent reference output;
3. a documented alternate supported model for the Apple M4 local-answer smoke suite.

Only after one of those is true should `apple-m4-cpu-neon` claim prompt-in, intelligible-answer-out behavior.

## Follow-Up Recommendation

Add a new prerequisite item before unblocking `M4-QA-001`:

```text
M4-QA-MODEL-001 — Validate supported local-answer model artifact
```

Acceptance should require a reference runner to produce a coherent short answer for the campaign prompt suite, record the model hash and tokenizer metadata, and fail if `tokenizer.ggml.pre` is missing for BPE models that require it.

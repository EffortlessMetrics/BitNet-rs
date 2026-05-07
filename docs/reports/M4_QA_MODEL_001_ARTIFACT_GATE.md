# M4-QA-MODEL-001 Artifact Gate

**Date:** 2026-05-07
**Campaign:** `apple-m4-local-answer`
**Status:** Current supported BitNet GGUF rejected for local-answer quality

## Summary

`M4-QA-MODEL-001` keeps `M4-QA-001` blocked. The current supported local BitNet artifact is structurally valid and has the expected SHA256, but it fails the Apple M4 local-answer prompt suite under the reference runner. This confirms the blocker is model artifact quality, not Apple M4 CPU/NEON routing.

The artifact must not be used to claim prompt-in, coherent-answer-out behavior. The next required step is `M4-QA-MODEL-002`: acquire or regenerate a supported GGUF/tokenizer artifact that passes the same reference-runner prompt suite.

## Tested Artifact

```text
repo=microsoft/bitnet-b1.58-2B-4T-gguf
repo_revision=a1f2f1c765812aa8af3f6eda4a313707064bba15
file=ggml-model-i2_s.gguf
local_path=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
sha256=4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
bytes=1187801280
architecture=bitnet-b1.58
quantization=i2_s
```

Structural checks pass:

```text
GGUF version: 3
tensors: 332
KV pairs: 24
vocab_size: 128256
hidden_size: 2560
layers: 30
```

This structural validity does not imply answer quality.

## Metadata Gate

The reference runner reports:

```text
llm_load_vocab: missing pre-tokenizer type, using: 'default'
llm_load_vocab: GENERATION QUALITY WILL BE DEGRADED!
llm_load_vocab: CONSIDER REGENERATING THE MODEL
```

The GGUF metadata contains `tokenizer.ggml.model`, `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, special-token IDs, and a chat template, but it does not provide the required pre-tokenizer authority for this local-answer gate.

## Reference Prompt Suite

Command shape:

```bash
cd /Users/steven/.cache/bitnet_cpp
build/bin/llama-cli \
  -m /Users/steven/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -n 16 \
  -t 8 \
  -p "<prompt>" \
  -ngl 0 \
  -c 2048 \
  --temp 0 \
  -b 1
```

Results:

| Case | Expected | Reference continuation | Result |
|---|---|---|---|
| `math_sentence` | contains `4` or `four` | `rewardededvest ************************************************************************ makeover glide answered bioned transitions byteesuitonet worse tit` | fail |
| `capital_france_sentence` | contains `Paris` | `FPDMAkara SAYmos small gy band adopt[emoji] standing direct passaidu markedifu` | fail |
| `rust_sentence` | contains Rust/programming terms | `v start,onlinecons f front make HQ [invalid-byte] alone bidsredo earned fractionsla` | fail |

All three runs exited successfully, produced non-empty text, and failed semantic quality. All three also carried the missing pre-tokenizer metadata warning.

## Decision

The current artifact is rejected for Apple M4 local-answer claims:

```text
status=rejected_for_local_answer
reason=structurally valid GGUF but failed reference prompt-suite quality and lacks required pre-tokenizer metadata
```

`M4-QA-001` remains blocked. Do not weaken the answer-corpus gate to pass this artifact.

## Next Step

`M4-QA-MODEL-002` must acquire or regenerate a supported local-answer artifact that satisfies:

```text
reference runner produces coherent short answers for the campaign prompt suite
exact SHA256 recorded
GGUF architecture and quantization recorded
tokenizer metadata recorded
pre-tokenizer authority present, or an explicit compatibility decision is documented
bad artifacts rejected rather than worked around
```

# Apple M3 MacBook Air Microsoft 2B I2_S

Date: 2026-05-14
Work item: `M3MBA-005A`

## Result

The official Microsoft BitNet 2B I2_S GGUF identity was recorded on the Apple M3 MacBook Air lane.

Evidence receipt: `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-i2s-identity.json`

This is identity, hash, cache, and storage evidence only. It does not accept the artifact for Apple answer behavior.

## Artifact

- Repository: `microsoft/bitnet-b1.58-2B-4T-gguf`
- Source: `https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf`
- Revision: `a1f2f1c765812aa8af3f6eda4a313707064bba15`
- Filename: `ggml-model-i2_s.gguf`
- Format: GGUF
- Architecture: `bitnet-b1.58`
- Quantization: `i2_s`
- Size: 1,187,801,280 bytes
- SHA-256: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- Header linked ETag: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`

The downloaded file size and local SHA-256 match the shared Apple candidate matrix and shared model artifact manifest.

## Cache And Storage

- Cache root: `/Users/sarahisaacs/Library/Caches/bitnet-rs/models`
- Local model path: `/Users/sarahisaacs/Library/Caches/bitnet-rs/models/microsoft-bitnet-2b-i2s/ggml-model-i2_s.gguf`
- Storage before download: 60,456,036 KiB available on `/System/Volumes/Data`
- Storage after hash verification: 59,294,360 KiB available on `/System/Volumes/Data`
- Hard free-space floor: 8,388,608 KiB
- Preferred free-space floor: 26,214,400 KiB
- Retention: retained for `M3MBA-005B` tokenizer authority and `M3MBA-005C` reference output decision
- Model binaries committed: false

Power was AC with the internal battery at 100% charged. `pmset -g therm` recorded no thermal, performance, or CPU power warning level.

## Authority References

- Shared answer gate: `docs/model-artifacts/ANSWER_ARTIFACT_GATE.md`
- Shared artifact manifest: `ci/model-artifacts/artifact-manifest.toml`
- Tokenizer authority ledger: `ci/model-artifacts/tokenizer-authority.toml`
- Model/kernel compatibility ledger: `ci/model-artifacts/model-kernel-compatibility.toml`
- Apple MacBook candidate matrix: `ci/hardware/apple-silicon-macbook/bitnet-candidate-matrix.toml`

The compatibility ledger records the Microsoft model as the official model family and records ARM `i2_s` as supported. The shared answer gate still requires external tokenizer/pre-tokenizer authority and reference prompt output before a backend lane can make local-answer claims.

## Claim Boundary

This report records official Microsoft 2B I2_S artifact identity for the M3 Air context only. It does not claim the artifact is accepted for Apple local answers, Rust M4 BitNet local answers, M4 Mac mini proof, Apple Metal BitNet inference, MPSGraph model inference, Neural Engine execution, QK256 on Apple Silicon, speedup, or broad Apple Silicon performance.

Dense Qwen M3 evidence does not prove BitNet behavior.

## Next

`M3MBA-005B` records tokenizer and pre-tokenizer authority for this cached artifact. `M3MBA-005C` records the reference output decision and is the first item in this sequence allowed to accept, reject, or block the candidate for the M3 Air reference context.

<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-REG-003 | #5030 | `codex/lunar-lake/LNL258V-REG-003-strict-v2` | Make strict Lunar Lake regression fail closed on the v2 quality/profile surface by requiring corpus v2 and route-profile comparison coverage, recording a strict regression surface summary, rejecting fallback or candidate GPU/NPU promotion drift, and preserving the no-new-inference, no-quality-claim, no-speedup, no-route-promotion, and no BitNet QK256/I2_S behavior-change boundary. |
| nvidia-5070ti | CUDA-MODEL-SMOLLM2-001 | #5029 | `model/cuda-smollm2-360m-artifact-contract` | Add an exact SmolLM2 360M artifact contract with source, file identity, SHA256, byte size, GGUF metadata, quantization, tokenizer, chat template, context length, license, storage envelope, and VRAM estimate where available, while keeping CPU answer, CUDA, product CLI, speedup, server, full-residency, broad dense GGUF, and BitNet QK256 claims false. |

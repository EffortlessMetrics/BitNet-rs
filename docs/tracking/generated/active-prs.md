<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-MODEL-006 | #4985 | `model/cuda-qwen3-0_6b-warm-session` | Add Qwen3 0.6B warm-session strict CUDA proof with model/tokenizer/CUDA context loaded once, runtime buffers reused where intended, weights_uploaded_once=true, per_turn_weight_upload=false, fallback_used=false for every turn, and no speed/server/full-residency/BitNet QK256 claim. |

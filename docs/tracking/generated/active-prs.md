<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-DENSE-REF-001 | #5653 | `codex/apple-m4-inference-excellence/M4-DENSE-REF-001-reference-vs-rust` | Add a bounded dense SLM reference-runner vs Rust M4 comparison for supported Qwen model identities, recording prompt template, token IDs where available, generated text, mechanical scores, and deltas so Rust/template/tokenizer regressions can be separated from model behavior. |
| intel-npu | NPU-012 | #5634 | `codex/intel-npu/NPU-012-source-of-truth-map` | Add the NPU source-of-truth map and implementation plan that make current NPU-002 through NPU-011 evidence, claim boundaries, Intel Lunar Lake/OpenVINO target scope, and future Apple/Qualcomm/AMD NPU family separation visible without runtime claims or route promotion. |
| nvidia-5070ti | CUDA-MODEL-011 | #5645 | `codex/cuda-model-011-qwen3-chat-user-path` | Record Qwen3 0.6B Q8_0 through the normal bitnet chat user path with model/tokenizer/CUDA context/weights loaded once across multiple prompts, selected_backend=nvidia-rtx-5070-ti-cuda, selected_route=dense_regular_llm_cuda, fallback_used=false, quality gate evidence, and no product, server, speedup, full-residency, or BitNet QK256 promotion. |

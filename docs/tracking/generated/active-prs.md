<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-MODEL-003 | #4903 | `model/cuda-qwen3-0_6b-all-layer-plan` | Add the Qwen3 0.6B CUDA all-layer plan with routed ops, unsupported ops, model-boundary gaps, and claim boundaries before any one-token CUDA proof. |
| slm-cpu | SLM-CPU-016 | #4918 | `codex/slm-cpu-016-kaby-operator-profile` | Add the Kaby Lake SLM operator appliance profile after SLM-CPU-015 by documenting and, where the command surface supports it, emitting a single strict Qwen3-0.6B Q8_0 operator receipt bundle that uses the selected thread envelope, records model SHA, GGUF tokenizer authority, prompt IDs, generated IDs, selected CPU backend/kernel identity, fallback_used=false, warm-session timing, resident memory, power, thermal, storage/free-space, and unsupported-path fields as measured or explicitly unavailable. The slice must preserve the SLM-CPU-009/010/011/015 generated-ID behavior oracle, keep any timing claim bounded to the recorded i5-8250U host/model/corpus/backend/thread context, and must not claim sustained throughput, Q4/Q5 quant expansion, a second model, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |

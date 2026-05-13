<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | SLM-OV258V-003 | #4584 | `codex/lunar-lake/SLM-OV258V-003-openvino-gpu-smoke` | Record the Qwen2.5 OpenVINO GPU LLMPipeline bounded smoke on Arc 140V now that the local OpenVINO GenAI runtime and INT4 symmetric IR export are available, recording selected_backend=openvino-gpu, runtime_api=openvino_genai, runtime_device=GPU.0, resolved Arc 140V identity, fallback_used=false, bounded answer gates, export file identity, and claim boundaries without claiming OpenVINO NPU execution, native OpenCL proof, GPU speedup or sustained phase performance, broad dense SLM quality, generated token IDs from GenAI internals, or BitNet QK256/I2_S proof. |
| nvidia-5070ti | CUDA-DENSE-050 | #4589 | `docs/cuda-dense-qwen25-product-audit` | Audit Qwen2.5 0.5B Q8_0 dense CUDA receipts to distinguish real hardware/user-path evidence from validators and contracts before new dense runtime work. |

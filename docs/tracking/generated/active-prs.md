<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-047 | #4492 | `codex/nvidia-5070ti/CUDA-DENSE-047-qwen15-all-layer-plan` | Record the first RTX 5070 Ti strict CUDA route audit for the verified Qwen2.5 1.5B Q4_K_M dense GGUF artifact after CUDA-DENSE-046, proving the artifact fetch and model verify gates pass, the 5070 Ti backend is visible, and the existing dense-gguf-all-layer-plan command fails closed with strict_cuda_ready=false instead of inheriting Qwen2.5 0.5B Q8_0 CUDA proof; update model coverage to require Q4_K_M CUDA route support or an explicit unsupported-route receipt before any larger-Qwen CUDA answer claim. |

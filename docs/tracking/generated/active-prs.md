<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-STATUS-001 | #5158 | `docs/cuda-capability-matrix` | Add a user-facing CUDA capability matrix status page for the 9950X3D + RTX 5070 Ti lane, sourced from the model coverage matrix and NVIDIA campaign proof ledger. The page must distinguish official BitNet I2_S/QK256, dense Qwen2.5, Qwen3, SmolLM2, and later dense candidates; preserve speedup_claim=false and server_ready=false where the model coverage row says false; and avoid creating any new model, runtime, CUDA, answer, speed, server, full-residency, or publication claim. |

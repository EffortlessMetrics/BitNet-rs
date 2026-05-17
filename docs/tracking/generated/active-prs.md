<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-GPU-QUAL-001 | #5327 | `codex/lunar-lake/LNL258V-GPU-QUAL-001-openvino-gpu-diagnosis` | Add a no-new-inference Lunar Lake OpenVINO GPU corpus-v2 diagnosis artifact that reads the existing OpenVINO CPU/GPU/NPU corpus-v2 receipt, selects runtime_device=GPU.0, classifies failed GPU cases by profile/category/failure class, records direct-versus-retokenized generated-token visibility, explains why the OpenVINO GPU route remains blocked for promotion, and preserves no route-promotion, no speedup, no power-advantage, no Arc acceleration, no NPU claim, and no BitNet QK256/I2_S behavior-change boundaries. |

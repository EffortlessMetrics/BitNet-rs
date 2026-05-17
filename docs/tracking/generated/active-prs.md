<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-NPU-QUAL-001 | #5335 | `codex/lunar-lake/LNL258V-NPU-QUAL-001-openvino-npu-diagnosis` | Add a no-new-inference Lunar Lake OpenVINO NPU corpus-v2 diagnosis artifact that reads the existing OpenVINO CPU/GPU/NPU corpus-v2 receipt, selects runtime_device=NPU, classifies failed NPU cases by profile/category/failure class, records direct-versus-retokenized generated-token visibility, explains why the OpenVINO NPU route remains blocked for promotion, and preserves no route-promotion, no speedup, no power-advantage, no NPU acceleration, no native NPU inference outside OpenVINO GenAI, no dynamic decode/beam/parallel sampling, and no BitNet QK256/I2_S behavior-change boundaries. |

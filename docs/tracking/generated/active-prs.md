<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-PROFILE-RUN-002 | #5803 | `codex/lunar-lake/LNL258V-PROFILE-RUN-002-validator` | Teach the Lunar Lake OpenVINO receipt validator to accept the explicit profile-run artifact from LNL258V-PROFILE-RUN-001, normalize that artifact to use runtime_api=openvino_genai with LLMPipeline identity recorded separately, add validator coverage for GPU/NPU profile-run cases with direct generated token IDs, validate the committed profile-run receipt, and preserve no route-promotion, no speedup, no power-advantage, no native accelerator, and no BitNet QK256/I2_S behavior-change claims. |

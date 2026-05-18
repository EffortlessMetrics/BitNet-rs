//! Receipt schema and artifact-kind constants.
//!
//! Kept separate from validators so provenance labels and receipt taxonomy can
//! evolve independently from validation logic.

pub const RECEIPT_SCHEMA_VERSION: &str = "1.0.0";

/// Alias for schema version (for consistency)
pub const RECEIPT_SCHEMA: &str = RECEIPT_SCHEMA_VERSION;

/// Reusable Apple M4 run identity contract for receipt families that need
/// matching-history comparison across eval, benchmark, warm, chat, serve, and
/// dashboard evidence.
pub const M4_RUN_IDENTITY_CONTRACT_VERSION: &str = "m4-run-identity-v1";

/// Artifact kind for the dense regular-LLM CUDA reference lane.
///
/// This is deliberately separate from BitNet packed I2_S/QK256 CUDA receipt
/// kinds. Dense CUDA evidence may share CUDA runtime plumbing, but it must not
/// satisfy BitNet packed-kernel proof gates.
pub const DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND: &str = "dense_regular_llm_cuda";

/// Artifact kind for descriptor-only dense GGUF tensor inspection.
///
/// This is not a CUDA execution receipt. It records that a dense GGUF reader
/// path can classify model tensor roles without claiming dense inference,
/// speedup, full residency, or BitNet packed-kernel proof.
pub const DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND: &str =
    "dense_gguf_tensor_descriptor_inspection";

/// Artifact kind for dense GGUF linear fixture extraction.
///
/// This remains below dense GGUF CUDA execution. It records that one recognized
/// dense GGUF linear tensor can be materialized as F32 and evaluated by a CPU
/// reference matvec, but it must not claim dense inference, CUDA parity,
/// speedup, full residency, or BitNet packed-kernel proof.
pub const DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND: &str = "dense_gguf_linear_fixture_extraction";

/// Artifact kind for dense GGUF norm fixture extraction.
///
/// This is a CPU-reference fixture for RMSNorm weights extracted from a dense
/// GGUF artifact. It is below CUDA parity and dense GGUF inference.
pub const DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND: &str = "dense_gguf_norm_fixture_extraction";

/// Artifact kind for dense GGUF RMSNorm CUDA parity.
///
/// This receipt proves descriptor-extracted dense GGUF norm fixtures can run
/// through the dense F32 CUDA RMSNorm path and match CPU references. It is
/// still fixture-level evidence, not dense GGUF inference.
pub const DENSE_GGUF_NORM_CUDA_PARITY_ARTIFACT_KIND: &str = "dense_gguf_norm_cuda_parity";

/// Artifact kind for dense GGUF RoPE CUDA parity.
///
/// This receipt proves metadata-derived dense GGUF Q/K RoPE fixtures can run
/// through the dense F32 CUDA RoPE path and match CPU references. It is still
/// fixture-level evidence, not dense GGUF inference.
pub const DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND: &str = "dense_gguf_rope_cuda_parity";

/// Artifact kind for dense GGUF attention-score fixture extraction.
///
/// This receipt records a CPU-reference attention score fixture derived from
/// metadata-based RoPE Q/K outputs. It is below CUDA parity and dense GGUF
/// inference. It does not by itself promote planner routing.
pub const DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND: &str =
    "dense_gguf_attention_score_fixture_extraction";

/// Artifact kind for dense GGUF attention-score CUDA parity.
///
/// This receipt proves metadata-derived dense GGUF Q/K score fixtures can run
/// through a strict F32 CUDA attention-score kernel and match CPU references.
/// It remains fixture-level evidence, not dense GGUF inference.
pub const DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND: &str =
    "dense_gguf_attention_score_cuda_parity";

/// Artifact kind for dense GGUF attention-softmax fixture extraction.
///
/// This receipt records CPU-reference softmax probabilities derived from the
/// metadata-based attention-score fixture. It is below CUDA parity and dense
/// GGUF inference.
pub const DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND: &str =
    "dense_gguf_attention_softmax_fixture_extraction";

/// Artifact kind for dense GGUF attention-softmax CUDA parity.
///
/// This receipt proves the metadata-derived attention-softmax fixture can run
/// through a strict CUDA F32 softmax kernel. It is below dense GGUF inference.
pub const DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND: &str =
    "dense_gguf_attention_softmax_cuda_parity";

/// Artifact kind for dense GGUF attention V-mix fixture extraction.
///
/// This receipt records CPU-reference context vectors derived from verified
/// attention-softmax probabilities and a deterministic attention-V fixture. It
/// is below CUDA parity and dense GGUF inference.
pub const DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND: &str =
    "dense_gguf_attention_v_mix_fixture_extraction";

/// Artifact kind for dense GGUF attention V-mix CUDA parity.
///
/// This receipt proves the metadata-derived attention V-mix fixture can run
/// through a strict CUDA F32 V-mix kernel. It is below dense GGUF inference and
/// does not by itself promote one-layer planner routing.
pub const DENSE_GGUF_ATTENTION_V_MIX_CUDA_PARITY_ARTIFACT_KIND: &str =
    "dense_gguf_attention_v_mix_cuda_parity";

/// Artifact kind for dense GGUF MLP activation fixture extraction.
///
/// This receipt records CPU-reference SiLU(gate) * up activation values derived
/// from verified dense GGUF MLP gate/up fixture outputs. It is below CUDA
/// parity and dense GGUF inference.
pub const DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND: &str =
    "dense_gguf_mlp_activation_fixture_extraction";

/// Artifact kind for dense GGUF MLP activation CUDA parity.
///
/// This receipt proves the metadata-derived SiLU(gate) * up activation fixture
/// can run through a strict CUDA F32 activation kernel. It is below dense GGUF
/// inference and does not by itself promote one-layer planner routing.
pub const DENSE_GGUF_MLP_ACTIVATION_CUDA_PARITY_ARTIFACT_KIND: &str =
    "dense_gguf_mlp_activation_cuda_parity";

/// Artifact kind for dense GGUF single-linear CUDA parity.
///
/// This receipt proves one descriptor-extracted dense GGUF linear fixture can
/// be routed through the dense FP16 CUDA GEMM path and compared against the
/// bridge CPU reference. It is not full dense GGUF inference.
pub const DENSE_GGUF_LINEAR_CUDA_PARITY_ARTIFACT_KIND: &str = "dense_gguf_linear_cuda_parity";

/// Artifact kind for dense GGUF linear role-sweep CUDA parity.
///
/// This receipt proves multiple descriptor-extracted dense GGUF linear fixtures
/// can be routed through the dense FP16 CUDA GEMM path in one model-aware
/// planner receipt. It is still not full dense GGUF inference.
pub const DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND: &str =
    "dense_gguf_linear_role_sweep_cuda_parity";

/// Artifact kind for dense GGUF one-layer execution-plan gap receipts.
///
/// This receipt proves planner routing and fail-closed strict CUDA behavior for
/// one dense transformer layer. It does not execute full dense GGUF inference.
pub const DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND: &str =
    "dense_gguf_one_layer_execution_plan";

/// Artifact kind for dense GGUF one-layer CPU reference harness receipts.
///
/// This receipt records a deterministic CPU-only layer-0 reference output for
/// the dense regular-LLM lane. It is the comparison anchor for later integrated
/// CUDA layer parity, not dense GGUF inference or CUDA execution.
pub const DENSE_GGUF_ONE_LAYER_CPU_REFERENCE_ARTIFACT_KIND: &str =
    "dense_gguf_one_layer_cpu_reference";

/// Artifact kind for integrated dense GGUF one-layer CUDA parity receipts.
///
/// This receipt runs the full governed layer-0 CUDA-routable plan against the
/// CPU reference harness. It proves one-layer CUDA parity only; it is not dense
/// GGUF inference, token generation, speedup, persistent residency, full CUDA
/// residency, or BitNet packed I2_S/QK256 proof.
pub const DENSE_GGUF_ONE_LAYER_CUDA_INTEGRATED_PARITY_ARTIFACT_KIND: &str =
    "dense_gguf_one_layer_cuda_integrated_parity";

/// Artifact kind for dense GGUF all-layer execution-plan receipts.
///
/// This receipt inspects the whole transformer-block stack and records whether
/// each layer matches the governed dense CUDA layer plan. It is not dense GGUF
/// inference, token generation, speedup, persistent residency, full CUDA
/// residency, or BitNet packed I2_S/QK256 proof.
pub const DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND: &str =
    "dense_gguf_all_layer_execution_plan";

/// Artifact kind for dense GGUF model-boundary fixture receipts.
///
/// This receipt records token embedding lookup, final model norm, LM head, and
/// logits diagnostics after the transformer-block plan is route-complete. It
/// is not Qwen one-token inference, sampling, KV cache policy, speedup, full
/// CUDA residency, or BitNet packed I2_S/QK256 proof.
pub const DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND: &str =
    "dense_gguf_model_boundary_fixtures";

/// Artifact kind for dense GGUF KV-cache policy receipts.
///
/// This receipt records the governed KV-cache shape, planned residency, and
/// byte estimates needed before Qwen one-token CUDA proof. It is not KV-cache
/// allocation, token generation, speedup, full CUDA residency, or BitNet
/// packed I2_S/QK256 proof.
pub const DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND: &str = "dense_gguf_kv_cache_policy";

/// Artifact kind for dense GGUF sampling-policy receipts.
///
/// This receipt records the governed logits-transfer and deterministic sampler
/// policy needed before Qwen one-token CUDA proof. It is not token generation,
/// runtime sampling integration, speedup, full CUDA residency, or BitNet packed
/// I2_S/QK256 proof.
pub const DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND: &str = "dense_gguf_sampling_policy";

/// Artifact kind for strict dense Qwen one-token CUDA proof receipts.
///
/// This is the first dense GGUF token-generation proof gate. It must consume
/// the governed all-layer plan, model-boundary fixtures, KV-cache policy, and
/// sampling policy receipts, compare CPU and CUDA selected-token evidence, and
/// keep short-decode, chat, speedup, full-residency, server, and BitNet packed
/// I2_S/QK256 proof claims false.
pub const DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND: &str =
    "dense_gguf_qwen_one_token_strict_cuda_proof";
/// Artifact kind for the governed dense Qwen short-decode strict CUDA proof.
///
/// This is a bounded 5-16 token proof layered after the one-token proof. It
/// must keep chat, speedup, server, full-residency, and BitNet packed I2_S/QK256
/// proof claims false.
pub const DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND: &str =
    "dense_gguf_qwen_short_decode_strict_cuda_proof";
/// Artifact kind for the governed dense Qwen warm-session strict CUDA proof.
///
/// This is a bounded multi-turn proof layered after the short-decode proof. It
/// may claim scoped warm-session reuse, but must keep ask/chat, speedup, server,
/// full-residency, and BitNet packed I2_S/QK256 proof claims false.
pub const DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND: &str =
    "dense_gguf_qwen_warm_session_strict_cuda_proof";
/// Artifact kind for the governed dense Qwen CUDA ask UX receipt.
///
/// This wraps the bounded short-decode and warm-session proof boundary into the
/// user-facing `bitnet ask --device cuda` path. It may claim the scoped ask UX
/// path, but must keep chat, server, speedup, full-residency, and BitNet packed
/// I2_S/QK256 proof claims false.
pub const DENSE_GGUF_QWEN_ASK_STRICT_CUDA_PROOF_ARTIFACT_KIND: &str =
    "dense_gguf_qwen_ask_strict_cuda_proof";
/// Artifact kind for the governed dense Qwen CUDA chat UX receipt.
///
/// This wraps the bounded warm-session proof boundary into the user-facing
/// `bitnet chat --device cuda` path. It may claim the scoped chat UX path, but
/// must keep server, speedup, full-residency, broad dense GGUF inference, and
/// BitNet packed I2_S/QK256 proof claims false.
pub const DENSE_GGUF_QWEN_CHAT_STRICT_CUDA_PROOF_ARTIFACT_KIND: &str =
    "dense_gguf_qwen_chat_strict_cuda_proof";

/// Receipt kind for server shared-engine chat-completion receipts.
///
/// This receipt kind is used by the server path, not the CLI ask/chat path.
/// It must keep server readiness false until an exact-profile promotion PR
/// updates the model coverage matrix.
pub const SERVER_SHARED_ENGINE_CHAT_COMPLETION_RECEIPT_KIND: &str =
    "server_shared_engine_chat_completion";
pub(crate) const QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID: &str = "qwen2.5-0.5b-instruct-q8_0";
pub(crate) const QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE: &str = "qwen2.5-0.5b-instruct-q8_0.gguf";
pub(crate) const QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256: &str =
    "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
pub(crate) const BITNET_B158_2B_4T_I2S_MODEL_ID: &str = "microsoft-bitnet-b1.58-2B-4T-i2s";
pub(crate) const BITNET_B158_2B_4T_I2S_MODEL_FILE: &str = "ggml-model-i2_s.gguf";
pub(crate) const BITNET_B158_2B_4T_I2S_MODEL_SHA256: &str =
    "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162";
pub(crate) const QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID: &str = "qwen3-0.6b-instruct-q8_0";
pub(crate) const QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE: &str = "Qwen3-0.6B-Q8_0.gguf";
pub(crate) const QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256: &str =
    "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031";
pub(crate) const DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER: &[&str] =
    &["attention_softmax", "attention_v_mix", "mlp_activation"];
pub(crate) const DENSE_ONE_LAYER_ATTENTION_V_MIX_FIXTURE_GAP_CANDIDATE_ORDER: &[&str] =
    &["attention_v_mix", "mlp_activation"];
pub(crate) const DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER: &[&str] = &["mlp_activation"];
pub(crate) const DENSE_ONE_LAYER_NO_REMAINING_GAP_CANDIDATE_ORDER: &[&str] = &[];

/// Model class label for CUDA receipts that exercise dense regular LLM kernels.
pub const DENSE_REGULAR_LLM_MODEL_CLASS: &str = "dense_regular_llm";

/// Planner receipt schema version currently emitted by CUDA execution receipts.
pub const CUDA_PLANNER_RECEIPT_VERSION: &str = "cuda-planner-004";

//! CLI command implementations

pub mod answer_corpus;
pub mod answer_parity;
#[cfg(feature = "cli-bench")]
pub mod benchmark;
pub mod chat;
pub mod convert;
pub mod dense_gguf_linear_parity;
#[allow(dead_code)]
pub mod eval;
pub mod external_reference_instrumentation;
pub mod first_token_divergence;
pub mod inference;
pub mod inspect;
pub mod lunar_lake;
pub mod output_head_logits_audit;
pub mod receipts;
pub mod reference_compare;
pub mod serve;
pub mod support;
pub mod template_util;
pub mod transformer_layer_parity;

pub use answer_corpus::AnswerCorpusCommand;
pub use answer_parity::AnswerParityCommand;
#[cfg(feature = "cli-bench")]
pub use benchmark::BenchmarkCommand;
pub use convert::ConvertCommand;
pub use dense_gguf_linear_parity::{
    DenseGgufAllLayerPlanCommand, DenseGgufAttentionScoreCudaParityCommand,
    DenseGgufAttentionScoreFixtureCommand, DenseGgufAttentionSoftmaxCudaParityCommand,
    DenseGgufAttentionSoftmaxFixtureCommand, DenseGgufAttentionVMixCudaParityCommand,
    DenseGgufAttentionVMixFixtureCommand, DenseGgufKvCachePolicyCommand,
    DenseGgufLinearParityCommand, DenseGgufLinearRoleSweepCommand,
    DenseGgufMlpActivationCudaParityCommand, DenseGgufMlpActivationFixtureCommand,
    DenseGgufModelBoundaryFixturesCommand, DenseGgufNormCudaParityCommand,
    DenseGgufNormFixtureCommand, DenseGgufOneLayerCpuReferenceCommand,
    DenseGgufOneLayerCudaParityCommand, DenseGgufOneLayerPlanCommand,
    DenseGgufQwenOneTokenStrictCudaCommand, DenseGgufQwenShortDecodeStrictCudaCommand,
    DenseGgufQwenWarmDecodeStrictCudaCommand, DenseGgufQwenWarmSessionStrictCudaCommand,
    DenseGgufRopeCudaParityCommand, DenseGgufSamplingPolicyCommand, DenseQwenCudaAskOptions,
    run_dense_qwen_cuda_ask,
};
pub use external_reference_instrumentation::ExternalReferenceInstrumentationCommand;
pub use first_token_divergence::FirstTokenDivergenceCommand;
pub use inference::InferenceCommand;
pub use inspect::InspectCommand;
pub use lunar_lake::{LunarLakeAction, LunarLakeCommand};
pub use output_head_logits_audit::OutputHeadLogitsAuditCommand;
pub use receipts::ReceiptsCommand;
pub use reference_compare::ReferenceCompareCommand;
pub use serve::ServeCommand;
pub use support::SupportCommand;
pub use transformer_layer_parity::TransformerLayerParityCommand;

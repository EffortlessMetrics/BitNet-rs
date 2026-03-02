#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::opencl_pipeline::{InferencePipeline, PipelineConfig, PipelineStage};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PipelineInput {
    num_layers: u8,
    hidden_dim_factor: u8,
    num_heads: u8,
    intermediate_factor: u8,
    vocab_size_factor: u8,
    max_seq_len: u8,
    use_gpu: bool,
    fallback_to_cpu: bool,
    input_ids: Vec<u8>,
    position_byte: u8,
    num_executions: u8,
}

fuzz_target!(|input: PipelineInput| {
    let num_layers = (input.num_layers as usize % 8) + 1;
    let num_heads = (input.num_heads as usize % 8) + 1;
    // head_dim must be > 0; we pick small values to keep it fast.
    let head_dim = (input.hidden_dim_factor as usize % 8) + 4;
    let hidden_dim = num_heads * head_dim;
    let intermediate_dim = ((input.intermediate_factor as usize % 8) + 1) * hidden_dim;
    let vocab_size = ((input.vocab_size_factor as usize % 16) + 1) * 64;
    let max_seq_len = (input.max_seq_len as usize % 128) + 1;

    let config = PipelineConfig {
        num_layers,
        hidden_dim,
        num_heads,
        head_dim,
        intermediate_dim,
        vocab_size,
        max_seq_len,
        use_gpu: input.use_gpu,
        fallback_to_cpu: input.fallback_to_cpu,
    };

    // Validation must not panic.
    if config.validate().is_err() {
        return;
    }

    let mut pipeline = match InferencePipeline::new(config.clone()) {
        Ok(p) => p,
        Err(_) => return,
    };

    // Verify stage_order and stages_per_token are consistent.
    let expected_stages = 1 + num_layers * 4 + 2;
    assert_eq!(pipeline.stages_per_token(), expected_stages, "stages_per_token mismatch");
    assert_eq!(pipeline.stage_order().len(), expected_stages, "stage_order length mismatch");

    // Verify stage order structure: Embedding, then per-layer (RmsNorm, Attention,
    // RmsNorm, FeedForward), then FinalNorm, LogitProjection.
    let order = pipeline.stage_order();
    assert_eq!(order[0], PipelineStage::Embedding);
    for layer in 0..num_layers {
        let base = 1 + layer * 4;
        assert_eq!(order[base], PipelineStage::RmsNorm);
        assert_eq!(order[base + 1], PipelineStage::Attention);
        assert_eq!(order[base + 2], PipelineStage::RmsNorm);
        assert_eq!(order[base + 3], PipelineStage::FeedForward);
    }
    assert_eq!(order[order.len() - 2], PipelineStage::FinalNorm);
    assert_eq!(order[order.len() - 1], PipelineStage::LogitProjection);

    // Build input_ids from fuzz data.
    let ids: Vec<u32> =
        input.input_ids.iter().take(8).map(|&b| b as u32 % vocab_size as u32).collect();

    if ids.is_empty() {
        // Empty input_ids should produce an error, not a panic.
        let result = pipeline.execute_single_token_cpu(&[], 0);
        assert!(result.is_err());
        return;
    }

    let position = input.position_byte as usize % max_seq_len;

    // Execute must not panic.
    match pipeline.execute_single_token_cpu(&ids, position) {
        Ok(execution) => {
            // Verify execution results are internally consistent.
            assert_eq!(execution.stages.len(), expected_stages);
            assert_eq!(execution.tokens_generated, 1);

            // Total time must be non-negative.
            assert!(execution.total_time_ns > 0 || execution.stages.is_empty());

            // GPU utilization must be in [0.0, 1.0].
            let util = execution.gpu_utilization();
            assert!((0.0..=1.0).contains(&util), "gpu_utilization out of range: {util}");

            // Slowest stage must not panic.
            if let Some(slowest) = execution.slowest_stage() {
                assert!(slowest.execution_time_ns <= execution.total_time_ns);
            }

            // Summary string must not panic.
            let _ = execution.summary();

            // Total fallbacks must not exceed total stages.
            assert!(execution.total_fallbacks() <= execution.stages.len());

            // Verify output shapes are plausible.
            for stage in &execution.stages {
                assert!(!stage.output_shape.is_empty(), "empty output_shape");
                for &s in &stage.output_shape {
                    assert!(s > 0, "zero dimension in output_shape");
                }
            }
        }
        Err(_) => {
            // Errors are acceptable; the important thing is no panic.
        }
    }

    // Execute multiple times to verify execution_count tracking.
    let num_extra = (input.num_executions as usize % 4).min(3);
    let initial_count = pipeline.execution_count();
    for _ in 0..num_extra {
        let _ = pipeline.execute_single_token_cpu(&ids, position);
    }
    assert!(pipeline.execution_count() >= initial_count, "execution_count did not increase");

    // Config accessor must not panic.
    let cfg = pipeline.config();
    assert_eq!(cfg.num_layers, num_layers);
    assert_eq!(cfg.hidden_dim, hidden_dim);

    // Position >= max_seq_len should produce error, not panic.
    let result = pipeline.execute_single_token_cpu(&ids, max_seq_len);
    assert!(result.is_err());
});

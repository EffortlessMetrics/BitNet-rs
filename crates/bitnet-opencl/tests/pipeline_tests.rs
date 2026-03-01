//! Integration tests for the GPU inference pipeline, memory planner,
//! execution graph, and pipeline builder.

use bitnet_opencl::execution_graph::ExecutionGraph;
use bitnet_opencl::memory_planner::MemoryPlanner;
use bitnet_opencl::pipeline::{
    GpuInferencePipeline, PipelineConfig, PipelineError, PipelineStage, QuantFormat,
};
use bitnet_opencl::pipeline_builder::PipelineBuilder;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn small_config() -> PipelineConfig {
    PipelineConfig {
        num_layers: 2,
        hidden_dim: 64,
        num_heads: 4,
        head_dim: 16,
        vocab_size: 128,
        max_seq_len: 32,
        use_flash_attention: false,
        fuse_layernorm: false,
    }
}

fn large_config() -> PipelineConfig {
    PipelineConfig {
        num_layers: 32,
        hidden_dim: 2048,
        num_heads: 32,
        head_dim: 64,
        vocab_size: 32000,
        max_seq_len: 2048,
        use_flash_attention: true,
        fuse_layernorm: false,
    }
}

// ===========================================================================
// Pipeline construction
// ===========================================================================

#[test]
fn test_pipeline_construction_small() {
    let pipe = GpuInferencePipeline::new(small_config()).unwrap();
    // embedding + 2*(norm+attn+norm+ffn) + linear + softmax = 11
    assert_eq!(pipe.stages().len(), 11);
}

#[test]
fn test_pipeline_construction_large() {
    let pipe = GpuInferencePipeline::new(large_config()).unwrap();
    // embedding + 32*4 + linear + softmax = 131
    let expected = 1 + 32 * 4 + 1 + 1;
    assert_eq!(pipe.stages().len(), expected);
}

#[test]
fn test_pipeline_rejects_zero_layers() {
    let mut cfg = small_config();
    cfg.num_layers = 0;
    let err = GpuInferencePipeline::new(cfg).unwrap_err();
    assert!(matches!(err, PipelineError::InvalidConfig(_)), "expected InvalidConfig, got {err:?}");
}

#[test]
fn test_pipeline_rejects_zero_vocab() {
    let mut cfg = small_config();
    cfg.vocab_size = 0;
    assert!(GpuInferencePipeline::new(cfg).is_err());
}

// ===========================================================================
// Stage ordering
// ===========================================================================

#[test]
fn test_stage_ordering_starts_with_embedding() {
    let pipe = GpuInferencePipeline::new(small_config()).unwrap();
    assert!(
        matches!(pipe.stages()[0], PipelineStage::Embedding { .. }),
        "first stage must be Embedding"
    );
}

#[test]
fn test_stage_ordering_ends_with_softmax() {
    let pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let last = pipe.stages().last().unwrap();
    assert!(matches!(last, PipelineStage::Softmax { .. }), "last stage must be Softmax");
}

#[test]
fn test_stage_ordering_second_to_last_is_linear() {
    let pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let n = pipe.stages().len();
    assert!(matches!(pipe.stages()[n - 2], PipelineStage::Linear { .. }));
}

// ===========================================================================
// Forward pass
// ===========================================================================

#[test]
fn test_forward_produces_vocab_sized_logits() {
    let cfg = small_config();
    let vocab = cfg.vocab_size;
    let mut pipe = GpuInferencePipeline::new(cfg).unwrap();
    let logits = pipe.forward(&[1, 2, 3]).unwrap();
    assert_eq!(logits.len(), vocab);
}

#[test]
fn test_forward_rejects_empty_input() {
    let mut pipe = GpuInferencePipeline::new(small_config()).unwrap();
    assert!(matches!(pipe.forward(&[]), Err(PipelineError::EmptyInput)));
}

#[test]
fn test_forward_layer_out_of_range() {
    let mut pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let mut hidden = vec![0.0f32; 64];
    let err = pipe.forward_layer(99, &mut hidden).unwrap_err();
    assert!(matches!(err, PipelineError::LayerOutOfRange { index: 99, .. }));
}

#[test]
fn test_forward_layer_dimension_mismatch() {
    let mut pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let mut bad = vec![0.0f32; 7]; // wrong dim
    assert!(matches!(
        pipe.forward_layer(0, &mut bad),
        Err(PipelineError::DimensionMismatch { .. })
    ));
}

// ===========================================================================
// Metrics
// ===========================================================================

#[test]
fn test_metrics_after_forward() {
    let mut pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let _ = pipe.forward(&[10, 20]).unwrap();
    let m = pipe.metrics();
    assert_eq!(m.tokens_processed, 2);
    assert!(!m.kernel_times.is_empty());
}

// ===========================================================================
// Memory planner
// ===========================================================================

#[test]
fn test_memory_planner_empty() {
    let plan = MemoryPlanner::plan(&[]);
    assert_eq!(plan.peak_memory(), 0);
    assert!(plan.allocations().is_empty());
}

#[test]
fn test_memory_planner_single_stage() {
    let stages = vec![PipelineStage::RmsNorm { dim: 256, eps: 1e-5 }];
    let plan = MemoryPlanner::plan(&stages);
    assert_eq!(plan.allocations().len(), 1);
    assert!(plan.peak_memory() > 0);
}

#[test]
fn test_memory_planner_reuses_buffers() {
    let stages = vec![
        PipelineStage::RmsNorm { dim: 256, eps: 1e-5 },
        PipelineStage::Linear { in_features: 256, out_features: 256 },
        PipelineStage::RmsNorm { dim: 256, eps: 1e-5 },
        PipelineStage::Linear { in_features: 256, out_features: 256 },
    ];
    let plan = MemoryPlanner::plan(&stages);
    assert!(plan.reuse_count() > 0, "memory planner should reuse at least one buffer");
}

#[test]
fn test_memory_planner_peak_bounded() {
    let pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let plan = MemoryPlanner::plan(pipe.stages());
    // Peak should not exceed sum of all outputs.
    let total: u64 = pipe.stages().iter().map(|s| s.output_bytes()).sum();
    assert!(plan.peak_memory() <= total);
}

// ===========================================================================
// Execution graph
// ===========================================================================

#[test]
fn test_execution_graph_topological_order() {
    let pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let graph = ExecutionGraph::from_pipeline(&pipe);
    let order = graph.topological_order();
    assert_eq!(order.len(), pipe.stages().len());
    // Each node should come after its dependencies.
    for &id in &order {
        let node = &graph.nodes()[id];
        for &dep in &node.dependencies {
            let dep_pos = order.iter().position(|&x| x == dep).unwrap();
            let node_pos = order.iter().position(|&x| x == id).unwrap();
            assert!(dep_pos < node_pos);
        }
    }
}

#[test]
fn test_execution_graph_critical_path_non_empty() {
    let pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let graph = ExecutionGraph::from_pipeline(&pipe);
    let cp = graph.critical_path();
    assert!(!cp.is_empty());
}

#[test]
fn test_execution_graph_parallelizable_groups() {
    let pipe = GpuInferencePipeline::new(small_config()).unwrap();
    let graph = ExecutionGraph::from_pipeline(&pipe);
    let groups = graph.parallelizable_groups();
    // Linear chain → each group has exactly one node.
    for group in &groups {
        assert_eq!(group.len(), 1);
    }
}

// ===========================================================================
// Pipeline builder
// ===========================================================================

#[test]
fn test_builder_default_stages() {
    let pipe = PipelineBuilder::with_config(small_config()).build().unwrap();
    // Same as GpuInferencePipeline::new: 1 + 2*4 + 1 + 1 = 11
    assert_eq!(pipe.stages().len(), 11);
}

#[test]
fn test_builder_add_layer_increments() {
    let cfg = PipelineConfig {
        num_layers: 1,
        hidden_dim: 64,
        num_heads: 4,
        head_dim: 16,
        vocab_size: 128,
        max_seq_len: 32,
        use_flash_attention: false,
        fuse_layernorm: false,
    };
    let pipe = PipelineBuilder::with_config(cfg).add_layer().build().unwrap();
    // 1 (original) + 1 (add_layer) = 2 layers
    assert_eq!(pipe.config().num_layers, 2);
}

#[test]
fn test_builder_with_custom_stages() {
    let cfg = small_config();
    let pipe = PipelineBuilder::with_config(cfg)
        .add_stage(PipelineStage::Dequantize { format: QuantFormat::QK256, block_size: 256 })
        .add_stage(PipelineStage::RmsNorm { dim: 64, eps: 1e-5 })
        .build()
        .unwrap();
    // embedding + 2 custom + linear + softmax = 5
    assert_eq!(pipe.stages().len(), 5);
}

#[test]
fn test_builder_enable_fusion() {
    let pipe = PipelineBuilder::with_config(small_config())
        .enable_fusion()
        .enable_profiling()
        .build()
        .unwrap();
    assert!(pipe.config().fuse_layernorm);
}

#[test]
fn test_builder_optimizations_default() {
    let builder = PipelineBuilder::new();
    let flags = builder.optimizations();
    assert!(!flags.kernel_fusion);
    assert!(flags.memory_reuse);
    assert!(!flags.async_execution);
    assert!(!flags.profiling);
}

// ===========================================================================
// Property-style tests for memory planner
// ===========================================================================

#[test]
fn test_memory_planner_random_stages() {
    // Generate a sequence of stages with varying sizes and verify
    // that the planner never reports negative peak memory and that
    // allocations ids are unique.
    let dims: Vec<usize> = vec![32, 128, 64, 256, 64, 128, 32, 512];
    let stages: Vec<PipelineStage> =
        dims.into_iter().map(|d| PipelineStage::RmsNorm { dim: d, eps: 1e-5 }).collect();
    let plan = MemoryPlanner::plan(&stages);
    assert!(plan.peak_memory() > 0);

    // IDs must be unique.
    let ids: Vec<usize> = plan.allocations().iter().map(|a| a.id).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(ids.len(), sorted.len(), "allocation IDs must be unique");
}

#[test]
fn test_memory_planner_monotonic_sizes() {
    // Monotonically increasing sizes → no reuse possible.
    let stages: Vec<PipelineStage> = (1..=5)
        .map(|i| PipelineStage::Linear { in_features: i * 100, out_features: i * 200 })
        .collect();
    let plan = MemoryPlanner::plan(&stages);
    assert!(plan.peak_memory() > 0);
    // Each new buffer is larger than any freed one, so no reuse.
    assert_eq!(plan.reuse_count(), 0);
}

// ===========================================================================
// Ignored (hardware-dependent) tests
// ===========================================================================

#[test]
#[ignore = "requires OpenCL runtime - run manually on GPU hardware"]
fn test_pipeline_forward_on_real_device() {
    let mut pipe = GpuInferencePipeline::new(large_config()).unwrap();
    let logits = pipe.forward(&[1]).unwrap();
    assert_eq!(logits.len(), 32000);
}

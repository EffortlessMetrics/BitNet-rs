//! Edge-case tests for execution planner: OpKind, OptimizationLevel, PlanConfig,
//! ExecutionNode, ExecutionGraph, MemoryPlanner, StreamScheduler, LaunchPlanner,
//! PipelinePartitioner, CostModel, PlanOptimizer, ExecutionPlannerEngine.

use bitnet_gpu_hal::execution_planner::{
    CostModel, ExecutionGraph, ExecutionNode, ExecutionPlannerEngine, LaunchConfig, LaunchPlanner,
    MemoryPlanner, OpKind, OptimizationLevel, PipelinePartitioner, PlanConfig, PlanOptimizer,
    StreamScheduler,
};

// ── OpKind ────────────────────────────────────────────────────────────────────

#[test]
fn op_kind_all_13_variants() {
    let ops = [
        OpKind::MatMul,
        OpKind::Conv,
        OpKind::Attention,
        OpKind::LayerNorm,
        OpKind::Activation,
        OpKind::Elementwise,
        OpKind::Reduce,
        OpKind::Transpose,
        OpKind::Gather,
        OpKind::Scatter,
        OpKind::Softmax,
        OpKind::Embedding,
        OpKind::Custom,
    ];
    assert_eq!(ops.len(), 13);
}

#[test]
fn op_kind_display() {
    let s = format!("{}", OpKind::MatMul);
    assert!(!s.is_empty());
}

#[test]
fn op_kind_clone_eq_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(OpKind::MatMul);
    set.insert(OpKind::Conv);
    set.insert(OpKind::MatMul); // duplicate
    assert_eq!(set.len(), 2);
}

// ── OptimizationLevel ─────────────────────────────────────────────────────────

#[test]
fn optimization_level_default() {
    let level = OptimizationLevel::default();
    assert_eq!(level, OptimizationLevel::Standard);
}

#[test]
fn optimization_level_all_variants() {
    let _a = OptimizationLevel::None;
    let _b = OptimizationLevel::Basic;
    let _c = OptimizationLevel::Standard;
    let _d = OptimizationLevel::Aggressive;
}

// ── PlanConfig ────────────────────────────────────────────────────────────────

#[test]
fn plan_config_default() {
    let cfg = PlanConfig::default();
    assert_eq!(cfg.max_memory_bytes, 4 * 1024 * 1024 * 1024); // 4 GiB
    assert_eq!(cfg.max_parallelism, 4);
    assert_eq!(cfg.memory_alignment, 256);
    assert!(cfg.enable_fusion);
    assert!(cfg.enable_memory_reuse);
}

#[test]
fn plan_config_new() {
    let cfg = PlanConfig::new(1024 * 1024, 8);
    assert_eq!(cfg.max_memory_bytes, 1024 * 1024);
    assert_eq!(cfg.max_parallelism, 8);
}

#[test]
fn plan_config_align() {
    let cfg = PlanConfig::default(); // alignment = 256
    assert_eq!(cfg.align(0), 0);
    assert_eq!(cfg.align(1), 256);
    assert_eq!(cfg.align(256), 256);
    assert_eq!(cfg.align(257), 512);
}

// ── ExecutionNode ─────────────────────────────────────────────────────────────

#[test]
fn execution_node_new() {
    let node = ExecutionNode::new(0, "matmul_0", OpKind::MatMul);
    assert_eq!(node.id, 0);
    assert_eq!(node.label, "matmul_0");
    assert_eq!(node.op, OpKind::MatMul);
    assert!(node.dependencies.is_empty());
    assert_eq!(node.output_bytes, 0);
    assert_eq!(node.workspace_bytes, 0);
    assert_eq!(node.flops, 0);
}

#[test]
fn execution_node_builder_chain() {
    let node = ExecutionNode::new(0, "matmul", OpKind::MatMul)
        .with_dep(1)
        .with_output_bytes(4096)
        .with_workspace_bytes(1024)
        .with_flops(1_000_000)
        .with_output_shape(vec![32, 128]);
    assert_eq!(node.dependencies, vec![1]);
    assert_eq!(node.output_bytes, 4096);
    assert_eq!(node.workspace_bytes, 1024);
    assert_eq!(node.flops, 1_000_000);
    assert_eq!(node.output_shape, vec![32, 128]);
}

#[test]
fn execution_node_total_memory() {
    let node = ExecutionNode::new(0, "op", OpKind::Activation)
        .with_output_bytes(1000)
        .with_workspace_bytes(500);
    assert_eq!(node.total_memory(), 1500);
}

#[test]
fn execution_node_multiple_deps() {
    let node = ExecutionNode::new(0, "op", OpKind::Elementwise).with_dep(1).with_dep(2).with_dep(3);
    assert_eq!(node.dependencies.len(), 3);
}

// ── ExecutionGraph ────────────────────────────────────────────────────────────

#[test]
fn execution_graph_empty() {
    let g = ExecutionGraph::new();
    assert!(g.is_empty());
    assert_eq!(g.len(), 0);
    assert_eq!(g.total_output_bytes(), 0);
    assert_eq!(g.total_flops(), 0);
}

#[test]
fn execution_graph_add_nodes() {
    let mut g = ExecutionGraph::new();
    let id0 = g.add_node(ExecutionNode::new(0, "a", OpKind::MatMul).with_output_bytes(100));
    let _id1 = g.add_node(
        ExecutionNode::new(1, "b", OpKind::Activation).with_dep(id0).with_output_bytes(200),
    );
    assert_eq!(g.len(), 2);
    assert_eq!(g.total_output_bytes(), 300);
}

#[test]
fn execution_graph_node_access() {
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "a", OpKind::LayerNorm));
    assert!(g.node(0).is_some());
    assert!(g.node(99).is_none());
}

#[test]
fn execution_graph_topological_sort_linear() {
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "a", OpKind::Embedding));
    g.add_node(ExecutionNode::new(1, "b", OpKind::MatMul).with_dep(0));
    g.add_node(ExecutionNode::new(2, "c", OpKind::Activation).with_dep(1));
    let order = g.topological_sort();
    assert!(order.is_some());
    let order = order.unwrap();
    assert_eq!(order, vec![0, 1, 2]);
}

#[test]
fn execution_graph_topological_sort_diamond() {
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "root", OpKind::Embedding));
    g.add_node(ExecutionNode::new(1, "left", OpKind::MatMul).with_dep(0));
    g.add_node(ExecutionNode::new(2, "right", OpKind::MatMul).with_dep(0));
    g.add_node(ExecutionNode::new(3, "join", OpKind::Elementwise).with_dep(1).with_dep(2));
    let order = g.topological_sort();
    assert!(order.is_some());
    let order = order.unwrap();
    assert_eq!(order.len(), 4);
    // Root must come first, join must come last
    assert_eq!(order[0], 0);
    assert_eq!(order[3], 3);
}

#[test]
fn execution_graph_roots_and_leaves() {
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "root", OpKind::Embedding));
    g.add_node(ExecutionNode::new(1, "mid", OpKind::MatMul).with_dep(0));
    g.add_node(ExecutionNode::new(2, "leaf", OpKind::Softmax).with_dep(1));
    let roots = g.roots();
    let leaves = g.leaves();
    assert_eq!(roots, vec![0]);
    assert_eq!(leaves, vec![2]);
}

#[test]
fn execution_graph_successors() {
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "a", OpKind::Embedding));
    g.add_node(ExecutionNode::new(1, "b", OpKind::MatMul).with_dep(0));
    g.add_node(ExecutionNode::new(2, "c", OpKind::Activation).with_dep(0));
    let succs = g.successors(0);
    assert!(succs.contains(&1));
    assert!(succs.contains(&2));
}

// ── MemoryPlanner ─────────────────────────────────────────────────────────────

#[test]
fn memory_planner_empty_graph() {
    let cfg = PlanConfig::default();
    let mut planner = MemoryPlanner::new(&cfg);
    let g = ExecutionGraph::new();
    let allocs = planner.plan(&g, &[]);
    assert!(allocs.is_empty());
    assert_eq!(planner.peak_usage(), 0);
}

#[test]
fn memory_planner_single_node() {
    let cfg = PlanConfig::default();
    let mut planner = MemoryPlanner::new(&cfg);
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "op", OpKind::MatMul).with_output_bytes(4096));
    let allocs = planner.plan(&g, &[0]);
    assert!(!allocs.is_empty());
    assert!(planner.peak_usage() > 0);
}

#[test]
fn memory_planner_linear_chain() {
    let cfg = PlanConfig::default();
    let mut planner = MemoryPlanner::new(&cfg);
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "a", OpKind::Embedding).with_output_bytes(1024));
    g.add_node(ExecutionNode::new(1, "b", OpKind::MatMul).with_dep(0).with_output_bytes(2048));
    g.add_node(ExecutionNode::new(2, "c", OpKind::Activation).with_dep(1).with_output_bytes(2048));
    let allocs = planner.plan(&g, &[0, 1, 2]);
    assert_eq!(allocs.len(), 3);
}

// ── StreamScheduler ───────────────────────────────────────────────────────────

#[test]
fn stream_scheduler_single_stream() {
    let mut sched = StreamScheduler::new(1);
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "a", OpKind::MatMul));
    g.add_node(ExecutionNode::new(1, "b", OpKind::Activation).with_dep(0));
    let assignments = sched.schedule(&g, &[0, 1]);
    assert_eq!(assignments.len(), 2);
    assert!(assignments.iter().all(|a| a.stream_id == 0));
}

#[test]
fn stream_scheduler_parallel_branches() {
    let mut sched = StreamScheduler::new(4);
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "root", OpKind::Embedding));
    g.add_node(ExecutionNode::new(1, "branch_a", OpKind::MatMul).with_dep(0));
    g.add_node(ExecutionNode::new(2, "branch_b", OpKind::MatMul).with_dep(0));
    let assignments = sched.schedule(&g, &[0, 1, 2]);
    assert_eq!(assignments.len(), 3);
    assert!(sched.streams_used() >= 1);
}

// ── LaunchConfig ──────────────────────────────────────────────────────────────

#[test]
fn launch_config_thread_calculations() {
    let lc = LaunchConfig { node_id: 0, grid: [4, 2, 1], block: [256, 1, 1], shared_mem_bytes: 0 };
    assert_eq!(lc.threads_per_block(), 256);
    assert_eq!(lc.total_blocks(), 8);
    assert_eq!(lc.total_threads(), 2048);
}

#[test]
fn launch_config_3d() {
    let lc = LaunchConfig { node_id: 0, grid: [2, 2, 2], block: [8, 8, 4], shared_mem_bytes: 4096 };
    assert_eq!(lc.threads_per_block(), 256);
    assert_eq!(lc.total_blocks(), 8);
    assert_eq!(lc.total_threads(), 2048);
    assert_eq!(lc.shared_mem_bytes, 4096);
}

// ── LaunchPlanner ─────────────────────────────────────────────────────────────

#[test]
fn launch_planner_cpu_default() {
    let planner = LaunchPlanner::cpu_default();
    let g = ExecutionGraph::new();
    let mut planner2 = planner;
    let configs = planner2.plan(&g);
    assert!(configs.is_empty());
}

#[test]
fn launch_planner_with_nodes() {
    let mut planner = LaunchPlanner::cpu_default();
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "matmul", OpKind::MatMul).with_flops(1000));
    g.add_node(ExecutionNode::new(1, "act", OpKind::Activation).with_dep(0).with_flops(500));
    let configs = planner.plan(&g);
    assert_eq!(configs.len(), 2);
}

// ── CostModel ─────────────────────────────────────────────────────────────────

#[test]
fn cost_model_default() {
    let cm = CostModel::default();
    assert!((cm.peak_gflops - 100.0).abs() < 1e-6);
    assert!((cm.peak_bandwidth_gbs - 50.0).abs() < 1e-6);
    assert_eq!(cm.launch_overhead_us, 5);
}

#[test]
fn cost_model_estimate() {
    let cm = CostModel::default();
    let node = ExecutionNode::new(0, "matmul", OpKind::MatMul)
        .with_flops(1_000_000_000)
        .with_output_bytes(1_000_000);
    let est = cm.estimate(&node);
    assert!(est.time_us > 0);
    assert_eq!(est.flops, 1_000_000_000);
}

#[test]
fn cost_model_total_cost() {
    let cm = CostModel::default();
    let nodes = vec![
        ExecutionNode::new(0, "a", OpKind::MatMul).with_flops(100),
        ExecutionNode::new(1, "b", OpKind::Activation).with_flops(50),
    ];
    let total = cm.total_cost(nodes.iter());
    assert_eq!(total.flops, 150);
}

// ── PipelinePartitioner ───────────────────────────────────────────────────────

#[test]
fn pipeline_partitioner_empty() {
    let cm = CostModel::default();
    let mut part = PipelinePartitioner::new(2);
    let g = ExecutionGraph::new();
    let stages = part.partition(&g, &[], &cm);
    assert!(stages.is_empty());
}

#[test]
fn pipeline_partitioner_single_stage() {
    let cm = CostModel::default();
    let mut part = PipelinePartitioner::new(1);
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "a", OpKind::MatMul).with_flops(1000));
    let stages = part.partition(&g, &[0], &cm);
    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].node_count(), 1);
}

#[test]
fn pipeline_partitioner_bottleneck() {
    let cm = CostModel::default();
    let mut part = PipelinePartitioner::new(2);
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "heavy", OpKind::MatMul).with_flops(1_000_000));
    g.add_node(ExecutionNode::new(1, "light", OpKind::Activation).with_dep(0).with_flops(100));
    let _stages = part.partition(&g, &[0, 1], &cm);
    let bottleneck = part.bottleneck();
    assert!(bottleneck.is_some());
}

// ── PlanOptimizer ─────────────────────────────────────────────────────────────

#[test]
fn plan_optimizer_empty_graph() {
    let opt = PlanOptimizer::new(PlanConfig::default());
    let g = ExecutionGraph::new();
    let result = opt.optimize(&g);
    assert!(result.exec_order.is_empty());
}

#[test]
fn plan_optimizer_linear() {
    let opt = PlanOptimizer::new(PlanConfig::default());
    let mut g = ExecutionGraph::new();
    g.add_node(ExecutionNode::new(0, "embed", OpKind::Embedding));
    g.add_node(ExecutionNode::new(1, "norm", OpKind::LayerNorm).with_dep(0));
    g.add_node(ExecutionNode::new(2, "matmul", OpKind::MatMul).with_dep(1));
    let result = opt.optimize(&g);
    // Optimizer may fuse adjacent nodes, reducing count
    assert!(!result.exec_order.is_empty());
    assert!(result.exec_order.len() <= 3);
}

// ── ExecutionPlannerEngine ────────────────────────────────────────────────────

#[test]
fn planner_engine_cpu_default() {
    let engine = ExecutionPlannerEngine::cpu_default();
    let g = ExecutionGraph::new();
    let plan = engine.plan(&g);
    assert!(plan.is_ok());
    let plan = plan.unwrap();
    assert_eq!(plan.peak_memory_bytes, 0);
}

#[test]
fn planner_engine_linear_graph() {
    let engine = ExecutionPlannerEngine::cpu_default();
    let mut g = ExecutionGraph::new();
    g.add_node(
        ExecutionNode::new(0, "embed", OpKind::Embedding).with_output_bytes(4096).with_flops(1000),
    );
    g.add_node(
        ExecutionNode::new(1, "matmul", OpKind::MatMul)
            .with_dep(0)
            .with_output_bytes(8192)
            .with_flops(1_000_000),
    );
    g.add_node(
        ExecutionNode::new(2, "softmax", OpKind::Softmax)
            .with_dep(1)
            .with_output_bytes(8192)
            .with_flops(5000),
    );
    let plan = engine.plan(&g).unwrap();
    assert_eq!(plan.exec_order.len(), 3);
    assert!(!plan.allocations.is_empty());
    assert!(!plan.launch_configs.is_empty());
    assert!(plan.peak_memory_bytes > 0);
}

#[test]
fn planner_engine_config_access() {
    let engine = ExecutionPlannerEngine::cpu_default();
    assert_eq!(engine.config().max_parallelism, 4);
    assert!((engine.cost_model().peak_gflops - 100.0).abs() < 1e-6);
}

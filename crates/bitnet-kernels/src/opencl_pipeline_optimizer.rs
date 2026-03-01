//! Pipeline execution optimizer for OpenCL compute/transfer overlap.
//!
//! Schedules kernel execution across multiple streams to maximize throughput
//! on Intel Arc A770 (2 compute queues + 1 copy queue). Provides CPU reference
//! implementations for sequential, pipelined, and double-buffered scheduling.

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ── Types ──────────────────────────────────────────────────────────

/// Category of a scheduled operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    Compute,
    H2DTransfer,
    D2HTransfer,
    Synchronize,
}

impl fmt::Display for OpCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpCategory::Compute => write!(f, "Compute"),
            OpCategory::H2DTransfer => write!(f, "H2D"),
            OpCategory::D2HTransfer => write!(f, "D2H"),
            OpCategory::Synchronize => write!(f, "Sync"),
        }
    }
}

/// A single stage in the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub id: usize,
    pub name: String,
    pub compute_us: u64,
    pub transfer_us: u64,
    pub dependencies: Vec<usize>,
}

/// Configuration for pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub num_streams: usize,
    pub enable_overlap: bool,
    pub prefetch_depth: usize,
    pub double_buffer: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_streams: 3, // A770: 2 compute + 1 copy
            enable_overlap: true,
            prefetch_depth: 1,
            double_buffer: false,
        }
    }
}

/// A pipeline of stages to be scheduled.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub stages: Vec<PipelineStage>,
    pub config: PipelineConfig,
}

/// A single scheduled operation within an execution plan.
#[derive(Debug, Clone)]
pub struct ScheduledOp {
    pub stage_id: usize,
    pub stream_id: usize,
    pub start_us: u64,
    pub end_us: u64,
    pub op_type: OpCategory,
}

/// Complete execution plan produced by a scheduler.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub scheduled_ops: Vec<ScheduledOp>,
    pub total_time_us: u64,
    pub pipeline_efficiency: f32,
    pub streams_used: usize,
}

/// Statistics from simulating an execution plan.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub compute_time_us: u64,
    pub transfer_time_us: u64,
    pub idle_time_us: u64,
    pub overlap_pct: f32,
    pub throughput_items_per_sec: f64,
}

/// Errors that can occur during pipeline optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizeError {
    CyclicDependency,
    InfeasibleSchedule(String),
    StreamLimitExceeded(usize),
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizeError::CyclicDependency => write!(f, "cyclic dependency detected"),
            OptimizeError::InfeasibleSchedule(msg) => {
                write!(f, "infeasible schedule: {msg}")
            }
            OptimizeError::StreamLimitExceeded(n) => {
                write!(f, "stream limit exceeded: {n}")
            }
        }
    }
}

impl std::error::Error for OptimizeError {}

// ── Pipeline construction ──────────────────────────────────────────

/// Create a new empty pipeline with the given configuration.
pub fn create_pipeline(config: PipelineConfig) -> Pipeline {
    Pipeline {
        stages: Vec::new(),
        config,
    }
}

/// Add a stage to the pipeline and return its assigned ID.
pub fn cpu_add_stage(
    pipeline: &mut Pipeline,
    name: &str,
    compute_us: u64,
    transfer_us: u64,
    deps: Vec<usize>,
) -> usize {
    let id = pipeline.stages.len();
    pipeline.stages.push(PipelineStage {
        id,
        name: name.to_string(),
        compute_us,
        transfer_us,
        dependencies: deps,
    });
    id
}

// ── Topological helpers ────────────────────────────────────────────

/// Return a topological order of stage IDs, or `Err` on cycle.
fn topological_sort(stages: &[PipelineStage]) -> Result<Vec<usize>, OptimizeError> {
    let n = stages.len();
    let mut in_degree = vec![0u32; n];
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];

    for s in stages {
        for &dep in &s.dependencies {
            successors[dep].push(s.id);
            in_degree[s.id] += 1;
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &succ in &successors[node] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    if order.len() != n {
        Err(OptimizeError::CyclicDependency)
    } else {
        Ok(order)
    }
}

/// Earliest start time for `stage_id` given already-computed finish times.
fn earliest_start(stage: &PipelineStage, finish: &HashMap<usize, u64>) -> u64 {
    stage
        .dependencies
        .iter()
        .map(|dep| finish.get(dep).copied().unwrap_or(0))
        .max()
        .unwrap_or(0)
}

// ── Schedulers ─────────────────────────────────────────────────────

/// Sequential (no-overlap) baseline scheduler.
pub fn cpu_schedule_sequential(pipeline: &Pipeline) -> ExecutionPlan {
    let order = topological_sort(&pipeline.stages).unwrap_or_else(|_| {
        (0..pipeline.stages.len()).collect()
    });

    let mut ops = Vec::new();
    let mut clock: u64 = 0;

    for &sid in &order {
        let stage = &pipeline.stages[sid];

        // H2D transfer first
        if stage.transfer_us > 0 {
            ops.push(ScheduledOp {
                stage_id: sid,
                stream_id: 0,
                start_us: clock,
                end_us: clock + stage.transfer_us,
                op_type: OpCategory::H2DTransfer,
            });
            clock += stage.transfer_us;
        }

        // Then compute
        if stage.compute_us > 0 {
            ops.push(ScheduledOp {
                stage_id: sid,
                stream_id: 0,
                start_us: clock,
                end_us: clock + stage.compute_us,
                op_type: OpCategory::Compute,
            });
            clock += stage.compute_us;
        }

        // D2H transfer after compute
        if stage.transfer_us > 0 {
            ops.push(ScheduledOp {
                stage_id: sid,
                stream_id: 0,
                start_us: clock,
                end_us: clock + stage.transfer_us,
                op_type: OpCategory::D2HTransfer,
            });
            clock += stage.transfer_us;
        }
    }

    let total = clock;
    let useful: u64 = pipeline
        .stages
        .iter()
        .map(|s| s.compute_us + s.transfer_us)
        .sum();
    let efficiency = if total > 0 {
        useful as f32 / total as f32
    } else {
        1.0
    };

    ExecutionPlan {
        scheduled_ops: ops,
        total_time_us: total,
        pipeline_efficiency: efficiency.min(1.0),
        streams_used: 1,
    }
}

/// Pipelined scheduler that overlaps compute and transfers across streams.
pub fn cpu_schedule_pipelined(pipeline: &Pipeline) -> ExecutionPlan {
    let order = topological_sort(&pipeline.stages).unwrap_or_else(|_| {
        (0..pipeline.stages.len()).collect()
    });

    let num_streams = pipeline.config.num_streams.max(2);
    // stream 0: copy queue, streams 1..N: compute queues
    let mut stream_avail = vec![0u64; num_streams];
    let mut stage_finish: HashMap<usize, u64> = HashMap::new();
    let mut ops = Vec::new();

    for &sid in &order {
        let stage = &pipeline.stages[sid];
        let dep_ready = earliest_start(stage, &stage_finish);

        // Schedule H2D on copy stream (0)
        let h2d_start = stream_avail[0].max(dep_ready);
        let h2d_end = h2d_start + stage.transfer_us;
        if stage.transfer_us > 0 {
            ops.push(ScheduledOp {
                stage_id: sid,
                stream_id: 0,
                start_us: h2d_start,
                end_us: h2d_end,
                op_type: OpCategory::H2DTransfer,
            });
            stream_avail[0] = h2d_end;
        }

        // Pick the compute stream that becomes free earliest (streams 1..N)
        let compute_stream = (1..num_streams)
            .min_by_key(|&s| stream_avail[s])
            .unwrap_or(1);

        let compute_ready = if stage.transfer_us > 0 {
            h2d_end
        } else {
            dep_ready
        };
        let compute_start = stream_avail[compute_stream].max(compute_ready);
        let compute_end = compute_start + stage.compute_us;
        if stage.compute_us > 0 {
            ops.push(ScheduledOp {
                stage_id: sid,
                stream_id: compute_stream,
                start_us: compute_start,
                end_us: compute_end,
                op_type: OpCategory::Compute,
            });
            stream_avail[compute_stream] = compute_end;
        }

        let finish = compute_end.max(h2d_end);
        stage_finish.insert(sid, finish);
    }

    let total = *stream_avail.iter().max().unwrap_or(&0);
    let useful: u64 = pipeline
        .stages
        .iter()
        .map(|s| s.compute_us + s.transfer_us)
        .sum();
    let efficiency = if total > 0 {
        (useful as f32 / total as f32).min(1.0)
    } else {
        1.0
    };
    let streams_used = stream_avail.iter().filter(|&&t| t > 0).count().max(1);

    ExecutionPlan {
        scheduled_ops: ops,
        total_time_us: total,
        pipeline_efficiency: efficiency,
        streams_used,
    }
}

/// Double-buffered scheduler: overlaps iteration N+1 transfers with N compute.
pub fn cpu_schedule_double_buffered(pipeline: &Pipeline) -> ExecutionPlan {
    let order = topological_sort(&pipeline.stages).unwrap_or_else(|_| {
        (0..pipeline.stages.len()).collect()
    });

    let num_streams = pipeline.config.num_streams.max(2);
    // Two buffer slots with their own copy-queue availability
    let mut buf_copy_avail = [0u64; 2];
    let mut compute_avail = vec![0u64; num_streams.saturating_sub(1).max(1)];
    let mut stage_finish: HashMap<usize, u64> = HashMap::new();
    let mut ops = Vec::new();

    for (idx, &sid) in order.iter().enumerate() {
        let stage = &pipeline.stages[sid];
        let dep_ready = earliest_start(stage, &stage_finish);
        let buf = idx % 2;

        // H2D on copy queue for this buffer slot
        let h2d_start = buf_copy_avail[buf].max(dep_ready);
        let h2d_end = h2d_start + stage.transfer_us;
        if stage.transfer_us > 0 {
            ops.push(ScheduledOp {
                stage_id: sid,
                stream_id: 0,
                start_us: h2d_start,
                end_us: h2d_end,
                op_type: OpCategory::H2DTransfer,
            });
            buf_copy_avail[buf] = h2d_end;
        }

        // Pick earliest-available compute stream
        let cs = compute_avail
            .iter()
            .enumerate()
            .min_by_key(|&(_, t)| *t)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let compute_ready = if stage.transfer_us > 0 {
            h2d_end
        } else {
            dep_ready
        };
        let compute_start = compute_avail[cs].max(compute_ready);
        let compute_end = compute_start + stage.compute_us;
        if stage.compute_us > 0 {
            ops.push(ScheduledOp {
                stage_id: sid,
                stream_id: cs + 1,
                start_us: compute_start,
                end_us: compute_end,
                op_type: OpCategory::Compute,
            });
            compute_avail[cs] = compute_end;
        }

        let finish = compute_end.max(h2d_end);
        stage_finish.insert(sid, finish);
    }

    let max_copy = buf_copy_avail.iter().max().copied().unwrap_or(0);
    let max_compute = compute_avail.iter().max().copied().unwrap_or(0);
    let total = max_copy.max(max_compute);

    let useful: u64 = pipeline
        .stages
        .iter()
        .map(|s| s.compute_us + s.transfer_us)
        .sum();
    let efficiency = if total > 0 {
        (useful as f32 / total as f32).min(1.0)
    } else {
        1.0
    };
    let streams_used = {
        let copy_used = if buf_copy_avail.iter().any(|&t| t > 0) {
            1
        } else {
            0
        };
        let compute_used = compute_avail.iter().filter(|&&t| t > 0).count();
        (copy_used + compute_used).max(1)
    };

    ExecutionPlan {
        scheduled_ops: ops,
        total_time_us: total,
        pipeline_efficiency: efficiency,
        streams_used,
    }
}

// ── Analysis functions ─────────────────────────────────────────────

/// Compute the critical (longest) path through the pipeline DAG.
/// Returns (path of stage IDs, total time in µs).
pub fn cpu_compute_critical_path(pipeline: &Pipeline) -> (Vec<usize>, u64) {
    if pipeline.stages.is_empty() {
        return (vec![], 0);
    }

    let order = match topological_sort(&pipeline.stages) {
        Ok(o) => o,
        Err(_) => return (vec![], 0),
    };

    let n = pipeline.stages.len();
    let mut dist = vec![0u64; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];

    for &sid in &order {
        let stage = &pipeline.stages[sid];
        let stage_cost = stage.compute_us + stage.transfer_us;
        for &dep in &stage.dependencies {
            let through = dist[dep] + stage_cost;
            if through > dist[sid] {
                dist[sid] = through;
                pred[sid] = Some(dep);
            }
        }
        // Root nodes: just their own cost
        if stage.dependencies.is_empty() {
            dist[sid] = stage_cost;
        }
    }

    // Find the endpoint with maximum distance
    let end = (0..n).max_by_key(|&i| dist[i]).unwrap_or(0);
    let total = dist[end];

    // Trace back the path
    let mut path = vec![end];
    let mut cur = end;
    while let Some(p) = pred[cur] {
        path.push(p);
        cur = p;
    }
    path.reverse();

    (path, total)
}

/// Average number of concurrently active operations.
pub fn cpu_compute_parallelism(plan: &ExecutionPlan) -> f32 {
    if plan.scheduled_ops.is_empty() || plan.total_time_us == 0 {
        return 0.0;
    }

    let total_busy: u64 = plan
        .scheduled_ops
        .iter()
        .map(|op| op.end_us - op.start_us)
        .sum();

    total_busy as f32 / plan.total_time_us as f32
}

/// Ratio of useful work to total wall-clock time.
pub fn cpu_compute_efficiency(plan: &ExecutionPlan) -> f32 {
    if plan.total_time_us == 0 {
        return if plan.scheduled_ops.is_empty() {
            1.0
        } else {
            0.0
        };
    }

    let useful: u64 = plan
        .scheduled_ops
        .iter()
        .map(|op| op.end_us - op.start_us)
        .sum();

    let max_possible = plan.total_time_us * plan.streams_used as u64;
    if max_possible == 0 {
        return 1.0;
    }

    (useful as f32 / max_possible as f32).clamp(0.0, 1.0)
}

/// Simulate the execution plan and return aggregate statistics.
pub fn cpu_simulate_execution(plan: &ExecutionPlan) -> PipelineStats {
    let mut compute_time_us = 0u64;
    let mut transfer_time_us = 0u64;

    for op in &plan.scheduled_ops {
        let dur = op.end_us - op.start_us;
        match op.op_type {
            OpCategory::Compute => compute_time_us += dur,
            OpCategory::H2DTransfer | OpCategory::D2HTransfer => transfer_time_us += dur,
            OpCategory::Synchronize => {}
        }
    }

    let useful = compute_time_us + transfer_time_us;
    let total_slot = plan.total_time_us * plan.streams_used.max(1) as u64;
    let idle_time_us = total_slot.saturating_sub(useful);

    let overlap_pct = if plan.total_time_us > 0 && useful > plan.total_time_us {
        let overlap = useful - plan.total_time_us;
        (overlap as f32 / useful as f32 * 100.0).min(100.0)
    } else {
        0.0
    };

    let throughput = if plan.total_time_us > 0 {
        1_000_000.0 / plan.total_time_us as f64
    } else {
        0.0
    };

    PipelineStats {
        compute_time_us,
        transfer_time_us,
        idle_time_us,
        overlap_pct,
        throughput_items_per_sec: throughput,
    }
}

/// Estimated speedup: sequential_time / optimized_time.
pub fn cpu_estimate_speedup(
    sequential: &ExecutionPlan,
    optimized: &ExecutionPlan,
) -> f64 {
    if optimized.total_time_us == 0 {
        return 1.0;
    }
    sequential.total_time_us as f64 / optimized.total_time_us as f64
}

/// Identify the bottleneck stage: the one contributing the most wall-clock time.
pub fn cpu_find_bottleneck(plan: &ExecutionPlan) -> (usize, String) {
    if plan.scheduled_ops.is_empty() {
        return (0, "empty plan".to_string());
    }

    let mut stage_time: HashMap<usize, u64> = HashMap::new();
    for op in &plan.scheduled_ops {
        *stage_time.entry(op.stage_id).or_insert(0) += op.end_us - op.start_us;
    }

    let (&stage_id, &time) = stage_time
        .iter()
        .max_by_key(|&(_, t)| *t)
        .unwrap_or((&0, &0));

    (
        stage_id,
        format!("stage {stage_id} takes {time}µs ({:.0}% of total)", {
            if plan.total_time_us > 0 {
                time as f64 / plan.total_time_us as f64 * 100.0
            } else {
                0.0
            }
        }),
    )
}

/// Suggest optimizations based on plan characteristics.
pub fn cpu_suggest_optimization(plan: &ExecutionPlan) -> Vec<String> {
    let mut suggestions = Vec::new();

    let stats = cpu_simulate_execution(plan);

    if stats.overlap_pct < 10.0 && plan.streams_used <= 1 {
        suggestions.push(
            "Enable multi-stream execution to overlap compute and transfers".to_string(),
        );
    }

    if stats.transfer_time_us > stats.compute_time_us && stats.transfer_time_us > 0 {
        suggestions.push("Transfer-bound: consider larger batch sizes or pinned memory".to_string());
    }

    if stats.compute_time_us > stats.transfer_time_us * 2 {
        suggestions.push("Compute-bound: consider kernel fusion or work-group tuning".to_string());
    }

    let efficiency = cpu_compute_efficiency(plan);
    if efficiency < 0.5 {
        suggestions.push("Low efficiency: increase pipeline depth or enable double-buffering".to_string());
    }

    let parallelism = cpu_compute_parallelism(plan);
    if parallelism < 1.5 && plan.scheduled_ops.len() > 2 {
        suggestions
            .push("Low parallelism: split independent stages across streams".to_string());
    }

    if plan.streams_used < 3 {
        suggestions.push(
            "A770 supports 2 compute + 1 copy queue: use all 3 streams".to_string(),
        );
    }

    if suggestions.is_empty() {
        suggestions.push("Pipeline is well-optimized".to_string());
    }

    suggestions
}

/// Format an execution plan as a human-readable string.
pub fn format_execution_plan(plan: &ExecutionPlan) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Execution Plan: {}µs total, {:.1}% efficiency, {} streams\n",
        plan.total_time_us,
        plan.pipeline_efficiency * 100.0,
        plan.streams_used,
    ));
    out.push_str(&format!(
        "{:<8} {:<10} {:<10} {:<10} {:<10}\n",
        "Stage", "Stream", "Start(µs)", "End(µs)", "Type"
    ));
    out.push_str(&"-".repeat(52));
    out.push('\n');
    for op in &plan.scheduled_ops {
        out.push_str(&format!(
            "{:<8} {:<10} {:<10} {:<10} {:<10}\n",
            op.stage_id, op.stream_id, op.start_us, op.end_us, op.op_type,
        ));
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PipelineConfig {
        PipelineConfig::default()
    }

    fn a770_config() -> PipelineConfig {
        PipelineConfig {
            num_streams: 3,
            enable_overlap: true,
            prefetch_depth: 1,
            double_buffer: false,
        }
    }

    // ── Construction ──────────────────────────────────────

    #[test]
    fn test_create_pipeline_empty() {
        let p = create_pipeline(default_config());
        assert!(p.stages.is_empty());
    }

    #[test]
    fn test_add_stages_correct_ids() {
        let mut p = create_pipeline(default_config());
        let a = cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let b = cpu_add_stage(&mut p, "B", 200, 50, vec![a]);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(p.stages.len(), 2);
    }

    #[test]
    fn test_add_stage_preserves_name() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "matmul_q4", 500, 100, vec![]);
        assert_eq!(p.stages[0].name, "matmul_q4");
    }

    #[test]
    fn test_add_stage_preserves_deps() {
        let mut p = create_pipeline(default_config());
        let a = cpu_add_stage(&mut p, "A", 10, 10, vec![]);
        let b = cpu_add_stage(&mut p, "B", 10, 10, vec![a]);
        assert_eq!(p.stages[1].dependencies, vec![0]);
        let _c = cpu_add_stage(&mut p, "C", 10, 10, vec![a, b]);
        assert_eq!(p.stages[2].dependencies, vec![0, 1]);
    }

    // ── Sequential schedule ───────────────────────────────

    #[test]
    fn test_sequential_total_time_single() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        // H2D(50) + Compute(100) + D2H(50) = 200
        assert_eq!(plan.total_time_us, 200);
    }

    #[test]
    fn test_sequential_total_time_chain() {
        let mut p = create_pipeline(default_config());
        let a = cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        cpu_add_stage(&mut p, "B", 100, 50, vec![a]);
        let plan = cpu_schedule_sequential(&p);
        assert_eq!(plan.total_time_us, 400);
    }

    #[test]
    fn test_sequential_uses_one_stream() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        cpu_add_stage(&mut p, "B", 200, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        assert_eq!(plan.streams_used, 1);
    }

    #[test]
    fn test_sequential_op_ordering() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        assert_eq!(plan.scheduled_ops.len(), 3);
        assert_eq!(plan.scheduled_ops[0].op_type, OpCategory::H2DTransfer);
        assert_eq!(plan.scheduled_ops[1].op_type, OpCategory::Compute);
        assert_eq!(plan.scheduled_ops[2].op_type, OpCategory::D2HTransfer);
    }

    #[test]
    fn test_sequential_no_transfer_stage() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 0, vec![]);
        let plan = cpu_schedule_sequential(&p);
        assert_eq!(plan.total_time_us, 100);
        assert_eq!(plan.scheduled_ops.len(), 1);
    }

    // ── Pipelined schedule ────────────────────────────────

    #[test]
    fn test_pipelined_overlaps_compute_and_transfer() {
        let mut p = create_pipeline(a770_config());
        let a = cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        cpu_add_stage(&mut p, "B", 100, 50, vec![a]);
        let seq = cpu_schedule_sequential(&p);
        let pipe = cpu_schedule_pipelined(&p);
        assert!(
            pipe.total_time_us <= seq.total_time_us,
            "pipelined {}µs should be <= sequential {}µs",
            pipe.total_time_us,
            seq.total_time_us
        );
    }

    #[test]
    fn test_pipelined_uses_multiple_streams() {
        let mut p = create_pipeline(a770_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        cpu_add_stage(&mut p, "B", 100, 50, vec![]);
        let plan = cpu_schedule_pipelined(&p);
        assert!(plan.streams_used >= 2, "should use >=2 streams");
    }

    #[test]
    fn test_pipelined_independent_stages_parallel() {
        let mut p = create_pipeline(a770_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        cpu_add_stage(&mut p, "B", 100, 50, vec![]);
        let plan = cpu_schedule_pipelined(&p);
        // With overlap, two independent stages can run in parallel
        let seq = cpu_schedule_sequential(&p);
        assert!(plan.total_time_us < seq.total_time_us);
    }

    #[test]
    fn test_pipelined_respects_dependencies() {
        let mut p = create_pipeline(a770_config());
        let a = cpu_add_stage(&mut p, "A", 100, 0, vec![]);
        cpu_add_stage(&mut p, "B", 100, 0, vec![a]);
        let plan = cpu_schedule_pipelined(&p);
        // B must start after A finishes
        let a_end = plan
            .scheduled_ops
            .iter()
            .filter(|op| op.stage_id == 0)
            .map(|op| op.end_us)
            .max()
            .unwrap();
        let b_start = plan
            .scheduled_ops
            .iter()
            .filter(|op| op.stage_id == 1)
            .map(|op| op.start_us)
            .min()
            .unwrap();
        assert!(b_start >= a_end);
    }

    // ── Double buffered ───────────────────────────────────

    #[test]
    fn test_double_buffered_further_overlap() {
        let mut p = create_pipeline(PipelineConfig {
            num_streams: 3,
            enable_overlap: true,
            prefetch_depth: 1,
            double_buffer: true,
        });
        for _ in 0..4 {
            cpu_add_stage(&mut p, "stage", 100, 50, vec![]);
        }
        let pipe = cpu_schedule_pipelined(&p);
        let db = cpu_schedule_double_buffered(&p);
        assert!(
            db.total_time_us <= pipe.total_time_us,
            "double-buffered {}µs should be <= pipelined {}µs",
            db.total_time_us,
            pipe.total_time_us,
        );
    }

    #[test]
    fn test_double_buffered_single_stage() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let db = cpu_schedule_double_buffered(&p);
        assert_eq!(db.total_time_us, 150); // H2D(50) + Compute(100)
    }

    // ── Critical path ─────────────────────────────────────

    #[test]
    fn test_critical_path_linear_chain() {
        let mut p = create_pipeline(default_config());
        let a = cpu_add_stage(&mut p, "A", 100, 0, vec![]);
        let b = cpu_add_stage(&mut p, "B", 200, 0, vec![a]);
        let _c = cpu_add_stage(&mut p, "C", 150, 0, vec![b]);
        let (path, cost) = cpu_compute_critical_path(&p);
        assert_eq!(path, vec![0, 1, 2]);
        assert_eq!(cost, 450);
    }

    #[test]
    fn test_critical_path_diamond_dag() {
        let mut p = create_pipeline(default_config());
        let a = cpu_add_stage(&mut p, "A", 100, 0, vec![]);
        let b = cpu_add_stage(&mut p, "B", 300, 0, vec![a]);
        let c = cpu_add_stage(&mut p, "C", 50, 0, vec![a]);
        let _d = cpu_add_stage(&mut p, "D", 100, 0, vec![b, c]);
        let (path, cost) = cpu_compute_critical_path(&p);
        // A(100) -> B(300) -> D(100) = 500
        assert_eq!(cost, 500);
        assert!(path.contains(&0));
        assert!(path.contains(&1));
        assert!(path.contains(&3));
        assert!(!path.contains(&2)); // C is not on critical path
    }

    #[test]
    fn test_critical_path_empty() {
        let p = create_pipeline(default_config());
        let (path, cost) = cpu_compute_critical_path(&p);
        assert!(path.is_empty());
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_critical_path_single_stage() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let (path, cost) = cpu_compute_critical_path(&p);
        assert_eq!(path, vec![0]);
        assert_eq!(cost, 150);
    }

    // ── Parallelism ───────────────────────────────────────

    #[test]
    fn test_parallelism_sequential_is_one() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 0, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let par = cpu_compute_parallelism(&plan);
        assert!((par - 1.0).abs() < 0.01, "sequential parallelism should be ~1.0, got {par}");
    }

    #[test]
    fn test_parallelism_independent_stages() {
        let mut p = create_pipeline(a770_config());
        cpu_add_stage(&mut p, "A", 100, 0, vec![]);
        cpu_add_stage(&mut p, "B", 100, 0, vec![]);
        let plan = cpu_schedule_pipelined(&p);
        let par = cpu_compute_parallelism(&plan);
        assert!(par > 1.0, "parallel stages should have parallelism > 1.0, got {par}");
    }

    #[test]
    fn test_parallelism_empty_plan() {
        let plan = ExecutionPlan {
            scheduled_ops: vec![],
            total_time_us: 0,
            pipeline_efficiency: 1.0,
            streams_used: 0,
        };
        assert_eq!(cpu_compute_parallelism(&plan), 0.0);
    }

    // ── Efficiency ────────────────────────────────────────

    #[test]
    fn test_efficiency_perfect() {
        let plan = ExecutionPlan {
            scheduled_ops: vec![ScheduledOp {
                stage_id: 0,
                stream_id: 0,
                start_us: 0,
                end_us: 100,
                op_type: OpCategory::Compute,
            }],
            total_time_us: 100,
            pipeline_efficiency: 1.0,
            streams_used: 1,
        };
        let eff = cpu_compute_efficiency(&plan);
        assert!((eff - 1.0).abs() < 0.01, "efficiency should be 1.0, got {eff}");
    }

    #[test]
    fn test_efficiency_in_range() {
        let mut p = create_pipeline(a770_config());
        let a = cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        cpu_add_stage(&mut p, "B", 200, 50, vec![a]);
        let plan = cpu_schedule_pipelined(&p);
        let eff = cpu_compute_efficiency(&plan);
        assert!(
            (0.0..=1.0).contains(&eff),
            "efficiency should be in [0,1], got {eff}"
        );
    }

    #[test]
    fn test_efficiency_half_utilized() {
        let plan = ExecutionPlan {
            scheduled_ops: vec![ScheduledOp {
                stage_id: 0,
                stream_id: 0,
                start_us: 0,
                end_us: 50,
                op_type: OpCategory::Compute,
            }],
            total_time_us: 100,
            pipeline_efficiency: 0.5,
            streams_used: 1,
        };
        let eff = cpu_compute_efficiency(&plan);
        assert!((eff - 0.5).abs() < 0.01, "should be ~0.5, got {eff}");
    }

    // ── Speedup ───────────────────────────────────────────

    #[test]
    fn test_speedup_pipelined_ge_one() {
        let mut p = create_pipeline(a770_config());
        let a = cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        cpu_add_stage(&mut p, "B", 100, 50, vec![a]);
        let seq = cpu_schedule_sequential(&p);
        let pipe = cpu_schedule_pipelined(&p);
        let speedup = cpu_estimate_speedup(&seq, &pipe);
        assert!(
            speedup >= 1.0,
            "pipelined speedup should be >= 1.0, got {speedup}"
        );
    }

    #[test]
    fn test_speedup_identity() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let seq = cpu_schedule_sequential(&p);
        let speedup = cpu_estimate_speedup(&seq, &seq);
        assert!((speedup - 1.0).abs() < 0.01);
    }

    // ── Bottleneck detection ──────────────────────────────

    #[test]
    fn test_bottleneck_finds_largest_stage() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "small", 10, 0, vec![]);
        cpu_add_stage(&mut p, "large", 1000, 0, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let (stage_id, desc) = cpu_find_bottleneck(&plan);
        assert_eq!(stage_id, 1);
        assert!(desc.contains("1000"));
    }

    #[test]
    fn test_bottleneck_empty_plan() {
        let plan = ExecutionPlan {
            scheduled_ops: vec![],
            total_time_us: 0,
            pipeline_efficiency: 1.0,
            streams_used: 0,
        };
        let (_, desc) = cpu_find_bottleneck(&plan);
        assert_eq!(desc, "empty plan");
    }

    // ── Optimization suggestions ──────────────────────────

    #[test]
    fn test_suggest_multi_stream() {
        let plan = ExecutionPlan {
            scheduled_ops: vec![
                ScheduledOp {
                    stage_id: 0,
                    stream_id: 0,
                    start_us: 0,
                    end_us: 100,
                    op_type: OpCategory::Compute,
                },
                ScheduledOp {
                    stage_id: 1,
                    stream_id: 0,
                    start_us: 100,
                    end_us: 200,
                    op_type: OpCategory::H2DTransfer,
                },
                ScheduledOp {
                    stage_id: 2,
                    stream_id: 0,
                    start_us: 200,
                    end_us: 300,
                    op_type: OpCategory::Compute,
                },
            ],
            total_time_us: 300,
            pipeline_efficiency: 0.3,
            streams_used: 1,
        };
        let suggestions = cpu_suggest_optimization(&plan);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_suggest_nonempty() {
        let mut p = create_pipeline(a770_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_pipelined(&p);
        let suggestions = cpu_suggest_optimization(&plan);
        assert!(!suggestions.is_empty());
    }

    // ── Simulate ──────────────────────────────────────────

    #[test]
    fn test_simulate_compute_time() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let stats = cpu_simulate_execution(&plan);
        assert_eq!(stats.compute_time_us, 100);
    }

    #[test]
    fn test_simulate_transfer_time() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let stats = cpu_simulate_execution(&plan);
        assert_eq!(stats.transfer_time_us, 100); // H2D(50) + D2H(50)
    }

    #[test]
    fn test_simulate_throughput_positive() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let stats = cpu_simulate_execution(&plan);
        assert!(stats.throughput_items_per_sec > 0.0);
    }

    #[test]
    fn test_simulate_overlap_pct_sequential() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let stats = cpu_simulate_execution(&plan);
        assert!(
            stats.overlap_pct < 1.0,
            "sequential should have ~0% overlap, got {}%",
            stats.overlap_pct
        );
    }

    // ── Format ────────────────────────────────────────────

    #[test]
    fn test_format_execution_plan_header() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let formatted = format_execution_plan(&plan);
        assert!(formatted.contains("Execution Plan"));
        assert!(formatted.contains("µs total"));
    }

    #[test]
    fn test_format_execution_plan_ops() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 50, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let formatted = format_execution_plan(&plan);
        assert!(formatted.contains("Compute"));
        assert!(formatted.contains("H2D"));
    }

    // ── Edge cases ────────────────────────────────────────

    #[test]
    fn test_edge_single_stage_pipeline() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "only", 500, 100, vec![]);
        let seq = cpu_schedule_sequential(&p);
        let pipe = cpu_schedule_pipelined(&p);
        // Single stage: pipelined can't improve over sequential
        assert!(pipe.total_time_us <= seq.total_time_us);
    }

    #[test]
    fn test_edge_all_independent_max_parallel() {
        let mut p = create_pipeline(a770_config());
        for i in 0..5 {
            cpu_add_stage(&mut p, &format!("s{i}"), 100, 0, vec![]);
        }
        let plan = cpu_schedule_pipelined(&p);
        let par = cpu_compute_parallelism(&plan);
        assert!(par > 1.0, "independent stages should yield parallelism > 1.0");
    }

    #[test]
    fn test_edge_all_sequential_min_parallel() {
        let mut p = create_pipeline(a770_config());
        let mut prev = cpu_add_stage(&mut p, "s0", 100, 0, vec![]);
        for i in 1..5 {
            prev = cpu_add_stage(&mut p, &format!("s{i}"), 100, 0, vec![prev]);
        }
        let plan = cpu_schedule_pipelined(&p);
        // All chained: parallelism should be low
        let par = cpu_compute_parallelism(&plan);
        assert!(par <= 2.0, "fully chained should have low parallelism, got {par}");
    }

    // ── Property tests ────────────────────────────────────

    #[test]
    fn test_property_pipelined_le_sequential() {
        let mut p = create_pipeline(a770_config());
        let a = cpu_add_stage(&mut p, "A", 200, 100, vec![]);
        let b = cpu_add_stage(&mut p, "B", 150, 80, vec![]);
        cpu_add_stage(&mut p, "C", 300, 50, vec![a, b]);
        let seq = cpu_schedule_sequential(&p);
        let pipe = cpu_schedule_pipelined(&p);
        assert!(
            pipe.total_time_us <= seq.total_time_us,
            "pipelined {}µs > sequential {}µs",
            pipe.total_time_us,
            seq.total_time_us,
        );
    }

    #[test]
    fn test_property_efficiency_in_unit_range() {
        let mut p = create_pipeline(a770_config());
        for i in 0..8 {
            cpu_add_stage(&mut p, &format!("s{i}"), 100 + i as u64 * 50, 30, vec![]);
        }
        let plan = cpu_schedule_pipelined(&p);
        let eff = cpu_compute_efficiency(&plan);
        assert!(
            (0.0..=1.0).contains(&eff),
            "efficiency must be in [0,1], got {eff}"
        );
    }

    // ── A770-specific tests ───────────────────────────────

    #[test]
    fn test_a770_two_compute_one_copy() {
        let cfg = a770_config();
        assert_eq!(cfg.num_streams, 3);
        let mut p = create_pipeline(cfg);
        cpu_add_stage(&mut p, "gemm", 500, 100, vec![]);
        cpu_add_stage(&mut p, "softmax", 50, 20, vec![]);
        cpu_add_stage(&mut p, "layernorm", 30, 10, vec![]);
        let plan = cpu_schedule_pipelined(&p);
        assert!(plan.streams_used >= 2, "A770 should use multiple streams");
    }

    #[test]
    fn test_a770_copy_stream_separate() {
        let mut p = create_pipeline(a770_config());
        cpu_add_stage(&mut p, "A", 200, 100, vec![]);
        cpu_add_stage(&mut p, "B", 200, 100, vec![]);
        let plan = cpu_schedule_pipelined(&p);
        // Copy ops should all be on stream 0
        let copy_ops: Vec<_> = plan
            .scheduled_ops
            .iter()
            .filter(|op| op.op_type == OpCategory::H2DTransfer)
            .collect();
        for op in &copy_ops {
            assert_eq!(op.stream_id, 0, "H2D transfers should be on copy stream 0");
        }
    }

    // ── Topological sort / cycles ─────────────────────────

    #[test]
    fn test_topological_sort_detects_cycle() {
        // Manually build a cycle: A -> B -> A
        let stages = vec![
            PipelineStage {
                id: 0,
                name: "A".to_string(),
                compute_us: 100,
                transfer_us: 0,
                dependencies: vec![1],
            },
            PipelineStage {
                id: 1,
                name: "B".to_string(),
                compute_us: 100,
                transfer_us: 0,
                dependencies: vec![0],
            },
        ];
        let result = topological_sort(&stages);
        assert_eq!(result, Err(OptimizeError::CyclicDependency));
    }

    #[test]
    fn test_topological_sort_no_cycle() {
        let stages = vec![
            PipelineStage {
                id: 0,
                name: "A".to_string(),
                compute_us: 100,
                transfer_us: 0,
                dependencies: vec![],
            },
            PipelineStage {
                id: 1,
                name: "B".to_string(),
                compute_us: 100,
                transfer_us: 0,
                dependencies: vec![0],
            },
        ];
        let result = topological_sort(&stages);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0, 1]);
    }

    // ── Error types ───────────────────────────────────────

    #[test]
    fn test_optimize_error_display() {
        assert_eq!(
            OptimizeError::CyclicDependency.to_string(),
            "cyclic dependency detected"
        );
        assert!(OptimizeError::InfeasibleSchedule("test".into())
            .to_string()
            .contains("test"));
        assert!(OptimizeError::StreamLimitExceeded(4)
            .to_string()
            .contains('4'));
    }

    #[test]
    fn test_op_category_display() {
        assert_eq!(OpCategory::Compute.to_string(), "Compute");
        assert_eq!(OpCategory::H2DTransfer.to_string(), "H2D");
        assert_eq!(OpCategory::D2HTransfer.to_string(), "D2H");
        assert_eq!(OpCategory::Synchronize.to_string(), "Sync");
    }

    #[test]
    fn test_pipeline_config_default() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.num_streams, 3);
        assert!(cfg.enable_overlap);
        assert_eq!(cfg.prefetch_depth, 1);
        assert!(!cfg.double_buffer);
    }

    // ── Additional coverage ───────────────────────────────

    #[test]
    fn test_zero_compute_stage() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "transfer_only", 0, 100, vec![]);
        let plan = cpu_schedule_sequential(&p);
        assert_eq!(plan.total_time_us, 200); // H2D(100) + D2H(100)
    }

    #[test]
    fn test_many_independent_stages() {
        let mut p = create_pipeline(a770_config());
        for i in 0..20 {
            cpu_add_stage(&mut p, &format!("s{i}"), 50, 10, vec![]);
        }
        let seq = cpu_schedule_sequential(&p);
        let pipe = cpu_schedule_pipelined(&p);
        assert!(pipe.total_time_us < seq.total_time_us);
    }

    #[test]
    fn test_simulate_idle_time_sequential() {
        let mut p = create_pipeline(default_config());
        cpu_add_stage(&mut p, "A", 100, 0, vec![]);
        let plan = cpu_schedule_sequential(&p);
        let stats = cpu_simulate_execution(&plan);
        // 1 stream, all utilized → 0 idle
        assert_eq!(stats.idle_time_us, 0);
    }

    #[test]
    fn test_speedup_with_many_stages() {
        let mut p = create_pipeline(a770_config());
        for _ in 0..10 {
            cpu_add_stage(&mut p, "s", 100, 50, vec![]);
        }
        let seq = cpu_schedule_sequential(&p);
        let pipe = cpu_schedule_pipelined(&p);
        let speedup = cpu_estimate_speedup(&seq, &pipe);
        assert!(speedup >= 1.0);
    }
}

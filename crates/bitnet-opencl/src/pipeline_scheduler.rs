//! Pipeline scheduler that overlaps transfer / compute / readback stages
//! across successive tokens for maximum GPU utilisation.
//!
//! The scheduler builds a DAG of [`PipelineStage`] operations and
//! dispatches them through an [`AsyncGpuExecutor`], allowing stage *N+1*
//! transfer to overlap with stage *N* compute and stage *N-1* readback.

use std::collections::VecDeque;
use std::fmt;

use crate::async_executor::{AsyncGpuExecutor, ExecutionPlan, GpuEvent, PipelineStage};
use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Minimum allowed pipeline depth.
pub const MIN_PIPELINE_DEPTH: usize = 1;

/// Maximum allowed pipeline depth.
pub const MAX_PIPELINE_DEPTH: usize = 4;

// ---------------------------------------------------------------------------
// DagNode
// ---------------------------------------------------------------------------

/// A single node in the operation DAG.
#[derive(Debug, Clone)]
pub struct DagNode {
    /// Logical pipeline stage.
    pub stage: PipelineStage,
    /// Human-readable label for diagnostics.
    pub label: String,
    /// Indices of nodes this node depends on.
    pub deps: Vec<usize>,
}

// ---------------------------------------------------------------------------
// TokenPipeline – per-token 3-stage sequence
// ---------------------------------------------------------------------------

/// Represents the three canonical stages for one token's processing.
#[derive(Debug, Clone)]
struct TokenPipeline {
    token_index: usize,
    /// Index of each stage's node inside the DAG (Transfer, Compute, Readback).
    node_indices: [usize; 3],
}

// ---------------------------------------------------------------------------
// PipelineScheduler
// ---------------------------------------------------------------------------

/// Schedules overlapping GPU pipeline stages across successive tokens.
///
/// ```text
/// Token 0:  [Transfer₀] → [Compute₀] → [Readback₀]
/// Token 1:       [Transfer₁] → [Compute₁] → [Readback₁]
/// Token 2:            [Transfer₂] → [Compute₂] → [Readback₂]
/// ```
///
/// The depth controls how many tokens may be in-flight simultaneously.
pub struct PipelineScheduler {
    /// Maximum number of in-flight token pipelines.
    depth: usize,
    /// DAG of all operations.
    dag: Vec<DagNode>,
    /// Per-token bookkeeping.
    pipelines: VecDeque<TokenPipeline>,
    /// Tokens that have been fully completed (readback done).
    completed_tokens: usize,
}

impl PipelineScheduler {
    /// Create a scheduler with the given pipeline `depth` (1–4).
    pub fn new(depth: usize) -> GpuResult<Self> {
        if !(MIN_PIPELINE_DEPTH..=MAX_PIPELINE_DEPTH).contains(&depth) {
            return Err(GpuError::InvalidPipelineDepth {
                depth,
                min: MIN_PIPELINE_DEPTH,
                max: MAX_PIPELINE_DEPTH,
            });
        }
        Ok(Self { depth, dag: Vec::new(), pipelines: VecDeque::new(), completed_tokens: 0 })
    }

    /// Pipeline depth.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Total DAG nodes.
    #[must_use]
    pub const fn dag_len(&self) -> usize {
        self.dag.len()
    }

    /// Number of token pipelines currently in-flight.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.pipelines.len()
    }

    /// Number of tokens whose readback has completed.
    #[must_use]
    pub const fn completed_tokens(&self) -> usize {
        self.completed_tokens
    }

    /// Returns `true` when no operations are scheduled.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.pipelines.is_empty()
    }

    // -- scheduling --------------------------------------------------------

    /// Schedule the three canonical stages for the next token.
    ///
    /// Returns `None` if the pipeline is at maximum depth (caller should
    /// drain at least one token first via [`drain_completed`]).
    pub fn schedule_token(&mut self) -> Option<usize> {
        if self.pipelines.len() >= self.depth {
            return None;
        }

        let token_index = self.completed_tokens + self.pipelines.len();

        // Dependencies: Transfer → Compute → Readback (within a token).
        // Cross-token: Compute[N] depends on Transfer[N] AND Compute[N-1].
        let prev_compute = self.pipelines.back().map(|p| p.node_indices[1]);

        let transfer_idx =
            self.add_dag_node(PipelineStage::Transfer, format!("transfer_tok{token_index}"), &[]);

        // Compute depends on this token's transfer and (optionally)
        // the previous token's compute to enforce ordering.
        let mut compute_deps = vec![transfer_idx];
        if let Some(prev) = prev_compute {
            compute_deps.push(prev);
        }
        let compute_idx = self.add_dag_node(
            PipelineStage::Compute,
            format!("compute_tok{token_index}"),
            &compute_deps,
        );

        let readback_idx = self.add_dag_node(
            PipelineStage::Readback,
            format!("readback_tok{token_index}"),
            &[compute_idx],
        );

        self.pipelines.push_back(TokenPipeline {
            token_index,
            node_indices: [transfer_idx, compute_idx, readback_idx],
        });

        Some(token_index)
    }

    /// Build an [`ExecutionPlan`] from the current DAG, suitable for
    /// submission to an [`AsyncGpuExecutor`].
    #[must_use]
    pub fn build_plan(&self) -> ExecutionPlan {
        let mut plan = ExecutionPlan::new();
        for node in &self.dag {
            plan.add_op(node.stage, &node.label, &node.deps);
        }
        plan
    }

    /// Execute the current DAG on the given executor and return events.
    pub fn execute(&self, executor: &AsyncGpuExecutor) -> GpuResult<Vec<GpuEvent>> {
        let plan = self.build_plan();
        executor.execute_plan(&plan)
    }

    /// Mark the oldest in-flight token as completed and remove it.
    ///
    /// Returns the token index that was drained, or `None` if empty.
    pub fn drain_completed(&mut self) -> Option<usize> {
        let pipeline = self.pipelines.pop_front()?;
        self.completed_tokens += 1;
        Some(pipeline.token_index)
    }

    /// Reset the scheduler, clearing all DAG nodes and in-flight state.
    pub fn reset(&mut self) {
        self.dag.clear();
        self.pipelines.clear();
        self.completed_tokens = 0;
    }

    // -- DAG helpers -------------------------------------------------------

    fn add_dag_node(&mut self, stage: PipelineStage, label: String, deps: &[usize]) -> usize {
        let idx = self.dag.len();
        self.dag.push(DagNode { stage, label, deps: deps.to_vec() });
        idx
    }
}

impl fmt::Debug for PipelineScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineScheduler")
            .field("depth", &self.depth)
            .field("dag_nodes", &self.dag.len())
            .field("in_flight", &self.pipelines.len())
            .field("completed_tokens", &self.completed_tokens)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_depth_bounds() {
        assert!(PipelineScheduler::new(0).is_err());
        assert!(PipelineScheduler::new(5).is_err());
        assert!(PipelineScheduler::new(1).is_ok());
        assert!(PipelineScheduler::new(4).is_ok());
    }

    #[test]
    fn schedule_single_token() {
        let mut sched = PipelineScheduler::new(1).unwrap();
        let tok = sched.schedule_token();
        assert_eq!(tok, Some(0));
        // At depth 1 the next schedule must fail.
        assert_eq!(sched.schedule_token(), None);
    }

    #[test]
    fn schedule_two_tokens_depth2() {
        let mut sched = PipelineScheduler::new(2).unwrap();
        assert_eq!(sched.schedule_token(), Some(0));
        assert_eq!(sched.schedule_token(), Some(1));
        assert_eq!(sched.schedule_token(), None);
    }

    #[test]
    fn drain_allows_more_tokens() {
        let mut sched = PipelineScheduler::new(1).unwrap();
        sched.schedule_token();
        assert_eq!(sched.drain_completed(), Some(0));
        assert_eq!(sched.completed_tokens(), 1);
        assert_eq!(sched.schedule_token(), Some(1));
    }

    #[test]
    fn dag_has_correct_structure() {
        let mut sched = PipelineScheduler::new(2).unwrap();
        sched.schedule_token(); // tok 0: nodes 0,1,2
        sched.schedule_token(); // tok 1: nodes 3,4,5

        // 6 DAG nodes total.
        assert_eq!(sched.dag_len(), 6);

        // Tok-0 transfer has no deps.
        assert!(sched.dag[0].deps.is_empty());
        // Tok-0 compute depends on tok-0 transfer.
        assert_eq!(sched.dag[1].deps, vec![0]);
        // Tok-0 readback depends on tok-0 compute.
        assert_eq!(sched.dag[2].deps, vec![1]);

        // Tok-1 transfer has no deps.
        assert!(sched.dag[3].deps.is_empty());
        // Tok-1 compute depends on tok-1 transfer AND tok-0 compute.
        assert_eq!(sched.dag[4].deps, vec![3, 1]);
        // Tok-1 readback depends on tok-1 compute.
        assert_eq!(sched.dag[5].deps, vec![4]);
    }

    #[test]
    fn build_plan_matches_dag() {
        let mut sched = PipelineScheduler::new(1).unwrap();
        sched.schedule_token();
        let plan = sched.build_plan();
        assert_eq!(plan.len(), 3);
    }

    #[test]
    fn execute_on_mock_executor() {
        let exec = AsyncGpuExecutor::new(3);
        let mut sched = PipelineScheduler::new(2).unwrap();
        sched.schedule_token();
        sched.schedule_token();

        let events = sched.execute(&exec).unwrap();
        assert_eq!(events.len(), 6);
        for e in &events {
            assert!(e.is_complete());
        }
    }

    #[test]
    fn reset_clears_everything() {
        let mut sched = PipelineScheduler::new(2).unwrap();
        sched.schedule_token();
        sched.schedule_token();
        sched.reset();
        assert!(sched.is_idle());
        assert_eq!(sched.dag_len(), 0);
        assert_eq!(sched.completed_tokens(), 0);
    }
}

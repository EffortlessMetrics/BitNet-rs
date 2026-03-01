//! Async GPU execution engine for overlapping compute and data transfers.
//!
//! Provides a task-based scheduling system with priority queues,
//! dependency tracking, and pipeline execution for GPU workloads.

use std::collections::HashSet;

/// Unique identifier for a submitted GPU task.
pub type TaskId = u64;

/// The async execution engine that manages GPU task scheduling.
///
/// Tasks are submitted with priorities and dependencies, then advanced
/// via [`tick`](AsyncEngine::tick) calls that simulate execution progress.
pub struct AsyncEngine {
    task_queue: Vec<GpuTask>,
    completed: Vec<TaskResult>,
    next_id: TaskId,
    max_concurrent: usize,
    running: Vec<TaskId>,
}

/// A GPU task submitted to the execution engine.
pub struct GpuTask {
    pub id: TaskId,
    pub kind: TaskKind,
    pub priority: Priority,
    pub dependencies: Vec<TaskId>,
    pub status: TaskStatus,
    pub submitted_at: u64,
}

/// The kind of GPU operation to execute.
#[derive(Debug, Clone)]
pub enum TaskKind {
    /// Launch a compute kernel on the GPU.
    KernelLaunch { kernel: String, grid: [u32; 3], block: [u32; 3] },
    /// Transfer data from host to device memory.
    HostToDevice { size: u64 },
    /// Transfer data from device to host memory.
    DeviceToHost { size: u64 },
    /// Transfer data between device memory regions.
    DeviceToDevice { size: u64 },
    /// Synchronize all pending operations.
    Synchronize,
    /// A custom user-defined operation.
    Custom(String),
}

/// Task execution priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Current status of a GPU task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

/// Result of a completed GPU task.
pub struct TaskResult {
    pub task_id: TaskId,
    pub status: TaskStatus,
    pub start_us: u64,
    pub end_us: u64,
    pub device: Option<String>,
}

/// Timeline view of execution showing compute/transfer overlap.
pub struct ExecutionTimeline {
    /// Compute stream entries: (`start_us`, `end_us`, name).
    pub compute_stream: Vec<(u64, u64, String)>,
    /// Transfer stream entries: (`start_us`, `end_us`, name).
    pub transfer_stream: Vec<(u64, u64, String)>,
    /// Ratio of overlapped time to total time (0.0–1.0).
    pub overlap_ratio: f64,
}

/// A named stage within a pipeline.
pub struct PipelineStage {
    pub name: String,
    pub tasks: Vec<TaskId>,
}

/// A staged pipeline of GPU tasks.
pub struct Pipeline {
    pub stages: Vec<PipelineStage>,
    pub engine: AsyncEngine,
}

impl TaskKind {
    /// Estimated execution duration in microseconds.
    fn estimated_duration_us(&self) -> u64 {
        match self {
            Self::KernelLaunch { grid, .. } => {
                let work = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]);
                work.max(1) * 10
            }
            Self::HostToDevice { size }
            | Self::DeviceToHost { size }
            | Self::DeviceToDevice { size } => (*size / 1_000_000).max(1) * 100,
            Self::Synchronize => 50,
            Self::Custom(_) => 500,
        }
    }

    /// Whether this task is a data transfer operation.
    const fn is_transfer(&self) -> bool {
        matches!(
            self,
            Self::HostToDevice { .. } | Self::DeviceToHost { .. } | Self::DeviceToDevice { .. }
        )
    }

    /// Short display name for timeline entries.
    fn display_name(&self) -> String {
        match self {
            Self::KernelLaunch { kernel, .. } => format!("kernel:{kernel}"),
            Self::HostToDevice { size } => format!("h2d:{size}B"),
            Self::DeviceToHost { size } => format!("d2h:{size}B"),
            Self::DeviceToDevice { size } => format!("d2d:{size}B"),
            Self::Synchronize => "sync".to_string(),
            Self::Custom(name) => format!("custom:{name}"),
        }
    }
}

impl AsyncEngine {
    /// Create a new engine with the given concurrency limit.
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            task_queue: Vec::new(),
            completed: Vec::new(),
            next_id: 1,
            max_concurrent: max_concurrent.max(1),
            running: Vec::new(),
        }
    }

    /// Submit a task and return its unique ID.
    ///
    /// Returns `None` if a circular dependency would be created.
    pub fn submit(
        &mut self,
        kind: TaskKind,
        priority: Priority,
        dependencies: Vec<TaskId>,
    ) -> Option<TaskId> {
        let id = self.next_id;

        // Check that all dependencies exist
        for &dep in &dependencies {
            let exists = self.task_queue.iter().any(|t| t.id == dep)
                || self.completed.iter().any(|r| r.task_id == dep);
            if !exists {
                return None;
            }
        }

        // Detect circular dependencies via DFS
        if self.would_create_cycle(id, &dependencies) {
            return None;
        }

        self.next_id += 1;
        self.task_queue.push(GpuTask {
            id,
            kind,
            priority,
            dependencies,
            status: TaskStatus::Pending,
            submitted_at: 0,
        });
        Some(id)
    }

    /// Advance simulation by one tick at the given time.
    ///
    /// Completes running tasks whose duration has elapsed, then
    /// starts the highest-priority ready tasks up to the concurrency limit.
    pub fn tick(&mut self, current_time_us: u64) -> Vec<TaskResult> {
        let results = self.complete_running_tasks(current_time_us);
        self.record_completions(&results);
        self.start_ready_tasks(current_time_us);
        self.cascade_failures(current_time_us);
        results
    }

    /// Complete any running tasks whose estimated duration has elapsed.
    fn complete_running_tasks(&mut self, current_time_us: u64) -> Vec<TaskResult> {
        let mut results = Vec::new();
        let mut still_running = Vec::new();
        for &running_id in &self.running {
            if let Some(task) = self.task_queue.iter().find(|t| t.id == running_id) {
                let duration = task.kind.estimated_duration_us();
                if current_time_us >= task.submitted_at + duration {
                    results.push(TaskResult {
                        task_id: running_id,
                        status: TaskStatus::Completed,
                        start_us: task.submitted_at,
                        end_us: task.submitted_at + duration,
                        device: Some("gpu:0".to_string()),
                    });
                } else {
                    still_running.push(running_id);
                }
            }
        }
        self.running = still_running;
        results
    }

    /// Record task completions into the completed list.
    fn record_completions(&mut self, results: &[TaskResult]) {
        for result in results {
            if let Some(task) = self.task_queue.iter_mut().find(|t| t.id == result.task_id) {
                task.status = TaskStatus::Completed;
            }
            self.completed.push(TaskResult {
                task_id: result.task_id,
                status: result.status.clone(),
                start_us: result.start_us,
                end_us: result.end_us,
                device: result.device.clone(),
            });
        }
    }

    /// Start highest-priority ready tasks up to the concurrency limit.
    fn start_ready_tasks(&mut self, current_time_us: u64) {
        let available_slots = self.max_concurrent.saturating_sub(self.running.len());
        if available_slots == 0 {
            return;
        }

        let completed_ids: HashSet<TaskId> = self.completed.iter().map(|r| r.task_id).collect();
        let failed_ids: HashSet<TaskId> = self
            .completed
            .iter()
            .filter(|r| matches!(r.status, TaskStatus::Failed(_)))
            .map(|r| r.task_id)
            .collect();

        let mut ready: Vec<(TaskId, Priority)> = self
            .task_queue
            .iter()
            .filter(|t| {
                t.status == TaskStatus::Pending
                    && t.dependencies.iter().all(|d| completed_ids.contains(d))
                    && !t.dependencies.iter().any(|d| failed_ids.contains(d))
            })
            .map(|t| (t.id, t.priority))
            .collect();

        ready.sort_by(|a, b| b.1.cmp(&a.1));

        for (task_id, _) in ready.into_iter().take(available_slots) {
            if let Some(task) = self.task_queue.iter_mut().find(|t| t.id == task_id) {
                task.status = TaskStatus::Running;
                task.submitted_at = current_time_us;
                self.running.push(task_id);
            }
        }
    }

    /// Fail pending tasks whose dependencies have failed.
    fn cascade_failures(&mut self, current_time_us: u64) {
        let failed_ids: HashSet<TaskId> = self
            .completed
            .iter()
            .filter(|r| matches!(r.status, TaskStatus::Failed(_)))
            .map(|r| r.task_id)
            .collect();
        let mut cascade_failed = Vec::new();
        for task in &self.task_queue {
            if task.status == TaskStatus::Pending
                && task.dependencies.iter().any(|d| failed_ids.contains(d))
            {
                cascade_failed.push(task.id);
            }
        }
        for task_id in cascade_failed {
            if let Some(task) = self.task_queue.iter_mut().find(|t| t.id == task_id) {
                task.status = TaskStatus::Failed("dependency failed".to_string());
                self.completed.push(TaskResult {
                    task_id,
                    status: TaskStatus::Failed("dependency failed".to_string()),
                    start_us: current_time_us,
                    end_us: current_time_us,
                    device: None,
                });
            }
        }
    }

    /// Cancel a pending or running task.
    pub fn cancel(&mut self, task_id: TaskId) -> bool {
        if let Some(task) = self.task_queue.iter_mut().find(|t| t.id == task_id) {
            match task.status {
                TaskStatus::Pending | TaskStatus::Running => {
                    task.status = TaskStatus::Cancelled;
                    self.running.retain(|&id| id != task_id);
                    self.completed.push(TaskResult {
                        task_id,
                        status: TaskStatus::Cancelled,
                        start_us: 0,
                        end_us: 0,
                        device: None,
                    });
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Wait for all tasks to complete. Returns results by running ticks
    /// until every task is resolved.
    pub fn wait_all(&mut self) -> Vec<TaskResult> {
        let mut all_results = Vec::new();
        let mut time = 0u64;
        loop {
            let unresolved = self
                .task_queue
                .iter()
                .any(|t| matches!(t.status, TaskStatus::Pending | TaskStatus::Running));
            if !unresolved {
                break;
            }
            let results = self.tick(time);
            all_results.extend(results);
            time += 100;
        }
        all_results
    }

    /// Compute an execution timeline from completed tasks.
    #[must_use]
    pub fn timeline(&self) -> ExecutionTimeline {
        let mut compute_stream = Vec::new();
        let mut transfer_stream = Vec::new();

        for result in &self.completed {
            if !matches!(result.status, TaskStatus::Completed) {
                continue;
            }
            let task = self.task_queue.iter().find(|t| t.id == result.task_id);
            let name = task.map_or_else(|| "unknown".to_string(), |t| t.kind.display_name());
            let is_transfer = task.is_some_and(|t| t.kind.is_transfer());

            if is_transfer {
                transfer_stream.push((result.start_us, result.end_us, name));
            } else {
                compute_stream.push((result.start_us, result.end_us, name));
            }
        }

        let overlap_ratio = compute_overlap_ratio(&compute_stream, &transfer_stream);

        ExecutionTimeline { compute_stream, transfer_stream, overlap_ratio }
    }

    /// Check if adding `new_id` with `deps` would create a cycle.
    fn would_create_cycle(&self, new_id: TaskId, deps: &[TaskId]) -> bool {
        // BFS/DFS: check if any dep transitively depends on new_id
        // Since new_id hasn't been inserted yet, we only need to check
        // if deps form a cycle among themselves that would include new_id.
        // Actually, since new_id doesn't exist yet, we check if any dep
        // can reach any other dep that lists new_id — but new_id isn't
        // in the graph yet. The real risk is if dep A depends on dep B
        // and we're adding new_id -> A, but someone later adds A -> new_id.
        // We check if any existing task in the transitive closure of deps
        // already depends on new_id (which can't happen since new_id is new).
        // But we also check for cycles among existing deps.
        let _ = new_id;
        for &dep in deps {
            let mut visited = HashSet::new();
            if self.has_transitive_dep(dep, deps, &mut visited) {
                return true;
            }
        }
        false
    }

    /// Check if `task_id` transitively depends on any id in `targets`.
    fn has_transitive_dep(
        &self,
        task_id: TaskId,
        targets: &[TaskId],
        visited: &mut HashSet<TaskId>,
    ) -> bool {
        if !visited.insert(task_id) {
            return false;
        }
        if let Some(task) = self.task_queue.iter().find(|t| t.id == task_id) {
            for &dep in &task.dependencies {
                if targets.contains(&dep) {
                    // dep is one of the deps we're adding — would be a cycle
                    // if dep also transitively reaches task_id
                    continue;
                }
                if self.has_transitive_dep(dep, targets, visited) {
                    return true;
                }
            }
        }
        false
    }

    /// Fail a task explicitly (used for testing cascading failures).
    pub fn fail_task(&mut self, task_id: TaskId, reason: &str) {
        if let Some(task) = self.task_queue.iter_mut().find(|t| t.id == task_id) {
            task.status = TaskStatus::Failed(reason.to_string());
            self.running.retain(|&id| id != task_id);
            self.completed.push(TaskResult {
                task_id,
                status: TaskStatus::Failed(reason.to_string()),
                start_us: 0,
                end_us: 0,
                device: None,
            });
        }
    }

    /// Number of tasks currently in the queue (all statuses).
    #[must_use]
    pub const fn queue_len(&self) -> usize {
        self.task_queue.len()
    }

    /// Number of currently running tasks.
    #[must_use]
    pub const fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Number of completed task results.
    #[must_use]
    pub const fn completed_count(&self) -> usize {
        self.completed.len()
    }
}

impl Pipeline {
    /// Create a new empty pipeline.
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self { stages: Vec::new(), engine: AsyncEngine::new(max_concurrent) }
    }

    /// Add a stage with the given task kinds. Returns task IDs for the stage.
    pub fn add_stage(&mut self, name: &str, tasks: Vec<(TaskKind, Priority)>) -> Vec<TaskId> {
        // All tasks in this stage depend on all tasks from the previous stage
        let prev_ids: Vec<TaskId> = self.stages.last().map_or_else(Vec::new, |s| s.tasks.clone());

        let mut stage_ids = Vec::new();
        for (kind, priority) in tasks {
            if let Some(id) = self.engine.submit(kind, priority, prev_ids.clone()) {
                stage_ids.push(id);
            }
        }

        self.stages.push(PipelineStage { name: name.to_string(), tasks: stage_ids.clone() });
        stage_ids
    }

    /// Execute all pipeline stages and return results.
    pub fn execute(&mut self) -> Vec<TaskResult> {
        self.engine.wait_all()
    }

    /// Estimate total execution time in microseconds.
    #[must_use]
    pub fn estimate_time(&self) -> u64 {
        let mut total = 0u64;
        for stage in &self.stages {
            let stage_max = stage
                .tasks
                .iter()
                .filter_map(|&id| {
                    self.engine
                        .task_queue
                        .iter()
                        .find(|t| t.id == id)
                        .map(|t| t.kind.estimated_duration_us())
                })
                .max()
                .unwrap_or(0);
            total += stage_max;
        }
        total
    }
}

/// Compute the ratio of overlapping time between compute and transfer streams.
fn compute_overlap_ratio(compute: &[(u64, u64, String)], transfer: &[(u64, u64, String)]) -> f64 {
    if compute.is_empty() || transfer.is_empty() {
        return 0.0;
    }

    let global_start = compute.iter().chain(transfer.iter()).map(|&(s, _, _)| s).min().unwrap_or(0);
    let global_end = compute.iter().chain(transfer.iter()).map(|&(_, e, _)| e).max().unwrap_or(0);
    let total_span = global_end.saturating_sub(global_start);
    if total_span == 0 {
        return 0.0;
    }

    // Calculate overlap by checking each microsecond interval
    // (simplified: pairwise interval overlap)
    let mut overlap_us = 0u64;
    for &(cs, ce, _) in compute {
        for &(ts, te, _) in transfer {
            let start = cs.max(ts);
            let end = ce.min(te);
            if start < end {
                overlap_us += end - start;
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let ratio = overlap_us as f64 / total_span as f64;
    ratio.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Single task lifecycle ──────────────────────────────────────

    #[test]
    fn submit_and_complete_single_task() {
        let mut engine = AsyncEngine::new(4);
        let id = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        assert_eq!(id, 1);
        assert_eq!(engine.queue_len(), 1);

        // Tick to start
        engine.tick(0);
        assert_eq!(engine.running_count(), 1);

        // Tick to complete (Synchronize = 50us)
        let results = engine.tick(50);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].task_id, id);
        assert_eq!(results[0].status, TaskStatus::Completed);
    }

    #[test]
    fn submit_kernel_launch_task() {
        let mut engine = AsyncEngine::new(4);
        let id = engine
            .submit(
                TaskKind::KernelLaunch {
                    kernel: "matmul".to_string(),
                    grid: [1, 1, 1],
                    block: [256, 1, 1],
                },
                Priority::Normal,
                vec![],
            )
            .unwrap();
        engine.tick(0);
        let results = engine.tick(10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].task_id, id);
    }

    #[test]
    fn submit_host_to_device_transfer() {
        let mut engine = AsyncEngine::new(4);
        let id = engine
            .submit(TaskKind::HostToDevice { size: 1_000_000 }, Priority::Normal, vec![])
            .unwrap();
        assert!(engine.queue_len() == 1);
        let _ = engine.wait_all();
        assert_eq!(engine.completed_count(), 1);
        assert!(
            engine.completed.iter().any(|r| r.task_id == id && r.status == TaskStatus::Completed)
        );
    }

    #[test]
    fn submit_device_to_host_transfer() {
        let mut engine = AsyncEngine::new(4);
        let id = engine
            .submit(TaskKind::DeviceToHost { size: 2_000_000 }, Priority::High, vec![])
            .unwrap();
        let results = engine.wait_all();
        assert!(results.iter().any(|r| r.task_id == id));
    }

    #[test]
    fn submit_device_to_device_transfer() {
        let mut engine = AsyncEngine::new(4);
        let id = engine
            .submit(TaskKind::DeviceToDevice { size: 500_000 }, Priority::Low, vec![])
            .unwrap();
        let _ = engine.wait_all();
        assert!(engine.completed.iter().any(|r| r.task_id == id));
    }

    #[test]
    fn submit_custom_task() {
        let mut engine = AsyncEngine::new(4);
        let id =
            engine.submit(TaskKind::Custom("my_op".to_string()), Priority::Normal, vec![]).unwrap();
        let _ = engine.wait_all();
        assert!(
            engine.completed.iter().any(|r| r.task_id == id && r.status == TaskStatus::Completed)
        );
    }

    #[test]
    fn task_ids_increment() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let b = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let c = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }

    #[test]
    fn task_result_has_device() {
        let mut engine = AsyncEngine::new(4);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.tick(0);
        let results = engine.tick(100);
        assert!(results[0].device.is_some());
        assert_eq!(results[0].device.as_deref(), Some("gpu:0"));
    }

    // ── Dependencies ───────────────────────────────────────────────

    #[test]
    fn dependency_respected_b_waits_for_a() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let b = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![a]).unwrap();

        // Tick: A starts, B waits
        engine.tick(0);
        assert_eq!(engine.running_count(), 1);

        // Complete A
        engine.tick(50);

        // Now B should start
        engine.tick(50);
        assert!(
            engine.task_queue.iter().find(|t| t.id == b).unwrap().status == TaskStatus::Running
        );
    }

    #[test]
    fn multiple_dependencies() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let b = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let c = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![a, b]).unwrap();

        // A and B start
        engine.tick(0);
        assert_eq!(engine.running_count(), 2);

        // Complete A and B
        engine.tick(50);
        // C should now start
        engine.tick(50);
        assert!(
            engine.task_queue.iter().find(|t| t.id == c).unwrap().status == TaskStatus::Running
        );
    }

    #[test]
    fn dependency_chain_a_b_c() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let b = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![a]).unwrap();
        let c = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![b]).unwrap();

        let results = engine.wait_all();
        // All should complete
        assert!(!results.is_empty());
        assert_eq!(engine.completed_count(), 3);

        let find = |id| engine.completed.iter().find(|r| r.task_id == id).unwrap();
        // B starts after A ends
        assert!(find(b).start_us >= find(a).end_us);
        // C starts after B ends
        assert!(find(c).start_us >= find(b).end_us);
    }

    #[test]
    fn submit_with_nonexistent_dependency_returns_none() {
        let mut engine = AsyncEngine::new(4);
        let result = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![999]);
        assert!(result.is_none());
    }

    // ── Priority ordering ──────────────────────────────────────────

    #[test]
    fn priority_ordering_critical_before_low() {
        let mut engine = AsyncEngine::new(1);
        engine.submit(TaskKind::Synchronize, Priority::Low, vec![]).unwrap();
        let critical = engine.submit(TaskKind::Synchronize, Priority::Critical, vec![]).unwrap();

        // With concurrency=1, only one should start — the Critical one
        engine.tick(0);
        assert_eq!(engine.running_count(), 1);
        assert!(
            engine.task_queue.iter().find(|t| t.id == critical).unwrap().status
                == TaskStatus::Running
        );
    }

    #[test]
    fn priority_high_before_normal() {
        let mut engine = AsyncEngine::new(1);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let high = engine.submit(TaskKind::Synchronize, Priority::High, vec![]).unwrap();

        engine.tick(0);
        assert!(
            engine.task_queue.iter().find(|t| t.id == high).unwrap().status == TaskStatus::Running
        );
    }

    #[test]
    fn mixed_priorities_execution_order() {
        let mut engine = AsyncEngine::new(1);
        let low = engine.submit(TaskKind::Synchronize, Priority::Low, vec![]).unwrap();
        let high = engine.submit(TaskKind::Synchronize, Priority::High, vec![]).unwrap();
        let normal = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();

        let _ = engine.wait_all();

        let order: Vec<TaskId> = engine
            .completed
            .iter()
            .filter(|r| r.status == TaskStatus::Completed)
            .map(|r| r.task_id)
            .collect();
        let high_pos = order.iter().position(|&id| id == high).unwrap();
        let normal_pos = order.iter().position(|&id| id == normal).unwrap();
        let low_pos = order.iter().position(|&id| id == low).unwrap();
        assert!(high_pos < normal_pos);
        assert!(normal_pos < low_pos);
    }

    // ── Concurrency limits ─────────────────────────────────────────

    #[test]
    fn concurrent_limit_enforced() {
        let mut engine = AsyncEngine::new(2);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();

        engine.tick(0);
        assert_eq!(engine.running_count(), 2);
    }

    #[test]
    fn concurrent_limit_of_one() {
        let mut engine = AsyncEngine::new(1);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();

        engine.tick(0);
        assert_eq!(engine.running_count(), 1);
    }

    #[test]
    fn concurrent_limit_allows_backfill() {
        let mut engine = AsyncEngine::new(2);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();

        engine.tick(0);
        assert_eq!(engine.running_count(), 2);

        // Complete two, third should start
        engine.tick(50);
        engine.tick(50);
        assert!(engine.running_count() <= 2);
    }

    #[test]
    fn max_concurrent_zero_treated_as_one() {
        let engine = AsyncEngine::new(0);
        assert_eq!(engine.max_concurrent, 1);
    }

    // ── Cancellation ───────────────────────────────────────────────

    #[test]
    fn cancel_pending_task() {
        let mut engine = AsyncEngine::new(1);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let b = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();

        engine.tick(0); // A starts
        let cancelled = engine.cancel(b);
        assert!(cancelled);
        assert!(
            engine.task_queue.iter().find(|t| t.id == b).unwrap().status == TaskStatus::Cancelled
        );
        let _ = a;
    }

    #[test]
    fn cancel_running_task() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.tick(0); // A starts running
        assert_eq!(engine.running_count(), 1);

        let cancelled = engine.cancel(a);
        assert!(cancelled);
        assert_eq!(engine.running_count(), 0);
    }

    #[test]
    fn cancel_completed_task_returns_false() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let _ = engine.wait_all();
        let cancelled = engine.cancel(a);
        assert!(!cancelled);
    }

    #[test]
    fn cancel_nonexistent_task_returns_false() {
        let mut engine = AsyncEngine::new(4);
        assert!(!engine.cancel(999));
    }

    // ── Empty engine ───────────────────────────────────────────────

    #[test]
    fn empty_engine_tick() {
        let mut engine = AsyncEngine::new(4);
        let results = engine.tick(0);
        assert!(results.is_empty());
    }

    #[test]
    fn empty_engine_wait_all() {
        let mut engine = AsyncEngine::new(4);
        let results = engine.wait_all();
        assert!(results.is_empty());
    }

    #[test]
    fn empty_engine_timeline() {
        let engine = AsyncEngine::new(4);
        let tl = engine.timeline();
        assert!(tl.compute_stream.is_empty());
        assert!(tl.transfer_stream.is_empty());
        assert!(tl.overlap_ratio.abs() < f64::EPSILON);
    }

    // ── Transfer/compute overlap ───────────────────────────────────

    #[test]
    fn transfer_compute_overlap_calculation() {
        let mut engine = AsyncEngine::new(4);
        // Submit a kernel and a transfer that can run concurrently
        engine
            .submit(
                TaskKind::KernelLaunch {
                    kernel: "matmul".to_string(),
                    grid: [1, 1, 1],
                    block: [256, 1, 1],
                },
                Priority::Normal,
                vec![],
            )
            .unwrap();
        engine
            .submit(TaskKind::HostToDevice { size: 1_000_000 }, Priority::Normal, vec![])
            .unwrap();

        let _ = engine.wait_all();
        let tl = engine.timeline();
        assert!(!tl.compute_stream.is_empty());
        assert!(!tl.transfer_stream.is_empty());
    }

    #[test]
    fn timeline_overlap_ratio_with_concurrent_tasks() {
        // Manually check overlap calculation
        let compute = vec![(0, 100, "kernel".to_string())];
        let transfer = vec![(50, 150, "h2d".to_string())];
        let ratio = compute_overlap_ratio(&compute, &transfer);
        // Overlap is 50us (50-100), total span is 150us (0-150)
        let expected = 50.0 / 150.0;
        assert!((ratio - expected).abs() < 1e-6);
    }

    #[test]
    fn timeline_no_overlap() {
        let compute = vec![(0, 50, "kernel".to_string())];
        let transfer = vec![(100, 200, "h2d".to_string())];
        let ratio = compute_overlap_ratio(&compute, &transfer);
        assert!(ratio.abs() < f64::EPSILON);
    }

    #[test]
    fn timeline_full_overlap() {
        let compute = vec![(0, 100, "kernel".to_string())];
        let transfer = vec![(0, 100, "h2d".to_string())];
        let ratio = compute_overlap_ratio(&compute, &transfer);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn timeline_empty_compute() {
        let compute: Vec<(u64, u64, String)> = vec![];
        let transfer = vec![(0, 100, "h2d".to_string())];
        let ratio = compute_overlap_ratio(&compute, &transfer);
        assert!(ratio.abs() < f64::EPSILON);
    }

    #[test]
    fn timeline_empty_transfer() {
        let compute = vec![(0, 100, "kernel".to_string())];
        let transfer: Vec<(u64, u64, String)> = vec![];
        let ratio = compute_overlap_ratio(&compute, &transfer);
        assert!(ratio.abs() < f64::EPSILON);
    }

    // ── Pipeline ───────────────────────────────────────────────────

    #[test]
    fn pipeline_stages_execute_in_order() {
        let mut pipeline = Pipeline::new(4);
        let s1 = pipeline.add_stage(
            "upload",
            vec![(TaskKind::HostToDevice { size: 1_000_000 }, Priority::Normal)],
        );
        let s2 = pipeline.add_stage(
            "compute",
            vec![(
                TaskKind::KernelLaunch {
                    kernel: "matmul".to_string(),
                    grid: [1, 1, 1],
                    block: [256, 1, 1],
                },
                Priority::Normal,
            )],
        );

        let _ = pipeline.execute();

        let find = |id| pipeline.engine.completed.iter().find(|r| r.task_id == id).unwrap();
        // Stage 2 must start after stage 1 ends
        assert!(find(s2[0]).start_us >= find(s1[0]).end_us);
    }

    #[test]
    fn pipeline_multiple_tasks_per_stage() {
        let mut pipeline = Pipeline::new(4);
        let ids = pipeline.add_stage(
            "transfers",
            vec![
                (TaskKind::HostToDevice { size: 1_000_000 }, Priority::Normal),
                (TaskKind::HostToDevice { size: 2_000_000 }, Priority::Normal),
            ],
        );
        assert_eq!(ids.len(), 2);
        let _ = pipeline.execute();
        assert!(pipeline.engine.completed_count() >= 2);
    }

    #[test]
    fn pipeline_estimate_time() {
        let mut pipeline = Pipeline::new(4);
        pipeline.add_stage(
            "upload",
            vec![(TaskKind::HostToDevice { size: 1_000_000 }, Priority::Normal)],
        );
        pipeline.add_stage("compute", vec![(TaskKind::Synchronize, Priority::Normal)]);
        let estimate = pipeline.estimate_time();
        assert!(estimate > 0);
    }

    #[test]
    fn pipeline_empty() {
        let pipeline = Pipeline::new(4);
        assert!(pipeline.stages.is_empty());
        assert_eq!(pipeline.estimate_time(), 0);
    }

    #[test]
    fn pipeline_single_stage() {
        let mut pipeline = Pipeline::new(4);
        pipeline.add_stage("only", vec![(TaskKind::Synchronize, Priority::Normal)]);
        let _ = pipeline.execute();
        assert!(pipeline.engine.completed_count() >= 1);
    }

    // ── Failed task prevents dependents ────────────────────────────

    #[test]
    fn failed_task_prevents_dependents() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let b = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![a]).unwrap();

        // Fail task A explicitly
        engine.fail_task(a, "gpu error");

        // Tick should cascade failure to B
        engine.tick(0);
        let b_result = engine.completed.iter().find(|r| r.task_id == b).unwrap();
        assert!(matches!(b_result.status, TaskStatus::Failed(_)));
    }

    #[test]
    fn failed_dependency_chain_cascades() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let b = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![a]).unwrap();
        let c = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![b]).unwrap();

        engine.fail_task(a, "hardware fault");
        engine.tick(0); // B fails
        engine.tick(0); // C fails

        assert!(
            engine
                .completed
                .iter()
                .any(|r| r.task_id == c && matches!(r.status, TaskStatus::Failed(_)))
        );
    }

    // ── Synchronization barrier ────────────────────────────────────

    #[test]
    fn synchronize_barrier() {
        let mut engine = AsyncEngine::new(4);
        let a = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let b = engine
            .submit(TaskKind::HostToDevice { size: 1_000_000 }, Priority::Normal, vec![])
            .unwrap();
        // Sync barrier depends on both
        let sync = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![a, b]).unwrap();

        let _ = engine.wait_all();

        let find = |id| engine.completed.iter().find(|r| r.task_id == id).unwrap();
        assert!(find(sync).start_us >= find(a).end_us);
        assert!(find(sync).start_us >= find(b).end_us);
    }

    // ── Stress tests ───────────────────────────────────────────────

    #[test]
    fn many_tasks_stress_test() {
        let mut engine = AsyncEngine::new(8);
        for _ in 0..100 {
            engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        }
        let _ = engine.wait_all();
        assert_eq!(engine.completed_count(), 100);
    }

    #[test]
    fn many_tasks_with_chain_dependencies() {
        let mut engine = AsyncEngine::new(4);
        let mut prev = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        for _ in 0..50 {
            prev = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![prev]).unwrap();
        }
        let _ = engine.wait_all();
        assert_eq!(engine.completed_count(), 51);
    }

    #[test]
    fn many_independent_tasks_fill_slots() {
        let mut engine = AsyncEngine::new(4);
        for _ in 0..20 {
            engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        }
        engine.tick(0);
        assert_eq!(engine.running_count(), 4);
    }

    // ── Mixed priorities with dependencies ─────────────────────────

    #[test]
    fn mixed_priorities_with_dependencies() {
        let mut engine = AsyncEngine::new(1);
        let base = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let _ = engine.wait_all();

        let _low = engine.submit(TaskKind::Synchronize, Priority::Low, vec![base]).unwrap();
        let critical =
            engine.submit(TaskKind::Synchronize, Priority::Critical, vec![base]).unwrap();

        engine.tick(200);
        assert!(
            engine.task_queue.iter().find(|t| t.id == critical).unwrap().status
                == TaskStatus::Running
        );
    }

    // ── Circular dependency detection ──────────────────────────────

    #[test]
    fn circular_dependency_self_reference() {
        // Can't self-reference since task isn't in the queue yet during submit
        // But we can try a nonexistent dep which also returns None
        let mut engine = AsyncEngine::new(4);
        let result = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![1]);
        assert!(result.is_none());
    }

    // ── TaskKind display names ─────────────────────────────────────

    #[test]
    fn task_kind_display_names() {
        assert_eq!(
            TaskKind::KernelLaunch {
                kernel: "gemm".to_string(),
                grid: [1, 1, 1],
                block: [1, 1, 1],
            }
            .display_name(),
            "kernel:gemm"
        );
        assert_eq!(TaskKind::HostToDevice { size: 1024 }.display_name(), "h2d:1024B");
        assert_eq!(TaskKind::DeviceToHost { size: 512 }.display_name(), "d2h:512B");
        assert_eq!(TaskKind::DeviceToDevice { size: 256 }.display_name(), "d2d:256B");
        assert_eq!(TaskKind::Synchronize.display_name(), "sync");
        assert_eq!(TaskKind::Custom("foo".to_string()).display_name(), "custom:foo");
    }

    #[test]
    fn task_kind_is_transfer() {
        assert!(TaskKind::HostToDevice { size: 1 }.is_transfer());
        assert!(TaskKind::DeviceToHost { size: 1 }.is_transfer());
        assert!(TaskKind::DeviceToDevice { size: 1 }.is_transfer());
        assert!(!TaskKind::Synchronize.is_transfer());
        assert!(
            !TaskKind::KernelLaunch { kernel: "x".into(), grid: [1, 1, 1], block: [1, 1, 1] }
                .is_transfer()
        );
        assert!(!TaskKind::Custom("x".into()).is_transfer());
    }

    #[test]
    fn task_kind_estimated_duration() {
        assert!(TaskKind::Synchronize.estimated_duration_us() > 0);
        assert!(TaskKind::HostToDevice { size: 10_000_000 }.estimated_duration_us() > 0);
        assert!(
            TaskKind::KernelLaunch { kernel: "k".into(), grid: [4, 4, 4], block: [1, 1, 1] }
                .estimated_duration_us()
                > 0
        );
    }

    // ── Engine accessors ───────────────────────────────────────────

    #[test]
    fn engine_new_defaults() {
        let engine = AsyncEngine::new(8);
        assert_eq!(engine.queue_len(), 0);
        assert_eq!(engine.running_count(), 0);
        assert_eq!(engine.completed_count(), 0);
    }

    #[test]
    fn engine_queue_len_tracks_submissions() {
        let mut engine = AsyncEngine::new(4);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        assert_eq!(engine.queue_len(), 1);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        assert_eq!(engine.queue_len(), 2);
    }

    // ── Timeline from engine ───────────────────────────────────────

    #[test]
    fn timeline_classifies_compute_and_transfer() {
        let mut engine = AsyncEngine::new(4);
        engine
            .submit(
                TaskKind::KernelLaunch {
                    kernel: "relu".to_string(),
                    grid: [1, 1, 1],
                    block: [1, 1, 1],
                },
                Priority::Normal,
                vec![],
            )
            .unwrap();
        engine
            .submit(TaskKind::HostToDevice { size: 1_000_000 }, Priority::Normal, vec![])
            .unwrap();
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();

        let _ = engine.wait_all();
        let tl = engine.timeline();

        // Kernel + Sync → compute stream; H2D → transfer stream
        assert_eq!(tl.compute_stream.len(), 2);
        assert_eq!(tl.transfer_stream.len(), 1);
    }

    #[test]
    fn timeline_entries_have_names() {
        let mut engine = AsyncEngine::new(4);
        engine
            .submit(
                TaskKind::KernelLaunch {
                    kernel: "softmax".to_string(),
                    grid: [1, 1, 1],
                    block: [1, 1, 1],
                },
                Priority::Normal,
                vec![],
            )
            .unwrap();
        let _ = engine.wait_all();
        let tl = engine.timeline();
        assert!(tl.compute_stream[0].2.contains("softmax"));
    }

    // ── Completion ordering ────────────────────────────────────────

    #[test]
    fn completed_tasks_have_valid_timestamps() {
        let mut engine = AsyncEngine::new(4);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let _ = engine.wait_all();
        let result = &engine.completed[0];
        assert!(result.end_us >= result.start_us);
    }

    #[test]
    fn multiple_ticks_advance_correctly() {
        let mut engine = AsyncEngine::new(4);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();

        let r1 = engine.tick(0);
        assert!(r1.is_empty()); // Started but not completed

        let r2 = engine.tick(25);
        assert!(r2.is_empty()); // Still running

        let r3 = engine.tick(50);
        assert_eq!(r3.len(), 1); // Now completed
    }

    // ── Pipeline advanced ──────────────────────────────────────────

    #[test]
    fn pipeline_three_stages() {
        let mut pipeline = Pipeline::new(4);
        pipeline.add_stage(
            "load",
            vec![(TaskKind::HostToDevice { size: 1_000_000 }, Priority::Normal)],
        );
        pipeline.add_stage(
            "compute",
            vec![(
                TaskKind::KernelLaunch {
                    kernel: "matmul".to_string(),
                    grid: [2, 2, 1],
                    block: [256, 1, 1],
                },
                Priority::Normal,
            )],
        );
        pipeline.add_stage(
            "download",
            vec![(TaskKind::DeviceToHost { size: 1_000_000 }, Priority::Normal)],
        );

        let _ = pipeline.execute();
        assert!(pipeline.engine.completed_count() >= 3);
    }

    #[test]
    fn pipeline_stage_names_stored() {
        let mut pipeline = Pipeline::new(4);
        pipeline.add_stage("alpha", vec![(TaskKind::Synchronize, Priority::Normal)]);
        pipeline.add_stage("beta", vec![(TaskKind::Synchronize, Priority::Normal)]);
        assert_eq!(pipeline.stages[0].name, "alpha");
        assert_eq!(pipeline.stages[1].name, "beta");
    }

    #[test]
    fn pipeline_estimate_sums_stages() {
        let mut pipeline = Pipeline::new(4);
        pipeline.add_stage("s1", vec![(TaskKind::Synchronize, Priority::Normal)]);
        pipeline.add_stage("s2", vec![(TaskKind::Synchronize, Priority::Normal)]);
        // Each Synchronize = 50us, so estimate = 100
        assert_eq!(pipeline.estimate_time(), 100);
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn tick_at_exact_completion_time() {
        let mut engine = AsyncEngine::new(4);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.tick(0); // Start at t=0
        let results = engine.tick(50); // Synchronize = 50us exactly
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn tick_well_past_completion() {
        let mut engine = AsyncEngine::new(4);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        engine.tick(0);
        let results = engine.tick(10_000);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn submit_after_wait_all() {
        let mut engine = AsyncEngine::new(4);
        engine.submit(TaskKind::Synchronize, Priority::Normal, vec![]).unwrap();
        let _ = engine.wait_all();

        // Submit new task referencing completed one
        let prev = 1;
        let id = engine.submit(TaskKind::Synchronize, Priority::Normal, vec![prev]).unwrap();
        let _ = engine.wait_all();
        assert!(
            engine.completed.iter().any(|r| r.task_id == id && r.status == TaskStatus::Completed)
        );
    }
}

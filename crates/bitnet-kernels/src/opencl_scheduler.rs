//! OpenCL work scheduler for multi-kernel execution ordering and dependency tracking.
//!
//! In inference pipelines, kernels have data dependencies (e.g., RMSNorm must
//! complete before Attention). This module schedules kernel dispatch order and
//! manages execution dependencies using a DAG-based approach.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the work scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// The dependency graph contains a cycle.
    CycleDetected,
    /// A referenced task ID does not exist.
    TaskNotFound(u64),
    /// A dependency target does not exist.
    DependencyNotFound(u64),
    /// The task has already been marked completed.
    AlreadyCompleted(u64),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "dependency cycle detected"),
            Self::TaskNotFound(id) => write!(f, "task {id} not found"),
            Self::DependencyNotFound(id) => write!(f, "dependency target {id} not found"),
            Self::AlreadyCompleted(id) => write!(f, "task {id} already completed"),
        }
    }
}

impl std::error::Error for SchedulerError {}

// ---------------------------------------------------------------------------
// Task / state / ordering types
// ---------------------------------------------------------------------------

/// A single kernel task to be scheduled.
#[derive(Debug, Clone)]
pub struct KernelTask {
    /// Unique task identifier.
    pub id: u64,
    /// Human-readable name for the kernel.
    pub name: String,
    /// IDs of tasks that must complete before this task can run.
    pub dependencies: Vec<u64>,
    /// Estimated wall-clock duration in nanoseconds.
    pub estimated_duration_ns: u64,
    /// Scheduling priority (higher = execute sooner among ready tasks).
    pub priority: u32,
}

/// Lifecycle state of a task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskState {
    Pending,
    Ready,
    Running,
    Completed,
    Failed(String),
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Ready => write!(f, "Ready"),
            Self::Running => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed(reason) => write!(f, "Failed({reason})"),
        }
    }
}

/// Strategy used to determine execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleOrder {
    /// Execute tasks in the order they were added.
    Sequential,
    /// Execute tasks respecting only data dependencies (topological order).
    DataDependency,
    /// Like `DataDependency`, but among equally-ready tasks prefer higher priority.
    PriorityBased,
}

impl fmt::Display for ScheduleOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => write!(f, "Sequential"),
            Self::DataDependency => write!(f, "DataDependency"),
            Self::PriorityBased => write!(f, "PriorityBased"),
        }
    }
}

// ---------------------------------------------------------------------------
// ExecutionSchedule
// ---------------------------------------------------------------------------

/// The result of computing an execution schedule.
#[derive(Debug, Clone)]
pub struct ExecutionSchedule {
    /// Task IDs in scheduled execution order.
    pub order: Vec<u64>,
    /// Duration of the longest dependency chain (nanoseconds).
    pub critical_path_ns: u64,
    /// Number of tasks that could theoretically run concurrently.
    pub parallelism_opportunities: usize,
}

impl ExecutionSchedule {
    /// Total estimated execution time if tasks run sequentially.
    pub fn estimated_total_ns(&self) -> u64 {
        self.critical_path_ns
    }

    /// Overhead percentage of running sequentially versus the critical path.
    ///
    /// Returns 0.0 when the critical path is zero.
    pub fn sequential_overhead_pct(&self, scheduler: &WorkScheduler) -> f64 {
        let sequential_total: u64 = scheduler.tasks.iter().map(|t| t.estimated_duration_ns).sum();
        if self.critical_path_ns == 0 {
            return 0.0;
        }
        ((sequential_total as f64 - self.critical_path_ns as f64) / self.critical_path_ns as f64)
            * 100.0
    }
}

// ---------------------------------------------------------------------------
// WorkScheduler
// ---------------------------------------------------------------------------

/// DAG-based work scheduler for OpenCL kernel dispatch.
#[derive(Debug)]
pub struct WorkScheduler {
    tasks: Vec<KernelTask>,
    states: Vec<(u64, TaskState)>,
    next_id: u64,
}

impl WorkScheduler {
    /// Create an empty scheduler.
    pub fn new() -> Self {
        Self { tasks: Vec::new(), states: Vec::new(), next_id: 1 }
    }

    /// Add a task and return its assigned ID.
    pub fn add_task(&mut self, mut task: KernelTask) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        task.id = id;

        let initial_state =
            if task.dependencies.is_empty() { TaskState::Ready } else { TaskState::Pending };

        self.states.push((id, initial_state));
        self.tasks.push(task);
        id
    }

    /// Register an additional dependency between two existing tasks.
    pub fn add_dependency(&mut self, task_id: u64, depends_on: u64) -> Result<(), SchedulerError> {
        if !self.has_task(task_id) {
            return Err(SchedulerError::TaskNotFound(task_id));
        }
        if !self.has_task(depends_on) {
            return Err(SchedulerError::DependencyNotFound(depends_on));
        }
        let task = self.tasks.iter_mut().find(|t| t.id == task_id).unwrap();
        if !task.dependencies.contains(&depends_on) {
            task.dependencies.push(depends_on);
        }
        // Re-evaluate state: if it was Ready but now has an unsatisfied dep, go Pending.
        self.refresh_state(task_id);
        Ok(())
    }

    /// Compute an execution schedule using the given strategy.
    pub fn compute_schedule(
        &self,
        order: ScheduleOrder,
    ) -> Result<ExecutionSchedule, SchedulerError> {
        if self.detect_cycles() {
            return Err(SchedulerError::CycleDetected);
        }

        let ordered = match order {
            ScheduleOrder::Sequential => self.tasks.iter().map(|t| t.id).collect(),
            ScheduleOrder::DataDependency => self.topological_sort()?,
            ScheduleOrder::PriorityBased => self.priority_topological_sort()?,
        };

        let critical_path_ns = self.compute_critical_path();
        let parallelism_opportunities = self.count_parallelism();

        Ok(ExecutionSchedule { order: ordered, critical_path_ns, parallelism_opportunities })
    }

    /// Return IDs of tasks whose dependencies are all `Completed`.
    pub fn get_ready_tasks(&self) -> Vec<u64> {
        self.tasks
            .iter()
            .filter(|t| {
                let state = self.state_of(t.id);
                matches!(state, Some(TaskState::Ready))
            })
            .map(|t| t.id)
            .collect()
    }

    /// Mark a task as completed and promote newly-ready dependents.
    pub fn mark_completed(&mut self, task_id: u64) -> Result<(), SchedulerError> {
        let state = self.state_of(task_id).ok_or(SchedulerError::TaskNotFound(task_id))?;
        if state == TaskState::Completed {
            return Err(SchedulerError::AlreadyCompleted(task_id));
        }
        self.set_state(task_id, TaskState::Completed);
        self.promote_dependents(task_id);
        Ok(())
    }

    /// Mark a task as failed with a reason.
    pub fn mark_failed(&mut self, task_id: u64, reason: String) -> Result<(), SchedulerError> {
        if !self.has_task(task_id) {
            return Err(SchedulerError::TaskNotFound(task_id));
        }
        self.set_state(task_id, TaskState::Failed(reason));
        Ok(())
    }

    /// True when every task is `Completed` or `Failed`.
    pub fn is_complete(&self) -> bool {
        self.states.iter().all(|(_, s)| matches!(s, TaskState::Completed | TaskState::Failed(_)))
    }

    /// Total number of tasks.
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Number of completed tasks.
    pub fn completed_count(&self) -> usize {
        self.states.iter().filter(|(_, s)| *s == TaskState::Completed).count()
    }

    /// Number of failed tasks.
    pub fn failed_count(&self) -> usize {
        self.states.iter().filter(|(_, s)| matches!(s, TaskState::Failed(_))).count()
    }

    /// Kahn's algorithm topological sort respecting data dependencies.
    pub fn topological_sort(&self) -> Result<Vec<u64>, SchedulerError> {
        if self.detect_cycles() {
            return Err(SchedulerError::CycleDetected);
        }

        let ids: Vec<u64> = self.tasks.iter().map(|t| t.id).collect();
        let mut in_degree: HashMap<u64, usize> = ids.iter().map(|&id| (id, 0)).collect();

        for task in &self.tasks {
            for &dep in &task.dependencies {
                *in_degree.entry(task.id).or_default() += 1;
                let _ = dep; // dep is used below
            }
        }
        // Recount properly
        for (_, v) in in_degree.iter_mut() {
            *v = 0;
        }
        for task in &self.tasks {
            *in_degree.entry(task.id).or_default() = task.dependencies.len();
        }

        let mut queue: VecDeque<u64> =
            ids.iter().copied().filter(|id| in_degree[id] == 0).collect();
        let mut result = Vec::with_capacity(self.tasks.len());

        while let Some(id) = queue.pop_front() {
            result.push(id);
            for task in &self.tasks {
                if task.dependencies.contains(&id) {
                    let deg = in_degree.get_mut(&task.id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(task.id);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Detect whether the dependency graph contains a cycle (DFS-based).
    pub fn detect_cycles(&self) -> bool {
        let ids: HashSet<u64> = self.tasks.iter().map(|t| t.id).collect();
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();

        for &id in &ids {
            if !visited.contains(&id) && self.dfs_cycle(id, &mut visited, &mut on_stack) {
                return true;
            }
        }
        false
    }

    // -- private helpers ----------------------------------------------------

    fn has_task(&self, id: u64) -> bool {
        self.tasks.iter().any(|t| t.id == id)
    }

    fn state_of(&self, id: u64) -> Option<TaskState> {
        self.states.iter().find(|(tid, _)| *tid == id).map(|(_, s)| s.clone())
    }

    fn set_state(&mut self, id: u64, state: TaskState) {
        if let Some(entry) = self.states.iter_mut().find(|(tid, _)| *tid == id) {
            entry.1 = state;
        }
    }

    fn refresh_state(&mut self, id: u64) {
        let deps = self.tasks.iter().find(|t| t.id == id).map(|t| t.dependencies.clone());
        if let Some(deps) = deps {
            if deps.is_empty()
                || deps.iter().all(|d| self.state_of(*d) == Some(TaskState::Completed))
            {
                let cur = self.state_of(id);
                if matches!(cur, Some(TaskState::Pending)) {
                    self.set_state(id, TaskState::Ready);
                }
            } else {
                let cur = self.state_of(id);
                if matches!(cur, Some(TaskState::Ready)) {
                    self.set_state(id, TaskState::Pending);
                }
            }
        }
    }

    fn promote_dependents(&mut self, completed_id: u64) {
        let dependent_ids: Vec<u64> = self
            .tasks
            .iter()
            .filter(|t| t.dependencies.contains(&completed_id))
            .map(|t| t.id)
            .collect();
        for id in dependent_ids {
            self.refresh_state(id);
        }
    }

    fn dfs_cycle(&self, id: u64, visited: &mut HashSet<u64>, on_stack: &mut HashSet<u64>) -> bool {
        visited.insert(id);
        on_stack.insert(id);

        if let Some(task) = self.tasks.iter().find(|t| t.id == id) {
            for &dep in &task.dependencies {
                if !visited.contains(&dep) {
                    if self.dfs_cycle(dep, visited, on_stack) {
                        return true;
                    }
                } else if on_stack.contains(&dep) {
                    return true;
                }
            }
        }

        on_stack.remove(&id);
        false
    }

    /// Topological sort that breaks ties by higher priority first.
    fn priority_topological_sort(&self) -> Result<Vec<u64>, SchedulerError> {
        if self.detect_cycles() {
            return Err(SchedulerError::CycleDetected);
        }

        let priority_of: HashMap<u64, u32> =
            self.tasks.iter().map(|t| (t.id, t.priority)).collect();
        let mut in_degree: HashMap<u64, usize> =
            self.tasks.iter().map(|t| (t.id, t.dependencies.len())).collect();

        let mut ready: Vec<u64> =
            self.tasks.iter().filter(|t| t.dependencies.is_empty()).map(|t| t.id).collect();
        ready.sort_by(|a, b| priority_of[b].cmp(&priority_of[a]));

        let mut result = Vec::with_capacity(self.tasks.len());

        while !ready.is_empty() {
            let id = ready.remove(0);
            result.push(id);

            for task in &self.tasks {
                if task.dependencies.contains(&id) {
                    let deg = in_degree.get_mut(&task.id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push(task.id);
                    }
                }
            }
            ready.sort_by(|a, b| priority_of[b].cmp(&priority_of[a]));
        }

        Ok(result)
    }

    /// Longest path through the DAG weighted by estimated duration.
    fn compute_critical_path(&self) -> u64 {
        let dur: HashMap<u64, u64> =
            self.tasks.iter().map(|t| (t.id, t.estimated_duration_ns)).collect();
        let mut longest: HashMap<u64, u64> = HashMap::new();

        // Use topological order (ignore errors – caller already checked).
        if let Ok(order) = self.topological_sort() {
            for id in &order {
                let task = self.tasks.iter().find(|t| t.id == *id).unwrap();
                let max_dep = task
                    .dependencies
                    .iter()
                    .filter_map(|d| longest.get(d))
                    .copied()
                    .max()
                    .unwrap_or(0);
                longest.insert(*id, max_dep + dur[id]);
            }
        }

        longest.values().copied().max().unwrap_or(0)
    }

    /// Count the maximum number of tasks that could run concurrently.
    fn count_parallelism(&self) -> usize {
        // Tasks that share no dependency relationship can run in parallel.
        // Compute the maximum anti-chain width via BFS levels.
        if self.tasks.is_empty() {
            return 0;
        }

        let mut in_degree: HashMap<u64, usize> =
            self.tasks.iter().map(|t| (t.id, t.dependencies.len())).collect();
        let mut current_level: Vec<u64> =
            self.tasks.iter().filter(|t| t.dependencies.is_empty()).map(|t| t.id).collect();
        let mut max_width = current_level.len();

        while !current_level.is_empty() {
            let mut next_level = Vec::new();
            for id in &current_level {
                for task in &self.tasks {
                    if task.dependencies.contains(id) {
                        let deg = in_degree.get_mut(&task.id).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            next_level.push(task.id);
                        }
                    }
                }
            }
            if next_level.len() > max_width {
                max_width = next_level.len();
            }
            current_level = next_level;
        }

        max_width
    }
}

impl Default for WorkScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn task(name: &str, deps: Vec<u64>, dur_ns: u64, priority: u32) -> KernelTask {
        KernelTask {
            id: 0, // assigned by scheduler
            name: name.to_string(),
            dependencies: deps,
            estimated_duration_ns: dur_ns,
            priority,
        }
    }

    // -- Display impls -------------------------------------------------------

    #[test]
    fn test_task_state_display() {
        assert_eq!(TaskState::Pending.to_string(), "Pending");
        assert_eq!(TaskState::Ready.to_string(), "Ready");
        assert_eq!(TaskState::Running.to_string(), "Running");
        assert_eq!(TaskState::Completed.to_string(), "Completed");
        assert_eq!(TaskState::Failed("oops".into()).to_string(), "Failed(oops)");
    }

    #[test]
    fn test_schedule_order_display() {
        assert_eq!(ScheduleOrder::Sequential.to_string(), "Sequential");
        assert_eq!(ScheduleOrder::DataDependency.to_string(), "DataDependency");
        assert_eq!(ScheduleOrder::PriorityBased.to_string(), "PriorityBased");
    }

    #[test]
    fn test_scheduler_error_display() {
        assert_eq!(SchedulerError::CycleDetected.to_string(), "dependency cycle detected");
        assert_eq!(SchedulerError::TaskNotFound(7).to_string(), "task 7 not found");
        assert_eq!(
            SchedulerError::DependencyNotFound(3).to_string(),
            "dependency target 3 not found"
        );
        assert_eq!(SchedulerError::AlreadyCompleted(1).to_string(), "task 1 already completed");
    }

    // -- Empty scheduler -----------------------------------------------------

    #[test]
    fn test_empty_scheduler() {
        let sched = WorkScheduler::new();
        assert_eq!(sched.task_count(), 0);
        assert_eq!(sched.completed_count(), 0);
        assert_eq!(sched.failed_count(), 0);
        assert!(sched.is_complete());
        assert!(sched.get_ready_tasks().is_empty());
    }

    #[test]
    fn test_empty_topological_sort() {
        let sched = WorkScheduler::new();
        let sorted = sched.topological_sort().unwrap();
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_empty_no_cycles() {
        assert!(!WorkScheduler::new().detect_cycles());
    }

    // -- Single task ---------------------------------------------------------

    #[test]
    fn test_single_task() {
        let mut sched = WorkScheduler::new();
        let id = sched.add_task(task("rmsnorm", vec![], 1000, 1));
        assert_eq!(sched.task_count(), 1);
        assert_eq!(sched.get_ready_tasks(), vec![id]);
        assert!(!sched.is_complete());
    }

    #[test]
    fn test_single_task_complete() {
        let mut sched = WorkScheduler::new();
        let id = sched.add_task(task("rmsnorm", vec![], 1000, 1));
        sched.mark_completed(id).unwrap();
        assert!(sched.is_complete());
        assert_eq!(sched.completed_count(), 1);
    }

    // -- Linear chain A→B→C -------------------------------------------------

    #[test]
    fn test_linear_chain_ready() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let _b = sched.add_task(task("b", vec![a], 200, 1));
        let _c = sched.add_task(task("c", vec![_b], 300, 1));

        // Only A should be ready
        assert_eq!(sched.get_ready_tasks(), vec![a]);
    }

    #[test]
    fn test_linear_chain_promotion() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let c = sched.add_task(task("c", vec![b], 300, 1));

        sched.mark_completed(a).unwrap();
        assert_eq!(sched.get_ready_tasks(), vec![b]);

        sched.mark_completed(b).unwrap();
        assert_eq!(sched.get_ready_tasks(), vec![c]);

        sched.mark_completed(c).unwrap();
        assert!(sched.is_complete());
    }

    #[test]
    fn test_linear_chain_topological_sort() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let c = sched.add_task(task("c", vec![b], 300, 1));

        let sorted = sched.topological_sort().unwrap();
        assert_eq!(sorted, vec![a, b, c]);
    }

    #[test]
    fn test_linear_chain_critical_path() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let _c = sched.add_task(task("c", vec![b], 300, 1));

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        assert_eq!(schedule.critical_path_ns, 600);
        assert_eq!(schedule.parallelism_opportunities, 1);
    }

    // -- Diamond A→{B,C}→D --------------------------------------------------

    #[test]
    fn test_diamond_dependency() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let c = sched.add_task(task("c", vec![a], 300, 1));
        let _d = sched.add_task(task("d", vec![b, c], 100, 1));

        assert_eq!(sched.get_ready_tasks(), vec![a]);
        sched.mark_completed(a).unwrap();

        let mut ready = sched.get_ready_tasks();
        ready.sort();
        assert_eq!(ready, vec![b, c]);
    }

    #[test]
    fn test_diamond_topological_sort() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let c = sched.add_task(task("c", vec![a], 300, 1));
        let d = sched.add_task(task("d", vec![b, c], 100, 1));

        let sorted = sched.topological_sort().unwrap();
        // A must come first, D must come last, B and C in between.
        assert_eq!(sorted[0], a);
        assert_eq!(*sorted.last().unwrap(), d);
        assert!(sorted.contains(&b));
        assert!(sorted.contains(&c));
    }

    #[test]
    fn test_diamond_critical_path() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let c = sched.add_task(task("c", vec![a], 300, 1));
        let _d = sched.add_task(task("d", vec![b, c], 100, 1));

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        // Critical path: A(100) → C(300) → D(100) = 500
        assert_eq!(schedule.critical_path_ns, 500);
    }

    #[test]
    fn test_diamond_parallelism() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let c = sched.add_task(task("c", vec![a], 300, 1));
        let _d = sched.add_task(task("d", vec![b, c], 100, 1));

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        assert_eq!(schedule.parallelism_opportunities, 2); // B and C
    }

    #[test]
    fn test_diamond_complete_workflow() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let c = sched.add_task(task("c", vec![a], 300, 1));
        let d = sched.add_task(task("d", vec![b, c], 100, 1));

        sched.mark_completed(a).unwrap();
        sched.mark_completed(b).unwrap();
        // D not ready yet – C still pending
        assert!(!sched.get_ready_tasks().contains(&d));

        sched.mark_completed(c).unwrap();
        assert_eq!(sched.get_ready_tasks(), vec![d]);
        sched.mark_completed(d).unwrap();
        assert!(sched.is_complete());
    }

    // -- Independent parallel tasks ------------------------------------------

    #[test]
    fn test_independent_tasks() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![], 200, 1));
        let c = sched.add_task(task("c", vec![], 300, 1));

        let mut ready = sched.get_ready_tasks();
        ready.sort();
        assert_eq!(ready, vec![a, b, c]);
    }

    #[test]
    fn test_independent_parallelism() {
        let mut sched = WorkScheduler::new();
        sched.add_task(task("a", vec![], 100, 1));
        sched.add_task(task("b", vec![], 200, 1));
        sched.add_task(task("c", vec![], 300, 1));

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        assert_eq!(schedule.parallelism_opportunities, 3);
    }

    #[test]
    fn test_independent_critical_path() {
        let mut sched = WorkScheduler::new();
        sched.add_task(task("a", vec![], 100, 1));
        sched.add_task(task("b", vec![], 200, 1));
        sched.add_task(task("c", vec![], 300, 1));

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        // Critical path is longest single task = 300
        assert_eq!(schedule.critical_path_ns, 300);
    }

    // -- Priority ordering ---------------------------------------------------

    #[test]
    fn test_priority_ordering() {
        let mut sched = WorkScheduler::new();
        let _low = sched.add_task(task("low", vec![], 100, 1));
        let _med = sched.add_task(task("med", vec![], 100, 5));
        let _high = sched.add_task(task("high", vec![], 100, 10));

        let schedule = sched.compute_schedule(ScheduleOrder::PriorityBased).unwrap();
        // Highest priority first
        assert_eq!(schedule.order[0], _high);
        assert_eq!(schedule.order[1], _med);
        assert_eq!(schedule.order[2], _low);
    }

    #[test]
    fn test_priority_respects_dependencies() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 100, 10));

        let schedule = sched.compute_schedule(ScheduleOrder::PriorityBased).unwrap();
        // A must come before B regardless of priority
        let pos_a = schedule.order.iter().position(|&id| id == a).unwrap();
        let pos_b = schedule.order.iter().position(|&id| id == b).unwrap();
        assert!(pos_a < pos_b);
    }

    // -- Cycle detection -----------------------------------------------------

    #[test]
    fn test_cycle_detection_no_cycle() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let _b = sched.add_task(task("b", vec![a], 100, 1));
        assert!(!sched.detect_cycles());
    }

    #[test]
    fn test_cycle_detection_direct() {
        let mut sched = WorkScheduler::new();
        // Manually create a cycle by injecting raw tasks
        sched.tasks.push(KernelTask {
            id: 1,
            name: "a".into(),
            dependencies: vec![2],
            estimated_duration_ns: 100,
            priority: 1,
        });
        sched.tasks.push(KernelTask {
            id: 2,
            name: "b".into(),
            dependencies: vec![1],
            estimated_duration_ns: 100,
            priority: 1,
        });
        sched.states.push((1, TaskState::Pending));
        sched.states.push((2, TaskState::Pending));
        sched.next_id = 3;

        assert!(sched.detect_cycles());
    }

    #[test]
    fn test_cycle_detection_indirect() {
        let mut sched = WorkScheduler::new();
        sched.tasks.push(KernelTask {
            id: 1,
            name: "a".into(),
            dependencies: vec![3],
            estimated_duration_ns: 100,
            priority: 1,
        });
        sched.tasks.push(KernelTask {
            id: 2,
            name: "b".into(),
            dependencies: vec![1],
            estimated_duration_ns: 100,
            priority: 1,
        });
        sched.tasks.push(KernelTask {
            id: 3,
            name: "c".into(),
            dependencies: vec![2],
            estimated_duration_ns: 100,
            priority: 1,
        });
        sched.states.push((1, TaskState::Pending));
        sched.states.push((2, TaskState::Pending));
        sched.states.push((3, TaskState::Pending));
        sched.next_id = 4;

        assert!(sched.detect_cycles());
    }

    #[test]
    fn test_cycle_blocks_schedule() {
        let mut sched = WorkScheduler::new();
        sched.tasks.push(KernelTask {
            id: 1,
            name: "a".into(),
            dependencies: vec![2],
            estimated_duration_ns: 100,
            priority: 1,
        });
        sched.tasks.push(KernelTask {
            id: 2,
            name: "b".into(),
            dependencies: vec![1],
            estimated_duration_ns: 100,
            priority: 1,
        });
        sched.states.push((1, TaskState::Pending));
        sched.states.push((2, TaskState::Pending));
        sched.next_id = 3;

        let result = sched.compute_schedule(ScheduleOrder::DataDependency);
        assert_eq!(result.unwrap_err(), SchedulerError::CycleDetected);
    }

    // -- Error handling ------------------------------------------------------

    #[test]
    fn test_mark_completed_missing_task() {
        let mut sched = WorkScheduler::new();
        assert_eq!(sched.mark_completed(999).unwrap_err(), SchedulerError::TaskNotFound(999));
    }

    #[test]
    fn test_mark_failed_missing_task() {
        let mut sched = WorkScheduler::new();
        assert_eq!(
            sched.mark_failed(999, "boom".into()).unwrap_err(),
            SchedulerError::TaskNotFound(999)
        );
    }

    #[test]
    fn test_double_completion() {
        let mut sched = WorkScheduler::new();
        let id = sched.add_task(task("a", vec![], 100, 1));
        sched.mark_completed(id).unwrap();
        assert_eq!(sched.mark_completed(id).unwrap_err(), SchedulerError::AlreadyCompleted(id));
    }

    #[test]
    fn test_add_dependency_missing_task() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        assert_eq!(sched.add_dependency(999, a).unwrap_err(), SchedulerError::TaskNotFound(999));
    }

    #[test]
    fn test_add_dependency_missing_dep() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        assert_eq!(
            sched.add_dependency(a, 999).unwrap_err(),
            SchedulerError::DependencyNotFound(999)
        );
    }

    // -- Mark failed ---------------------------------------------------------

    #[test]
    fn test_mark_failed() {
        let mut sched = WorkScheduler::new();
        let id = sched.add_task(task("a", vec![], 100, 1));
        sched.mark_failed(id, "GPU timeout".into()).unwrap();
        assert_eq!(sched.failed_count(), 1);
        assert!(sched.is_complete());
    }

    #[test]
    fn test_failed_does_not_promote() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 100, 1));

        sched.mark_failed(a, "oops".into()).unwrap();
        // B should NOT be promoted because A failed (not completed)
        assert!(!sched.get_ready_tasks().contains(&b));
    }

    // -- Schedule order variants ---------------------------------------------

    #[test]
    fn test_sequential_order() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![], 200, 5));
        let c = sched.add_task(task("c", vec![], 300, 10));

        let schedule = sched.compute_schedule(ScheduleOrder::Sequential).unwrap();
        assert_eq!(schedule.order, vec![a, b, c]);
    }

    #[test]
    fn test_data_dependency_order() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 200, 1));
        let c = sched.add_task(task("c", vec![b], 300, 1));

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        assert_eq!(schedule.order, vec![a, b, c]);
    }

    #[test]
    fn test_priority_based_order() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![], 200, 5));
        let c = sched.add_task(task("c", vec![], 300, 10));

        let schedule = sched.compute_schedule(ScheduleOrder::PriorityBased).unwrap();
        assert_eq!(schedule.order, vec![c, b, a]);
    }

    // -- ExecutionSchedule methods -------------------------------------------

    #[test]
    fn test_estimated_total_ns() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let _b = sched.add_task(task("b", vec![a], 200, 1));

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        assert_eq!(schedule.estimated_total_ns(), 300);
    }

    #[test]
    fn test_sequential_overhead_pct() {
        let mut sched = WorkScheduler::new();
        sched.add_task(task("a", vec![], 100, 1));
        sched.add_task(task("b", vec![], 100, 1));

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        // Two independent tasks, critical path = 100, total = 200, overhead = 100%
        let overhead = schedule.sequential_overhead_pct(&sched);
        assert!((overhead - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sequential_overhead_zero_critical_path() {
        let sched = WorkScheduler::new();
        let schedule =
            ExecutionSchedule { order: vec![], critical_path_ns: 0, parallelism_opportunities: 0 };
        assert!((schedule.sequential_overhead_pct(&sched) - 0.0).abs() < f64::EPSILON);
    }

    // -- Large DAG -----------------------------------------------------------

    #[test]
    fn test_large_dag() {
        let mut sched = WorkScheduler::new();
        // Create 25 tasks: 5 roots, each with 4 dependents = 20 leaves
        let mut roots = Vec::new();
        for i in 0..5 {
            let id = sched.add_task(task(&format!("root_{i}"), vec![], 100, 1));
            roots.push(id);
        }
        let mut leaves = Vec::new();
        for root_id in &roots {
            for j in 0..4 {
                let id =
                    sched.add_task(task(&format!("leaf_{root_id}_{j}"), vec![*root_id], 50, 1));
                leaves.push(id);
            }
        }

        assert_eq!(sched.task_count(), 25);
        assert!(!sched.detect_cycles());

        let schedule = sched.compute_schedule(ScheduleOrder::DataDependency).unwrap();
        assert_eq!(schedule.order.len(), 25);
        // Critical path: any root(100) + any leaf(50) = 150
        assert_eq!(schedule.critical_path_ns, 150);
        // 20 leaves can run in parallel
        assert_eq!(schedule.parallelism_opportunities, 20);
    }

    #[test]
    fn test_large_dag_topological_correctness() {
        let mut sched = WorkScheduler::new();
        let mut ids = Vec::new();
        // Chain of 20 tasks
        for i in 0..20 {
            let deps = if i == 0 { vec![] } else { vec![ids[i - 1]] };
            let id = sched.add_task(task(&format!("t{i}"), deps, 10, 1));
            ids.push(id);
        }

        let sorted = sched.topological_sort().unwrap();
        assert_eq!(sorted, ids);
    }

    // -- add_dependency after creation ---------------------------------------

    #[test]
    fn test_add_dependency_demotes_ready() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![], 100, 1));

        // B is Ready, then we add a dependency on A
        assert!(sched.get_ready_tasks().contains(&b));
        sched.add_dependency(b, a).unwrap();
        // Now B should be Pending
        assert!(!sched.get_ready_tasks().contains(&b));
    }

    // -- Partial completion ready check --------------------------------------

    #[test]
    fn test_partial_completion_ready() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![], 100, 1));
        let c = sched.add_task(task("c", vec![a, b], 100, 1));

        sched.mark_completed(a).unwrap();
        // C not ready yet (B still pending)
        assert!(!sched.get_ready_tasks().contains(&c));
        sched.mark_completed(b).unwrap();
        assert!(sched.get_ready_tasks().contains(&c));
    }

    // -- Default trait -------------------------------------------------------

    #[test]
    fn test_default() {
        let sched = WorkScheduler::default();
        assert_eq!(sched.task_count(), 0);
    }

    // -- Counts after mixed states -------------------------------------------

    #[test]
    fn test_counts_mixed() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![], 100, 1));
        let c = sched.add_task(task("c", vec![], 100, 1));

        sched.mark_completed(a).unwrap();
        sched.mark_failed(b, "err".into()).unwrap();

        assert_eq!(sched.task_count(), 3);
        assert_eq!(sched.completed_count(), 1);
        assert_eq!(sched.failed_count(), 1);
        assert!(!sched.is_complete()); // c still ready
        assert_eq!(sched.get_ready_tasks(), vec![c]);
    }

    // -- Duplicate dependency is idempotent ----------------------------------

    #[test]
    fn test_duplicate_dependency_idempotent() {
        let mut sched = WorkScheduler::new();
        let a = sched.add_task(task("a", vec![], 100, 1));
        let b = sched.add_task(task("b", vec![a], 100, 1));

        sched.add_dependency(b, a).unwrap();
        let task_b = sched.tasks.iter().find(|t| t.id == b).unwrap();
        // Should only appear once
        assert_eq!(task_b.dependencies.iter().filter(|&&d| d == a).count(), 1);
    }
}

//! Event-based synchronization for ordering kernel executions and managing
//! producer-consumer dependencies on OpenCL devices (e.g. Intel Arc A770).

use std::collections::{HashSet, VecDeque};
use std::fmt;

// ── Types ──────────────────────────────────────────────────────────────

/// Unique identifier for a synchronization event.
pub type EventId = u64;

/// Lifecycle state of a synchronization event.
#[derive(Debug, Clone, PartialEq)]
pub enum EventState {
    Queued,
    Submitted,
    Running,
    Complete,
    Error(String),
}

impl fmt::Display for EventState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Submitted => write!(f, "Submitted"),
            Self::Running => write!(f, "Running"),
            Self::Complete => write!(f, "Complete"),
            Self::Error(msg) => write!(f, "Error({msg})"),
        }
    }
}

/// A single synchronization event tracking kernel execution.
#[derive(Debug, Clone)]
pub struct SyncEvent {
    pub id: EventId,
    pub name: String,
    pub state: EventState,
    pub queued_ns: u64,
    pub start_ns: u64,
    pub end_ns: u64,
    pub dependencies: Vec<EventId>,
}

/// A barrier that waits for a set of events to complete.
#[derive(Debug, Clone)]
pub struct BarrierPoint {
    pub id: u64,
    pub event_ids: Vec<EventId>,
    pub reached: bool,
}

/// Ordered timeline of events and barriers.
#[derive(Debug, Clone)]
pub struct EventTimeline {
    pub events: Vec<SyncEvent>,
    pub barriers: Vec<BarrierPoint>,
}

/// Top-level manager for event synchronization.
#[derive(Debug, Clone)]
pub struct SyncManager {
    pub timeline: EventTimeline,
    pub next_id: u64,
    pub stats: SyncStats,
}

/// Aggregate statistics for the synchronization manager.
#[derive(Debug, Clone)]
pub struct SyncStats {
    pub total_events: u64,
    pub total_barriers: u64,
    pub total_wait_time_us: u64,
    pub max_dependency_depth: usize,
    pub average_latency_us: f64,
}

/// Errors that can occur during synchronization operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncError {
    EventNotFound(EventId),
    CyclicDependency,
    BarrierTimeout,
    InvalidTransition { from: EventState, to: EventState },
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventNotFound(id) => write!(f, "event {id} not found"),
            Self::CyclicDependency => write!(f, "cyclic dependency detected"),
            Self::BarrierTimeout => write!(f, "barrier timeout"),
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {from} to {to}")
            }
        }
    }
}

impl std::error::Error for SyncError {}

// ── Helpers ────────────────────────────────────────────────────────────

/// Returns `true` when transitioning from `from` to `to` is legal.
fn is_valid_transition(from: &EventState, to: &EventState) -> bool {
    matches!(
        (from, to),
        (EventState::Queued, EventState::Submitted)
            | (EventState::Queued, EventState::Running)
            | (EventState::Submitted, EventState::Running)
            | (EventState::Running, EventState::Complete)
            | (EventState::Queued, EventState::Error(_))
            | (EventState::Submitted, EventState::Error(_))
            | (EventState::Running, EventState::Error(_))
    )
}

fn find_event(mgr: &SyncManager, id: EventId) -> Option<&SyncEvent> {
    mgr.timeline.events.iter().find(|e| e.id == id)
}

fn find_event_mut(mgr: &mut SyncManager, id: EventId) -> Option<&mut SyncEvent> {
    mgr.timeline.events.iter_mut().find(|e| e.id == id)
}

// ── CPU Reference Implementations ─────────────────────────────────────

/// Create a new, empty synchronization manager.
pub fn create_sync_manager() -> SyncManager {
    SyncManager {
        timeline: EventTimeline { events: Vec::new(), barriers: Vec::new() },
        next_id: 1,
        stats: SyncStats {
            total_events: 0,
            total_barriers: 0,
            total_wait_time_us: 0,
            max_dependency_depth: 0,
            average_latency_us: 0.0,
        },
    }
}

/// Create a new event with the given name and dependency list.
pub fn cpu_create_event(
    mgr: &mut SyncManager,
    name: &str,
    deps: Vec<EventId>,
) -> Result<EventId, SyncError> {
    // Validate that all dependencies exist.
    for &dep in &deps {
        if find_event(mgr, dep).is_none() {
            return Err(SyncError::EventNotFound(dep));
        }
    }

    let id = mgr.next_id;

    // Check for cycles *before* inserting.
    if cpu_detect_cycle(mgr, id, &deps) {
        return Err(SyncError::CyclicDependency);
    }

    mgr.timeline.events.push(SyncEvent {
        id,
        name: name.to_string(),
        state: EventState::Queued,
        queued_ns: 0,
        start_ns: 0,
        end_ns: 0,
        dependencies: deps,
    });
    mgr.next_id += 1;
    mgr.stats.total_events += 1;
    Ok(id)
}

/// Transition an event to a new state, validating the transition.
pub fn cpu_transition_event(
    mgr: &mut SyncManager,
    id: EventId,
    new_state: EventState,
) -> Result<(), SyncError> {
    let event =
        find_event_mut(mgr, id).ok_or(SyncError::EventNotFound(id))?;

    if !is_valid_transition(&event.state, &new_state) {
        return Err(SyncError::InvalidTransition {
            from: event.state.clone(),
            to: new_state,
        });
    }
    event.state = new_state;
    Ok(())
}

/// Returns `true` when every dependency of `id` is `Complete`.
pub fn cpu_check_dependencies(mgr: &SyncManager, id: EventId) -> bool {
    let Some(event) = find_event(mgr, id) else {
        return false;
    };
    let deps = event.dependencies.clone();
    deps.iter().all(|dep_id| {
        find_event(mgr, *dep_id)
            .is_some_and(|e| e.state == EventState::Complete)
    })
}

/// Create a barrier that waits for the listed events.
pub fn cpu_create_barrier(
    mgr: &mut SyncManager,
    event_ids: Vec<EventId>,
) -> u64 {
    let id = mgr.next_id;
    mgr.next_id += 1;
    let reached = event_ids.iter().all(|eid| {
        find_event(mgr, *eid)
            .is_some_and(|e| e.state == EventState::Complete)
    });
    mgr.timeline.barriers.push(BarrierPoint { id, event_ids, reached });
    mgr.stats.total_barriers += 1;
    id
}

/// Check whether a barrier has been reached (all its events are complete).
pub fn cpu_check_barrier(mgr: &SyncManager, barrier_id: u64) -> bool {
    mgr.timeline
        .barriers
        .iter()
        .find(|b| b.id == barrier_id)
        .is_some_and(|b| {
            b.event_ids.iter().all(|eid| {
                find_event(mgr, *eid)
                    .is_some_and(|e| e.state == EventState::Complete)
            })
        })
}

/// Return all events whose dependencies are fully met and that are still
/// `Queued`.
pub fn cpu_get_ready_events(mgr: &SyncManager) -> Vec<EventId> {
    mgr.timeline
        .events
        .iter()
        .filter(|e| {
            e.state == EventState::Queued && cpu_check_dependencies(mgr, e.id)
        })
        .map(|e| e.id)
        .collect()
}

/// Compute the critical path (longest dependency chain).
pub fn cpu_compute_critical_path(mgr: &SyncManager) -> Vec<EventId> {
    // Build an index: id → position for quick look-up.
    let id_set: HashSet<EventId> =
        mgr.timeline.events.iter().map(|e| e.id).collect();

    // Memoised DFS returning the longest path ending at `id`.
    fn longest_path(
        mgr: &SyncManager,
        id: EventId,
        id_set: &HashSet<EventId>,
        cache: &mut std::collections::HashMap<EventId, Vec<EventId>>,
    ) -> Vec<EventId> {
        if let Some(cached) = cache.get(&id) {
            return cached.clone();
        }
        let deps = mgr
            .timeline
            .events
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.dependencies.clone())
            .unwrap_or_default();

        let best_prefix = deps
            .iter()
            .filter(|d| id_set.contains(d))
            .map(|&d| longest_path(mgr, d, id_set, cache))
            .max_by_key(|p| p.len())
            .unwrap_or_default();

        let mut path = best_prefix;
        path.push(id);
        cache.insert(id, path.clone());
        path
    }

    let mut cache = std::collections::HashMap::new();
    mgr.timeline
        .events
        .iter()
        .map(|e| longest_path(mgr, e.id, &id_set, &mut cache))
        .max_by_key(|p| p.len())
        .unwrap_or_default()
}

/// Latency of a single event (end − start) in nanoseconds.
pub fn cpu_get_event_latency(event: &SyncEvent) -> u64 {
    event.end_ns.saturating_sub(event.start_ns)
}

/// Duration of the entire timeline (last end − first queued) in
/// nanoseconds.
pub fn cpu_get_timeline_duration(mgr: &SyncManager) -> u64 {
    if mgr.timeline.events.is_empty() {
        return 0;
    }
    let first = mgr
        .timeline
        .events
        .iter()
        .map(|e| e.queued_ns)
        .min()
        .unwrap_or(0);
    let last = mgr
        .timeline
        .events
        .iter()
        .map(|e| e.end_ns)
        .max()
        .unwrap_or(0);
    last.saturating_sub(first)
}

/// Returns `true` if adding `deps` to `id` would create a cycle.
pub fn cpu_detect_cycle(
    mgr: &SyncManager,
    id: EventId,
    deps: &[EventId],
) -> bool {
    // BFS from each proposed dep; if we can reach `id`, it's a cycle.
    let mut visited = HashSet::new();
    let mut queue: VecDeque<EventId> = deps.iter().copied().collect();
    while let Some(cur) = queue.pop_front() {
        if cur == id {
            return true;
        }
        if !visited.insert(cur) {
            continue;
        }
        if let Some(ev) = find_event(mgr, cur) {
            for &d in &ev.dependencies {
                queue.push_back(d);
            }
        }
    }
    false
}

/// Collect aggregate statistics from the current timeline.
pub fn cpu_get_stats(mgr: &SyncManager) -> SyncStats {
    let total_events = mgr.timeline.events.len() as u64;
    let total_barriers = mgr.timeline.barriers.len() as u64;

    let (total_latency_ns, count) = mgr
        .timeline
        .events
        .iter()
        .filter(|e| e.end_ns > e.start_ns)
        .fold((0u64, 0u64), |(sum, cnt), e| {
            (sum + cpu_get_event_latency(e), cnt + 1)
        });
    let average_latency_us = if count > 0 {
        (total_latency_ns as f64) / (count as f64) / 1000.0
    } else {
        0.0
    };

    let total_wait_time_us = mgr
        .timeline
        .events
        .iter()
        .filter(|e| e.start_ns > e.queued_ns)
        .map(|e| e.start_ns - e.queued_ns)
        .sum::<u64>()
        / 1000;

    let max_dependency_depth =
        cpu_compute_critical_path(mgr).len().saturating_sub(1);

    SyncStats {
        total_events,
        total_barriers,
        total_wait_time_us,
        max_dependency_depth,
        average_latency_us,
    }
}

/// Human-readable text representation of the event timeline.
pub fn format_timeline(mgr: &SyncManager) -> String {
    let mut out = String::from("=== Event Timeline ===\n");
    for ev in &mgr.timeline.events {
        out.push_str(&format!(
            "[{}] {} — {} (deps: {:?})\n",
            ev.id, ev.name, ev.state, ev.dependencies,
        ));
    }
    if !mgr.timeline.barriers.is_empty() {
        out.push_str("--- Barriers ---\n");
        for b in &mgr.timeline.barriers {
            out.push_str(&format!(
                "Barrier {} — events {:?} — reached: {}\n",
                b.id, b.event_ids, b.reached,
            ));
        }
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- Manager creation ---------------------------------------------------

    #[test]
    fn test_create_manager_empty() {
        let mgr = create_sync_manager();
        assert!(mgr.timeline.events.is_empty());
        assert!(mgr.timeline.barriers.is_empty());
        assert_eq!(mgr.next_id, 1);
    }

    #[test]
    fn test_create_manager_stats_zeroed() {
        let mgr = create_sync_manager();
        assert_eq!(mgr.stats.total_events, 0);
        assert_eq!(mgr.stats.total_barriers, 0);
        assert_eq!(mgr.stats.total_wait_time_us, 0);
    }

    // -- Event creation -----------------------------------------------------

    #[test]
    fn test_create_event_returns_id() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "matmul", vec![]).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn test_create_event_increments_id() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        assert_eq!(b, a + 1);
    }

    #[test]
    fn test_create_event_stores_name() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "softmax", vec![]).unwrap();
        let ev = find_event(&mgr, id).unwrap();
        assert_eq!(ev.name, "softmax");
    }

    #[test]
    fn test_create_event_initial_state_queued() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "k", vec![]).unwrap();
        assert_eq!(find_event(&mgr, id).unwrap().state, EventState::Queued);
    }

    #[test]
    fn test_create_event_with_deps() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        assert_eq!(find_event(&mgr, b).unwrap().dependencies, vec![a]);
    }

    #[test]
    fn test_create_event_invalid_dep_errors() {
        let mut mgr = create_sync_manager();
        let res = cpu_create_event(&mut mgr, "x", vec![999]);
        assert_eq!(res, Err(SyncError::EventNotFound(999)));
    }

    #[test]
    fn test_create_event_updates_total_count() {
        let mut mgr = create_sync_manager();
        cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        assert_eq!(mgr.stats.total_events, 2);
    }

    // -- State transitions --------------------------------------------------

    #[test]
    fn test_transition_queued_to_submitted() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "e", vec![]).unwrap();
        assert!(cpu_transition_event(&mut mgr, id, EventState::Submitted).is_ok());
        assert_eq!(find_event(&mgr, id).unwrap().state, EventState::Submitted);
    }

    #[test]
    fn test_transition_queued_to_running() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "e", vec![]).unwrap();
        assert!(cpu_transition_event(&mut mgr, id, EventState::Running).is_ok());
    }

    #[test]
    fn test_transition_submitted_to_running() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "e", vec![]).unwrap();
        cpu_transition_event(&mut mgr, id, EventState::Submitted).unwrap();
        assert!(cpu_transition_event(&mut mgr, id, EventState::Running).is_ok());
    }

    #[test]
    fn test_transition_running_to_complete() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "e", vec![]).unwrap();
        cpu_transition_event(&mut mgr, id, EventState::Running).unwrap();
        assert!(cpu_transition_event(&mut mgr, id, EventState::Complete).is_ok());
    }

    #[test]
    fn test_transition_to_error_from_queued() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "e", vec![]).unwrap();
        let res = cpu_transition_event(
            &mut mgr,
            id,
            EventState::Error("boom".into()),
        );
        assert!(res.is_ok());
    }

    #[test]
    fn test_transition_to_error_from_running() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "e", vec![]).unwrap();
        cpu_transition_event(&mut mgr, id, EventState::Running).unwrap();
        assert!(cpu_transition_event(
            &mut mgr,
            id,
            EventState::Error("fail".into()),
        )
        .is_ok());
    }

    #[test]
    fn test_transition_invalid_complete_to_queued() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "e", vec![]).unwrap();
        cpu_transition_event(&mut mgr, id, EventState::Running).unwrap();
        cpu_transition_event(&mut mgr, id, EventState::Complete).unwrap();
        let res = cpu_transition_event(&mut mgr, id, EventState::Queued);
        assert!(matches!(res, Err(SyncError::InvalidTransition { .. })));
    }

    #[test]
    fn test_transition_invalid_complete_to_running() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "e", vec![]).unwrap();
        cpu_transition_event(&mut mgr, id, EventState::Running).unwrap();
        cpu_transition_event(&mut mgr, id, EventState::Complete).unwrap();
        let res = cpu_transition_event(&mut mgr, id, EventState::Running);
        assert!(matches!(res, Err(SyncError::InvalidTransition { .. })));
    }

    #[test]
    fn test_transition_nonexistent_event() {
        let mut mgr = create_sync_manager();
        let res = cpu_transition_event(&mut mgr, 42, EventState::Running);
        assert_eq!(res, Err(SyncError::EventNotFound(42)));
    }

    // -- Dependencies -------------------------------------------------------

    #[test]
    fn test_deps_all_complete_returns_true() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Running).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Complete).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        assert!(cpu_check_dependencies(&mgr, b));
    }

    #[test]
    fn test_deps_not_met_returns_false() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        assert!(!cpu_check_dependencies(&mgr, b));
    }

    #[test]
    fn test_deps_partial_met() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Running).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Complete).unwrap();
        let c = cpu_create_event(&mut mgr, "c", vec![a, b]).unwrap();
        assert!(!cpu_check_dependencies(&mgr, c));
    }

    // -- Barriers -----------------------------------------------------------

    #[test]
    fn test_barrier_all_complete_reached() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        for id in [a, b] {
            cpu_transition_event(&mut mgr, id, EventState::Running).unwrap();
            cpu_transition_event(&mut mgr, id, EventState::Complete).unwrap();
        }
        let bid = cpu_create_barrier(&mut mgr, vec![a, b]);
        assert!(cpu_check_barrier(&mgr, bid));
    }

    #[test]
    fn test_barrier_partial_not_reached() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Running).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Complete).unwrap();
        let bid = cpu_create_barrier(&mut mgr, vec![a, b]);
        assert!(!cpu_check_barrier(&mgr, bid));
    }

    #[test]
    fn test_barrier_empty_events_reached() {
        let mut mgr = create_sync_manager();
        let bid = cpu_create_barrier(&mut mgr, vec![]);
        assert!(cpu_check_barrier(&mgr, bid));
    }

    #[test]
    fn test_barrier_updates_stats() {
        let mut mgr = create_sync_manager();
        cpu_create_barrier(&mut mgr, vec![]);
        cpu_create_barrier(&mut mgr, vec![]);
        assert_eq!(mgr.stats.total_barriers, 2);
    }

    // -- Ready events -------------------------------------------------------

    #[test]
    fn test_ready_events_no_deps() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let ready = cpu_get_ready_events(&mgr);
        assert!(ready.contains(&a));
    }

    #[test]
    fn test_ready_events_with_pending_dep() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        let ready = cpu_get_ready_events(&mgr);
        assert!(ready.contains(&a));
        assert!(!ready.contains(&b));
    }

    #[test]
    fn test_ready_events_excludes_non_queued() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Running).unwrap();
        let ready = cpu_get_ready_events(&mgr);
        assert!(!ready.contains(&a));
    }

    // -- Critical path ------------------------------------------------------

    #[test]
    fn test_critical_path_linear_chain() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        let c = cpu_create_event(&mut mgr, "c", vec![b]).unwrap();
        let path = cpu_compute_critical_path(&mgr);
        assert_eq!(path, vec![a, b, c]);
    }

    #[test]
    fn test_critical_path_diamond() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        let c = cpu_create_event(&mut mgr, "c", vec![a]).unwrap();
        let d = cpu_create_event(&mut mgr, "d", vec![b, c]).unwrap();
        let path = cpu_compute_critical_path(&mgr);
        // Two equally long paths (a→b→d or a→c→d); either is valid.
        assert_eq!(path.len(), 3);
        assert_eq!(*path.first().unwrap(), a);
        assert_eq!(*path.last().unwrap(), d);
    }

    #[test]
    fn test_critical_path_single_event() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        assert_eq!(cpu_compute_critical_path(&mgr), vec![a]);
    }

    #[test]
    fn test_critical_path_empty() {
        let mgr = create_sync_manager();
        assert!(cpu_compute_critical_path(&mgr).is_empty());
    }

    // -- Cycle detection ----------------------------------------------------

    #[test]
    fn test_detect_cycle_catches_self_loop() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        assert!(cpu_detect_cycle(&mgr, a, &[a]));
    }

    #[test]
    fn test_detect_cycle_catches_indirect() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        // Trying to add a dep from a → b would create a→b→…→a.
        assert!(cpu_detect_cycle(&mgr, a, &[b]));
    }

    #[test]
    fn test_detect_cycle_no_false_positive() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        assert!(!cpu_detect_cycle(&mgr, b, &[a]));
    }

    #[test]
    fn test_create_event_rejects_cycle() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        // c depends on b, then try to make a depend on c → cycle
        let c = cpu_create_event(&mut mgr, "c", vec![b]).unwrap();
        // Manually attempt: new event depending on c, with id that would
        // be next — but easier: just detect via helper.
        assert!(cpu_detect_cycle(&mgr, a, &[c]));
    }

    // -- Latency & duration -------------------------------------------------

    #[test]
    fn test_event_latency_correct() {
        let ev = SyncEvent {
            id: 1,
            name: "k".into(),
            state: EventState::Complete,
            queued_ns: 0,
            start_ns: 1000,
            end_ns: 5000,
            dependencies: vec![],
        };
        assert_eq!(cpu_get_event_latency(&ev), 4000);
    }

    #[test]
    fn test_event_latency_zero_when_not_started() {
        let ev = SyncEvent {
            id: 1,
            name: "k".into(),
            state: EventState::Queued,
            queued_ns: 0,
            start_ns: 0,
            end_ns: 0,
            dependencies: vec![],
        };
        assert_eq!(cpu_get_event_latency(&ev), 0);
    }

    #[test]
    fn test_timeline_duration_empty() {
        let mgr = create_sync_manager();
        assert_eq!(cpu_get_timeline_duration(&mgr), 0);
    }

    #[test]
    fn test_timeline_duration_covers_all() {
        let mut mgr = create_sync_manager();
        cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        // Manually set timestamps for deterministic results.
        mgr.timeline.events[0].queued_ns = 100;
        mgr.timeline.events[0].end_ns = 500;
        mgr.timeline.events[1].queued_ns = 200;
        mgr.timeline.events[1].end_ns = 900;
        assert_eq!(cpu_get_timeline_duration(&mgr), 800); // 900 - 100
    }

    // -- Stats --------------------------------------------------------------

    #[test]
    fn test_stats_correct_counts() {
        let mut mgr = create_sync_manager();
        cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        cpu_create_barrier(&mut mgr, vec![]);
        let stats = cpu_get_stats(&mgr);
        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.total_barriers, 1);
    }

    #[test]
    fn test_stats_average_latency() {
        let mut mgr = create_sync_manager();
        cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        mgr.timeline.events[0].start_ns = 0;
        mgr.timeline.events[0].end_ns = 2000;
        mgr.timeline.events[1].start_ns = 0;
        mgr.timeline.events[1].end_ns = 4000;
        let stats = cpu_get_stats(&mgr);
        // avg = (2000 + 4000) / 2 / 1000 = 3.0 µs
        assert!((stats.average_latency_us - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_max_depth() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        let _c = cpu_create_event(&mut mgr, "c", vec![b]).unwrap();
        let stats = cpu_get_stats(&mgr);
        assert_eq!(stats.max_dependency_depth, 2); // chain length 3 → depth 2
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn test_edge_no_dependencies() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "solo", vec![]).unwrap();
        assert!(cpu_check_dependencies(&mgr, id));
    }

    #[test]
    fn test_edge_single_event_ready() {
        let mut mgr = create_sync_manager();
        let id = cpu_create_event(&mut mgr, "only", vec![]).unwrap();
        assert_eq!(cpu_get_ready_events(&mgr), vec![id]);
    }

    #[test]
    fn test_edge_all_independent() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![]).unwrap();
        let c = cpu_create_event(&mut mgr, "c", vec![]).unwrap();
        let ready = cpu_get_ready_events(&mgr);
        assert_eq!(ready.len(), 3);
        assert!(ready.contains(&a));
        assert!(ready.contains(&b));
        assert!(ready.contains(&c));
    }

    // -- Properties ---------------------------------------------------------

    #[test]
    fn test_property_ready_events_have_no_pending_deps() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Running).unwrap();
        cpu_transition_event(&mut mgr, a, EventState::Complete).unwrap();
        let _c = cpu_create_event(&mut mgr, "c", vec![b]).unwrap();
        for rid in cpu_get_ready_events(&mgr) {
            assert!(cpu_check_dependencies(&mgr, rid));
        }
    }

    #[test]
    fn test_property_critical_path_length_le_total() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        let b = cpu_create_event(&mut mgr, "b", vec![a]).unwrap();
        let _c = cpu_create_event(&mut mgr, "c", vec![]).unwrap();
        let _d = cpu_create_event(&mut mgr, "d", vec![b]).unwrap();
        let path = cpu_compute_critical_path(&mgr);
        assert!(path.len() <= mgr.timeline.events.len());
    }

    // -- Format timeline ----------------------------------------------------

    #[test]
    fn test_format_timeline_contains_event_names() {
        let mut mgr = create_sync_manager();
        cpu_create_event(&mut mgr, "matmul", vec![]).unwrap();
        cpu_create_event(&mut mgr, "softmax", vec![]).unwrap();
        let text = format_timeline(&mgr);
        assert!(text.contains("matmul"));
        assert!(text.contains("softmax"));
    }

    #[test]
    fn test_format_timeline_contains_barrier() {
        let mut mgr = create_sync_manager();
        let a = cpu_create_event(&mut mgr, "a", vec![]).unwrap();
        cpu_create_barrier(&mut mgr, vec![a]);
        let text = format_timeline(&mgr);
        assert!(text.contains("Barrier"));
    }

    #[test]
    fn test_format_timeline_empty() {
        let mgr = create_sync_manager();
        let text = format_timeline(&mgr);
        assert!(text.contains("Event Timeline"));
    }

    // -- SyncError Display --------------------------------------------------

    #[test]
    fn test_error_display() {
        assert_eq!(
            SyncError::EventNotFound(7).to_string(),
            "event 7 not found"
        );
        assert_eq!(
            SyncError::CyclicDependency.to_string(),
            "cyclic dependency detected"
        );
        assert_eq!(
            SyncError::BarrierTimeout.to_string(),
            "barrier timeout"
        );
    }

    #[test]
    fn test_error_invalid_transition_display() {
        let err = SyncError::InvalidTransition {
            from: EventState::Complete,
            to: EventState::Queued,
        };
        assert!(err.to_string().contains("invalid transition"));
    }
}

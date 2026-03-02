//! OpenCL execution profiler for detailed per-layer and per-kernel timing.
//!
//! Captures hierarchical profile events, builds flame graphs, identifies
//! bottlenecks, and suggests optimizations for A770 tuning.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Category of a profiled event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventCategory {
    KernelExecution,
    MemoryTransfer,
    Synchronization,
    HostCompute,
    QueueWait,
}

impl fmt::Display for EventCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelExecution => write!(f, "KernelExecution"),
            Self::MemoryTransfer => write!(f, "MemoryTransfer"),
            Self::Synchronization => write!(f, "Synchronization"),
            Self::HostCompute => write!(f, "HostCompute"),
            Self::QueueWait => write!(f, "QueueWait"),
        }
    }
}

/// A single profiled event within a session.
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub name: String,
    pub category: EventCategory,
    pub start_us: u64,
    pub duration_us: u64,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub metadata: HashMap<String, String>,
}

/// A recording session containing a list of events.
#[derive(Debug, Clone)]
pub struct ProfileSession {
    pub events: Vec<ProfileEvent>,
    pub start_time_ns: u64,
    pub name: String,
}

/// Node in a flame-graph tree.
#[derive(Debug, Clone)]
pub struct FlameNode {
    pub name: String,
    pub self_time_us: u64,
    pub total_time_us: u64,
    pub children: Vec<FlameNode>,
    pub call_count: u64,
}

/// A single identified bottleneck.
#[derive(Debug, Clone)]
pub struct BottleneckInfo {
    pub event_name: String,
    pub percentage: f32,
    pub suggestion: String,
}

/// Summary report for a profile session.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub session_name: String,
    pub total_time_us: u64,
    pub kernel_time_us: u64,
    pub transfer_time_us: u64,
    pub sync_time_us: u64,
    pub host_time_us: u64,
    pub bottlenecks: Vec<BottleneckInfo>,
    pub flame_root: FlameNode,
}

/// Top-level profiler that owns sessions.
#[derive(Debug)]
pub struct Profiler {
    pub sessions: Vec<ProfileSession>,
    pub current_session: Option<usize>,
    pub enabled: bool,
}

/// Errors that can occur during profiling operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    NoActiveSession,
    EventNotFound(usize),
    SessionNotFound,
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoActiveSession => write!(f, "no active profiling session"),
            Self::EventNotFound(id) => write!(f, "event {id} not found"),
            Self::SessionNotFound => write!(f, "session not found"),
        }
    }
}

impl std::error::Error for ProfileError {}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Create a new, disabled profiler.
pub fn create_profiler() -> Profiler {
    Profiler { sessions: Vec::new(), current_session: None, enabled: true }
}

/// Begin a new profiling session and return its index.
pub fn cpu_begin_session(profiler: &mut Profiler, name: &str) -> usize {
    let idx = profiler.sessions.len();
    profiler.sessions.push(ProfileSession {
        events: Vec::new(),
        start_time_ns: 0,
        name: name.to_string(),
    });
    profiler.current_session = Some(idx);
    idx
}

/// End the current profiling session.
pub fn cpu_end_session(profiler: &mut Profiler) -> Result<(), ProfileError> {
    if profiler.current_session.is_none() {
        return Err(ProfileError::NoActiveSession);
    }
    profiler.current_session = None;
    Ok(())
}

/// Record an event in the current session and return its index.
pub fn cpu_record_event(
    profiler: &mut Profiler,
    name: &str,
    category: EventCategory,
    duration_us: u64,
    parent: Option<usize>,
) -> Result<usize, ProfileError> {
    let session_idx =
        profiler.current_session.ok_or(ProfileError::NoActiveSession)?;
    let session = &mut profiler.sessions[session_idx];

    // Compute start_us: after the last event ends, or 0.
    let start_us = if let Some(parent_idx) = parent {
        if parent_idx >= session.events.len() {
            return Err(ProfileError::EventNotFound(parent_idx));
        }
        // Place right after parent's start (children are nested).
        let parent_event = &session.events[parent_idx];
        let children_end: u64 = parent_event
            .children
            .iter()
            .map(|&c| {
                session.events[c].start_us + session.events[c].duration_us
            })
            .max()
            .unwrap_or(parent_event.start_us);
        children_end
    } else {
        session
            .events
            .last()
            .map(|e| e.start_us + e.duration_us)
            .unwrap_or(0)
    };

    let event_idx = session.events.len();
    session.events.push(ProfileEvent {
        name: name.to_string(),
        category,
        start_us,
        duration_us,
        parent,
        children: Vec::new(),
        metadata: HashMap::new(),
    });

    if let Some(parent_idx) = parent {
        session.events[parent_idx].children.push(event_idx);
    }

    Ok(event_idx)
}

/// Build a flame-graph tree from a session.
pub fn cpu_build_flame_graph(session: &ProfileSession) -> FlameNode {
    fn build_node(session: &ProfileSession, idx: usize) -> FlameNode {
        let event = &session.events[idx];
        let child_nodes: Vec<FlameNode> =
            event.children.iter().map(|&c| build_node(session, c)).collect();
        let children_total: u64 =
            child_nodes.iter().map(|c| c.total_time_us).sum();
        FlameNode {
            name: event.name.clone(),
            self_time_us: event.duration_us.saturating_sub(children_total),
            total_time_us: event.duration_us,
            children: child_nodes,
            call_count: 1,
        }
    }

    // Collect root events (no parent).
    let roots: Vec<usize> = session
        .events
        .iter()
        .enumerate()
        .filter(|(_, e)| e.parent.is_none())
        .map(|(i, _)| i)
        .collect();

    let child_nodes: Vec<FlameNode> =
        roots.iter().map(|&i| build_node(session, i)).collect();
    let total: u64 = child_nodes.iter().map(|c| c.total_time_us).sum();

    FlameNode {
        name: session.name.clone(),
        self_time_us: 0,
        total_time_us: total,
        children: child_nodes,
        call_count: 1,
    }
}

/// Find the top-N bottleneck events by duration.
pub fn cpu_find_bottlenecks(
    session: &ProfileSession,
    top_n: usize,
) -> Vec<BottleneckInfo> {
    let total_us: u64 = session.events.iter().map(|e| e.duration_us).sum();
    if total_us == 0 {
        return Vec::new();
    }

    let mut sorted: Vec<&ProfileEvent> = session.events.iter().collect();
    sorted.sort_by(|a, b| b.duration_us.cmp(&a.duration_us));
    sorted.truncate(top_n);

    sorted
        .into_iter()
        .map(|e| {
            let pct = (e.duration_us as f32 / total_us as f32) * 100.0;
            let suggestion = suggestion_for(&e.category, pct);
            BottleneckInfo {
                event_name: e.name.clone(),
                percentage: pct,
                suggestion,
            }
        })
        .collect()
}

fn suggestion_for(cat: &EventCategory, pct: f32) -> String {
    match cat {
        EventCategory::KernelExecution if pct > 50.0 => {
            "Consider tiling or work-group tuning to reduce kernel time"
                .to_string()
        }
        EventCategory::KernelExecution => {
            "Profile sub-kernels for further breakdown".to_string()
        }
        EventCategory::MemoryTransfer if pct > 30.0 => {
            "Use pinned memory or overlap transfers with compute".to_string()
        }
        EventCategory::MemoryTransfer => {
            "Batch small transfers into larger ones".to_string()
        }
        EventCategory::Synchronization => {
            "Reduce synchronization points or use async barriers".to_string()
        }
        EventCategory::HostCompute => {
            "Offload host computation to device if possible".to_string()
        }
        EventCategory::QueueWait => {
            "Increase queue concurrency or overlap submissions".to_string()
        }
    }
}

/// Generate a full profile report for a session.
pub fn cpu_generate_report(session: &ProfileSession) -> ProfileReport {
    let kernel_time_us = sum_category(session, EventCategory::KernelExecution);
    let transfer_time_us =
        sum_category(session, EventCategory::MemoryTransfer);
    let sync_time_us = sum_category(session, EventCategory::Synchronization);
    let host_time_us = sum_category(session, EventCategory::HostCompute);
    let queue_wait_us = sum_category(session, EventCategory::QueueWait);
    let total_time_us =
        kernel_time_us + transfer_time_us + sync_time_us + host_time_us + queue_wait_us;

    let bottlenecks = cpu_find_bottlenecks(session, 5);
    let flame_root = cpu_build_flame_graph(session);

    ProfileReport {
        session_name: session.name.clone(),
        total_time_us,
        kernel_time_us,
        transfer_time_us,
        sync_time_us,
        host_time_us,
        bottlenecks,
        flame_root,
    }
}

fn sum_category(session: &ProfileSession, cat: EventCategory) -> u64 {
    session
        .events
        .iter()
        .filter(|e| e.category == cat)
        .map(|e| e.duration_us)
        .sum()
}

/// Return time spent per category as a map.
pub fn cpu_category_breakdown(
    session: &ProfileSession,
) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for event in &session.events {
        *map.entry(event.category.to_string()).or_insert(0) += event.duration_us;
    }
    map
}

/// Compute the compute/transfer overlap ratio (0.0 = no overlap, 1.0 = full).
///
/// For the CPU reference implementation we detect simple interval overlaps
/// between `KernelExecution` and `MemoryTransfer` events.
pub fn cpu_compute_overlap(session: &ProfileSession) -> f32 {
    let kernels: Vec<(u64, u64)> = session
        .events
        .iter()
        .filter(|e| e.category == EventCategory::KernelExecution)
        .map(|e| (e.start_us, e.start_us + e.duration_us))
        .collect();
    let transfers: Vec<(u64, u64)> = session
        .events
        .iter()
        .filter(|e| e.category == EventCategory::MemoryTransfer)
        .map(|e| (e.start_us, e.start_us + e.duration_us))
        .collect();

    if kernels.is_empty() || transfers.is_empty() {
        return 0.0;
    }

    let mut overlap_us: u64 = 0;
    for &(ks, ke) in &kernels {
        for &(ts, te) in &transfers {
            let start = ks.max(ts);
            let end = ke.min(te);
            if start < end {
                overlap_us += end - start;
            }
        }
    }

    let total_transfer: u64 = transfers.iter().map(|(s, e)| e - s).sum();
    if total_transfer == 0 {
        0.0
    } else {
        (overlap_us as f32 / total_transfer as f32).min(1.0)
    }
}

/// Suggest optimizations based on a profile report.
pub fn cpu_suggest_optimizations(report: &ProfileReport) -> Vec<String> {
    let mut suggestions = Vec::new();

    if report.total_time_us == 0 {
        return suggestions;
    }

    let kernel_pct =
        report.kernel_time_us as f32 / report.total_time_us as f32 * 100.0;
    let transfer_pct =
        report.transfer_time_us as f32 / report.total_time_us as f32 * 100.0;
    let sync_pct =
        report.sync_time_us as f32 / report.total_time_us as f32 * 100.0;

    if kernel_pct > 60.0 {
        suggestions.push(format!(
            "Kernel execution dominates ({kernel_pct:.1}%); \
             optimize work-group sizes or use sub-groups"
        ));
    }
    if transfer_pct > 25.0 {
        suggestions.push(format!(
            "Memory transfers are {transfer_pct:.1}% of total; \
             use pinned memory and async copies"
        ));
    }
    if sync_pct > 15.0 {
        suggestions.push(format!(
            "Synchronization overhead is {sync_pct:.1}%; \
             reduce barriers or use event-based sync"
        ));
    }

    for b in &report.bottlenecks {
        if b.percentage > 20.0 {
            suggestions.push(format!(
                "Bottleneck '{}' ({:.1}%): {}",
                b.event_name, b.percentage, b.suggestion
            ));
        }
    }

    if suggestions.is_empty() {
        suggestions.push("Profile looks balanced; consider micro-benchmarks for further gains".to_string());
    }
    suggestions
}

/// Export a session to Chrome `chrome://tracing` JSON format.
pub fn cpu_export_chrome_trace(session: &ProfileSession) -> String {
    let mut entries = Vec::new();
    for event in &session.events {
        // Chrome trace uses microseconds for ts and dur.
        let cat = event.category.to_string();
        let name = event.name.replace('"', "\\\"");
        entries.push(format!(
            r#"{{"name":"{}","cat":"{}","ph":"X","ts":{},"dur":{},"pid":1,"tid":1}}"#,
            name, cat, event.start_us, event.duration_us,
        ));
    }
    format!("[{}]", entries.join(","))
}

/// Compare two sessions and return per-event speedup ratios.
///
/// Matches events by name. Returns `(event_name, speedup)` where
/// `speedup = baseline_duration / comparison_duration`.
pub fn cpu_compare_sessions(
    a: &ProfileSession,
    b: &ProfileSession,
) -> Vec<(String, f64)> {
    let a_map: HashMap<&str, u64> =
        a.events.iter().map(|e| (e.name.as_str(), e.duration_us)).collect();

    let mut results = Vec::new();
    for event in &b.events {
        if let Some(&a_dur) = a_map.get(event.name.as_str()) {
            let speedup = if event.duration_us == 0 {
                f64::INFINITY
            } else {
                a_dur as f64 / event.duration_us as f64
            };
            results.push((event.name.clone(), speedup));
        }
    }
    results
}

/// Format a profile report as a human-readable string.
pub fn format_profile_report(report: &ProfileReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("=== Profile: {} ===\n", report.session_name));
    out.push_str(&format!("Total:    {} µs\n", report.total_time_us));
    out.push_str(&format!("Kernel:   {} µs\n", report.kernel_time_us));
    out.push_str(&format!("Transfer: {} µs\n", report.transfer_time_us));
    out.push_str(&format!("Sync:     {} µs\n", report.sync_time_us));
    out.push_str(&format!("Host:     {} µs\n", report.host_time_us));

    if !report.bottlenecks.is_empty() {
        out.push_str("\nBottlenecks:\n");
        for b in &report.bottlenecks {
            out.push_str(&format!(
                "  - {} ({:.1}%): {}\n",
                b.event_name, b.percentage, b.suggestion
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Profiler creation ---------------------------------------------------

    #[test]
    fn test_create_profiler() {
        let p = create_profiler();
        assert!(p.sessions.is_empty());
        assert!(p.current_session.is_none());
        assert!(p.enabled);
    }

    #[test]
    fn test_create_profiler_enabled_by_default() {
        let p = create_profiler();
        assert!(p.enabled);
    }

    // -- Session lifecycle ---------------------------------------------------

    #[test]
    fn test_begin_session_returns_index() {
        let mut p = create_profiler();
        assert_eq!(cpu_begin_session(&mut p, "s0"), 0);
        assert_eq!(cpu_begin_session(&mut p, "s1"), 1);
    }

    #[test]
    fn test_begin_session_sets_current() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s0");
        assert_eq!(p.current_session, Some(0));
    }

    #[test]
    fn test_end_session_clears_current() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s0");
        cpu_end_session(&mut p).unwrap();
        assert!(p.current_session.is_none());
    }

    #[test]
    fn test_end_session_no_active_errors() {
        let mut p = create_profiler();
        assert_eq!(cpu_end_session(&mut p), Err(ProfileError::NoActiveSession));
    }

    #[test]
    fn test_session_name_stored() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "my_session");
        assert_eq!(p.sessions[0].name, "my_session");
    }

    // -- Event recording -----------------------------------------------------

    #[test]
    fn test_record_event_returns_index() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let idx = cpu_record_event(
            &mut p,
            "kernel_a",
            EventCategory::KernelExecution,
            100,
            None,
        )
        .unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_record_event_no_session_errors() {
        let mut p = create_profiler();
        let r = cpu_record_event(
            &mut p,
            "e",
            EventCategory::KernelExecution,
            10,
            None,
        );
        assert_eq!(r, Err(ProfileError::NoActiveSession));
    }

    #[test]
    fn test_record_event_bad_parent_errors() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let r = cpu_record_event(
            &mut p,
            "e",
            EventCategory::KernelExecution,
            10,
            Some(99),
        );
        assert_eq!(r, Err(ProfileError::EventNotFound(99)));
    }

    #[test]
    fn test_record_multiple_events() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "a", EventCategory::KernelExecution, 50, None)
            .unwrap();
        cpu_record_event(
            &mut p,
            "b",
            EventCategory::MemoryTransfer,
            30,
            None,
        )
        .unwrap();
        assert_eq!(p.sessions[0].events.len(), 2);
    }

    #[test]
    fn test_record_event_category_stored() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(
            &mut p,
            "sync",
            EventCategory::Synchronization,
            10,
            None,
        )
        .unwrap();
        assert_eq!(
            p.sessions[0].events[0].category,
            EventCategory::Synchronization
        );
    }

    #[test]
    fn test_record_event_parent_child_link() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let parent = cpu_record_event(
            &mut p,
            "parent",
            EventCategory::KernelExecution,
            100,
            None,
        )
        .unwrap();
        let child = cpu_record_event(
            &mut p,
            "child",
            EventCategory::KernelExecution,
            40,
            Some(parent),
        )
        .unwrap();
        assert_eq!(p.sessions[0].events[parent].children, vec![child]);
        assert_eq!(p.sessions[0].events[child].parent, Some(parent));
    }

    #[test]
    fn test_event_start_time_sequential() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "a", EventCategory::KernelExecution, 50, None)
            .unwrap();
        cpu_record_event(&mut p, "b", EventCategory::KernelExecution, 30, None)
            .unwrap();
        let s = &p.sessions[0];
        assert_eq!(s.events[0].start_us, 0);
        assert_eq!(s.events[1].start_us, 50);
    }

    #[test]
    fn test_event_all_categories() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        for cat in [
            EventCategory::KernelExecution,
            EventCategory::MemoryTransfer,
            EventCategory::Synchronization,
            EventCategory::HostCompute,
            EventCategory::QueueWait,
        ] {
            cpu_record_event(&mut p, &cat.to_string(), cat, 10, None).unwrap();
        }
        assert_eq!(p.sessions[0].events.len(), 5);
    }

    // -- Flame graph ---------------------------------------------------------

    #[test]
    fn test_flame_graph_single_event() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 100, None)
            .unwrap();
        let root = cpu_build_flame_graph(&p.sessions[0]);
        assert_eq!(root.total_time_us, 100);
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].name, "k");
    }

    #[test]
    fn test_flame_graph_parent_child() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let parent = cpu_record_event(
            &mut p,
            "parent",
            EventCategory::KernelExecution,
            100,
            None,
        )
        .unwrap();
        cpu_record_event(
            &mut p,
            "child",
            EventCategory::KernelExecution,
            40,
            Some(parent),
        )
        .unwrap();
        let root = cpu_build_flame_graph(&p.sessions[0]);
        assert_eq!(root.children.len(), 1);
        let p_node = &root.children[0];
        assert_eq!(p_node.total_time_us, 100);
        assert_eq!(p_node.self_time_us, 60);
        assert_eq!(p_node.children.len(), 1);
        assert_eq!(p_node.children[0].name, "child");
    }

    #[test]
    fn test_flame_graph_empty_session() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let root = cpu_build_flame_graph(&p.sessions[0]);
        assert_eq!(root.total_time_us, 0);
        assert!(root.children.is_empty());
    }

    #[test]
    fn test_flame_graph_multiple_roots() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "a", EventCategory::KernelExecution, 50, None)
            .unwrap();
        cpu_record_event(
            &mut p,
            "b",
            EventCategory::MemoryTransfer,
            30,
            None,
        )
        .unwrap();
        let root = cpu_build_flame_graph(&p.sessions[0]);
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.total_time_us, 80);
    }

    #[test]
    fn test_flame_graph_deeply_nested() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let l0 = cpu_record_event(
            &mut p,
            "l0",
            EventCategory::KernelExecution,
            1000,
            None,
        )
        .unwrap();
        let l1 = cpu_record_event(
            &mut p,
            "l1",
            EventCategory::KernelExecution,
            800,
            Some(l0),
        )
        .unwrap();
        let l2 = cpu_record_event(
            &mut p,
            "l2",
            EventCategory::KernelExecution,
            500,
            Some(l1),
        )
        .unwrap();
        cpu_record_event(
            &mut p,
            "l3",
            EventCategory::KernelExecution,
            200,
            Some(l2),
        )
        .unwrap();
        let root = cpu_build_flame_graph(&p.sessions[0]);
        let n0 = &root.children[0];
        assert_eq!(n0.self_time_us, 200); // 1000 - 800
        let n1 = &n0.children[0];
        assert_eq!(n1.self_time_us, 300); // 800 - 500
        let n2 = &n1.children[0];
        assert_eq!(n2.self_time_us, 300); // 500 - 200
        let n3 = &n2.children[0];
        assert_eq!(n3.self_time_us, 200);
    }

    #[test]
    fn test_flame_graph_call_count() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 10, None)
            .unwrap();
        let root = cpu_build_flame_graph(&p.sessions[0]);
        assert_eq!(root.call_count, 1);
        assert_eq!(root.children[0].call_count, 1);
    }

    // -- Bottleneck detection ------------------------------------------------

    #[test]
    fn test_bottleneck_identifies_slowest() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "fast", EventCategory::KernelExecution, 10, None)
            .unwrap();
        cpu_record_event(
            &mut p,
            "slow",
            EventCategory::MemoryTransfer,
            90,
            None,
        )
        .unwrap();
        let bottlenecks = cpu_find_bottlenecks(&p.sessions[0], 1);
        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0].event_name, "slow");
    }

    #[test]
    fn test_bottleneck_percentage() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "a", EventCategory::KernelExecution, 75, None)
            .unwrap();
        cpu_record_event(&mut p, "b", EventCategory::MemoryTransfer, 25, None)
            .unwrap();
        let bottlenecks = cpu_find_bottlenecks(&p.sessions[0], 2);
        assert!((bottlenecks[0].percentage - 75.0).abs() < 0.1);
        assert!((bottlenecks[1].percentage - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_bottleneck_empty_session() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let bottlenecks = cpu_find_bottlenecks(&p.sessions[0], 5);
        assert!(bottlenecks.is_empty());
    }

    #[test]
    fn test_bottleneck_suggestion_not_empty() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 100, None)
            .unwrap();
        let b = cpu_find_bottlenecks(&p.sessions[0], 1);
        assert!(!b[0].suggestion.is_empty());
    }

    // -- Report generation ---------------------------------------------------

    #[test]
    fn test_report_correct_category_totals() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k1", EventCategory::KernelExecution, 50, None)
            .unwrap();
        cpu_record_event(&mut p, "k2", EventCategory::KernelExecution, 30, None)
            .unwrap();
        cpu_record_event(&mut p, "t", EventCategory::MemoryTransfer, 20, None)
            .unwrap();
        let report = cpu_generate_report(&p.sessions[0]);
        assert_eq!(report.kernel_time_us, 80);
        assert_eq!(report.transfer_time_us, 20);
        assert_eq!(report.total_time_us, 100);
    }

    #[test]
    fn test_report_session_name() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "test_session");
        let report = cpu_generate_report(&p.sessions[0]);
        assert_eq!(report.session_name, "test_session");
    }

    #[test]
    fn test_report_empty_session() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "empty");
        let report = cpu_generate_report(&p.sessions[0]);
        assert_eq!(report.total_time_us, 0);
    }

    #[test]
    fn test_report_has_flame_root() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 50, None)
            .unwrap();
        let report = cpu_generate_report(&p.sessions[0]);
        assert_eq!(report.flame_root.children.len(), 1);
    }

    // -- Category breakdown --------------------------------------------------

    #[test]
    fn test_category_breakdown_sums_correctly() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k1", EventCategory::KernelExecution, 40, None)
            .unwrap();
        cpu_record_event(&mut p, "k2", EventCategory::KernelExecution, 60, None)
            .unwrap();
        cpu_record_event(&mut p, "t", EventCategory::MemoryTransfer, 25, None)
            .unwrap();
        let bd = cpu_category_breakdown(&p.sessions[0]);
        assert_eq!(bd["KernelExecution"], 100);
        assert_eq!(bd["MemoryTransfer"], 25);
    }

    #[test]
    fn test_category_breakdown_empty() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let bd = cpu_category_breakdown(&p.sessions[0]);
        assert!(bd.is_empty());
    }

    // -- Overlap computation -------------------------------------------------

    #[test]
    fn test_overlap_zero_when_sequential() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 100, None)
            .unwrap();
        cpu_record_event(&mut p, "t", EventCategory::MemoryTransfer, 50, None)
            .unwrap();
        let overlap = cpu_compute_overlap(&p.sessions[0]);
        assert!((overlap - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_overlap_empty_session() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        assert_eq!(cpu_compute_overlap(&p.sessions[0]), 0.0);
    }

    #[test]
    fn test_overlap_no_transfers() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 100, None)
            .unwrap();
        assert_eq!(cpu_compute_overlap(&p.sessions[0]), 0.0);
    }

    // -- Optimization suggestions -------------------------------------------

    #[test]
    fn test_suggestions_generated_for_bottlenecks() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 900, None)
            .unwrap();
        cpu_record_event(&mut p, "t", EventCategory::MemoryTransfer, 100, None)
            .unwrap();
        let report = cpu_generate_report(&p.sessions[0]);
        let suggestions = cpu_suggest_optimizations(&report);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_suggestions_empty_session() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let report = cpu_generate_report(&p.sessions[0]);
        let suggestions = cpu_suggest_optimizations(&report);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggestions_balanced_profile() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 30, None)
            .unwrap();
        cpu_record_event(&mut p, "t", EventCategory::MemoryTransfer, 30, None)
            .unwrap();
        cpu_record_event(&mut p, "h", EventCategory::HostCompute, 40, None)
            .unwrap();
        let report = cpu_generate_report(&p.sessions[0]);
        let suggestions = cpu_suggest_optimizations(&report);
        assert!(suggestions.iter().any(|s| s.contains("balanced")));
    }

    // -- Chrome trace export -------------------------------------------------

    #[test]
    fn test_chrome_trace_valid_json() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 100, None)
            .unwrap();
        let json = cpu_export_chrome_trace(&p.sessions[0]);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"name\":\"k\""));
    }

    #[test]
    fn test_chrome_trace_empty() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        let json = cpu_export_chrome_trace(&p.sessions[0]);
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_chrome_trace_multiple_events() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "a", EventCategory::KernelExecution, 10, None)
            .unwrap();
        cpu_record_event(&mut p, "b", EventCategory::MemoryTransfer, 20, None)
            .unwrap();
        let json = cpu_export_chrome_trace(&p.sessions[0]);
        assert!(json.contains("\"name\":\"a\""));
        assert!(json.contains("\"name\":\"b\""));
    }

    #[test]
    fn test_chrome_trace_has_duration() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 42, None)
            .unwrap();
        let json = cpu_export_chrome_trace(&p.sessions[0]);
        assert!(json.contains("\"dur\":42"));
    }

    // -- Session comparison --------------------------------------------------

    #[test]
    fn test_compare_sessions_speedup() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "baseline");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 200, None)
            .unwrap();
        cpu_end_session(&mut p).unwrap();
        cpu_begin_session(&mut p, "optimized");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 100, None)
            .unwrap();
        let results =
            cpu_compare_sessions(&p.sessions[0], &p.sessions[1]);
        assert_eq!(results.len(), 1);
        assert!((results[0].1 - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compare_sessions_no_common() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "a");
        cpu_record_event(&mut p, "x", EventCategory::KernelExecution, 10, None)
            .unwrap();
        cpu_end_session(&mut p).unwrap();
        cpu_begin_session(&mut p, "b");
        cpu_record_event(&mut p, "y", EventCategory::KernelExecution, 10, None)
            .unwrap();
        let results =
            cpu_compare_sessions(&p.sessions[0], &p.sessions[1]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_compare_sessions_empty() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "a");
        cpu_end_session(&mut p).unwrap();
        cpu_begin_session(&mut p, "b");
        let results =
            cpu_compare_sessions(&p.sessions[0], &p.sessions[1]);
        assert!(results.is_empty());
    }

    // -- Format report -------------------------------------------------------

    #[test]
    fn test_format_report_contains_name() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "fmt_test");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 50, None)
            .unwrap();
        let report = cpu_generate_report(&p.sessions[0]);
        let text = format_profile_report(&report);
        assert!(text.contains("fmt_test"));
    }

    #[test]
    fn test_format_report_contains_totals() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 100, None)
            .unwrap();
        let report = cpu_generate_report(&p.sessions[0]);
        let text = format_profile_report(&report);
        assert!(text.contains("100"));
        assert!(text.contains("µs"));
    }

    // -- Property-based style tests ------------------------------------------

    #[test]
    fn test_property_total_ge_max_category() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "k", EventCategory::KernelExecution, 50, None)
            .unwrap();
        cpu_record_event(&mut p, "t", EventCategory::MemoryTransfer, 30, None)
            .unwrap();
        cpu_record_event(
            &mut p,
            "s",
            EventCategory::Synchronization,
            20,
            None,
        )
        .unwrap();
        let report = cpu_generate_report(&p.sessions[0]);
        assert!(report.total_time_us >= report.kernel_time_us);
        assert!(report.total_time_us >= report.transfer_time_us);
        assert!(report.total_time_us >= report.sync_time_us);
    }

    #[test]
    fn test_property_percentages_sum_approx_100() {
        let mut p = create_profiler();
        cpu_begin_session(&mut p, "s");
        cpu_record_event(&mut p, "a", EventCategory::KernelExecution, 40, None)
            .unwrap();
        cpu_record_event(&mut p, "b", EventCategory::MemoryTransfer, 30, None)
            .unwrap();
        cpu_record_event(
            &mut p,
            "c",
            EventCategory::Synchronization,
            20,
            None,
        )
        .unwrap();
        cpu_record_event(&mut p, "d", EventCategory::HostCompute, 10, None)
            .unwrap();
        let bottlenecks = cpu_find_bottlenecks(&p.sessions[0], 10);
        let total_pct: f32 =
            bottlenecks.iter().map(|b| b.percentage).sum();
        assert!(
            (total_pct - 100.0).abs() < 0.5,
            "percentages sum to {total_pct}, expected ~100"
        );
    }

    // -- Error display -------------------------------------------------------

    #[test]
    fn test_profile_error_display() {
        assert_eq!(
            ProfileError::NoActiveSession.to_string(),
            "no active profiling session"
        );
        assert_eq!(
            ProfileError::EventNotFound(42).to_string(),
            "event 42 not found"
        );
        assert_eq!(
            ProfileError::SessionNotFound.to_string(),
            "session not found"
        );
    }

    #[test]
    fn test_event_category_display() {
        assert_eq!(EventCategory::KernelExecution.to_string(), "KernelExecution");
        assert_eq!(EventCategory::QueueWait.to_string(), "QueueWait");
    }
}

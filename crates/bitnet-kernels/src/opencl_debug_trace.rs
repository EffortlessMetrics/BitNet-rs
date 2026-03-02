//! Debug tracing system for A770 OpenCL kernels.
//!
//! Captures kernel dispatch parameters, buffer states, intermediate tensor
//! values, and execution traces for debugging numerical issues and performance
//! problems.  All operations have CPU reference implementations so the module
//! compiles and tests without an actual OpenCL runtime.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Trace level
// ---------------------------------------------------------------------------

/// Verbosity level for trace capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TraceLevel {
    /// No tracing.
    Off,
    /// Kernel name + status only.
    Summary,
    /// Summary + args and work sizes.
    Detailed,
    /// Everything including buffer snapshots.
    Full,
}

impl fmt::Display for TraceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "Off"),
            Self::Summary => write!(f, "Summary"),
            Self::Detailed => write!(f, "Detailed"),
            Self::Full => write!(f, "Full"),
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel trace
// ---------------------------------------------------------------------------

/// Execution status of a kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelStatus {
    Success,
    Error,
    Timeout,
}

/// Captured information about a single kernel dispatch.
#[derive(Debug, Clone)]
pub struct KernelTrace {
    /// Name of the dispatched kernel.
    pub kernel_name: String,
    /// Global work size per dimension.
    pub global_size: Vec<usize>,
    /// Local work size per dimension (may be empty for auto).
    pub local_size: Vec<usize>,
    /// Human-readable argument descriptions.
    pub args_info: Vec<String>,
    /// Wall-clock execution time in microseconds.
    pub duration_us: u64,
    /// Outcome of the dispatch.
    pub status: KernelStatus,
}

impl KernelTrace {
    /// Create a new trace for a successful kernel dispatch.
    pub fn new(
        kernel_name: impl Into<String>,
        global_size: Vec<usize>,
        local_size: Vec<usize>,
        duration_us: u64,
    ) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            global_size,
            local_size,
            args_info: Vec::new(),
            duration_us,
            status: KernelStatus::Success,
        }
    }

    /// Attach argument information.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args_info = args;
        self
    }

    /// Set the dispatch status.
    pub fn with_status(mut self, status: KernelStatus) -> Self {
        self.status = status;
        self
    }
}

impl fmt::Display for KernelTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} global={:?} local={:?} {:.1}ms {:?}",
            self.kernel_name,
            self.global_size,
            self.local_size,
            self.duration_us as f64 / 1000.0,
            self.status,
        )
    }
}

// ---------------------------------------------------------------------------
// Buffer snapshot
// ---------------------------------------------------------------------------

/// Snapshot of a GPU buffer's contents for debugging.
#[derive(Debug, Clone)]
pub struct BufferSnapshot {
    /// Human-readable buffer name.
    pub name: String,
    /// Shape of the tensor (e.g. `[batch, seq_len, hidden]`).
    pub shape: Vec<usize>,
    /// Data type description (e.g. `"f32"`, `"i2_s"`).
    pub dtype: String,
    /// First N sample values.
    pub head_values: Vec<f64>,
    /// Last N sample values.
    pub tail_values: Vec<f64>,
    /// Deterministic checksum over all values.
    pub checksum: u64,
    /// Count of NaN values detected.
    pub nan_count: usize,
    /// Count of infinity values detected.
    pub inf_count: usize,
    /// Total number of elements.
    pub num_elements: usize,
}

impl BufferSnapshot {
    /// Number of sample values to capture from each end by default.
    const DEFAULT_SAMPLE_N: usize = 8;

    /// Create a snapshot from an `f32` buffer.
    pub fn from_f32(name: impl Into<String>, shape: Vec<usize>, data: &[f32]) -> Self {
        Self::from_f32_with_n(name, shape, data, Self::DEFAULT_SAMPLE_N)
    }

    /// Create a snapshot with a custom sample count.
    pub fn from_f32_with_n(
        name: impl Into<String>,
        shape: Vec<usize>,
        data: &[f32],
        n: usize,
    ) -> Self {
        let num_elements = data.len();
        let head_values: Vec<f64> = data.iter().take(n).map(|&v| v as f64).collect();
        let tail_values: Vec<f64> = if num_elements > n {
            data.iter().rev().take(n).rev().map(|&v| v as f64).collect()
        } else {
            head_values.clone()
        };

        let checksum = compute_checksum_f32(data);
        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();

        Self {
            name: name.into(),
            shape,
            dtype: "f32".into(),
            head_values,
            tail_values,
            checksum,
            nan_count,
            inf_count,
            num_elements,
        }
    }

    /// Create a snapshot from an `f64` buffer.
    pub fn from_f64(name: impl Into<String>, shape: Vec<usize>, data: &[f64]) -> Self {
        let n = Self::DEFAULT_SAMPLE_N;
        let num_elements = data.len();
        let head_values: Vec<f64> = data.iter().take(n).copied().collect();
        let tail_values: Vec<f64> = if num_elements > n {
            data.iter().rev().take(n).rev().copied().collect()
        } else {
            head_values.clone()
        };

        let checksum = compute_checksum_f64(data);
        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();

        Self {
            name: name.into(),
            shape,
            dtype: "f64".into(),
            head_values,
            tail_values,
            checksum,
            nan_count,
            inf_count,
            num_elements,
        }
    }

    /// Returns `true` if the snapshot contains any NaN values.
    pub fn has_nan(&self) -> bool {
        self.nan_count > 0
    }

    /// Returns `true` if the snapshot contains any infinity values.
    pub fn has_inf(&self) -> bool {
        self.inf_count > 0
    }

    /// Returns `true` if the snapshot is numerically clean (no NaN/Inf).
    pub fn is_clean(&self) -> bool {
        !self.has_nan() && !self.has_inf()
    }
}

impl fmt::Display for BufferSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} shape={:?} dtype={} elems={} nan={} inf={} cksum={:#x}",
            self.name,
            self.shape,
            self.dtype,
            self.num_elements,
            self.nan_count,
            self.inf_count,
            self.checksum,
        )
    }
}

// ---------------------------------------------------------------------------
// Checksum helpers (CPU reference)
// ---------------------------------------------------------------------------

/// Compute a deterministic 64-bit checksum over `f32` data.
///
/// Uses bit-level representation to ensure NaN values are handled
/// deterministically (all NaN bit patterns hash identically).
pub fn compute_checksum_f32(data: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for &v in data {
        let bits = canonicalize_f32(v).to_bits();
        h ^= bits as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    h
}

/// Compute a deterministic 64-bit checksum over `f64` data.
pub fn compute_checksum_f64(data: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        let bits = canonicalize_f64(v).to_bits();
        h ^= bits;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Canonicalize an `f32`: map all NaN patterns to a single canonical NaN.
fn canonicalize_f32(v: f32) -> f32 {
    if v.is_nan() { f32::NAN } else { v }
}

/// Canonicalize an `f64`: map all NaN patterns to a single canonical NaN.
fn canonicalize_f64(v: f64) -> f64 {
    if v.is_nan() { f64::NAN } else { v }
}

// ---------------------------------------------------------------------------
// Trace filter
// ---------------------------------------------------------------------------

/// Filter criteria for selecting which traces to capture or display.
#[derive(Debug, Clone, Default)]
pub struct TraceFilter {
    /// If `Some`, only capture kernels whose name matches one of these.
    pub kernel_names: Option<Vec<String>>,
    /// Minimum execution duration (µs) to include.
    pub min_duration_us: u64,
    /// Whether to capture buffer snapshots (expensive).
    pub capture_buffers: bool,
}

impl TraceFilter {
    /// Create a filter that captures everything.
    pub fn capture_all() -> Self {
        Self { kernel_names: None, min_duration_us: 0, capture_buffers: true }
    }

    /// Create a filter targeting specific kernel names.
    pub fn for_kernels(names: Vec<String>) -> Self {
        Self { kernel_names: Some(names), min_duration_us: 0, capture_buffers: false }
    }

    /// Set the minimum duration threshold.
    pub fn with_min_duration(mut self, us: u64) -> Self {
        self.min_duration_us = us;
        self
    }

    /// Enable buffer snapshot capture.
    pub fn with_buffers(mut self) -> Self {
        self.capture_buffers = true;
        self
    }

    /// Check if a trace passes this filter.
    pub fn matches(&self, trace: &KernelTrace) -> bool {
        if let Some(ref names) = self.kernel_names
            && !names.iter().any(|n| n == &trace.kernel_name)
        {
            return false;
        }
        if trace.duration_us < self.min_duration_us {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Trace session
// ---------------------------------------------------------------------------

/// A debug trace session that collects kernel traces and buffer snapshots.
#[derive(Debug, Clone)]
pub struct TraceSession {
    /// Unique session identifier.
    pub id: String,
    /// When the session started.
    pub started_at: SystemTime,
    /// Collected kernel traces.
    pub traces: Vec<KernelTrace>,
    /// Collected buffer snapshots.
    pub snapshots: Vec<BufferSnapshot>,
    /// Active trace level.
    pub level: TraceLevel,
    /// Optional filter applied during capture.
    pub filter: Option<TraceFilter>,
}

impl TraceSession {
    /// Create a new session with the given ID and trace level.
    pub fn new(id: impl Into<String>, level: TraceLevel) -> Self {
        Self {
            id: id.into(),
            started_at: SystemTime::now(),
            traces: Vec::new(),
            snapshots: Vec::new(),
            level,
            filter: None,
        }
    }

    /// Attach a filter to this session.
    pub fn with_filter(mut self, filter: TraceFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Record a kernel trace if the current level and filter allow it.
    pub fn record_trace(&mut self, trace: KernelTrace) {
        if self.level == TraceLevel::Off {
            return;
        }
        if let Some(ref f) = self.filter
            && !f.matches(&trace)
        {
            return;
        }
        self.traces.push(trace);
    }

    /// Record a buffer snapshot if the current level is `Full`.
    pub fn record_snapshot(&mut self, snapshot: BufferSnapshot) {
        if self.level != TraceLevel::Full {
            return;
        }
        if let Some(ref f) = self.filter
            && !f.capture_buffers
        {
            return;
        }
        self.snapshots.push(snapshot);
    }

    /// Number of recorded kernel traces.
    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    /// Number of recorded buffer snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Total wall-clock time across all traced kernels in microseconds.
    pub fn total_duration_us(&self) -> u64 {
        self.traces.iter().map(|t| t.duration_us).sum()
    }

    /// Return traces sorted by duration descending (hotspot first).
    pub fn hotspots(&self) -> Vec<&KernelTrace> {
        let mut sorted: Vec<&KernelTrace> = self.traces.iter().collect();
        sorted.sort_by(|a, b| b.duration_us.cmp(&a.duration_us));
        sorted
    }

    /// Return only snapshots that contain NaN or Inf values.
    pub fn problematic_snapshots(&self) -> Vec<&BufferSnapshot> {
        self.snapshots.iter().filter(|s| !s.is_clean()).collect()
    }

    /// Merge another session's traces and snapshots into this one.
    pub fn merge(&mut self, other: &TraceSession) {
        self.traces.extend(other.traces.iter().cloned());
        self.snapshots.extend(other.snapshots.iter().cloned());
    }

    /// Return a summary of kernel invocation counts by name.
    pub fn kernel_histogram(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        for t in &self.traces {
            *map.entry(t.kernel_name.clone()).or_insert(0) += 1;
        }
        map
    }

    /// Elapsed wall-clock time since session start.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed().unwrap_or(Duration::ZERO)
    }
}

impl fmt::Display for TraceSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TraceSession({}, level={}, traces={}, snapshots={})",
            self.id,
            self.level,
            self.traces.len(),
            self.snapshots.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Trace exporter
// ---------------------------------------------------------------------------

/// Exports a [`TraceSession`] to various formats.
pub struct TraceExporter;

impl TraceExporter {
    /// Export the session as a JSON string.
    ///
    /// This is a lightweight, dependency-free serializer (no `serde` needed at
    /// runtime) that produces a human-readable JSON object.
    pub fn to_json(session: &TraceSession) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"id\": \"{}\",\n", session.id));
        out.push_str(&format!("  \"level\": \"{}\",\n", session.level));
        out.push_str(&format!("  \"total_duration_us\": {},\n", session.total_duration_us()));

        // traces
        out.push_str("  \"traces\": [\n");
        for (i, t) in session.traces.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"kernel_name\": \"{}\",\n", t.kernel_name));
            out.push_str(&format!("      \"global_size\": {:?},\n", t.global_size));
            out.push_str(&format!("      \"local_size\": {:?},\n", t.local_size));
            out.push_str(&format!("      \"duration_us\": {},\n", t.duration_us));
            out.push_str(&format!("      \"status\": \"{:?}\"\n", t.status));
            if i + 1 < session.traces.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  ],\n");

        // snapshots
        out.push_str("  \"snapshots\": [\n");
        for (i, s) in session.snapshots.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"name\": \"{}\",\n", s.name));
            out.push_str(&format!("      \"shape\": {:?},\n", s.shape));
            out.push_str(&format!("      \"dtype\": \"{}\",\n", s.dtype));
            out.push_str(&format!("      \"num_elements\": {},\n", s.num_elements));
            out.push_str(&format!("      \"nan_count\": {},\n", s.nan_count));
            out.push_str(&format!("      \"inf_count\": {},\n", s.inf_count));
            out.push_str(&format!("      \"checksum\": \"{:#x}\"\n", s.checksum));
            if i + 1 < session.snapshots.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  ]\n");

        out.push('}');
        out
    }

    /// Export the session in Chrome Trace Event format (JSON).
    ///
    /// The output can be loaded into `chrome://tracing` or Perfetto for
    /// timeline visualisation.  Each kernel dispatch becomes a duration event
    /// (`ph: "X"`) on a single thread.
    pub fn to_chrome_trace(session: &TraceSession) -> String {
        let mut events = String::from("[");
        let mut ts: u64 = 0; // running timestamp in µs
        for (i, t) in session.traces.iter().enumerate() {
            if i > 0 {
                events.push(',');
            }
            events.push_str(&format!(
                concat!(
                    "{{",
                    "\"name\":\"{name}\",",
                    "\"cat\":\"kernel\",",
                    "\"ph\":\"X\",",
                    "\"ts\":{ts},",
                    "\"dur\":{dur},",
                    "\"pid\":1,",
                    "\"tid\":1,",
                    "\"args\":{{\"global_size\":\"{gs:?}\",\"local_size\":\"{ls:?}\",\"status\":\"{st:?}\"}}",
                    "}}"
                ),
                name = t.kernel_name,
                ts = ts,
                dur = t.duration_us,
                gs = t.global_size,
                ls = t.local_size,
                st = t.status,
            ));
            ts += t.duration_us;
        }
        events.push(']');
        events
    }
}

// ---------------------------------------------------------------------------
// Divergence detector
// ---------------------------------------------------------------------------

/// Result of comparing two trace sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Divergence {
    /// Sessions are equivalent (same kernels, same checksums).
    None,
    /// Different number of traces.
    TraceCountMismatch { expected: usize, actual: usize },
    /// Kernel name differs at a specific trace index.
    KernelNameMismatch { index: usize, expected: String, actual: String },
    /// Buffer checksum differs at a specific snapshot index.
    ChecksumMismatch { index: usize, name: String, expected: u64, actual: u64 },
    /// Snapshot count differs.
    SnapshotCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "no divergence"),
            Self::TraceCountMismatch { expected, actual } => {
                write!(f, "trace count: expected {expected}, got {actual}")
            }
            Self::KernelNameMismatch { index, expected, actual } => {
                write!(f, "kernel name at [{index}]: expected \"{expected}\", got \"{actual}\"")
            }
            Self::ChecksumMismatch { index, name, expected, actual } => {
                write!(
                    f,
                    "checksum for \"{name}\" at [{index}]: expected {expected:#x}, \
                     got {actual:#x}"
                )
            }
            Self::SnapshotCountMismatch { expected, actual } => {
                write!(f, "snapshot count: expected {expected}, got {actual}")
            }
        }
    }
}

/// Compares two [`TraceSession`]s and finds the first point of divergence.
pub struct DivergenceDetector;

impl DivergenceDetector {
    /// Compare two sessions and return the first divergence found.
    ///
    /// Checks in order:
    /// 1. Trace count
    /// 2. Kernel names (pairwise)
    /// 3. Snapshot count
    /// 4. Snapshot checksums (pairwise)
    pub fn compare(expected: &TraceSession, actual: &TraceSession) -> Divergence {
        // 1. Trace count
        if expected.traces.len() != actual.traces.len() {
            return Divergence::TraceCountMismatch {
                expected: expected.traces.len(),
                actual: actual.traces.len(),
            };
        }

        // 2. Kernel names
        for (i, (e, a)) in expected.traces.iter().zip(actual.traces.iter()).enumerate() {
            if e.kernel_name != a.kernel_name {
                return Divergence::KernelNameMismatch {
                    index: i,
                    expected: e.kernel_name.clone(),
                    actual: a.kernel_name.clone(),
                };
            }
        }

        // 3. Snapshot count
        if expected.snapshots.len() != actual.snapshots.len() {
            return Divergence::SnapshotCountMismatch {
                expected: expected.snapshots.len(),
                actual: actual.snapshots.len(),
            };
        }

        // 4. Snapshot checksums
        for (i, (e, a)) in expected.snapshots.iter().zip(actual.snapshots.iter()).enumerate() {
            if e.checksum != a.checksum {
                return Divergence::ChecksumMismatch {
                    index: i,
                    name: e.name.clone(),
                    expected: e.checksum,
                    actual: a.checksum,
                };
            }
        }

        Divergence::None
    }

    /// Return *all* divergences between two sessions (not just the first).
    pub fn compare_all(expected: &TraceSession, actual: &TraceSession) -> Vec<Divergence> {
        let mut divs = Vec::new();

        if expected.traces.len() != actual.traces.len() {
            divs.push(Divergence::TraceCountMismatch {
                expected: expected.traces.len(),
                actual: actual.traces.len(),
            });
        }

        let min_traces = expected.traces.len().min(actual.traces.len());
        for i in 0..min_traces {
            if expected.traces[i].kernel_name != actual.traces[i].kernel_name {
                divs.push(Divergence::KernelNameMismatch {
                    index: i,
                    expected: expected.traces[i].kernel_name.clone(),
                    actual: actual.traces[i].kernel_name.clone(),
                });
            }
        }

        if expected.snapshots.len() != actual.snapshots.len() {
            divs.push(Divergence::SnapshotCountMismatch {
                expected: expected.snapshots.len(),
                actual: actual.snapshots.len(),
            });
        }

        let min_snaps = expected.snapshots.len().min(actual.snapshots.len());
        for i in 0..min_snaps {
            if expected.snapshots[i].checksum != actual.snapshots[i].checksum {
                divs.push(Divergence::ChecksumMismatch {
                    index: i,
                    name: expected.snapshots[i].name.clone(),
                    expected: expected.snapshots[i].checksum,
                    actual: actual.snapshots[i].checksum,
                });
            }
        }

        divs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TraceLevel ---------------------------------------------------------

    #[test]
    fn trace_level_ordering() {
        assert!(TraceLevel::Off < TraceLevel::Summary);
        assert!(TraceLevel::Summary < TraceLevel::Detailed);
        assert!(TraceLevel::Detailed < TraceLevel::Full);
    }

    #[test]
    fn trace_level_display() {
        assert_eq!(TraceLevel::Off.to_string(), "Off");
        assert_eq!(TraceLevel::Summary.to_string(), "Summary");
        assert_eq!(TraceLevel::Detailed.to_string(), "Detailed");
        assert_eq!(TraceLevel::Full.to_string(), "Full");
    }

    #[test]
    fn trace_level_equality() {
        assert_eq!(TraceLevel::Full, TraceLevel::Full);
        assert_ne!(TraceLevel::Off, TraceLevel::Full);
    }

    // -- KernelTrace --------------------------------------------------------

    #[test]
    fn kernel_trace_new() {
        let t = KernelTrace::new("matmul", vec![1024], vec![256], 500);
        assert_eq!(t.kernel_name, "matmul");
        assert_eq!(t.global_size, vec![1024]);
        assert_eq!(t.local_size, vec![256]);
        assert_eq!(t.duration_us, 500);
        assert_eq!(t.status, KernelStatus::Success);
        assert!(t.args_info.is_empty());
    }

    #[test]
    fn kernel_trace_with_args() {
        let t = KernelTrace::new("add", vec![512], vec![], 100)
            .with_args(vec!["buf_a: f32[512]".into(), "buf_b: f32[512]".into()]);
        assert_eq!(t.args_info.len(), 2);
    }

    #[test]
    fn kernel_trace_with_status() {
        let t =
            KernelTrace::new("bad_kernel", vec![1], vec![1], 0).with_status(KernelStatus::Error);
        assert_eq!(t.status, KernelStatus::Error);
    }

    #[test]
    fn kernel_trace_display() {
        let t = KernelTrace::new("softmax", vec![64, 128], vec![16], 1234);
        let s = t.to_string();
        assert!(s.contains("softmax"));
        assert!(s.contains("Success"));
    }

    #[test]
    fn kernel_trace_timeout_status() {
        let t =
            KernelTrace::new("stall", vec![1], vec![1], 999999).with_status(KernelStatus::Timeout);
        assert_eq!(t.status, KernelStatus::Timeout);
    }

    // -- BufferSnapshot -----------------------------------------------------

    #[test]
    fn snapshot_from_f32_basic() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let snap = BufferSnapshot::from_f32("weights", vec![10, 10], &data);
        assert_eq!(snap.name, "weights");
        assert_eq!(snap.shape, vec![10, 10]);
        assert_eq!(snap.dtype, "f32");
        assert_eq!(snap.num_elements, 100);
        assert_eq!(snap.nan_count, 0);
        assert_eq!(snap.inf_count, 0);
        assert!(snap.is_clean());
    }

    #[test]
    fn snapshot_head_tail_values() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let snap = BufferSnapshot::from_f32("x", vec![20], &data);
        assert_eq!(snap.head_values.len(), 8);
        assert_eq!(snap.tail_values.len(), 8);
        assert_eq!(snap.head_values[0], 0.0);
        assert_eq!(snap.head_values[7], 7.0);
        assert_eq!(snap.tail_values[7], 19.0);
    }

    #[test]
    fn snapshot_small_buffer_head_tail_same() {
        let data = vec![1.0f32, 2.0, 3.0];
        let snap = BufferSnapshot::from_f32("tiny", vec![3], &data);
        assert_eq!(snap.head_values, snap.tail_values);
    }

    #[test]
    fn snapshot_detects_nan() {
        let data = vec![1.0f32, f32::NAN, 3.0, f32::NAN];
        let snap = BufferSnapshot::from_f32("nan_buf", vec![4], &data);
        assert_eq!(snap.nan_count, 2);
        assert!(snap.has_nan());
        assert!(!snap.is_clean());
    }

    #[test]
    fn snapshot_detects_inf() {
        let data = vec![f32::INFINITY, 1.0, f32::NEG_INFINITY];
        let snap = BufferSnapshot::from_f32("inf_buf", vec![3], &data);
        assert_eq!(snap.inf_count, 2);
        assert!(snap.has_inf());
        assert!(!snap.is_clean());
    }

    #[test]
    fn snapshot_nan_and_inf_combined() {
        let data = vec![f32::NAN, f32::INFINITY, 0.0];
        let snap = BufferSnapshot::from_f32("mixed", vec![3], &data);
        assert_eq!(snap.nan_count, 1);
        assert_eq!(snap.inf_count, 1);
        assert!(!snap.is_clean());
    }

    #[test]
    fn snapshot_empty_buffer() {
        let data: Vec<f32> = vec![];
        let snap = BufferSnapshot::from_f32("empty", vec![0], &data);
        assert_eq!(snap.num_elements, 0);
        assert!(snap.head_values.is_empty());
        assert!(snap.is_clean());
    }

    #[test]
    fn snapshot_from_f64() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let snap = BufferSnapshot::from_f64("f64_buf", vec![3], &data);
        assert_eq!(snap.dtype, "f64");
        assert_eq!(snap.num_elements, 3);
        assert!(snap.is_clean());
    }

    #[test]
    fn snapshot_display() {
        let data = vec![1.0f32; 10];
        let snap = BufferSnapshot::from_f32("buf", vec![10], &data);
        let s = snap.to_string();
        assert!(s.contains("buf"));
        assert!(s.contains("elems=10"));
    }

    #[test]
    fn snapshot_custom_sample_count() {
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let snap = BufferSnapshot::from_f32_with_n("x", vec![50], &data, 3);
        assert_eq!(snap.head_values.len(), 3);
        assert_eq!(snap.tail_values.len(), 3);
        assert_eq!(snap.head_values, vec![0.0, 1.0, 2.0]);
        assert_eq!(snap.tail_values, vec![47.0, 48.0, 49.0]);
    }

    // -- Checksum -----------------------------------------------------------

    #[test]
    fn checksum_deterministic() {
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let c1 = compute_checksum_f32(&data);
        let c2 = compute_checksum_f32(&data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn checksum_differs_on_change() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![1.0f32, 2.0, 3.001];
        assert_ne!(compute_checksum_f32(&data1), compute_checksum_f32(&data2));
    }

    #[test]
    fn checksum_empty() {
        let c = compute_checksum_f32(&[]);
        // Must be the FNV offset basis (no data mixed in)
        assert_eq!(c, 0xcbf29ce484222325);
    }

    #[test]
    fn checksum_nan_canonical() {
        // Different NaN bit patterns should produce the same checksum.
        let nan1 = f32::from_bits(0x7fc00001);
        let nan2 = f32::from_bits(0x7fc00002);
        assert!(nan1.is_nan());
        assert!(nan2.is_nan());
        assert_eq!(compute_checksum_f32(&[nan1]), compute_checksum_f32(&[nan2]),);
    }

    #[test]
    fn checksum_f64_deterministic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        assert_eq!(compute_checksum_f64(&data), compute_checksum_f64(&data));
    }

    // -- TraceFilter --------------------------------------------------------

    #[test]
    fn filter_default_matches_all() {
        let f = TraceFilter::default();
        let t = KernelTrace::new("anything", vec![1], vec![], 0);
        assert!(f.matches(&t));
    }

    #[test]
    fn filter_by_kernel_name() {
        let f = TraceFilter::for_kernels(vec!["matmul".into()]);
        let t1 = KernelTrace::new("matmul", vec![1], vec![], 100);
        let t2 = KernelTrace::new("softmax", vec![1], vec![], 100);
        assert!(f.matches(&t1));
        assert!(!f.matches(&t2));
    }

    #[test]
    fn filter_by_min_duration() {
        let f = TraceFilter::default().with_min_duration(500);
        let fast = KernelTrace::new("k", vec![1], vec![], 100);
        let slow = KernelTrace::new("k", vec![1], vec![], 1000);
        assert!(!f.matches(&fast));
        assert!(f.matches(&slow));
    }

    #[test]
    fn filter_combined_name_and_duration() {
        let f = TraceFilter::for_kernels(vec!["matmul".into()]).with_min_duration(200);
        let t1 = KernelTrace::new("matmul", vec![1], vec![], 100); // wrong duration
        let t2 = KernelTrace::new("softmax", vec![1], vec![], 300); // wrong name
        let t3 = KernelTrace::new("matmul", vec![1], vec![], 300); // match
        assert!(!f.matches(&t1));
        assert!(!f.matches(&t2));
        assert!(f.matches(&t3));
    }

    #[test]
    fn filter_capture_all() {
        let f = TraceFilter::capture_all();
        assert!(f.capture_buffers);
        assert!(f.kernel_names.is_none());
    }

    #[test]
    fn filter_with_buffers() {
        let f = TraceFilter::default().with_buffers();
        assert!(f.capture_buffers);
    }

    // -- TraceSession -------------------------------------------------------

    #[test]
    fn session_new() {
        let s = TraceSession::new("test-1", TraceLevel::Detailed);
        assert_eq!(s.id, "test-1");
        assert_eq!(s.level, TraceLevel::Detailed);
        assert_eq!(s.trace_count(), 0);
        assert_eq!(s.snapshot_count(), 0);
    }

    #[test]
    fn session_record_trace_summary_level() {
        let mut s = TraceSession::new("s", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("k1", vec![64], vec![16], 100));
        assert_eq!(s.trace_count(), 1);
    }

    #[test]
    fn session_off_ignores_traces() {
        let mut s = TraceSession::new("s", TraceLevel::Off);
        s.record_trace(KernelTrace::new("k", vec![1], vec![], 10));
        assert_eq!(s.trace_count(), 0);
    }

    #[test]
    fn session_snapshot_only_at_full() {
        let snap = BufferSnapshot::from_f32("b", vec![2], &[1.0, 2.0]);

        let mut detailed = TraceSession::new("d", TraceLevel::Detailed);
        detailed.record_snapshot(snap.clone());
        assert_eq!(detailed.snapshot_count(), 0);

        let mut full = TraceSession::new("f", TraceLevel::Full);
        full.record_snapshot(snap);
        assert_eq!(full.snapshot_count(), 1);
    }

    #[test]
    fn session_snapshot_respects_filter() {
        let snap = BufferSnapshot::from_f32("b", vec![2], &[1.0, 2.0]);

        // capture_buffers = false → snapshot not recorded even at Full
        let filter = TraceFilter::default(); // capture_buffers is false
        let mut s = TraceSession::new("s", TraceLevel::Full).with_filter(filter);
        s.record_snapshot(snap);
        assert_eq!(s.snapshot_count(), 0);
    }

    #[test]
    fn session_total_duration() {
        let mut s = TraceSession::new("s", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("a", vec![1], vec![], 100));
        s.record_trace(KernelTrace::new("b", vec![1], vec![], 200));
        s.record_trace(KernelTrace::new("c", vec![1], vec![], 300));
        assert_eq!(s.total_duration_us(), 600);
    }

    #[test]
    fn session_hotspots_sorted() {
        let mut s = TraceSession::new("s", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("fast", vec![1], vec![], 10));
        s.record_trace(KernelTrace::new("slow", vec![1], vec![], 999));
        s.record_trace(KernelTrace::new("mid", vec![1], vec![], 500));
        let hot = s.hotspots();
        assert_eq!(hot[0].kernel_name, "slow");
        assert_eq!(hot[1].kernel_name, "mid");
        assert_eq!(hot[2].kernel_name, "fast");
    }

    #[test]
    fn session_problematic_snapshots() {
        let mut s = TraceSession::new("s", TraceLevel::Full);
        let filter = TraceFilter::capture_all();
        s.filter = Some(filter);
        s.record_snapshot(BufferSnapshot::from_f32("ok", vec![2], &[1.0, 2.0]));
        s.record_snapshot(BufferSnapshot::from_f32("bad", vec![2], &[f32::NAN, 1.0]));
        let problems = s.problematic_snapshots();
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].name, "bad");
    }

    #[test]
    fn session_merge() {
        let mut a = TraceSession::new("a", TraceLevel::Summary);
        a.record_trace(KernelTrace::new("k1", vec![1], vec![], 10));

        let mut b = TraceSession::new("b", TraceLevel::Summary);
        b.record_trace(KernelTrace::new("k2", vec![1], vec![], 20));

        a.merge(&b);
        assert_eq!(a.trace_count(), 2);
    }

    #[test]
    fn session_kernel_histogram() {
        let mut s = TraceSession::new("s", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("matmul", vec![1], vec![], 10));
        s.record_trace(KernelTrace::new("matmul", vec![1], vec![], 20));
        s.record_trace(KernelTrace::new("softmax", vec![1], vec![], 30));
        let hist = s.kernel_histogram();
        assert_eq!(hist["matmul"], 2);
        assert_eq!(hist["softmax"], 1);
    }

    #[test]
    fn session_display() {
        let s = TraceSession::new("demo", TraceLevel::Full);
        let d = s.to_string();
        assert!(d.contains("demo"));
        assert!(d.contains("Full"));
    }

    #[test]
    fn session_with_filter_applies() {
        let filter = TraceFilter::for_kernels(vec!["target".into()]);
        let mut s = TraceSession::new("s", TraceLevel::Summary).with_filter(filter);
        s.record_trace(KernelTrace::new("target", vec![1], vec![], 10));
        s.record_trace(KernelTrace::new("other", vec![1], vec![], 10));
        assert_eq!(s.trace_count(), 1);
        assert_eq!(s.traces[0].kernel_name, "target");
    }

    #[test]
    fn session_empty_total_duration() {
        let s = TraceSession::new("empty", TraceLevel::Summary);
        assert_eq!(s.total_duration_us(), 0);
    }

    // -- TraceExporter (JSON) -----------------------------------------------

    #[test]
    fn export_json_empty_session() {
        let s = TraceSession::new("empty", TraceLevel::Summary);
        let json = TraceExporter::to_json(&s);
        assert!(json.contains("\"id\": \"empty\""));
        assert!(json.contains("\"traces\": ["));
        assert!(json.contains("\"snapshots\": ["));
    }

    #[test]
    fn export_json_with_traces() {
        let mut s = TraceSession::new("j", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("k1", vec![64], vec![16], 200));
        let json = TraceExporter::to_json(&s);
        assert!(json.contains("\"kernel_name\": \"k1\""));
        assert!(json.contains("\"duration_us\": 200"));
    }

    #[test]
    fn export_json_with_snapshots() {
        let mut s = TraceSession::new("j", TraceLevel::Full);
        let filter = TraceFilter::capture_all();
        s.filter = Some(filter);
        s.record_snapshot(BufferSnapshot::from_f32("buf", vec![3], &[1.0, 2.0, 3.0]));
        let json = TraceExporter::to_json(&s);
        assert!(json.contains("\"name\": \"buf\""));
        assert!(json.contains("\"num_elements\": 3"));
    }

    // -- TraceExporter (Chrome trace) ---------------------------------------

    #[test]
    fn export_chrome_trace_empty() {
        let s = TraceSession::new("e", TraceLevel::Summary);
        let ct = TraceExporter::to_chrome_trace(&s);
        assert_eq!(ct, "[]");
    }

    #[test]
    fn export_chrome_trace_events() {
        let mut s = TraceSession::new("c", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("a", vec![1], vec![], 100));
        s.record_trace(KernelTrace::new("b", vec![1], vec![], 200));
        let ct = TraceExporter::to_chrome_trace(&s);
        assert!(ct.contains("\"name\":\"a\""));
        assert!(ct.contains("\"name\":\"b\""));
        assert!(ct.contains("\"ph\":\"X\""));
        assert!(ct.contains("\"cat\":\"kernel\""));
    }

    #[test]
    fn export_chrome_trace_timestamps_sequential() {
        let mut s = TraceSession::new("c", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("a", vec![1], vec![], 100));
        s.record_trace(KernelTrace::new("b", vec![1], vec![], 200));
        let ct = TraceExporter::to_chrome_trace(&s);
        // First event at ts=0, second at ts=100
        assert!(ct.contains("\"ts\":0"));
        assert!(ct.contains("\"ts\":100"));
    }

    #[test]
    fn export_chrome_trace_is_valid_json_array() {
        let mut s = TraceSession::new("c", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("k", vec![1], vec![], 50));
        let ct = TraceExporter::to_chrome_trace(&s);
        assert!(ct.starts_with('['));
        assert!(ct.ends_with(']'));
        // Should parse as valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&ct).unwrap();
        assert!(parsed.is_array());
    }

    // -- DivergenceDetector -------------------------------------------------

    #[test]
    fn divergence_none_for_identical() {
        let mut a = TraceSession::new("a", TraceLevel::Full);
        let filter = TraceFilter::capture_all();
        a.filter = Some(filter);
        a.record_trace(KernelTrace::new("k", vec![1], vec![], 10));
        a.record_snapshot(BufferSnapshot::from_f32("b", vec![2], &[1.0, 2.0]));

        let mut b = TraceSession::new("b", TraceLevel::Full);
        let filter2 = TraceFilter::capture_all();
        b.filter = Some(filter2);
        b.record_trace(KernelTrace::new("k", vec![1], vec![], 10));
        b.record_snapshot(BufferSnapshot::from_f32("b", vec![2], &[1.0, 2.0]));

        assert_eq!(DivergenceDetector::compare(&a, &b), Divergence::None);
    }

    #[test]
    fn divergence_trace_count_mismatch() {
        let mut a = TraceSession::new("a", TraceLevel::Summary);
        a.record_trace(KernelTrace::new("k", vec![1], vec![], 10));

        let b = TraceSession::new("b", TraceLevel::Summary);

        let d = DivergenceDetector::compare(&a, &b);
        assert_eq!(d, Divergence::TraceCountMismatch { expected: 1, actual: 0 });
    }

    #[test]
    fn divergence_kernel_name_mismatch() {
        let mut a = TraceSession::new("a", TraceLevel::Summary);
        a.record_trace(KernelTrace::new("matmul", vec![1], vec![], 10));

        let mut b = TraceSession::new("b", TraceLevel::Summary);
        b.record_trace(KernelTrace::new("softmax", vec![1], vec![], 10));

        match DivergenceDetector::compare(&a, &b) {
            Divergence::KernelNameMismatch { index, expected, actual } => {
                assert_eq!(index, 0);
                assert_eq!(expected, "matmul");
                assert_eq!(actual, "softmax");
            }
            other => panic!("expected KernelNameMismatch, got {other:?}"),
        }
    }

    #[test]
    fn divergence_checksum_mismatch() {
        let mut a = TraceSession::new("a", TraceLevel::Full);
        let fa = TraceFilter::capture_all();
        a.filter = Some(fa);
        a.record_snapshot(BufferSnapshot::from_f32("w", vec![2], &[1.0, 2.0]));

        let mut b = TraceSession::new("b", TraceLevel::Full);
        let fb = TraceFilter::capture_all();
        b.filter = Some(fb);
        b.record_snapshot(BufferSnapshot::from_f32("w", vec![2], &[1.0, 3.0]));

        match DivergenceDetector::compare(&a, &b) {
            Divergence::ChecksumMismatch { index, name, .. } => {
                assert_eq!(index, 0);
                assert_eq!(name, "w");
            }
            other => panic!("expected ChecksumMismatch, got {other:?}"),
        }
    }

    #[test]
    fn divergence_snapshot_count_mismatch() {
        let mut a = TraceSession::new("a", TraceLevel::Full);
        let fa = TraceFilter::capture_all();
        a.filter = Some(fa);
        a.record_snapshot(BufferSnapshot::from_f32("x", vec![1], &[1.0]));

        let b = TraceSession::new("b", TraceLevel::Full);
        // no snapshots

        let d = DivergenceDetector::compare(&a, &b);
        assert_eq!(d, Divergence::SnapshotCountMismatch { expected: 1, actual: 0 });
    }

    #[test]
    fn divergence_display() {
        let d = Divergence::TraceCountMismatch { expected: 5, actual: 3 };
        let s = d.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    #[test]
    fn divergence_compare_all_multiple() {
        let mut a = TraceSession::new("a", TraceLevel::Summary);
        a.record_trace(KernelTrace::new("k1", vec![1], vec![], 10));
        a.record_trace(KernelTrace::new("k2", vec![1], vec![], 20));

        let mut b = TraceSession::new("b", TraceLevel::Summary);
        b.record_trace(KernelTrace::new("k1", vec![1], vec![], 10));
        b.record_trace(KernelTrace::new("WRONG", vec![1], vec![], 20));

        let divs = DivergenceDetector::compare_all(&a, &b);
        assert_eq!(divs.len(), 1);
        match &divs[0] {
            Divergence::KernelNameMismatch { index, .. } => assert_eq!(*index, 1),
            other => panic!("expected KernelNameMismatch, got {other:?}"),
        }
    }

    #[test]
    fn divergence_compare_empty_sessions() {
        let a = TraceSession::new("a", TraceLevel::Summary);
        let b = TraceSession::new("b", TraceLevel::Summary);
        assert_eq!(DivergenceDetector::compare(&a, &b), Divergence::None);
    }

    #[test]
    fn divergence_none_display() {
        assert_eq!(Divergence::None.to_string(), "no divergence");
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn single_trace_session() {
        let mut s = TraceSession::new("one", TraceLevel::Summary);
        s.record_trace(KernelTrace::new("only", vec![1], vec![], 42));
        assert_eq!(s.trace_count(), 1);
        assert_eq!(s.total_duration_us(), 42);
        assert_eq!(s.hotspots().len(), 1);
    }

    #[test]
    fn huge_buffer_snapshot() {
        let data: Vec<f32> = (0..100_000).map(|i| i as f32).collect();
        let snap = BufferSnapshot::from_f32("big", vec![100_000], &data);
        assert_eq!(snap.num_elements, 100_000);
        assert_eq!(snap.head_values.len(), 8);
        assert_eq!(snap.tail_values.len(), 8);
        assert!(snap.is_clean());
    }

    #[test]
    fn session_elapsed_is_non_negative() {
        let s = TraceSession::new("t", TraceLevel::Off);
        // elapsed() should not panic and should return >= 0
        let _ = s.elapsed();
    }

    // -- Property-style tests -----------------------------------------------

    #[test]
    fn checksum_determinism_property() {
        // Repeated checksums on the same data must be identical.
        for size in [0, 1, 7, 64, 1000] {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.123).collect();
            let c1 = compute_checksum_f32(&data);
            let c2 = compute_checksum_f32(&data);
            assert_eq!(c1, c2, "checksum not deterministic for size {size}");
        }
    }

    #[test]
    fn checksum_sensitivity_property() {
        // Flipping a single element should change the checksum.
        let mut data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let original = compute_checksum_f32(&data);
        data[50] += 0.001;
        let modified = compute_checksum_f32(&data);
        assert_ne!(original, modified);
    }

    #[test]
    fn snapshot_element_count_matches_shape() {
        let data: Vec<f32> = vec![0.0; 24];
        let snap = BufferSnapshot::from_f32("t", vec![2, 3, 4], &data);
        // num_elements == data.len(), shape is metadata only
        assert_eq!(snap.num_elements, 24);
    }
}

//! Rich GPU error diagnostics for debugging OpenCL kernel failures.
//!
//! Provides [`GpuDiagnosticReport`] which collects full context when a GPU
//! operation fails: device info, driver version, kernel source, arguments,
//! buffer sizes, and a recent event log of the last GPU operations.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Maximum number of recent events retained in the event log.
const DEFAULT_EVENT_LOG_CAPACITY: usize = 10;

/// A single GPU operation recorded in the event log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuEvent {
    /// Sequential event index.
    pub index: u64,
    /// Timestamp when the event occurred.
    pub timestamp: SystemTime,
    /// Operation type (e.g. `"kernel_dispatch"`, `"buffer_write"`, `"buffer_read"`).
    pub operation: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this operation succeeded.
    pub success: bool,
    /// Wall-clock duration of the operation, if measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<Duration>,
}

/// Information about a kernel argument at the time of failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelArgInfo {
    /// Argument index (0-based).
    pub index: usize,
    /// Argument name if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Argument type description (e.g. `"buffer<f32>"`, `"u32"`).
    pub type_desc: String,
    /// Buffer size in bytes (for buffer arguments).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_size: Option<usize>,
}

/// GPU device information captured at error time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device name (e.g. `"Intel Arc A770"`).
    pub name: String,
    /// Device vendor.
    pub vendor: String,
    /// Driver version string.
    pub driver_version: String,
    /// OpenCL version string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opencl_version: Option<String>,
    /// Global memory size in bytes.
    pub global_memory_bytes: u64,
    /// Maximum work group size.
    pub max_work_group_size: usize,
    /// Number of compute units.
    pub compute_units: usize,
}

/// Full diagnostic report collected when a GPU error occurs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDiagnosticReport {
    /// The error message that triggered this report.
    pub error_message: String,
    /// Device information.
    pub device_info: GpuDeviceInfo,
    /// Kernel that failed (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_name: Option<String>,
    /// Kernel source snippet (first 2048 chars).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_source: Option<String>,
    /// Kernel arguments at the time of failure.
    pub kernel_args: Vec<KernelArgInfo>,
    /// Global work size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_work_size: Option<Vec<usize>>,
    /// Local work size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_work_size: Option<Vec<usize>>,
    /// Recent GPU event log (up to last 10 operations before the error).
    pub recent_events: Vec<GpuEvent>,
    /// Timestamp when this report was generated.
    pub generated_at: SystemTime,
}

impl GpuDiagnosticReport {
    /// Export this report as a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export this report as a compact JSON string.
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize a report from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for GpuDiagnosticReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== GPU Diagnostic Report ===")?;
        writeln!(f, "Error: {}", self.error_message)?;
        writeln!(f)?;
        writeln!(f, "--- Device ---")?;
        writeln!(f, "  Name:            {}", self.device_info.name)?;
        writeln!(f, "  Vendor:          {}", self.device_info.vendor)?;
        writeln!(f, "  Driver:          {}", self.device_info.driver_version)?;
        if let Some(ref ocl) = self.device_info.opencl_version {
            writeln!(f, "  OpenCL:          {ocl}")?;
        }
        writeln!(
            f,
            "  Global Memory:   {} MB",
            self.device_info.global_memory_bytes / (1024 * 1024)
        )?;
        writeln!(
            f,
            "  Compute Units:   {}",
            self.device_info.compute_units
        )?;
        writeln!(
            f,
            "  Max WG Size:     {}",
            self.device_info.max_work_group_size
        )?;

        if let Some(ref name) = self.kernel_name {
            writeln!(f)?;
            writeln!(f, "--- Kernel ---")?;
            writeln!(f, "  Name: {name}")?;
            if let Some(ref gws) = self.global_work_size {
                writeln!(f, "  Global WS: {gws:?}")?;
            }
            if let Some(ref lws) = self.local_work_size {
                writeln!(f, "  Local WS:  {lws:?}")?;
            }
        }

        if !self.kernel_args.is_empty() {
            writeln!(f)?;
            writeln!(f, "--- Kernel Arguments ---")?;
            for arg in &self.kernel_args {
                let name = arg.name.as_deref().unwrap_or("(unnamed)");
                if let Some(sz) = arg.buffer_size {
                    writeln!(
                        f,
                        "  [{:>2}] {name}: {} ({} bytes)",
                        arg.index, arg.type_desc, sz
                    )?;
                } else {
                    writeln!(
                        f,
                        "  [{:>2}] {name}: {}",
                        arg.index, arg.type_desc
                    )?;
                }
            }
        }

        if let Some(ref src) = self.kernel_source {
            writeln!(f)?;
            writeln!(f, "--- Kernel Source (truncated) ---")?;
            for line in src.lines().take(20) {
                writeln!(f, "  {line}")?;
            }
            let total_lines = src.lines().count();
            if total_lines > 20 {
                writeln!(f, "  ... ({} more lines)", total_lines - 20)?;
            }
        }

        if !self.recent_events.is_empty() {
            writeln!(f)?;
            writeln!(f, "--- Recent Events ---")?;
            for evt in &self.recent_events {
                let status = if evt.success { "OK" } else { "FAIL" };
                let dur = evt
                    .duration
                    .map(|d| format!(" ({:.2?})", d))
                    .unwrap_or_default();
                writeln!(
                    f,
                    "  [{}] [{status}] {}: {}{}",
                    evt.index, evt.operation, evt.description, dur
                )?;
            }
        }

        writeln!(f, "=== End Report ===")?;
        Ok(())
    }
}

/// Builder for constructing a [`GpuDiagnosticReport`].
#[derive(Debug)]
pub struct DiagnosticReportBuilder {
    error_message: String,
    device_info: GpuDeviceInfo,
    kernel_name: Option<String>,
    kernel_source: Option<String>,
    kernel_args: Vec<KernelArgInfo>,
    global_work_size: Option<Vec<usize>>,
    local_work_size: Option<Vec<usize>>,
    recent_events: Vec<GpuEvent>,
}

impl DiagnosticReportBuilder {
    /// Start building a report for the given error and device.
    pub fn new(error_message: impl Into<String>, device_info: GpuDeviceInfo) -> Self {
        Self {
            error_message: error_message.into(),
            device_info,
            kernel_name: None,
            kernel_source: None,
            kernel_args: Vec::new(),
            global_work_size: None,
            local_work_size: None,
            recent_events: Vec::new(),
        }
    }

    /// Set the kernel name.
    pub fn kernel_name(mut self, name: impl Into<String>) -> Self {
        self.kernel_name = Some(name.into());
        self
    }

    /// Set the kernel source (truncated to 2048 chars).
    pub fn kernel_source(mut self, source: impl Into<String>) -> Self {
        let src: String = source.into();
        self.kernel_source = Some(if src.len() > 2048 {
            src[..2048].to_string()
        } else {
            src
        });
        self
    }

    /// Add a kernel argument.
    pub fn add_arg(mut self, arg: KernelArgInfo) -> Self {
        self.kernel_args.push(arg);
        self
    }

    /// Set global work size.
    pub fn global_work_size(mut self, gws: Vec<usize>) -> Self {
        self.global_work_size = Some(gws);
        self
    }

    /// Set local work size.
    pub fn local_work_size(mut self, lws: Vec<usize>) -> Self {
        self.local_work_size = Some(lws);
        self
    }

    /// Set recent events from an [`EventLog`].
    pub fn events_from_log(mut self, log: &EventLog) -> Self {
        self.recent_events = log.snapshot();
        self
    }

    /// Consume the builder and produce the report.
    pub fn build(self) -> GpuDiagnosticReport {
        GpuDiagnosticReport {
            error_message: self.error_message,
            device_info: self.device_info,
            kernel_name: self.kernel_name,
            kernel_source: self.kernel_source,
            kernel_args: self.kernel_args,
            global_work_size: self.global_work_size,
            local_work_size: self.local_work_size,
            recent_events: self.recent_events,
            generated_at: SystemTime::now(),
        }
    }
}

/// Thread-safe circular event log that retains the last N GPU operations.
#[derive(Debug, Clone)]
pub struct EventLog {
    inner: Arc<Mutex<EventLogInner>>,
}

#[derive(Debug)]
struct EventLogInner {
    events: VecDeque<GpuEvent>,
    capacity: usize,
    next_index: u64,
}

impl EventLog {
    /// Create an event log with the default capacity (10).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_EVENT_LOG_CAPACITY)
    }

    /// Create an event log with a custom capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EventLogInner {
                events: VecDeque::with_capacity(capacity),
                capacity,
                next_index: 0,
            })),
        }
    }

    /// Record a GPU event.
    pub fn record(
        &self,
        operation: impl Into<String>,
        description: impl Into<String>,
        success: bool,
        duration: Option<Duration>,
    ) {
        let mut inner = self.inner.lock().unwrap();
        let event = GpuEvent {
            index: inner.next_index,
            timestamp: SystemTime::now(),
            operation: operation.into(),
            description: description.into(),
            success,
            duration,
        };
        inner.next_index += 1;
        if inner.events.len() >= inner.capacity {
            inner.events.pop_front();
        }
        inner.events.push_back(event);
    }

    /// Number of events currently in the log.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a snapshot of all events in order.
    pub fn snapshot(&self) -> Vec<GpuEvent> {
        self.inner.lock().unwrap().events.iter().cloned().collect()
    }

    /// Clear all events.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.events.clear();
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> GpuDeviceInfo {
        GpuDeviceInfo {
            name: "Intel Arc A770".to_string(),
            vendor: "Intel".to_string(),
            driver_version: "23.43.27642.40".to_string(),
            opencl_version: Some("OpenCL 3.0".to_string()),
            global_memory_bytes: 16 * 1024 * 1024 * 1024,
            max_work_group_size: 1024,
            compute_units: 512,
        }
    }

    #[test]
    fn test_diagnostic_report_builder() {
        let report =
            DiagnosticReportBuilder::new("CL_OUT_OF_RESOURCES", test_device())
                .kernel_name("bitnet_matmul_i2s")
                .global_work_size(vec![1024, 1024])
                .local_work_size(vec![16, 16])
                .build();

        assert_eq!(report.error_message, "CL_OUT_OF_RESOURCES");
        assert_eq!(
            report.kernel_name.as_deref(),
            Some("bitnet_matmul_i2s")
        );
        assert_eq!(report.global_work_size, Some(vec![1024, 1024]));
        assert_eq!(report.local_work_size, Some(vec![16, 16]));
        assert_eq!(report.device_info.name, "Intel Arc A770");
    }

    #[test]
    fn test_diagnostic_report_json_roundtrip() {
        let report = DiagnosticReportBuilder::new(
            "CL_MEM_OBJECT_ALLOCATION_FAILURE",
            test_device(),
        )
        .kernel_name("layernorm_fp32")
        .add_arg(KernelArgInfo {
            index: 0,
            name: Some("input".to_string()),
            type_desc: "buffer<f32>".to_string(),
            buffer_size: Some(4096),
        })
        .build();

        let json = report.to_json().unwrap();
        let restored = GpuDiagnosticReport::from_json(&json).unwrap();

        assert_eq!(restored.error_message, report.error_message);
        assert_eq!(restored.kernel_name, report.kernel_name);
        assert_eq!(restored.kernel_args.len(), 1);
        assert_eq!(restored.kernel_args[0].buffer_size, Some(4096));
    }

    #[test]
    fn test_diagnostic_report_display_formatting() {
        let report =
            DiagnosticReportBuilder::new("CL_OUT_OF_RESOURCES", test_device())
                .kernel_name("matmul")
                .add_arg(KernelArgInfo {
                    index: 0,
                    name: Some("A".to_string()),
                    type_desc: "buffer<f32>".to_string(),
                    buffer_size: Some(8192),
                })
                .build();

        let display = format!("{report}");
        assert!(display.contains("GPU Diagnostic Report"));
        assert!(display.contains("CL_OUT_OF_RESOURCES"));
        assert!(display.contains("Intel Arc A770"));
        assert!(display.contains("matmul"));
        assert!(display.contains("8192 bytes"));
    }

    #[test]
    fn test_event_log_capacity() {
        let log = EventLog::with_capacity(3);
        for i in 0..5 {
            log.record("op", format!("event {i}"), true, None);
        }

        assert_eq!(log.len(), 3);
        let events = log.snapshot();
        assert_eq!(events[0].description, "event 2");
        assert_eq!(events[2].description, "event 4");
    }

    #[test]
    fn test_event_log_indices_monotonic() {
        let log = EventLog::new();
        log.record("a", "first", true, None);
        log.record("b", "second", false, None);
        log.record(
            "c",
            "third",
            true,
            Some(Duration::from_millis(5)),
        );

        let events = log.snapshot();
        assert_eq!(events[0].index, 0);
        assert_eq!(events[1].index, 1);
        assert_eq!(events[2].index, 2);
        assert!(!events[1].success);
        assert_eq!(
            events[2].duration,
            Some(Duration::from_millis(5))
        );
    }

    #[test]
    fn test_report_with_events() {
        let log = EventLog::new();
        log.record(
            "kernel_dispatch",
            "matmul enqueued",
            true,
            Some(Duration::from_millis(10)),
        );
        log.record("buffer_read", "read output", false, None);

        let report = DiagnosticReportBuilder::new(
            "CL_INVALID_KERNEL_ARGS",
            test_device(),
        )
        .events_from_log(&log)
        .build();

        assert_eq!(report.recent_events.len(), 2);
        assert!(report.recent_events[0].success);
        assert!(!report.recent_events[1].success);
    }

    #[test]
    fn test_kernel_source_truncation() {
        let long_source = "x".repeat(5000);
        let report = DiagnosticReportBuilder::new("err", test_device())
            .kernel_source(long_source)
            .build();

        assert_eq!(
            report.kernel_source.as_ref().unwrap().len(),
            2048
        );
    }

    #[test]
    fn test_event_log_thread_safe() {
        let log = EventLog::new();
        let log2 = log.clone();

        log.record("op1", "from log1", true, None);
        log2.record("op2", "from log2", true, None);

        assert_eq!(log.len(), 2);
        assert_eq!(log2.len(), 2);
    }

    #[test]
    fn test_event_log_clear() {
        let log = EventLog::new();
        log.record("op", "test", true, None);
        assert_eq!(log.len(), 1);

        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_report_compact_json() {
        let report =
            DiagnosticReportBuilder::new("err", test_device()).build();
        let compact = report.to_json_compact().unwrap();
        let pretty = report.to_json().unwrap();

        assert!(compact.len() < pretty.len());
        let r1 = GpuDiagnosticReport::from_json(&compact).unwrap();
        let r2 = GpuDiagnosticReport::from_json(&pretty).unwrap();
        assert_eq!(r1.error_message, r2.error_message);
    }
}

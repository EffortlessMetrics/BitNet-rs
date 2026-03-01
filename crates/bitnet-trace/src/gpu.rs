//! GPU kernel tracing for cross-validation against reference implementations.
//!
//! Provides [`GpuTracer`] which records tensor activations from GPU kernels,
//! capturing kernel name, input/output shapes, execution time, and device info.
//! Output is compatible with the existing [`TraceRecord`](crate::TraceRecord)
//! format used by the crossval comparison framework.

use crate::TraceRecord;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Metadata for a single GPU kernel execution captured during tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuKernelTrace {
    /// Kernel function name (e.g. `"bitnet_matmul_i2s"`)
    pub kernel_name: String,
    /// Device identifier (e.g. `"Intel Arc A770"`, `"cuda:0"`)
    pub device: String,
    /// Shapes of input tensors
    pub input_shapes: Vec<Vec<usize>>,
    /// Shape of the output tensor
    pub output_shape: Vec<usize>,
    /// Wall-clock execution time of the kernel
    pub execution_time: Duration,
    /// Underlying trace record (hash, RMS, etc.) for crossval comparison
    pub trace_record: TraceRecord,
}

/// Configuration for [`GpuTracer`].
#[derive(Debug, Clone)]
pub struct GpuTracerConfig {
    /// Maximum number of traces to retain in memory (0 = unlimited).
    pub max_traces: usize,
    /// Whether to capture execution timing.
    pub capture_timing: bool,
    /// Device name to tag on all traces.
    pub device_name: String,
}

impl Default for GpuTracerConfig {
    fn default() -> Self {
        Self {
            max_traces: 0,
            capture_timing: true,
            device_name: "gpu:0".to_string(),
        }
    }
}

/// Collects GPU kernel traces in memory for later export.
///
/// Thread-safe via internal `Arc<Mutex<…>>` so it can be shared across
/// kernel dispatch threads.
#[derive(Debug, Clone)]
pub struct GpuTracer {
    inner: Arc<Mutex<GpuTracerInner>>,
    config: GpuTracerConfig,
}

#[derive(Debug)]
struct GpuTracerInner {
    traces: Vec<GpuKernelTrace>,
    enabled: bool,
}

impl GpuTracer {
    /// Create a new tracer with the given configuration.
    pub fn new(config: GpuTracerConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(GpuTracerInner {
                traces: Vec::new(),
                enabled: true,
            })),
            config,
        }
    }

    /// Create a tracer with default configuration.
    pub fn default_tracer() -> Self {
        Self::new(GpuTracerConfig::default())
    }

    /// Enable or disable trace collection.
    pub fn set_enabled(&self, enabled: bool) {
        self.inner.lock().unwrap().enabled = enabled;
    }

    /// Returns `true` if the tracer is currently collecting traces.
    pub fn is_enabled(&self) -> bool {
        self.inner.lock().unwrap().enabled
    }

    /// Record a GPU kernel execution.
    ///
    /// `output_data` is the raw f32 output used to compute blake3 hash and RMS
    /// for crossval comparison. If timing is enabled, `execution_time` is used
    /// as-is; callers should measure wall-clock time around the kernel dispatch.
    pub fn record_kernel(
        &self,
        kernel_name: &str,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        output_data: &[f32],
        execution_time: Duration,
    ) {
        let mut inner = self.inner.lock().unwrap();
        if !inner.enabled {
            return;
        }

        let num_elements = output_shape.iter().product();
        let blake3_hash = {
            let bytes: Vec<u8> = output_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            blake3::hash(&bytes).to_hex().to_string()
        };
        let rms = compute_rms(output_data);

        let trace_record = TraceRecord {
            name: kernel_name.to_string(),
            shape: output_shape.clone(),
            dtype: "F32".to_string(),
            blake3: blake3_hash,
            rms,
            num_elements,
            seq: None,
            layer: None,
            stage: Some("gpu_kernel".to_string()),
        };

        let kernel_trace = GpuKernelTrace {
            kernel_name: kernel_name.to_string(),
            device: self.config.device_name.clone(),
            input_shapes,
            output_shape,
            execution_time: if self.config.capture_timing {
                execution_time
            } else {
                Duration::ZERO
            },
            trace_record,
        };

        if self.config.max_traces > 0 && inner.traces.len() >= self.config.max_traces {
            inner.traces.remove(0);
        }
        inner.traces.push(kernel_trace);
    }

    /// Start a timing scope; returns an [`Instant`] for use with [`record_kernel`](Self::record_kernel).
    pub fn start_timer(&self) -> Instant {
        Instant::now()
    }

    /// Number of traces currently held.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().traces.len()
    }

    /// Whether the tracer holds no traces.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a snapshot of all collected traces.
    pub fn snapshot(&self) -> Vec<GpuKernelTrace> {
        self.inner.lock().unwrap().traces.clone()
    }

    /// Extract all traces, leaving the tracer empty.
    pub fn drain(&self) -> Vec<GpuKernelTrace> {
        let mut inner = self.inner.lock().unwrap();
        std::mem::take(&mut inner.traces)
    }

    /// Clear all collected traces.
    pub fn clear(&self) {
        self.inner.lock().unwrap().traces.clear();
    }

    /// Export traces as [`TraceRecord`]s for crossval comparison.
    pub fn to_trace_records(&self) -> Vec<TraceRecord> {
        self.inner
            .lock()
            .unwrap()
            .traces
            .iter()
            .map(|t| t.trace_record.clone())
            .collect()
    }

    /// Serialize all traces to a JSON string compatible with crossval import.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let traces = self.snapshot();
        serde_json::to_string_pretty(&traces)
    }

    /// Filter traces by kernel name substring.
    pub fn filter_by_kernel(&self, substr: &str) -> Vec<GpuKernelTrace> {
        self.inner
            .lock()
            .unwrap()
            .traces
            .iter()
            .filter(|t| t.kernel_name.contains(substr))
            .cloned()
            .collect()
    }

    /// Total GPU execution time across all captured traces.
    pub fn total_execution_time(&self) -> Duration {
        self.inner
            .lock()
            .unwrap()
            .traces
            .iter()
            .map(|t| t.execution_time)
            .sum()
    }
}

/// Compute root-mean-square of a float slice.
fn compute_rms(data: &[f32]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = data.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum_sq / data.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.1).collect()
    }

    #[test]
    fn test_gpu_tracer_record_and_snapshot() {
        let tracer = GpuTracer::default_tracer();
        tracer.record_kernel(
            "matmul_i2s",
            vec![vec![4, 8], vec![8, 16]],
            vec![4, 16],
            &sample_data(64),
            Duration::from_millis(5),
        );

        assert_eq!(tracer.len(), 1);
        let snap = tracer.snapshot();
        assert_eq!(snap[0].kernel_name, "matmul_i2s");
        assert_eq!(snap[0].output_shape, vec![4, 16]);
        assert_eq!(snap[0].device, "gpu:0");
    }

    #[test]
    fn test_gpu_tracer_disabled() {
        let tracer = GpuTracer::default_tracer();
        tracer.set_enabled(false);
        tracer.record_kernel("k", vec![], vec![2], &[1.0, 2.0], Duration::from_millis(1));

        assert!(tracer.is_empty());
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_gpu_tracer_max_traces() {
        let config = GpuTracerConfig {
            max_traces: 2,
            ..Default::default()
        };
        let tracer = GpuTracer::new(config);
        for i in 0..5 {
            tracer.record_kernel(
                &format!("kernel_{i}"),
                vec![],
                vec![1],
                &[i as f32],
                Duration::from_millis(1),
            );
        }
        assert_eq!(tracer.len(), 2);
        let snap = tracer.snapshot();
        assert_eq!(snap[0].kernel_name, "kernel_3");
        assert_eq!(snap[1].kernel_name, "kernel_4");
    }

    #[test]
    fn test_gpu_tracer_drain() {
        let tracer = GpuTracer::default_tracer();
        tracer.record_kernel("k1", vec![], vec![1], &[1.0], Duration::from_millis(1));
        tracer.record_kernel("k2", vec![], vec![1], &[2.0], Duration::from_millis(2));

        let drained = tracer.drain();
        assert_eq!(drained.len(), 2);
        assert!(tracer.is_empty());
    }

    #[test]
    fn test_gpu_tracer_crossval_export() {
        let tracer = GpuTracer::default_tracer();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        tracer.record_kernel(
            "layernorm",
            vec![vec![1, 4]],
            vec![1, 4],
            &data,
            Duration::from_millis(3),
        );

        let records = tracer.to_trace_records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, "layernorm");
        assert_eq!(records[0].shape, vec![1, 4]);
        assert_eq!(records[0].dtype, "F32");
        assert!(!records[0].blake3.is_empty());
        assert!(records[0].rms > 0.0);
        assert_eq!(records[0].num_elements, 4);
        assert_eq!(records[0].stage.as_deref(), Some("gpu_kernel"));
    }

    #[test]
    fn test_gpu_tracer_json_export() {
        let tracer = GpuTracer::default_tracer();
        tracer.record_kernel("k", vec![], vec![2], &[1.0, 2.0], Duration::from_millis(1));

        let json = tracer.export_json().unwrap();
        let parsed: Vec<GpuKernelTrace> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].kernel_name, "k");
    }

    #[test]
    fn test_gpu_tracer_filter_by_kernel() {
        let tracer = GpuTracer::default_tracer();
        tracer.record_kernel("matmul_i2s", vec![], vec![1], &[1.0], Duration::from_millis(1));
        tracer.record_kernel("layernorm", vec![], vec![1], &[2.0], Duration::from_millis(1));
        tracer.record_kernel("matmul_qk256", vec![], vec![1], &[3.0], Duration::from_millis(1));

        let matmuls = tracer.filter_by_kernel("matmul");
        assert_eq!(matmuls.len(), 2);
    }

    #[test]
    fn test_gpu_tracer_total_execution_time() {
        let tracer = GpuTracer::default_tracer();
        tracer.record_kernel("k1", vec![], vec![1], &[1.0], Duration::from_millis(10));
        tracer.record_kernel("k2", vec![], vec![1], &[2.0], Duration::from_millis(20));

        assert_eq!(tracer.total_execution_time(), Duration::from_millis(30));
    }

    #[test]
    fn test_gpu_tracer_timing_disabled() {
        let config = GpuTracerConfig {
            capture_timing: false,
            ..Default::default()
        };
        let tracer = GpuTracer::new(config);
        tracer.record_kernel("k", vec![], vec![1], &[1.0], Duration::from_millis(999));

        let snap = tracer.snapshot();
        assert_eq!(snap[0].execution_time, Duration::ZERO);
    }

    #[test]
    fn test_gpu_tracer_thread_safe_clone() {
        let tracer = GpuTracer::default_tracer();
        let t2 = tracer.clone();

        tracer.record_kernel("from_t1", vec![], vec![1], &[1.0], Duration::from_millis(1));
        t2.record_kernel("from_t2", vec![], vec![1], &[2.0], Duration::from_millis(1));

        // Both clones share the same inner state
        assert_eq!(tracer.len(), 2);
        assert_eq!(t2.len(), 2);
    }

    #[test]
    fn test_compute_rms() {
        assert_eq!(compute_rms(&[]), 0.0);
        let rms = compute_rms(&[3.0, 4.0]);
        let expected = ((9.0 + 16.0) / 2.0_f64).sqrt();
        assert!((rms - expected).abs() < 1e-10);
    }
}

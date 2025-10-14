//! # Kernel Recorder
//!
//! Thread-safe kernel execution recorder for inference receipts.
//! Records kernel IDs during inference to provide honest compute evidence.

use std::sync::{Arc, Mutex};

/// Thread-safe kernel recorder that tracks executed kernel IDs
///
/// Used to record which compute kernels were actually executed during inference,
/// providing verifiable evidence for honest compute gates in CI/CD.
#[derive(Debug, Clone)]
pub struct KernelRecorder {
    inner: Arc<Mutex<Vec<String>>>,
}

impl KernelRecorder {
    /// Create a new kernel recorder
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(Vec::new())) }
    }

    /// Record execution of a kernel with the given ID
    ///
    /// Thread-safe: can be called concurrently from multiple threads.
    /// IDs should be static strings like "i2s_gemv", "gemm_fp16", etc.
    pub fn record(&self, id: &'static str) {
        if let Ok(mut kernels) = self.inner.lock() {
            kernels.push(id.to_string());
        }
    }

    /// Get a snapshot of all recorded kernel IDs
    ///
    /// Returns a deduplicated, sorted list of kernel IDs in the order they were first executed.
    pub fn snapshot(&self) -> Vec<String> {
        if let Ok(kernels) = self.inner.lock() {
            // Keep insertion order but deduplicate
            let mut seen = std::collections::HashSet::new();
            kernels.iter().filter(|k| seen.insert(k.as_str())).cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get the number of kernel executions recorded (including duplicates)
    pub fn count(&self) -> usize {
        self.inner.lock().map(|k| k.len()).unwrap_or(0)
    }

    /// Clear all recorded kernels
    pub fn clear(&self) {
        if let Ok(mut kernels) = self.inner.lock() {
            kernels.clear();
        }
    }
}

impl Default for KernelRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_recorder_basic() {
        let recorder = KernelRecorder::new();

        recorder.record("i2s_gemv");
        recorder.record("rope_apply");
        recorder.record("attention_real");

        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.len(), 3);
        assert!(snapshot.contains(&"i2s_gemv".to_string()));
        assert!(snapshot.contains(&"rope_apply".to_string()));
        assert!(snapshot.contains(&"attention_real".to_string()));
    }

    #[test]
    fn test_kernel_recorder_deduplication() {
        let recorder = KernelRecorder::new();

        recorder.record("i2s_gemv");
        recorder.record("rope_apply");
        recorder.record("i2s_gemv"); // duplicate
        recorder.record("attention_real");
        recorder.record("rope_apply"); // duplicate

        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.len(), 3); // deduplicated
        assert_eq!(snapshot[0], "i2s_gemv"); // first occurrence
        assert_eq!(snapshot[1], "rope_apply");
        assert_eq!(snapshot[2], "attention_real");
    }

    #[test]
    fn test_kernel_recorder_count() {
        let recorder = KernelRecorder::new();

        recorder.record("i2s_gemv");
        recorder.record("rope_apply");
        recorder.record("i2s_gemv");

        assert_eq!(recorder.count(), 3); // includes duplicates
        assert_eq!(recorder.snapshot().len(), 2); // deduplicated
    }

    #[test]
    fn test_kernel_recorder_clear() {
        let recorder = KernelRecorder::new();

        recorder.record("i2s_gemv");
        recorder.record("rope_apply");

        assert_eq!(recorder.count(), 2);

        recorder.clear();

        assert_eq!(recorder.count(), 0);
        assert_eq!(recorder.snapshot().len(), 0);
    }

    #[test]
    fn test_kernel_recorder_thread_safe() {
        use std::thread;

        let recorder = KernelRecorder::new();
        let mut handles = vec![];

        // Spawn 10 threads that each record the same kernels
        for _ in 0..10 {
            let recorder_clone = recorder.clone();
            let handle = thread::spawn(move || {
                recorder_clone.record("test_kernel");
                recorder_clone.record("another_kernel");
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Should have recorded from all threads
        assert_eq!(recorder.count(), 20); // 10 threads * 2 records each

        let snapshot = recorder.snapshot();
        assert!(snapshot.contains(&"test_kernel".to_string()));
        assert!(snapshot.contains(&"another_kernel".to_string()));
        // Should have 2 unique kernels
        assert_eq!(snapshot.len(), 2);
    }
}

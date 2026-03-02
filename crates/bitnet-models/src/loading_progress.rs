//! Model loading progress tracking.
//!
//! Reports progress during multi-shard model loading.

use std::time::Instant;

/// Progress event types during model loading.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadEvent {
    /// Starting to load a file/shard.
    ShardStart { index: usize, total: usize, name: String },
    /// Finished loading a shard.
    ShardDone { index: usize, total: usize, bytes: u64 },
    /// Starting to load a tensor.
    TensorStart { name: String, elements: usize },
    /// Finished loading a tensor.
    TensorDone { name: String, elements: usize },
    /// Conversion step (e.g., BF16→F32).
    Conversion { from_dtype: String, to_dtype: String, elements: usize },
    /// Loading complete.
    Complete { total_tensors: usize, total_bytes: u64, elapsed_ms: u64 },
    /// Error during loading.
    Error { message: String },
}

/// Tracks loading progress and computes metrics.
#[derive(Debug)]
pub struct LoadingProgress {
    start: Instant,
    total_shards: usize,
    shards_done: usize,
    total_tensors: usize,
    tensors_done: usize,
    #[allow(dead_code)]
    total_bytes: u64,
    bytes_done: u64,
    events: Vec<LoadEvent>,
}

impl LoadingProgress {
    pub fn new(total_shards: usize) -> Self {
        Self {
            start: Instant::now(),
            total_shards,
            shards_done: 0,
            total_tensors: 0,
            tensors_done: 0,
            total_bytes: 0,
            bytes_done: 0,
            events: Vec::new(),
        }
    }

    pub fn shard_start(&mut self, index: usize, name: &str) {
        self.events.push(LoadEvent::ShardStart {
            index,
            total: self.total_shards,
            name: name.to_string(),
        });
    }

    pub fn shard_done(&mut self, index: usize, bytes: u64) {
        self.shards_done += 1;
        self.bytes_done += bytes;
        self.events.push(LoadEvent::ShardDone { index, total: self.total_shards, bytes });
    }

    pub fn tensor_start(&mut self, name: &str, elements: usize) {
        self.total_tensors += 1;
        self.events.push(LoadEvent::TensorStart { name: name.to_string(), elements });
    }

    pub fn tensor_done(&mut self, name: &str, elements: usize) {
        self.tensors_done += 1;
        self.events.push(LoadEvent::TensorDone { name: name.to_string(), elements });
    }

    pub fn conversion(&mut self, from_dtype: &str, to_dtype: &str, elements: usize) {
        self.events.push(LoadEvent::Conversion {
            from_dtype: from_dtype.to_string(),
            to_dtype: to_dtype.to_string(),
            elements,
        });
    }

    pub fn error(&mut self, message: &str) {
        self.events.push(LoadEvent::Error { message: message.to_string() });
    }

    pub fn complete(&mut self) -> LoadEvent {
        let elapsed = self.start.elapsed().as_millis() as u64;
        let evt = LoadEvent::Complete {
            total_tensors: self.tensors_done,
            total_bytes: self.bytes_done,
            elapsed_ms: elapsed,
        };
        self.events.push(evt.clone());
        evt
    }

    /// Fraction of shards loaded (0.0 to 1.0).
    pub fn shard_progress(&self) -> f32 {
        if self.total_shards == 0 {
            return 1.0;
        }
        self.shards_done as f32 / self.total_shards as f32
    }

    /// Fraction of tensors loaded (0.0 to 1.0).
    pub fn tensor_progress(&self) -> f32 {
        if self.total_tensors == 0 {
            return 0.0;
        }
        self.tensors_done as f32 / self.total_tensors as f32
    }

    /// Bytes loaded per second.
    pub fn throughput_bytes_per_sec(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed < 1e-9 {
            return 0.0;
        }
        self.bytes_done as f64 / elapsed
    }

    pub fn shards_done(&self) -> usize {
        self.shards_done
    }
    pub fn tensors_done(&self) -> usize {
        self.tensors_done
    }
    pub fn bytes_done(&self) -> u64 {
        self.bytes_done
    }
    pub fn events(&self) -> &[LoadEvent] {
        &self.events
    }
    pub fn has_errors(&self) -> bool {
        self.events.iter().any(|e| matches!(e, LoadEvent::Error { .. }))
    }
}

/// Summary of a completed model load.
#[derive(Debug, Clone)]
pub struct LoadSummary {
    pub shards_loaded: usize,
    pub tensors_loaded: usize,
    pub total_bytes: u64,
    pub elapsed_ms: u64,
    pub throughput_mb_per_sec: f64,
    pub had_errors: bool,
}

impl From<&LoadingProgress> for LoadSummary {
    fn from(p: &LoadingProgress) -> Self {
        let elapsed = p.start.elapsed().as_millis() as u64;
        let tp = if elapsed > 0 {
            p.bytes_done as f64 / (elapsed as f64 / 1000.0) / 1_048_576.0
        } else {
            0.0
        };
        Self {
            shards_loaded: p.shards_done,
            tensors_loaded: p.tensors_done,
            total_bytes: p.bytes_done,
            elapsed_ms: elapsed,
            throughput_mb_per_sec: tp,
            had_errors: p.has_errors(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_progress() {
        let p = LoadingProgress::new(6);
        assert_eq!(p.shards_done(), 0);
        assert_eq!(p.tensors_done(), 0);
    }

    #[test]
    fn test_shard_tracking() {
        let mut p = LoadingProgress::new(3);
        p.shard_start(0, "shard-00000.safetensors");
        p.shard_done(0, 5_000_000);
        assert_eq!(p.shards_done(), 1);
        assert!((p.shard_progress() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_tensor_tracking() {
        let mut p = LoadingProgress::new(1);
        p.tensor_start("model.embed.weight", 100_000);
        p.tensor_done("model.embed.weight", 100_000);
        p.tensor_start("model.lm_head.weight", 50_000);
        p.tensor_done("model.lm_head.weight", 50_000);
        assert_eq!(p.tensors_done(), 2);
        assert!((p.tensor_progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_complete_event() {
        let mut p = LoadingProgress::new(1);
        p.shard_done(0, 1_000_000);
        p.tensor_done("x", 100);
        let evt = p.complete();
        match evt {
            LoadEvent::Complete { total_tensors, total_bytes, .. } => {
                assert_eq!(total_tensors, 1);
                assert_eq!(total_bytes, 1_000_000);
            }
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn test_error_tracking() {
        let mut p = LoadingProgress::new(1);
        assert!(!p.has_errors());
        p.error("corrupt header");
        assert!(p.has_errors());
    }

    #[test]
    fn test_events_recorded() {
        let mut p = LoadingProgress::new(2);
        p.shard_start(0, "a.safetensors");
        p.shard_done(0, 100);
        assert_eq!(p.events().len(), 2);
    }

    #[test]
    fn test_conversion_event() {
        let mut p = LoadingProgress::new(1);
        p.conversion("bf16", "f32", 1000);
        assert_eq!(p.events().len(), 1);
        match &p.events()[0] {
            LoadEvent::Conversion { from_dtype, to_dtype, elements } => {
                assert_eq!(from_dtype, "bf16");
                assert_eq!(to_dtype, "f32");
                assert_eq!(*elements, 1000);
            }
            _ => panic!("expected Conversion"),
        }
    }

    #[test]
    fn test_progress_zero_shards() {
        let p = LoadingProgress::new(0);
        assert!((p.shard_progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_zero_tensors() {
        let p = LoadingProgress::new(1);
        assert!((p.tensor_progress() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let mut p = LoadingProgress::new(2);
        p.shard_done(0, 500_000);
        p.shard_done(1, 500_000);
        p.tensor_done("a", 100);
        let s = LoadSummary::from(&p);
        assert_eq!(s.shards_loaded, 2);
        assert_eq!(s.tensors_loaded, 1);
        assert_eq!(s.total_bytes, 1_000_000);
        assert!(!s.had_errors);
    }

    #[test]
    fn test_bytes_done() {
        let mut p = LoadingProgress::new(3);
        p.shard_done(0, 100);
        p.shard_done(1, 200);
        assert_eq!(p.bytes_done(), 300);
    }

    #[test]
    fn test_throughput_no_time() {
        let p = LoadingProgress::new(1);
        let tp = p.throughput_bytes_per_sec();
        assert!(tp >= 0.0);
    }
}

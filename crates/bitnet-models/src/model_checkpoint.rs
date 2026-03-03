//! Model checkpoint management.
//!
//! Track model loading state, progress reporting, and checkpoint
//! metadata for resumable model loading.

use std::time::{Duration, Instant};

/// Loading stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStage {
    NotStarted,
    ReadingHeader,
    LoadingTensors,
    ConvertingDtypes,
    ValidatingWeights,
    BuildingModel,
    Complete,
    Failed,
}

impl LoadStage {
    pub fn name(&self) -> &'static str {
        match self {
            LoadStage::NotStarted => "not_started",
            LoadStage::ReadingHeader => "reading_header",
            LoadStage::LoadingTensors => "loading_tensors",
            LoadStage::ConvertingDtypes => "converting_dtypes",
            LoadStage::ValidatingWeights => "validating_weights",
            LoadStage::BuildingModel => "building_model",
            LoadStage::Complete => "complete",
            LoadStage::Failed => "failed",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, LoadStage::Complete | LoadStage::Failed)
    }
}

/// Progress tracker for model loading.
#[derive(Debug)]
pub struct LoadProgress {
    pub stage: LoadStage,
    pub tensors_loaded: usize,
    pub tensors_total: usize,
    pub bytes_loaded: u64,
    pub bytes_total: u64,
    started_at: Option<Instant>,
    stage_times: Vec<(LoadStage, Duration)>,
}

impl LoadProgress {
    pub fn new() -> Self {
        Self {
            stage: LoadStage::NotStarted,
            tensors_loaded: 0,
            tensors_total: 0,
            bytes_loaded: 0,
            bytes_total: 0,
            started_at: None,
            stage_times: Vec::new(),
        }
    }

    /// Start tracking.
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
        self.set_stage(LoadStage::ReadingHeader);
    }

    /// Advance to a new stage.
    pub fn set_stage(&mut self, stage: LoadStage) {
        if let Some(start) = self.started_at {
            self.stage_times.push((self.stage, start.elapsed()));
        }
        self.stage = stage;
    }

    /// Update tensor loading progress.
    pub fn update_tensor(&mut self, tensor_bytes: u64) {
        self.tensors_loaded += 1;
        self.bytes_loaded += tensor_bytes;
    }

    /// Set total expected counts.
    pub fn set_totals(&mut self, tensors: usize, bytes: u64) {
        self.tensors_total = tensors;
        self.bytes_total = bytes;
    }

    /// Fraction complete (0.0 to 1.0).
    pub fn fraction(&self) -> f64 {
        if self.tensors_total == 0 {
            return 0.0;
        }
        self.tensors_loaded as f64 / self.tensors_total as f64
    }

    /// Percentage complete (0 to 100).
    pub fn percent(&self) -> u32 {
        (self.fraction() * 100.0) as u32
    }

    /// Elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.started_at.map_or(Duration::ZERO, |s| s.elapsed())
    }

    /// Estimated time remaining.
    pub fn eta(&self) -> Option<Duration> {
        let frac = self.fraction();
        if frac <= 0.0 || frac >= 1.0 {
            return None;
        }
        let elapsed = self.elapsed().as_secs_f64();
        let total_est = elapsed / frac;
        let remaining = total_est - elapsed;
        Some(Duration::from_secs_f64(remaining.max(0.0)))
    }

    /// Loading throughput in MB/s.
    pub fn throughput_mbps(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.bytes_loaded as f64 / secs / (1024.0 * 1024.0)
    }

    /// Check if loading is complete.
    pub fn is_complete(&self) -> bool {
        self.stage == LoadStage::Complete
    }

    /// Check if loading failed.
    pub fn is_failed(&self) -> bool {
        self.stage == LoadStage::Failed
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        format!(
            "[{}] {}/{} tensors ({:.1}%), {:.1} MB/s",
            self.stage.name(),
            self.tensors_loaded,
            self.tensors_total,
            self.fraction() * 100.0,
            self.throughput_mbps(),
        )
    }
}

impl Default for LoadProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for a model checkpoint file.
#[derive(Debug, Clone)]
pub struct CheckpointMeta {
    pub path: String,
    pub format: String,
    pub num_tensors: usize,
    pub file_size_bytes: u64,
    pub num_shards: usize,
}

impl CheckpointMeta {
    pub fn new(path: impl Into<String>, format: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            format: format.into(),
            num_tensors: 0,
            file_size_bytes: 0,
            num_shards: 1,
        }
    }

    pub fn with_tensors(mut self, n: usize) -> Self {
        self.num_tensors = n;
        self
    }

    pub fn with_size(mut self, bytes: u64) -> Self {
        self.file_size_bytes = bytes;
        self
    }

    pub fn with_shards(mut self, n: usize) -> Self {
        self.num_shards = n;
        self
    }

    pub fn size_gb(&self) -> f64 {
        self.file_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn is_sharded(&self) -> bool {
        self.num_shards > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_basic() {
        let mut p = LoadProgress::new();
        p.set_totals(100, 1000);
        p.start();
        p.update_tensor(10);
        assert_eq!(p.tensors_loaded, 1);
        assert_eq!(p.percent(), 1);
    }

    #[test]
    fn test_progress_fraction() {
        let mut p = LoadProgress::new();
        p.set_totals(4, 400);
        p.update_tensor(100);
        p.update_tensor(100);
        assert!((p.fraction() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stage_progression() {
        let mut p = LoadProgress::new();
        p.start();
        assert_eq!(p.stage, LoadStage::ReadingHeader);
        p.set_stage(LoadStage::LoadingTensors);
        assert_eq!(p.stage, LoadStage::LoadingTensors);
    }

    #[test]
    fn test_is_terminal() {
        assert!(LoadStage::Complete.is_terminal());
        assert!(LoadStage::Failed.is_terminal());
        assert!(!LoadStage::LoadingTensors.is_terminal());
    }

    #[test]
    fn test_complete_check() {
        let mut p = LoadProgress::new();
        p.set_stage(LoadStage::Complete);
        assert!(p.is_complete());
        assert!(!p.is_failed());
    }

    #[test]
    fn test_summary() {
        let mut p = LoadProgress::new();
        p.set_totals(100, 1000);
        p.update_tensor(50);
        let s = p.summary();
        assert!(s.contains("1/100"));
    }

    #[test]
    fn test_zero_totals() {
        let p = LoadProgress::new();
        assert_eq!(p.fraction(), 0.0);
        assert_eq!(p.percent(), 0);
    }

    #[test]
    fn test_checkpoint_meta() {
        let m = CheckpointMeta::new("model.gguf", "GGUF")
            .with_tensors(200)
            .with_size(5 * 1024 * 1024 * 1024)
            .with_shards(1);
        assert!(!m.is_sharded());
        assert!((m.size_gb() - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_sharded_checkpoint() {
        let m = CheckpointMeta::new("model-00001.safetensors", "SafeTensors").with_shards(6);
        assert!(m.is_sharded());
    }

    #[test]
    fn test_eta_early() {
        let mut p = LoadProgress::new();
        p.set_totals(100, 1000);
        // No tensors loaded yet
        assert!(p.eta().is_none());
    }

    #[test]
    fn test_stage_name() {
        assert_eq!(LoadStage::ReadingHeader.name(), "reading_header");
        assert_eq!(LoadStage::Complete.name(), "complete");
    }

    #[test]
    fn test_failed_state() {
        let mut p = LoadProgress::new();
        p.set_stage(LoadStage::Failed);
        assert!(p.is_failed());
        assert!(!p.is_complete());
    }
}

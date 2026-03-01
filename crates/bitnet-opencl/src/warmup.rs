//! GPU warmup and JIT compilation at startup.
//!
//! [`GpuWarmup`] pre-compiles registered kernel sources, optionally launches
//! dummy kernels to prime GPU caches, and reports progress through a
//! caller-supplied callback.

use std::collections::HashMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How aggressively the GPU should be warmed up at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupLevel {
    /// Skip warmup entirely.
    None,
    /// Compile all registered kernel sources but do not launch them.
    KernelCompile,
    /// Compile **and** launch dummy kernels to prime GPU caches.
    FullWarmup,
}

impl Default for WarmupLevel {
    fn default() -> Self {
        Self::KernelCompile
    }
}

/// Progress event emitted during warmup.
#[derive(Debug, Clone)]
pub struct WarmupProgress {
    /// 0-based index of the current step.
    pub step: usize,
    /// Total number of steps expected.
    pub total: usize,
    /// Name of the kernel being processed.
    pub kernel_name: String,
    /// Phase description (e.g. "compiling", "launching").
    pub phase: String,
}

impl WarmupProgress {
    /// Fraction complete in `[0.0, 1.0]`.
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        (self.step + 1) as f64 / self.total as f64
    }
}

/// Result of compiling a single kernel source.
#[derive(Debug, Clone)]
pub struct CompileResult {
    /// Kernel name.
    pub name: String,
    /// Whether compilation succeeded.
    pub success: bool,
    /// Time taken to compile.
    pub duration: Duration,
    /// Error message (if any).
    pub error: Option<String>,
}

/// Result of a dummy kernel launch.
#[derive(Debug, Clone)]
pub struct LaunchResult {
    /// Kernel name.
    pub name: String,
    /// Whether the launch succeeded.
    pub success: bool,
    /// Time taken for the launch + sync.
    pub duration: Duration,
}

/// Summary returned after the warmup phase completes.
#[derive(Debug, Clone)]
pub struct WarmupReport {
    /// Warmup level that was used.
    pub level: WarmupLevel,
    /// Per-kernel compilation results (empty if level == None).
    pub compile_results: Vec<CompileResult>,
    /// Per-kernel launch results (empty unless level == FullWarmup).
    pub launch_results: Vec<LaunchResult>,
    /// Wall-clock duration of the entire warmup.
    pub total_duration: Duration,
}

impl WarmupReport {
    /// Number of kernels that failed compilation.
    pub fn compile_failures(&self) -> usize {
        self.compile_results.iter().filter(|r| !r.success).count()
    }

    /// Number of kernels that failed the dummy launch.
    pub fn launch_failures(&self) -> usize {
        self.launch_results.iter().filter(|r| !r.success).count()
    }

    /// `true` when every step succeeded.
    pub fn all_ok(&self) -> bool {
        self.compile_failures() == 0 && self.launch_failures() == 0
    }
}

/// GPU warmup engine.
///
/// Register kernel sources, then call [`run`](GpuWarmup::run) to compile and
/// optionally launch them. Progress is reported through a callback or channel.
pub struct GpuWarmup {
    /// (name, source) pairs registered for warmup.
    kernels: Vec<(String, String)>,
    /// Warmup level.
    level: WarmupLevel,
    /// Optional progress callback.
    progress_cb: Option<Box<dyn Fn(WarmupProgress) + Send>>,
    /// Simulated compilation results — test hook.
    /// Maps kernel name → simulated compile success/failure.
    simulated_compile: HashMap<String, bool>,
    /// Simulated launch results — test hook.
    simulated_launch: HashMap<String, bool>,
}

impl GpuWarmup {
    /// Create a new warmup engine with the given level.
    pub fn new(level: WarmupLevel) -> Self {
        Self {
            kernels: Vec::new(),
            level,
            progress_cb: None,
            simulated_compile: HashMap::new(),
            simulated_launch: HashMap::new(),
        }
    }

    /// Register a kernel source for warmup.
    pub fn register_kernel(
        &mut self,
        name: impl Into<String>,
        source: impl Into<String>,
    ) {
        self.kernels.push((name.into(), source.into()));
    }

    /// Set a progress callback.
    pub fn on_progress<F>(&mut self, cb: F)
    where
        F: Fn(WarmupProgress) + Send + 'static,
    {
        self.progress_cb = Some(Box::new(cb));
    }

    /// Inject a simulated compile outcome for testing.
    pub fn simulate_compile(&mut self, name: &str, success: bool) {
        self.simulated_compile.insert(name.to_string(), success);
    }

    /// Inject a simulated launch outcome for testing.
    pub fn simulate_launch(&mut self, name: &str, success: bool) {
        self.simulated_launch.insert(name.to_string(), success);
    }

    /// Number of registered kernels.
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Run the warmup and return a report.
    pub fn run(&self) -> WarmupReport {
        let start = Instant::now();

        if self.level == WarmupLevel::None {
            return WarmupReport {
                level: self.level,
                compile_results: Vec::new(),
                launch_results: Vec::new(),
                total_duration: start.elapsed(),
            };
        }

        let total_steps = match self.level {
            WarmupLevel::KernelCompile => self.kernels.len(),
            WarmupLevel::FullWarmup => self.kernels.len() * 2,
            WarmupLevel::None => 0,
        };

        let mut compile_results = Vec::with_capacity(self.kernels.len());
        let mut step = 0usize;

        // ── Compilation phase ───────────────────────────────────────
        for (name, _source) in &self.kernels {
            self.emit_progress(step, total_steps, name, "compiling");

            let t = Instant::now();
            let success = self
                .simulated_compile
                .get(name)
                .copied()
                .unwrap_or(true);
            let error = if success {
                None
            } else {
                Some(format!("simulated compile error for {name}"))
            };

            compile_results.push(CompileResult {
                name: name.clone(),
                success,
                duration: t.elapsed(),
                error,
            });
            step += 1;
        }

        // ── Launch phase (FullWarmup only) ──────────────────────────
        let mut launch_results = Vec::new();
        if self.level == WarmupLevel::FullWarmup {
            for (name, _source) in &self.kernels {
                self.emit_progress(step, total_steps, name, "launching");

                let t = Instant::now();
                let success = self
                    .simulated_launch
                    .get(name)
                    .copied()
                    .unwrap_or(true);

                launch_results.push(LaunchResult {
                    name: name.clone(),
                    success,
                    duration: t.elapsed(),
                });
                step += 1;
            }
        }

        WarmupReport {
            level: self.level,
            compile_results,
            launch_results,
            total_duration: start.elapsed(),
        }
    }

    /// Create a channel-based progress receiver. Call before [`run`].
    pub fn progress_channel(&mut self) -> mpsc::Receiver<WarmupProgress> {
        let (tx, rx) = mpsc::channel();
        self.on_progress(move |p| {
            let _ = tx.send(p);
        });
        rx
    }

    // ── internal helpers ────────────────────────────────────────────

    fn emit_progress(
        &self,
        step: usize,
        total: usize,
        kernel_name: &str,
        phase: &str,
    ) {
        if let Some(ref cb) = self.progress_cb {
            cb(WarmupProgress {
                step,
                total,
                kernel_name: kernel_name.to_string(),
                phase: phase.to_string(),
            });
        }
    }
}

/// Measure the wall-clock time of a closure (for before/after warmup timing).
pub fn measure_startup<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_warmup_none_returns_empty_report() {
        let mut w = GpuWarmup::new(WarmupLevel::None);
        w.register_kernel("k1", "__kernel void k1(){}");
        let report = w.run();
        assert!(report.compile_results.is_empty());
        assert!(report.launch_results.is_empty());
        assert!(report.all_ok());
    }

    #[test]
    fn test_kernel_compile_only() {
        let mut w = GpuWarmup::new(WarmupLevel::KernelCompile);
        w.register_kernel("matmul", "src");
        w.register_kernel("relu", "src");
        let report = w.run();
        assert_eq!(report.compile_results.len(), 2);
        assert!(report.launch_results.is_empty());
        assert!(report.all_ok());
    }

    #[test]
    fn test_full_warmup_compiles_and_launches() {
        let mut w = GpuWarmup::new(WarmupLevel::FullWarmup);
        w.register_kernel("k1", "src");
        w.register_kernel("k2", "src");
        let report = w.run();
        assert_eq!(report.compile_results.len(), 2);
        assert_eq!(report.launch_results.len(), 2);
        assert!(report.all_ok());
    }

    #[test]
    fn test_simulated_compile_failure() {
        let mut w = GpuWarmup::new(WarmupLevel::KernelCompile);
        w.register_kernel("good", "src");
        w.register_kernel("bad", "src");
        w.simulate_compile("bad", false);
        let report = w.run();
        assert_eq!(report.compile_failures(), 1);
        assert!(!report.all_ok());
        assert!(report.compile_results[1].error.is_some());
    }

    #[test]
    fn test_simulated_launch_failure() {
        let mut w = GpuWarmup::new(WarmupLevel::FullWarmup);
        w.register_kernel("k", "src");
        w.simulate_launch("k", false);
        let report = w.run();
        assert_eq!(report.launch_failures(), 1);
        assert!(!report.all_ok());
    }

    #[test]
    fn test_progress_callback_fires() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let mut w = GpuWarmup::new(WarmupLevel::FullWarmup);
        w.register_kernel("a", "src");
        w.register_kernel("b", "src");
        w.on_progress(move |_p| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });
        w.run();
        // 2 compile + 2 launch = 4 progress events
        assert_eq!(count.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_progress_channel() {
        let mut w = GpuWarmup::new(WarmupLevel::KernelCompile);
        w.register_kernel("x", "src");
        let rx = w.progress_channel();
        w.run();
        let p = rx.recv().unwrap();
        assert_eq!(p.kernel_name, "x");
        assert_eq!(p.phase, "compiling");
    }

    #[test]
    fn test_progress_fraction() {
        let p = WarmupProgress {
            step: 1,
            total: 4,
            kernel_name: "k".into(),
            phase: "compiling".into(),
        };
        assert!((p.fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_measure_startup_timing() {
        let (val, dur) = measure_startup(|| 42);
        assert_eq!(val, 42);
        assert!(dur < Duration::from_secs(1));
    }

    #[test]
    fn test_warmup_level_default() {
        assert_eq!(WarmupLevel::default(), WarmupLevel::KernelCompile);
    }

    #[test]
    fn test_kernel_count() {
        let mut w = GpuWarmup::new(WarmupLevel::None);
        assert_eq!(w.kernel_count(), 0);
        w.register_kernel("a", "src");
        w.register_kernel("b", "src");
        assert_eq!(w.kernel_count(), 2);
    }
}

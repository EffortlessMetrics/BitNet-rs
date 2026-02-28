//! GPU kernel hot-reload for development.
//!
//! [`KernelWatcher`] monitors `.cl` kernel source files for changes and
//! recompiles them on the fly. Only active when `BITNET_DEV_MODE=1`.
//!
//! Each kernel tracks a monotonically increasing version counter so
//! consumers can detect when a new compilation is available.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Compiled kernel representation.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Kernel source code that was compiled.
    pub source: String,
    /// Monotonically increasing version number.
    pub version: u64,
    /// Path to the source file.
    pub source_path: PathBuf,
    /// Timestamp of compilation.
    pub compiled_at: SystemTime,
}

/// Error type for hot-reload operations.
#[derive(Debug, thiserror::Error)]
pub enum HotReloadError {
    #[error("hot-reload requires BITNET_DEV_MODE=1")]
    DevModeNotEnabled,
    #[error("IO error watching {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("kernel compilation failed for {path}: {reason}")]
    CompilationFailed { path: PathBuf, reason: String },
    #[error("kernel not found: {0}")]
    KernelNotFound(String),
}

/// Callback invoked after a kernel is successfully recompiled.
pub type ReloadCallback = Box<dyn Fn(&str, &CompiledKernel) + Send + Sync>;

/// State for a single watched kernel.
#[derive(Debug)]
struct WatchedKernel {
    path: PathBuf,
    last_modified: Option<SystemTime>,
    current: Option<CompiledKernel>,
    version: u64,
}

/// Simulated kernel compiler (in production this would invoke OpenCL
/// `clCreateProgramWithSource` + `clBuildProgram`).
pub trait KernelCompiler: Send + Sync {
    /// Compile kernel source and return compiled bytes or an error message.
    fn compile(&self, name: &str, source: &str) -> Result<Vec<u8>, String>;
}

/// Default pass-through compiler that accepts any valid source.
#[derive(Debug, Clone, Default)]
pub struct PassthroughCompiler;

impl KernelCompiler for PassthroughCompiler {
    fn compile(&self, _name: &str, source: &str) -> Result<Vec<u8>, String> {
        if source.trim().is_empty() {
            return Err("empty kernel source".into());
        }
        Ok(source.as_bytes().to_vec())
    }
}

/// Watches `.cl` kernel files for changes and triggers recompilation.
///
/// Only active when `BITNET_DEV_MODE=1` environment variable is set.
pub struct KernelWatcher {
    kernels: Arc<Mutex<HashMap<String, WatchedKernel>>>,
    compiler: Arc<dyn KernelCompiler>,
    poll_interval: Duration,
    callback: Option<Arc<ReloadCallback>>,
    dev_mode: bool,
}

impl std::fmt::Debug for KernelWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelWatcher")
            .field("poll_interval", &self.poll_interval)
            .field("dev_mode", &self.dev_mode)
            .field(
                "kernel_count",
                &self.kernels.lock().unwrap_or_else(|e| e.into_inner()).len(),
            )
            .finish()
    }
}

/// Check if dev mode is enabled via environment variable.
pub fn is_dev_mode() -> bool {
    matches!(
        std::env::var("BITNET_DEV_MODE").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

impl KernelWatcher {
    /// Create a new watcher. Returns `DevModeNotEnabled` unless
    /// `BITNET_DEV_MODE=1` or `force` is true.
    pub fn new(force: bool) -> Result<Self, HotReloadError> {
        if !force && !is_dev_mode() {
            return Err(HotReloadError::DevModeNotEnabled);
        }
        Ok(Self {
            kernels: Arc::new(Mutex::new(HashMap::new())),
            compiler: Arc::new(PassthroughCompiler),
            poll_interval: Duration::from_millis(500),
            callback: None,
            dev_mode: true,
        })
    }

    /// Set a custom kernel compiler.
    pub fn with_compiler(mut self, compiler: Arc<dyn KernelCompiler>) -> Self {
        self.compiler = compiler;
        self
    }

    /// Set the polling interval for file change detection.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Set a callback invoked after successful recompilation.
    pub fn on_reload(mut self, cb: ReloadCallback) -> Self {
        self.callback = Some(Arc::new(cb));
        self
    }

    /// Register a kernel file to watch.
    pub fn watch(&self, name: &str, path: &Path) -> Result<(), HotReloadError> {
        let mut kernels = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        kernels.insert(
            name.to_string(),
            WatchedKernel {
                path: path.to_path_buf(),
                last_modified: None,
                current: None,
                version: 0,
            },
        );
        Ok(())
    }

    /// Remove a kernel from the watch list.
    pub fn unwatch(&self, name: &str) -> bool {
        let mut kernels = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        kernels.remove(name).is_some()
    }

    /// Get the current compiled version of a kernel.
    pub fn get_kernel(&self, name: &str) -> Result<CompiledKernel, HotReloadError> {
        let kernels = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        let watched = kernels
            .get(name)
            .ok_or_else(|| HotReloadError::KernelNotFound(name.into()))?;
        watched
            .current
            .clone()
            .ok_or_else(|| HotReloadError::KernelNotFound(name.into()))
    }

    /// Get the version number for a kernel (0 = never compiled).
    pub fn kernel_version(&self, name: &str) -> u64 {
        let kernels = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        kernels.get(name).map_or(0, |k| k.version)
    }

    /// List all watched kernel names.
    pub fn watched_kernels(&self) -> Vec<String> {
        let kernels = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        kernels.keys().cloned().collect()
    }

    /// Poll all watched kernels for changes and recompile if needed.
    /// Returns the names of kernels that were recompiled.
    pub fn poll(&self) -> Result<Vec<String>, HotReloadError> {
        let mut recompiled = Vec::new();
        let mut kernels = self.kernels.lock().unwrap_or_else(|e| e.into_inner());

        let names: Vec<String> = kernels.keys().cloned().collect();
        for name in names {
            let watched = kernels.get_mut(&name).unwrap();
            let modified = std::fs::metadata(&watched.path)
                .and_then(|m| m.modified())
                .map_err(|e| HotReloadError::Io {
                    path: watched.path.clone(),
                    source: e,
                })?;

            let needs_recompile = match watched.last_modified {
                None => true,
                Some(prev) => modified > prev,
            };

            if needs_recompile {
                let source = std::fs::read_to_string(&watched.path).map_err(|e| {
                    HotReloadError::Io {
                        path: watched.path.clone(),
                        source: e,
                    }
                })?;

                let _compiled = self.compiler.compile(&name, &source).map_err(|reason| {
                    HotReloadError::CompilationFailed {
                        path: watched.path.clone(),
                        reason,
                    }
                })?;

                watched.version += 1;
                let compiled_kernel = CompiledKernel {
                    source: source.clone(),
                    version: watched.version,
                    source_path: watched.path.clone(),
                    compiled_at: SystemTime::now(),
                };
                watched.current = Some(compiled_kernel.clone());
                watched.last_modified = Some(modified);

                if let Some(cb) = &self.callback {
                    cb(&name, &compiled_kernel);
                }

                recompiled.push(name);
            }
        }

        Ok(recompiled)
    }

    /// Returns whether dev mode is active.
    pub fn is_active(&self) -> bool {
        self.dev_mode
    }

    /// Returns the configured poll interval.
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn make_watcher() -> KernelWatcher {
        KernelWatcher::new(true).expect("forced watcher should succeed")
    }

    #[test]
    fn test_new_requires_dev_mode() {
        // Without force and without env var, should fail.
        temp_env::with_var("BITNET_DEV_MODE", None::<&str>, || {
            let result = KernelWatcher::new(false);
            assert!(matches!(result, Err(HotReloadError::DevModeNotEnabled)));
        });
    }

    #[test]
    fn test_new_with_force() {
        let watcher = KernelWatcher::new(true);
        assert!(watcher.is_ok());
        assert!(watcher.unwrap().is_active());
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_new_with_env_var() {
        temp_env::with_var("BITNET_DEV_MODE", Some("1"), || {
            let watcher = KernelWatcher::new(false);
            assert!(watcher.is_ok());
        });
    }

    #[test]
    fn test_watch_and_list() {
        let watcher = make_watcher();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("matmul.cl");
        std::fs::write(&path, "__kernel void matmul() {}").unwrap();

        watcher.watch("matmul", &path).unwrap();
        let names = watcher.watched_kernels();
        assert!(names.contains(&"matmul".to_string()));
    }

    #[test]
    fn test_unwatch() {
        let watcher = make_watcher();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.cl");
        std::fs::write(&path, "__kernel void test() {}").unwrap();

        watcher.watch("test", &path).unwrap();
        assert!(watcher.unwatch("test"));
        assert!(!watcher.unwatch("test"));
        assert!(watcher.watched_kernels().is_empty());
    }

    #[test]
    fn test_poll_compiles_on_first_run() {
        let watcher = make_watcher();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relu.cl");
        std::fs::write(&path, "__kernel void relu() {}").unwrap();

        watcher.watch("relu", &path).unwrap();
        let recompiled = watcher.poll().unwrap();
        assert_eq!(recompiled, vec!["relu".to_string()]);
        assert_eq!(watcher.kernel_version("relu"), 1);
    }

    #[test]
    fn test_poll_no_change_no_recompile() {
        let watcher = make_watcher();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("add.cl");
        std::fs::write(&path, "__kernel void add() {}").unwrap();

        watcher.watch("add", &path).unwrap();
        watcher.poll().unwrap();

        // Second poll with no file change
        let recompiled = watcher.poll().unwrap();
        assert!(recompiled.is_empty());
        assert_eq!(watcher.kernel_version("add"), 1);
    }

    #[test]
    fn test_poll_recompiles_on_file_change() {
        let watcher = make_watcher();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conv.cl");
        std::fs::write(&path, "__kernel void conv_v1() {}").unwrap();

        watcher.watch("conv", &path).unwrap();
        watcher.poll().unwrap();
        assert_eq!(watcher.kernel_version("conv"), 1);

        // Modify file (ensure different mtime)
        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&path, "__kernel void conv_v2() {}").unwrap();

        let recompiled = watcher.poll().unwrap();
        assert_eq!(recompiled, vec!["conv".to_string()]);
        assert_eq!(watcher.kernel_version("conv"), 2);

        let kernel = watcher.get_kernel("conv").unwrap();
        assert!(kernel.source.contains("conv_v2"));
    }

    #[test]
    fn test_compilation_failure() {
        struct FailCompiler;
        impl KernelCompiler for FailCompiler {
            fn compile(&self, _name: &str, _source: &str) -> Result<Vec<u8>, String> {
                Err("syntax error at line 1".into())
            }
        }

        let watcher = make_watcher().with_compiler(Arc::new(FailCompiler));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.cl");
        std::fs::write(&path, "invalid kernel").unwrap();

        watcher.watch("bad", &path).unwrap();
        let err = watcher.poll().unwrap_err();
        assert!(matches!(err, HotReloadError::CompilationFailed { .. }));
    }

    #[test]
    fn test_callback_invoked_on_reload() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let watcher = make_watcher().on_reload(Box::new(move |_name, _kernel| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cb.cl");
        std::fs::write(&path, "__kernel void cb() {}").unwrap();

        watcher.watch("cb", &path).unwrap();
        watcher.poll().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_get_kernel_not_found() {
        let watcher = make_watcher();
        let err = watcher.get_kernel("nonexistent").unwrap_err();
        assert!(matches!(err, HotReloadError::KernelNotFound(_)));
    }

    #[test]
    fn test_version_monotonically_increases() {
        let watcher = make_watcher();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mono.cl");
        std::fs::write(&path, "__kernel void v1() {}").unwrap();

        watcher.watch("mono", &path).unwrap();
        watcher.poll().unwrap();
        assert_eq!(watcher.kernel_version("mono"), 1);

        for i in 2..=5 {
            std::thread::sleep(Duration::from_millis(50));
            std::fs::write(&path, format!("__kernel void v{i}() {{}}")).unwrap();
            watcher.poll().unwrap();
            assert_eq!(watcher.kernel_version("mono"), i);
        }
    }

    #[test]
    fn test_poll_interval_config() {
        let watcher = make_watcher().with_poll_interval(Duration::from_secs(2));
        assert_eq!(watcher.poll_interval(), Duration::from_secs(2));
    }
}

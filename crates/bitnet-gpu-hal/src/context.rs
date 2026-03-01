//! GPU context manager for multi-backend lifecycle management.
//!
//! Provides [`GpuContext`] with a state machine (6 states),
//! [`ContextManager`] with auto-select and CPU fallback, and
//! [`ContextGuard`] for RAII-style context acquisition.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Backend types ─────────────────────────────────────────────────────────

/// Supported GPU/compute backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// CPU fallback (always available).
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// `OpenCL`.
    OpenCL,
    /// Vulkan compute.
    Vulkan,
    /// Apple Metal.
    Metal,
    /// AMD `ROCm`.
    Rocm,
    /// WebGPU.
    WebGpu,
    /// Intel Level Zero.
    LevelZero,
}

impl BackendType {
    /// Returns `true` if this backend represents a GPU (not CPU).
    pub const fn is_gpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Priority for auto-selection (higher = preferred).
    const fn priority(self) -> u8 {
        match self {
            Self::Cuda => 100,
            Self::Rocm => 90,
            Self::Metal => 85,
            Self::LevelZero => 80,
            Self::Vulkan => 70,
            Self::OpenCL => 60,
            Self::WebGpu => 50,
            Self::Cpu => 0,
        }
    }
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda => write!(f, "CUDA"),
            Self::OpenCL => write!(f, "OpenCL"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::Rocm => write!(f, "ROCm"),
            Self::WebGpu => write!(f, "WebGPU"),
            Self::LevelZero => write!(f, "Level Zero"),
        }
    }
}

// ── Context state machine ─────────────────────────────────────────────────

/// States a [`GpuContext`] can be in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextState {
    /// Context has been created but not yet initialized.
    Uninitialized,
    /// Backend driver is being loaded / device is being probed.
    Initializing,
    /// Context is ready to accept work.
    Ready,
    /// Context is currently executing work.
    Busy,
    /// Context encountered an error.
    Error(String),
    /// Context has been torn down and cannot be reused.
    Destroyed,
}

impl ContextState {
    /// Returns `true` if a transition to `target` is allowed.
    const fn can_transition_to(&self, target: &Self) -> bool {
        matches!(
            (self, target),
            (Self::Uninitialized, Self::Initializing)
                | (Self::Initializing, Self::Ready | Self::Error(_))
                | (Self::Ready, Self::Busy | Self::Destroyed | Self::Error(_))
                | (Self::Busy, Self::Ready | Self::Error(_) | Self::Destroyed)
                | (Self::Error(_), Self::Destroyed | Self::Initializing)
        )
    }
}

impl fmt::Display for ContextState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uninitialized => write!(f, "Uninitialized"),
            Self::Initializing => write!(f, "Initializing"),
            Self::Ready => write!(f, "Ready"),
            Self::Busy => write!(f, "Busy"),
            Self::Error(msg) => write!(f, "Error({msg})"),
            Self::Destroyed => write!(f, "Destroyed"),
        }
    }
}

// ── Context errors ────────────────────────────────────────────────────────

/// Errors produced by context operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// Attempted an invalid state transition.
    InvalidTransition { from: String, to: String },
    /// Context index does not exist.
    NotFound { index: usize },
    /// Context is in the wrong state for the requested operation.
    InvalidState { expected: String, actual: String },
    /// Context is busy and cannot be acquired.
    Busy { index: usize },
    /// No suitable backend is available.
    NoBackendAvailable,
    /// Context has been destroyed.
    Destroyed { index: usize },
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid state transition: {from} -> {to}")
            }
            Self::NotFound { index } => write!(f, "context {index} not found"),
            Self::InvalidState { expected, actual } => {
                write!(f, "expected state {expected}, got {actual}")
            }
            Self::Busy { index } => {
                write!(f, "context {index} is busy")
            }
            Self::NoBackendAvailable => {
                write!(f, "no suitable backend available")
            }
            Self::Destroyed { index } => {
                write!(f, "context {index} has been destroyed")
            }
        }
    }
}

impl std::error::Error for ContextError {}

// ── Context properties ────────────────────────────────────────────────────

/// Hardware and capability properties of a device backing a context.
#[derive(Debug, Clone)]
pub struct ContextProperties {
    /// Human-readable device name.
    pub device_name: String,
    /// Compute capability string (e.g. "8.6" for CUDA).
    pub compute_capability: Option<String>,
    /// Driver version string.
    pub driver_version: Option<String>,
    /// Maximum device memory in bytes.
    pub max_memory: u64,
    /// Maximum concurrent threads.
    pub max_threads: u32,
    /// Whether the device supports FP16 arithmetic.
    pub supports_fp16: bool,
    /// Whether the device supports INT8 arithmetic.
    pub supports_int8: bool,
}

impl Default for ContextProperties {
    fn default() -> Self {
        Self {
            device_name: String::from("Unknown"),
            compute_capability: None,
            driver_version: None,
            max_memory: 0,
            max_threads: 1,
            supports_fp16: false,
            supports_int8: false,
        }
    }
}

impl ContextProperties {
    /// Create properties for a CPU backend.
    pub fn cpu_default() -> Self {
        Self {
            device_name: String::from("CPU"),
            max_threads: 1,
            supports_fp16: false,
            supports_int8: true,
            ..Self::default()
        }
    }
}

// ── GpuContext ─────────────────────────────────────────────────────────────

/// A single GPU (or CPU) execution context.
pub struct GpuContext {
    backend: BackendType,
    device_index: usize,
    state: ContextState,
    properties: ContextProperties,
    created_at: u64,
}

impl GpuContext {
    /// Create a new context in [`ContextState::Uninitialized`].
    pub fn new(backend: BackendType, device_index: usize) -> Self {
        let properties = if backend == BackendType::Cpu {
            ContextProperties::cpu_default()
        } else {
            ContextProperties::default()
        };
        Self {
            backend,
            device_index,
            state: ContextState::Uninitialized,
            properties,
            created_at: now_epoch_secs(),
        }
    }

    /// Create a context with explicit properties.
    pub fn with_properties(
        backend: BackendType,
        device_index: usize,
        properties: ContextProperties,
    ) -> Self {
        Self {
            backend,
            device_index,
            state: ContextState::Uninitialized,
            properties,
            created_at: now_epoch_secs(),
        }
    }

    /// Transition to a new state, validating the transition is legal.
    pub fn transition(&mut self, new_state: ContextState) -> Result<(), ContextError> {
        if !self.state.can_transition_to(&new_state) {
            return Err(ContextError::InvalidTransition {
                from: self.state.to_string(),
                to: new_state.to_string(),
            });
        }
        self.state = new_state;
        Ok(())
    }

    /// The backend type for this context.
    pub const fn backend(&self) -> BackendType {
        self.backend
    }

    /// The device index within the backend.
    pub const fn device_index(&self) -> usize {
        self.device_index
    }

    /// Current state.
    pub const fn state(&self) -> &ContextState {
        &self.state
    }

    /// Device properties.
    pub const fn properties(&self) -> &ContextProperties {
        &self.properties
    }

    /// Epoch seconds when this context was created.
    pub const fn created_at(&self) -> u64 {
        self.created_at
    }

    /// Whether this context is in the [`ContextState::Ready`] state.
    pub fn is_ready(&self) -> bool {
        self.state == ContextState::Ready
    }

    /// Whether this context is in the [`ContextState::Busy`] state.
    pub fn is_busy(&self) -> bool {
        self.state == ContextState::Busy
    }

    /// Whether this context has been destroyed.
    pub fn is_destroyed(&self) -> bool {
        self.state == ContextState::Destroyed
    }

    /// Convenience: initialize the context (Uninitialized → Initializing → Ready).
    pub fn initialize(&mut self) -> Result<(), ContextError> {
        self.transition(ContextState::Initializing)?;
        self.transition(ContextState::Ready)
    }
}

// ── ContextGuard ──────────────────────────────────────────────────────────

/// RAII guard representing exclusive access to a context.
///
/// While held, the context is in the [`ContextState::Busy`] state.
/// Dropping or explicitly releasing the guard returns it to
/// [`ContextState::Ready`].
#[derive(Debug)]
pub struct ContextGuard {
    context_index: usize,
    acquired_at: u64,
}

impl ContextGuard {
    /// The index of the acquired context.
    pub const fn context_index(&self) -> usize {
        self.context_index
    }

    /// Epoch seconds when the context was acquired.
    pub const fn acquired_at(&self) -> u64 {
        self.acquired_at
    }
}

// ── ContextManagerConfig ──────────────────────────────────────────────────

/// Configuration for [`ContextManager`].
#[derive(Debug, Clone)]
pub struct ContextManagerConfig {
    /// If `true`, `auto_select_best` is called on creation.
    pub auto_select: bool,
    /// Preferred backend for auto-selection.
    pub prefer_backend: Option<BackendType>,
    /// If `true`, auto-select will fall back to CPU when no GPU is available.
    pub fallback_to_cpu: bool,
}

impl Default for ContextManagerConfig {
    fn default() -> Self {
        Self { auto_select: false, prefer_backend: None, fallback_to_cpu: true }
    }
}

// ── ContextManager ────────────────────────────────────────────────────────

/// Manages multiple [`GpuContext`] instances and their lifecycles.
pub struct ContextManager {
    contexts: Vec<Option<GpuContext>>,
    default_context: Option<usize>,
    config: ContextManagerConfig,
}

impl ContextManager {
    /// Create a new manager with the given configuration.
    pub const fn new(config: ContextManagerConfig) -> Self {
        Self { contexts: Vec::new(), default_context: None, config }
    }

    /// Create a context for `backend` on `device_index`, returning its
    /// manager-local index.
    pub fn create_context(
        &mut self,
        backend: BackendType,
        device_index: usize,
    ) -> Result<usize, ContextError> {
        let ctx = GpuContext::new(backend, device_index);
        let index = self.contexts.len();
        self.contexts.push(Some(ctx));
        if self.default_context.is_none() {
            self.default_context = Some(index);
        }
        Ok(index)
    }

    /// Create a context with explicit properties.
    pub fn create_context_with_properties(
        &mut self,
        backend: BackendType,
        device_index: usize,
        properties: ContextProperties,
    ) -> Result<usize, ContextError> {
        let ctx = GpuContext::with_properties(backend, device_index, properties);
        let index = self.contexts.len();
        self.contexts.push(Some(ctx));
        if self.default_context.is_none() {
            self.default_context = Some(index);
        }
        Ok(index)
    }

    /// Destroy the context at `index`.
    pub fn destroy_context(&mut self, index: usize) -> Result<(), ContextError> {
        let ctx = self.get_context_mut(index)?;
        // Allow destroying from any non-Destroyed state
        if ctx.is_destroyed() {
            return Err(ContextError::Destroyed { index });
        }
        // Transition through valid path
        match ctx.state() {
            ContextState::Uninitialized => {
                ctx.transition(ContextState::Initializing)?;
                ctx.transition(ContextState::Ready)?;
                ctx.transition(ContextState::Destroyed)?;
            }
            ContextState::Initializing => {
                ctx.transition(ContextState::Ready)?;
                ctx.transition(ContextState::Destroyed)?;
            }
            ContextState::Ready | ContextState::Busy | ContextState::Error(_) => {
                ctx.transition(ContextState::Destroyed)?;
            }
            ContextState::Destroyed => unreachable!(),
        }
        if self.default_context == Some(index) {
            self.default_context = self.first_live_index();
        }
        Ok(())
    }

    /// Get a reference to the context at `index`.
    pub fn get_context(&self, index: usize) -> Option<&GpuContext> {
        self.contexts.get(index).and_then(|slot| slot.as_ref().filter(|ctx| !ctx.is_destroyed()))
    }

    /// Get the default context, if one is set and alive.
    pub fn default_context(&self) -> Option<&GpuContext> {
        self.default_context.and_then(|i| self.get_context(i))
    }

    /// Index of the default context.
    pub const fn default_context_index(&self) -> Option<usize> {
        self.default_context
    }

    /// Set the default context to `index`.
    pub fn set_default(&mut self, index: usize) -> Result<(), ContextError> {
        if self.get_context(index).is_none() {
            return Err(ContextError::NotFound { index });
        }
        self.default_context = Some(index);
        Ok(())
    }

    /// Auto-select the best available context.
    ///
    /// Prefers `config.prefer_backend` if set, then highest-priority GPU,
    /// then CPU if `fallback_to_cpu` is enabled.
    pub fn auto_select_best(&mut self) -> Result<usize, ContextError> {
        // Gather live contexts with their indices
        let mut candidates: Vec<(usize, &GpuContext)> = self
            .contexts
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().filter(|c| !c.is_destroyed()).map(|c| (i, c)))
            .collect();

        if candidates.is_empty() {
            return Err(ContextError::NoBackendAvailable);
        }

        // If a preferred backend is configured, try it first
        if let Some(preferred) = self.config.prefer_backend
            && let Some(&(idx, _)) = candidates.iter().find(|(_, c)| c.backend() == preferred)
        {
            self.default_context = Some(idx);
            return Ok(idx);
        }

        // Sort by priority (descending)
        candidates.sort_by(|(_, a), (_, b)| b.backend().priority().cmp(&a.backend().priority()));

        let (best_idx, best_ctx) = candidates[0];

        // If the best is CPU and fallback_to_cpu is false, and we have no GPU, fail
        if !best_ctx.backend().is_gpu() && !self.config.fallback_to_cpu {
            return Err(ContextError::NoBackendAvailable);
        }

        self.default_context = Some(best_idx);
        Ok(best_idx)
    }

    /// Acquire exclusive access to the context at `index`.
    ///
    /// The context must be in the [`ContextState::Ready`] state.
    pub fn acquire(&mut self, index: usize) -> Result<ContextGuard, ContextError> {
        let ctx = self.get_context_mut(index)?;
        if ctx.is_destroyed() {
            return Err(ContextError::Destroyed { index });
        }
        if ctx.is_busy() {
            return Err(ContextError::Busy { index });
        }
        if !ctx.is_ready() {
            return Err(ContextError::InvalidState {
                expected: String::from("Ready"),
                actual: ctx.state().to_string(),
            });
        }
        ctx.transition(ContextState::Busy)?;
        Ok(ContextGuard { context_index: index, acquired_at: now_epoch_secs() })
    }

    /// Release a previously acquired context guard.
    pub fn release(&mut self, guard: &ContextGuard) -> Result<(), ContextError> {
        let index = guard.context_index;
        let ctx = self.get_context_mut(index)?;
        if ctx.is_destroyed() {
            return Err(ContextError::Destroyed { index });
        }
        ctx.transition(ContextState::Ready)?;
        Ok(())
    }

    /// List all backend types that have at least one live context.
    pub fn available_backends(&self) -> Vec<BackendType> {
        let mut backends: Vec<BackendType> = self
            .contexts
            .iter()
            .filter_map(|slot| slot.as_ref().filter(|c| !c.is_destroyed()).map(GpuContext::backend))
            .collect();
        backends.sort_by_key(|b| std::cmp::Reverse(b.priority()));
        backends.dedup();
        backends
    }

    /// Number of live (non-destroyed) contexts.
    pub fn live_count(&self) -> usize {
        self.contexts.iter().filter(|slot| slot.as_ref().is_some_and(|c| !c.is_destroyed())).count()
    }

    /// Configuration reference.
    pub const fn config(&self) -> &ContextManagerConfig {
        &self.config
    }

    // ── internal helpers ──────────────────────────────────────────────────

    fn get_context_mut(&mut self, index: usize) -> Result<&mut GpuContext, ContextError> {
        self.contexts
            .get_mut(index)
            .and_then(|slot| slot.as_mut())
            .ok_or(ContextError::NotFound { index })
    }

    fn first_live_index(&self) -> Option<usize> {
        self.contexts
            .iter()
            .enumerate()
            .find_map(|(i, slot)| slot.as_ref().filter(|c| !c.is_destroyed()).map(|_| i))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn now_epoch_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BackendType tests ─────────────────────────────────────────────

    #[test]
    fn backend_cpu_is_not_gpu() {
        assert!(!BackendType::Cpu.is_gpu());
    }

    #[test]
    fn backend_cuda_is_gpu() {
        assert!(BackendType::Cuda.is_gpu());
    }

    #[test]
    fn backend_opencl_is_gpu() {
        assert!(BackendType::OpenCL.is_gpu());
    }

    #[test]
    fn backend_vulkan_is_gpu() {
        assert!(BackendType::Vulkan.is_gpu());
    }

    #[test]
    fn backend_metal_is_gpu() {
        assert!(BackendType::Metal.is_gpu());
    }

    #[test]
    fn backend_rocm_is_gpu() {
        assert!(BackendType::Rocm.is_gpu());
    }

    #[test]
    fn backend_webgpu_is_gpu() {
        assert!(BackendType::WebGpu.is_gpu());
    }

    #[test]
    fn backend_levelzero_is_gpu() {
        assert!(BackendType::LevelZero.is_gpu());
    }

    #[test]
    fn backend_display() {
        assert_eq!(BackendType::Cpu.to_string(), "CPU");
        assert_eq!(BackendType::Cuda.to_string(), "CUDA");
        assert_eq!(BackendType::LevelZero.to_string(), "Level Zero");
    }

    #[test]
    fn backend_priority_cuda_over_cpu() {
        assert!(BackendType::Cuda.priority() > BackendType::Cpu.priority());
    }

    #[test]
    fn backend_priority_rocm_over_vulkan() {
        assert!(BackendType::Rocm.priority() > BackendType::Vulkan.priority());
    }

    // ── ContextState transition tests ─────────────────────────────────

    #[test]
    fn state_uninitialized_to_initializing() {
        let s = ContextState::Uninitialized;
        assert!(s.can_transition_to(&ContextState::Initializing));
    }

    #[test]
    fn state_initializing_to_ready() {
        let s = ContextState::Initializing;
        assert!(s.can_transition_to(&ContextState::Ready));
    }

    #[test]
    fn state_ready_to_busy() {
        let s = ContextState::Ready;
        assert!(s.can_transition_to(&ContextState::Busy));
    }

    #[test]
    fn state_busy_to_ready() {
        let s = ContextState::Busy;
        assert!(s.can_transition_to(&ContextState::Ready));
    }

    #[test]
    fn state_ready_to_destroyed() {
        let s = ContextState::Ready;
        assert!(s.can_transition_to(&ContextState::Destroyed));
    }

    #[test]
    fn state_busy_to_destroyed() {
        let s = ContextState::Busy;
        assert!(s.can_transition_to(&ContextState::Destroyed));
    }

    #[test]
    fn state_error_to_destroyed() {
        let s = ContextState::Error(String::from("oops"));
        assert!(s.can_transition_to(&ContextState::Destroyed));
    }

    #[test]
    fn state_error_to_initializing() {
        let s = ContextState::Error(String::from("retry"));
        assert!(s.can_transition_to(&ContextState::Initializing));
    }

    #[test]
    fn state_invalid_ready_to_uninitialized() {
        let s = ContextState::Ready;
        assert!(!s.can_transition_to(&ContextState::Uninitialized));
    }

    #[test]
    fn state_invalid_destroyed_to_ready() {
        let s = ContextState::Destroyed;
        assert!(!s.can_transition_to(&ContextState::Ready));
    }

    #[test]
    fn state_invalid_uninitialized_to_ready() {
        let s = ContextState::Uninitialized;
        assert!(!s.can_transition_to(&ContextState::Ready));
    }

    #[test]
    fn state_invalid_destroyed_to_initializing() {
        let s = ContextState::Destroyed;
        assert!(!s.can_transition_to(&ContextState::Initializing));
    }

    #[test]
    fn state_display() {
        assert_eq!(ContextState::Ready.to_string(), "Ready");
        assert_eq!(ContextState::Destroyed.to_string(), "Destroyed");
        assert_eq!(ContextState::Error(String::from("fail")).to_string(), "Error(fail)");
    }

    // ── GpuContext tests ──────────────────────────────────────────────

    #[test]
    fn context_new_is_uninitialized() {
        let ctx = GpuContext::new(BackendType::Cuda, 0);
        assert_eq!(*ctx.state(), ContextState::Uninitialized);
        assert_eq!(ctx.backend(), BackendType::Cuda);
        assert_eq!(ctx.device_index(), 0);
    }

    #[test]
    fn context_created_at_is_recent() {
        let ctx = GpuContext::new(BackendType::Cpu, 0);
        let now = now_epoch_secs();
        assert!(ctx.created_at() <= now);
        assert!(now - ctx.created_at() < 5);
    }

    #[test]
    fn context_transition_valid() {
        let mut ctx = GpuContext::new(BackendType::Cuda, 0);
        assert!(ctx.transition(ContextState::Initializing).is_ok());
        assert_eq!(*ctx.state(), ContextState::Initializing);
        assert!(ctx.transition(ContextState::Ready).is_ok());
        assert!(ctx.is_ready());
    }

    #[test]
    fn context_transition_invalid_rejected() {
        let mut ctx = GpuContext::new(BackendType::Cuda, 0);
        ctx.transition(ContextState::Initializing).unwrap();
        ctx.transition(ContextState::Ready).unwrap();
        let err = ctx.transition(ContextState::Uninitialized).unwrap_err();
        assert!(matches!(err, ContextError::InvalidTransition { .. }));
    }

    #[test]
    fn context_initialize_convenience() {
        let mut ctx = GpuContext::new(BackendType::Cpu, 0);
        assert!(ctx.initialize().is_ok());
        assert!(ctx.is_ready());
    }

    #[test]
    fn context_busy_then_ready() {
        let mut ctx = GpuContext::new(BackendType::Cuda, 0);
        ctx.initialize().unwrap();
        ctx.transition(ContextState::Busy).unwrap();
        assert!(ctx.is_busy());
        ctx.transition(ContextState::Ready).unwrap();
        assert!(ctx.is_ready());
    }

    #[test]
    fn context_error_state() {
        let mut ctx = GpuContext::new(BackendType::Cuda, 0);
        ctx.initialize().unwrap();
        ctx.transition(ContextState::Error(String::from("oom"))).unwrap();
        assert!(matches!(ctx.state(), ContextState::Error(_)));
    }

    #[test]
    fn context_destroyed_is_final() {
        let mut ctx = GpuContext::new(BackendType::Cuda, 0);
        ctx.initialize().unwrap();
        ctx.transition(ContextState::Destroyed).unwrap();
        assert!(ctx.is_destroyed());
        let err = ctx.transition(ContextState::Ready).unwrap_err();
        assert!(matches!(err, ContextError::InvalidTransition { .. }));
    }

    #[test]
    fn context_with_properties() {
        let props = ContextProperties {
            device_name: String::from("RTX 4090"),
            compute_capability: Some(String::from("8.9")),
            driver_version: Some(String::from("535.129")),
            max_memory: 24_576 * 1024 * 1024,
            max_threads: 16384,
            supports_fp16: true,
            supports_int8: true,
        };
        let ctx = GpuContext::with_properties(BackendType::Cuda, 0, props);
        assert_eq!(ctx.properties().device_name, "RTX 4090");
        assert_eq!(ctx.properties().compute_capability.as_deref(), Some("8.9"));
        assert!(ctx.properties().supports_fp16);
    }

    #[test]
    fn context_cpu_default_properties() {
        let ctx = GpuContext::new(BackendType::Cpu, 0);
        assert_eq!(ctx.properties().device_name, "CPU");
        assert!(ctx.properties().supports_int8);
    }

    #[test]
    fn context_error_to_reinitialize() {
        let mut ctx = GpuContext::new(BackendType::Cuda, 0);
        ctx.initialize().unwrap();
        ctx.transition(ContextState::Error(String::from("timeout"))).unwrap();
        ctx.transition(ContextState::Initializing).unwrap();
        ctx.transition(ContextState::Ready).unwrap();
        assert!(ctx.is_ready());
    }

    // ── ContextProperties tests ───────────────────────────────────────

    #[test]
    fn properties_default() {
        let p = ContextProperties::default();
        assert_eq!(p.device_name, "Unknown");
        assert!(p.compute_capability.is_none());
        assert_eq!(p.max_memory, 0);
        assert!(!p.supports_fp16);
    }

    #[test]
    fn properties_cpu_default() {
        let p = ContextProperties::cpu_default();
        assert_eq!(p.device_name, "CPU");
        assert!(p.supports_int8);
        assert!(!p.supports_fp16);
    }

    // ── ContextManager tests ──────────────────────────────────────────

    #[test]
    fn manager_new_empty() {
        let mgr = ContextManager::new(ContextManagerConfig::default());
        assert_eq!(mgr.live_count(), 0);
        assert!(mgr.default_context().is_none());
    }

    #[test]
    fn manager_create_context() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(mgr.live_count(), 1);
    }

    #[test]
    fn manager_first_context_is_default() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        assert_eq!(mgr.default_context_index(), Some(idx));
    }

    #[test]
    fn manager_get_context() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        let ctx = mgr.get_context(idx).unwrap();
        assert_eq!(ctx.backend(), BackendType::Cuda);
    }

    #[test]
    fn manager_get_context_nonexistent() {
        let mgr = ContextManager::new(ContextManagerConfig::default());
        assert!(mgr.get_context(99).is_none());
    }

    #[test]
    fn manager_destroy_context() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.destroy_context(idx).unwrap();
        assert!(mgr.get_context(idx).is_none());
        assert_eq!(mgr.live_count(), 0);
    }

    #[test]
    fn manager_destroy_updates_default() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx0 = mgr.create_context(BackendType::Cuda, 0).unwrap();
        let idx1 = mgr.create_context(BackendType::Cpu, 0).unwrap();
        assert_eq!(mgr.default_context_index(), Some(idx0));
        mgr.destroy_context(idx0).unwrap();
        assert_eq!(mgr.default_context_index(), Some(idx1));
    }

    #[test]
    fn manager_destroy_already_destroyed() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.destroy_context(idx).unwrap();
        let err = mgr.destroy_context(idx).unwrap_err();
        assert!(matches!(err, ContextError::Destroyed { .. }));
    }

    #[test]
    fn manager_multiple_contexts() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let i0 = mgr.create_context(BackendType::Cuda, 0).unwrap();
        let i1 = mgr.create_context(BackendType::Cpu, 0).unwrap();
        let i2 = mgr.create_context(BackendType::Vulkan, 0).unwrap();
        assert_eq!(mgr.live_count(), 3);
        assert_ne!(i0, i1);
        assert_ne!(i1, i2);
    }

    #[test]
    fn manager_set_default() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        mgr.create_context(BackendType::Cuda, 0).unwrap();
        let idx1 = mgr.create_context(BackendType::Cpu, 0).unwrap();
        mgr.set_default(idx1).unwrap();
        assert_eq!(mgr.default_context_index(), Some(idx1));
    }

    #[test]
    fn manager_set_default_invalid() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let err = mgr.set_default(42).unwrap_err();
        assert!(matches!(err, ContextError::NotFound { .. }));
    }

    // ── Auto-select tests ─────────────────────────────────────────────

    #[test]
    fn auto_select_prefers_gpu_over_cpu() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let cpu_idx = mgr.create_context(BackendType::Cpu, 0).unwrap();
        let cuda_idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        let best = mgr.auto_select_best().unwrap();
        assert_eq!(best, cuda_idx);
        assert_ne!(best, cpu_idx);
    }

    #[test]
    fn auto_select_falls_back_to_cpu() {
        let mut mgr = ContextManager::new(ContextManagerConfig {
            fallback_to_cpu: true,
            ..Default::default()
        });
        let cpu_idx = mgr.create_context(BackendType::Cpu, 0).unwrap();
        let best = mgr.auto_select_best().unwrap();
        assert_eq!(best, cpu_idx);
    }

    #[test]
    fn auto_select_no_fallback_fails() {
        let mut mgr = ContextManager::new(ContextManagerConfig {
            fallback_to_cpu: false,
            ..Default::default()
        });
        mgr.create_context(BackendType::Cpu, 0).unwrap();
        let err = mgr.auto_select_best().unwrap_err();
        assert!(matches!(err, ContextError::NoBackendAvailable));
    }

    #[test]
    fn auto_select_empty_manager() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let err = mgr.auto_select_best().unwrap_err();
        assert!(matches!(err, ContextError::NoBackendAvailable));
    }

    #[test]
    fn auto_select_prefers_configured_backend() {
        let mut mgr = ContextManager::new(ContextManagerConfig {
            prefer_backend: Some(BackendType::Vulkan),
            ..Default::default()
        });
        mgr.create_context(BackendType::Cuda, 0).unwrap();
        let vk_idx = mgr.create_context(BackendType::Vulkan, 0).unwrap();
        let best = mgr.auto_select_best().unwrap();
        assert_eq!(best, vk_idx);
    }

    #[test]
    fn auto_select_cuda_over_rocm() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        mgr.create_context(BackendType::Rocm, 0).unwrap();
        let cuda_idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        let best = mgr.auto_select_best().unwrap();
        assert_eq!(best, cuda_idx);
    }

    #[test]
    fn auto_select_skips_destroyed() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let cuda_idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        let cpu_idx = mgr.create_context(BackendType::Cpu, 0).unwrap();
        mgr.destroy_context(cuda_idx).unwrap();
        let best = mgr.auto_select_best().unwrap();
        assert_eq!(best, cpu_idx);
    }

    // ── Acquire / release tests ───────────────────────────────────────

    #[test]
    fn acquire_ready_context() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        // Initialize to Ready
        mgr.get_context_mut(idx).unwrap().initialize().unwrap();
        let guard = mgr.acquire(idx).unwrap();
        assert_eq!(guard.context_index(), idx);
        // Context is still accessible but in Busy state
        let ctx = mgr.get_context(idx).unwrap();
        assert!(ctx.is_busy());
        mgr.release(&guard).unwrap();
    }

    #[test]
    fn acquire_busy_context_fails() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.get_context_mut(idx).unwrap().initialize().unwrap();
        let _guard = mgr.acquire(idx).unwrap();
        let err = mgr.acquire(idx).unwrap_err();
        assert!(matches!(err, ContextError::Busy { .. }));
    }

    #[test]
    fn release_returns_to_ready() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.get_context_mut(idx).unwrap().initialize().unwrap();
        let guard = mgr.acquire(idx).unwrap();
        mgr.release(&guard).unwrap();
        let ctx = mgr.get_context(idx).unwrap();
        assert!(ctx.is_ready());
    }

    #[test]
    fn acquire_uninitialized_fails() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        let err = mgr.acquire(idx).unwrap_err();
        assert!(matches!(err, ContextError::InvalidState { .. }));
    }

    #[test]
    fn acquire_destroyed_fails() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.destroy_context(idx).unwrap();
        let err = mgr.acquire(idx).unwrap_err();
        assert!(matches!(err, ContextError::Destroyed { .. }));
    }

    #[test]
    fn guard_acquired_at_is_recent() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.get_context_mut(idx).unwrap().initialize().unwrap();
        let guard = mgr.acquire(idx).unwrap();
        let now = now_epoch_secs();
        assert!(guard.acquired_at() <= now);
        assert!(now - guard.acquired_at() < 5);
        mgr.release(&guard).unwrap();
    }

    // ── Available backends test ────────────────────────────────────────

    #[test]
    fn available_backends_empty() {
        let mgr = ContextManager::new(ContextManagerConfig::default());
        assert!(mgr.available_backends().is_empty());
    }

    #[test]
    fn available_backends_lists_unique() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.create_context(BackendType::Cuda, 1).unwrap();
        mgr.create_context(BackendType::Cpu, 0).unwrap();
        let backends = mgr.available_backends();
        assert_eq!(backends.len(), 2);
        assert!(backends.contains(&BackendType::Cuda));
        assert!(backends.contains(&BackendType::Cpu));
    }

    #[test]
    fn available_backends_excludes_destroyed() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.create_context(BackendType::Cpu, 0).unwrap();
        mgr.destroy_context(idx).unwrap();
        let backends = mgr.available_backends();
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0], BackendType::Cpu);
    }

    #[test]
    fn available_backends_sorted_by_priority() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        mgr.create_context(BackendType::Cpu, 0).unwrap();
        mgr.create_context(BackendType::Vulkan, 0).unwrap();
        mgr.create_context(BackendType::Cuda, 0).unwrap();
        let backends = mgr.available_backends();
        assert_eq!(backends[0], BackendType::Cuda);
        assert_eq!(backends[1], BackendType::Vulkan);
        assert_eq!(backends[2], BackendType::Cpu);
    }

    // ── Config tests ──────────────────────────────────────────────────

    #[test]
    fn config_default() {
        let cfg = ContextManagerConfig::default();
        assert!(!cfg.auto_select);
        assert!(cfg.prefer_backend.is_none());
        assert!(cfg.fallback_to_cpu);
    }

    #[test]
    fn config_custom() {
        let cfg = ContextManagerConfig {
            auto_select: true,
            prefer_backend: Some(BackendType::Metal),
            fallback_to_cpu: false,
        };
        assert!(cfg.auto_select);
        assert_eq!(cfg.prefer_backend, Some(BackendType::Metal));
        assert!(!cfg.fallback_to_cpu);
    }

    #[test]
    fn manager_exposes_config() {
        let cfg =
            ContextManagerConfig { prefer_backend: Some(BackendType::Cuda), ..Default::default() };
        let mgr = ContextManager::new(cfg);
        assert_eq!(mgr.config().prefer_backend, Some(BackendType::Cuda));
    }

    // ── ContextError display tests ────────────────────────────────────

    #[test]
    fn error_display_invalid_transition() {
        let err = ContextError::InvalidTransition {
            from: String::from("Ready"),
            to: String::from("Uninitialized"),
        };
        assert_eq!(err.to_string(), "invalid state transition: Ready -> Uninitialized");
    }

    #[test]
    fn error_display_not_found() {
        let err = ContextError::NotFound { index: 5 };
        assert_eq!(err.to_string(), "context 5 not found");
    }

    #[test]
    fn error_display_busy() {
        let err = ContextError::Busy { index: 3 };
        assert_eq!(err.to_string(), "context 3 is busy");
    }

    #[test]
    fn error_display_destroyed() {
        let err = ContextError::Destroyed { index: 1 };
        assert_eq!(err.to_string(), "context 1 has been destroyed");
    }

    #[test]
    fn error_display_no_backend() {
        let err = ContextError::NoBackendAvailable;
        assert_eq!(err.to_string(), "no suitable backend available");
    }

    // ── Create with properties via manager ────────────────────────────

    #[test]
    fn manager_create_with_properties() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let props = ContextProperties {
            device_name: String::from("A100"),
            max_memory: 80 * 1024 * 1024 * 1024,
            supports_fp16: true,
            ..Default::default()
        };
        let idx = mgr.create_context_with_properties(BackendType::Cuda, 0, props).unwrap();
        let ctx = mgr.get_context(idx).unwrap();
        assert_eq!(ctx.properties().device_name, "A100");
        assert!(ctx.properties().supports_fp16);
    }

    // ── Lifecycle integration test ────────────────────────────────────

    #[test]
    fn full_lifecycle_create_init_use_destroy() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();

        // Initialize
        mgr.get_context_mut(idx).unwrap().initialize().unwrap();
        assert!(mgr.get_context(idx).unwrap().is_ready());

        // Acquire → Busy
        let guard = mgr.acquire(idx).unwrap();

        // Release → Ready
        mgr.release(&guard).unwrap();
        assert!(mgr.get_context(idx).unwrap().is_ready());

        // Destroy
        mgr.destroy_context(idx).unwrap();
        assert!(mgr.get_context(idx).is_none());
        assert_eq!(mgr.live_count(), 0);
    }

    #[test]
    fn lifecycle_error_recovery() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.get_context_mut(idx).unwrap().initialize().unwrap();

        // Simulate error
        mgr.get_context_mut(idx)
            .unwrap()
            .transition(ContextState::Error(String::from("driver crash")))
            .unwrap();

        // Recover
        mgr.get_context_mut(idx).unwrap().transition(ContextState::Initializing).unwrap();
        mgr.get_context_mut(idx).unwrap().transition(ContextState::Ready).unwrap();
        assert!(mgr.get_context(idx).unwrap().is_ready());
    }

    #[test]
    fn destroy_busy_context() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.get_context_mut(idx).unwrap().initialize().unwrap();
        let _guard = mgr.acquire(idx).unwrap();
        // Destroy while busy should still work
        mgr.destroy_context(idx).unwrap();
        assert_eq!(mgr.live_count(), 0);
    }

    #[test]
    fn destroy_error_context() {
        let mut mgr = ContextManager::new(ContextManagerConfig::default());
        let idx = mgr.create_context(BackendType::Cuda, 0).unwrap();
        mgr.get_context_mut(idx).unwrap().initialize().unwrap();
        mgr.get_context_mut(idx)
            .unwrap()
            .transition(ContextState::Error(String::from("fatal")))
            .unwrap();
        mgr.destroy_context(idx).unwrap();
        assert_eq!(mgr.live_count(), 0);
    }
}

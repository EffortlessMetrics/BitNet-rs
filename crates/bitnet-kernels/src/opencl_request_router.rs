//! Inference request router for multi-backend dispatch.
//!
//! Routes incoming inference requests to the optimal backend (OpenCL GPU,
//! CPU SIMD, or hybrid) based on workload characteristics, backend availability,
//! and load.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Backend target for inference execution.
#[derive(Debug, Clone, PartialEq)]
pub enum Backend {
    OpenClGpu { device_id: usize },
    CpuSimd,
    Hybrid { gpu_layers: usize, cpu_layers: usize },
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenClGpu { device_id } => write!(f, "OpenCL GPU (device {device_id})"),
            Self::CpuSimd => write!(f, "CPU SIMD"),
            Self::Hybrid { gpu_layers, cpu_layers } => {
                write!(f, "Hybrid (GPU:{gpu_layers} CPU:{cpu_layers})")
            }
        }
    }
}

/// Priority level of an inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Realtime,
    Interactive,
    Batch,
    Background,
}

/// Characteristics describing a single inference request.
#[derive(Debug, Clone)]
pub struct RequestCharacteristics {
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_tokens: usize,
    pub model_size_mb: usize,
    pub priority: RequestPriority,
}

/// Result of the routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub backend: Backend,
    pub reason: String,
    pub estimated_latency_ms: f64,
    pub estimated_throughput_tps: f64,
}

/// Runtime status snapshot for a registered backend.
#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub backend: Backend,
    pub available: bool,
    pub load_pct: f32,
    pub queue_depth: usize,
    pub avg_latency_ms: f64,
}

/// Configuration knobs for the request router.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub prefer_gpu: bool,
    pub latency_threshold_ms: f64,
    pub throughput_threshold_tps: f64,
    pub fallback_to_cpu: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            latency_threshold_ms: 100.0,
            throughput_threshold_tps: 1.0,
            fallback_to_cpu: true,
        }
    }
}

/// Aggregate routing statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RouterStats {
    pub total_routed: u64,
    pub gpu_routed: u64,
    pub cpu_routed: u64,
    pub hybrid_routed: u64,
    pub fallbacks: u64,
}

/// The request router itself.
#[derive(Debug)]
pub struct RequestRouter {
    pub config: RouterConfig,
    pub backends: Vec<BackendStatus>,
    pub stats: RouterStats,
}

/// Errors that may occur during routing.
#[derive(Debug, Clone, PartialEq)]
pub enum RouterError {
    NoBackendAvailable,
    AllBackendsBusy,
    RequestTooLarge { size_mb: usize, max_mb: usize },
}

impl fmt::Display for RouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBackendAvailable => write!(f, "no backend available"),
            Self::AllBackendsBusy => write!(f, "all backends busy"),
            Self::RequestTooLarge { size_mb, max_mb } => {
                write!(f, "request too large: {size_mb} MB exceeds {max_mb} MB limit")
            }
        }
    }
}

impl std::error::Error for RouterError {}

// ---------------------------------------------------------------------------
// Constants (used by latency estimators)
// ---------------------------------------------------------------------------

/// Maximum model size the router will accept (MB).
const MAX_MODEL_SIZE_MB: usize = 32_768;

/// Base GPU latency for a 1-batch, 1 MB model (ms), includes dispatch overhead.
const GPU_BASE_LATENCY_MS: f64 = 5.0;
/// Per-MB factor for GPU latency.
const GPU_PER_MB_MS: f64 = 0.003;
/// Per-batch-element factor for GPU latency.
const GPU_PER_BATCH_MS: f64 = 0.2;

/// Base CPU latency for a 1-batch, 1 MB model (ms).
const CPU_BASE_LATENCY_MS: f64 = 3.0;
/// Per-MB factor for CPU latency.
const CPU_PER_MB_MS: f64 = 0.02;
/// Per-batch-element factor for CPU latency.
const CPU_PER_BATCH_MS: f64 = 1.0;

/// GPU load threshold above which we consider the device busy.
const BUSY_LOAD_THRESHOLD: f32 = 95.0;

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Create a new [`RequestRouter`] from the given configuration.
pub fn create_request_router(config: RouterConfig) -> RequestRouter {
    RequestRouter { config, backends: Vec::new(), stats: RouterStats::default() }
}

/// Register (or replace) a backend in the router.
pub fn cpu_register_backend(router: &mut RequestRouter, status: BackendStatus) {
    if let Some(existing) = router.backends.iter_mut().find(|b| b.backend == status.backend) {
        *existing = status;
    } else {
        router.backends.push(status);
    }
}

/// Route a request to the best available backend.
pub fn cpu_route_request(
    router: &mut RequestRouter,
    request: &RequestCharacteristics,
) -> Result<RoutingDecision, RouterError> {
    if router.backends.is_empty() {
        return Err(RouterError::NoBackendAvailable);
    }

    if request.model_size_mb > MAX_MODEL_SIZE_MB {
        return Err(RouterError::RequestTooLarge {
            size_mb: request.model_size_mb,
            max_mb: MAX_MODEL_SIZE_MB,
        });
    }

    // Check if every backend is unavailable or busy.
    let any_available =
        router.backends.iter().any(|b| b.available && b.load_pct < BUSY_LOAD_THRESHOLD);
    if !any_available {
        let has_cpu = router.backends.iter().any(|b| matches!(b.backend, Backend::CpuSimd));
        if !(router.config.fallback_to_cpu && has_cpu) {
            return Err(RouterError::AllBackendsBusy);
        }
    }

    // Determine ideal backend: if GPU is preferred, try GPU first.
    let mut fell_back = false;
    let backend = if router.config.prefer_gpu {
        if let Some(gpu) =
            router.backends.iter().find(|b| matches!(b.backend, Backend::OpenClGpu { .. }))
        {
            if gpu.available && gpu.load_pct < BUSY_LOAD_THRESHOLD {
                gpu.backend.clone()
            } else if router.config.fallback_to_cpu {
                fell_back = true;
                cpu_select_best_backend(router, request)
            } else {
                cpu_select_best_backend(router, request)
            }
        } else {
            cpu_select_best_backend(router, request)
        }
    } else {
        cpu_select_best_backend(router, request)
    };

    // Secondary fallback check for non-preferred paths.
    let backend = if !fell_back && cpu_should_fallback(router, &backend) {
        fell_back = true;
        Backend::CpuSimd
    } else {
        backend
    };

    if fell_back {
        router.stats.fallbacks += 1;
    }

    let (latency, throughput) = match &backend {
        Backend::OpenClGpu { .. } => {
            let lat = cpu_estimate_gpu_latency(request);
            (lat, tokens_per_sec(request, lat))
        }
        Backend::CpuSimd => {
            let lat = cpu_estimate_cpu_latency(request);
            (lat, tokens_per_sec(request, lat))
        }
        Backend::Hybrid { gpu_layers, .. } => {
            let lat = cpu_estimate_hybrid_latency(request, *gpu_layers);
            (lat, tokens_per_sec(request, lat))
        }
    };

    let reason = build_reason(&backend, request, &router.config);

    // Update stats.
    router.stats.total_routed += 1;
    match &backend {
        Backend::OpenClGpu { .. } => router.stats.gpu_routed += 1,
        Backend::CpuSimd => router.stats.cpu_routed += 1,
        Backend::Hybrid { .. } => router.stats.hybrid_routed += 1,
    }

    Ok(RoutingDecision {
        backend,
        reason,
        estimated_latency_ms: latency,
        estimated_throughput_tps: throughput,
    })
}

/// Estimate GPU latency for a request (ms).
pub fn cpu_estimate_gpu_latency(request: &RequestCharacteristics) -> f64 {
    GPU_BASE_LATENCY_MS
        + GPU_PER_MB_MS * request.model_size_mb as f64
        + GPU_PER_BATCH_MS * request.batch_size as f64
        + 0.001 * request.seq_len as f64
}

/// Estimate CPU latency for a request (ms).
pub fn cpu_estimate_cpu_latency(request: &RequestCharacteristics) -> f64 {
    CPU_BASE_LATENCY_MS
        + CPU_PER_MB_MS * request.model_size_mb as f64
        + CPU_PER_BATCH_MS * request.batch_size as f64
        + 0.005 * request.seq_len as f64
}

/// Estimate latency for hybrid execution splitting `gpu_layers` to the GPU.
pub fn cpu_estimate_hybrid_latency(request: &RequestCharacteristics, gpu_layers: usize) -> f64 {
    let total_layers = 32_usize; // typical transformer depth
    let gpu_frac = (gpu_layers as f64) / (total_layers as f64);
    let cpu_frac = 1.0 - gpu_frac;
    let gpu_part = cpu_estimate_gpu_latency(request) * gpu_frac;
    let cpu_part = cpu_estimate_cpu_latency(request) * cpu_frac;
    // Overhead for cross-device synchronisation.
    let sync_overhead = 1.5;
    gpu_part + cpu_part + sync_overhead
}

/// Pick the backend with the lowest load-adjusted estimated latency among available ones.
pub fn cpu_select_best_backend(
    router: &RequestRouter,
    request: &RequestCharacteristics,
) -> Backend {
    let available: Vec<&BackendStatus> = router
        .backends
        .iter()
        .filter(|b| b.available && b.load_pct < BUSY_LOAD_THRESHOLD)
        .collect();

    if available.is_empty() {
        return Backend::CpuSimd;
    }

    let mut best: Option<(&BackendStatus, f64)> = None;
    for status in &available {
        let est = estimate_for_backend(&status.backend, request);
        // Quadratic load penalty — high load is disproportionately penalised.
        let load_factor = 1.0 + (status.load_pct as f64 / 50.0).powi(2);
        let adjusted = est * load_factor;
        if best.is_none() || adjusted < best.unwrap().1 {
            best = Some((status, adjusted));
        }
    }

    best.map(|(b, _)| b.backend.clone()).unwrap_or(Backend::CpuSimd)
}

/// Return `true` when the primary choice is not viable and we should fall back.
pub fn cpu_should_fallback(router: &RequestRouter, primary: &Backend) -> bool {
    if !router.config.fallback_to_cpu {
        return false;
    }
    if matches!(primary, Backend::CpuSimd) {
        return false; // already CPU
    }
    let status = router.backends.iter().find(|b| b.backend == *primary);
    match status {
        Some(s) => !s.available || s.load_pct >= BUSY_LOAD_THRESHOLD,
        None => true, // backend not registered → fallback
    }
}

/// Update a backend's average latency after a completed request.
pub fn cpu_update_backend_status(router: &mut RequestRouter, backend: &Backend, latency_ms: f64) {
    if let Some(status) = router.backends.iter_mut().find(|b| b.backend == *backend) {
        // Exponential moving average (α = 0.3).
        status.avg_latency_ms = status.avg_latency_ms * 0.7 + latency_ms * 0.3;
    }
}

/// Return per-backend load distribution `(backend, fraction)`.
pub fn cpu_get_backend_load_balance(router: &RequestRouter) -> Vec<(Backend, f32)> {
    let total: u64 = router.stats.total_routed.max(1);
    router
        .backends
        .iter()
        .map(|b| {
            let count = match &b.backend {
                Backend::OpenClGpu { .. } => router.stats.gpu_routed,
                Backend::CpuSimd => router.stats.cpu_routed,
                Backend::Hybrid { .. } => router.stats.hybrid_routed,
            };
            (b.backend.clone(), count as f32 / total as f32)
        })
        .collect()
}

/// Return a snapshot of the current router stats.
pub fn cpu_get_stats(router: &RequestRouter) -> RouterStats {
    router.stats.clone()
}

/// Produce a human-readable summary of a routing decision.
pub fn format_routing_decision(decision: &RoutingDecision) -> String {
    format!(
        "Routed to {} — latency ~{:.1} ms, throughput ~{:.1} tok/s ({})",
        decision.backend,
        decision.estimated_latency_ms,
        decision.estimated_throughput_tps,
        decision.reason,
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn estimate_for_backend(backend: &Backend, request: &RequestCharacteristics) -> f64 {
    match backend {
        Backend::OpenClGpu { .. } => cpu_estimate_gpu_latency(request),
        Backend::CpuSimd => cpu_estimate_cpu_latency(request),
        Backend::Hybrid { gpu_layers, .. } => cpu_estimate_hybrid_latency(request, *gpu_layers),
    }
}

fn tokens_per_sec(request: &RequestCharacteristics, latency_ms: f64) -> f64 {
    if latency_ms <= 0.0 {
        return 0.0;
    }
    (request.max_tokens as f64) / (latency_ms / 1000.0)
}

fn build_reason(backend: &Backend, req: &RequestCharacteristics, cfg: &RouterConfig) -> String {
    match backend {
        Backend::OpenClGpu { .. } => {
            if cfg.prefer_gpu {
                format!("GPU preferred; model {}MB, batch {}", req.model_size_mb, req.batch_size)
            } else {
                format!("GPU lowest latency; model {}MB", req.model_size_mb)
            }
        }
        Backend::CpuSimd => {
            format!("CPU selected; model {}MB, batch {}", req.model_size_mb, req.batch_size)
        }
        Backend::Hybrid { gpu_layers, cpu_layers } => {
            format!("Hybrid split {gpu_layers}G/{cpu_layers}C; model {}MB", req.model_size_mb)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    fn default_config() -> RouterConfig {
        RouterConfig::default()
    }

    fn gpu_status(device_id: usize, available: bool, load: f32) -> BackendStatus {
        BackendStatus {
            backend: Backend::OpenClGpu { device_id },
            available,
            load_pct: load,
            queue_depth: 0,
            avg_latency_ms: 5.0,
        }
    }

    fn cpu_status(available: bool, load: f32) -> BackendStatus {
        BackendStatus {
            backend: Backend::CpuSimd,
            available,
            load_pct: load,
            queue_depth: 0,
            avg_latency_ms: 10.0,
        }
    }

    fn hybrid_status(gpu_layers: usize, cpu_layers: usize, load: f32) -> BackendStatus {
        BackendStatus {
            backend: Backend::Hybrid { gpu_layers, cpu_layers },
            available: true,
            load_pct: load,
            queue_depth: 0,
            avg_latency_ms: 7.0,
        }
    }

    fn small_request() -> RequestCharacteristics {
        RequestCharacteristics {
            batch_size: 1,
            seq_len: 64,
            max_tokens: 16,
            model_size_mb: 50,
            priority: RequestPriority::Interactive,
        }
    }

    fn large_batch_request() -> RequestCharacteristics {
        RequestCharacteristics {
            batch_size: 32,
            seq_len: 512,
            max_tokens: 256,
            model_size_mb: 4096,
            priority: RequestPriority::Batch,
        }
    }

    fn realtime_request() -> RequestCharacteristics {
        RequestCharacteristics {
            batch_size: 1,
            seq_len: 128,
            max_tokens: 32,
            model_size_mb: 2048,
            priority: RequestPriority::Realtime,
        }
    }

    // -- register backends ------------------------------------------------

    #[test]
    fn test_register_gpu_backend() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        assert_eq!(router.backends.len(), 1);
        assert!(matches!(router.backends[0].backend, Backend::OpenClGpu { device_id: 0 }));
    }

    #[test]
    fn test_register_cpu_backend() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        assert_eq!(router.backends.len(), 1);
        assert!(matches!(router.backends[0].backend, Backend::CpuSimd));
    }

    #[test]
    fn test_register_replaces_existing() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, gpu_status(0, false, 80.0));
        assert_eq!(router.backends.len(), 1);
        assert!(!router.backends[0].available);
    }

    #[test]
    fn test_register_multiple_backends() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        assert_eq!(router.backends.len(), 2);
    }

    // -- route: small request to CPU --------------------------------------

    #[test]
    fn test_route_small_request_to_cpu_when_no_gpu_preference() {
        let mut router =
            create_request_router(RouterConfig { prefer_gpu: false, ..default_config() });
        cpu_register_backend(&mut router, gpu_status(0, true, 50.0));
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let req = small_request();
        let decision = cpu_route_request(&mut router, &req).unwrap();
        // CPU should win for a tiny request when GPU is not preferred.
        assert!(matches!(decision.backend, Backend::CpuSimd));
    }

    #[test]
    fn test_route_small_request_cpu_only() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let req = small_request();
        let decision = cpu_route_request(&mut router, &req).unwrap();
        assert!(matches!(decision.backend, Backend::CpuSimd));
    }

    // -- route: large batch to GPU ----------------------------------------

    #[test]
    fn test_route_large_batch_to_gpu() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, cpu_status(true, 10.0));
        let req = large_batch_request();
        let decision = cpu_route_request(&mut router, &req).unwrap();
        assert!(matches!(decision.backend, Backend::OpenClGpu { .. }));
    }

    #[test]
    fn test_route_large_batch_gpu_preferred() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 20.0));
        cpu_register_backend(&mut router, cpu_status(true, 20.0));
        let req = large_batch_request();
        let decision = cpu_route_request(&mut router, &req).unwrap();
        assert!(matches!(decision.backend, Backend::OpenClGpu { .. }));
    }

    // -- route: realtime priority to fastest ------------------------------

    #[test]
    fn test_route_realtime_to_fastest() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 5.0));
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let req = realtime_request();
        let decision = cpu_route_request(&mut router, &req).unwrap();
        // GPU has lower latency for the model size.
        assert!(matches!(decision.backend, Backend::OpenClGpu { .. }));
    }

    #[test]
    fn test_route_realtime_prefers_lower_load() {
        let mut router =
            create_request_router(RouterConfig { prefer_gpu: false, ..default_config() });
        cpu_register_backend(&mut router, gpu_status(0, true, 80.0));
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        // Use a small model so CPU and GPU raw latencies are close,
        // letting the load penalty on GPU tip the balance to CPU.
        let req = RequestCharacteristics { priority: RequestPriority::Realtime, ..small_request() };
        let decision = cpu_route_request(&mut router, &req).unwrap();
        // CPU has much lower load; load adjustment should tip the scales.
        assert!(matches!(decision.backend, Backend::CpuSimd));
    }

    // -- latency estimates ------------------------------------------------

    #[test]
    fn test_gpu_latency_lower_than_cpu_for_large_model() {
        let req = large_batch_request();
        let gpu_lat = cpu_estimate_gpu_latency(&req);
        let cpu_lat = cpu_estimate_cpu_latency(&req);
        assert!(gpu_lat < cpu_lat, "GPU {gpu_lat} should be < CPU {cpu_lat}");
    }

    #[test]
    fn test_gpu_latency_positive() {
        let req = small_request();
        assert!(cpu_estimate_gpu_latency(&req) > 0.0);
    }

    #[test]
    fn test_cpu_latency_positive() {
        let req = small_request();
        assert!(cpu_estimate_cpu_latency(&req) > 0.0);
    }

    #[test]
    fn test_hybrid_latency_positive() {
        let req = small_request();
        assert!(cpu_estimate_hybrid_latency(&req, 16) > 0.0);
    }

    #[test]
    fn test_gpu_latency_scales_with_model_size() {
        let small = RequestCharacteristics { model_size_mb: 100, ..small_request() };
        let large = RequestCharacteristics { model_size_mb: 4000, ..small_request() };
        assert!(cpu_estimate_gpu_latency(&large) > cpu_estimate_gpu_latency(&small));
    }

    #[test]
    fn test_cpu_latency_scales_with_batch() {
        let b1 = RequestCharacteristics { batch_size: 1, ..small_request() };
        let b32 = RequestCharacteristics { batch_size: 32, ..small_request() };
        assert!(cpu_estimate_cpu_latency(&b32) > cpu_estimate_cpu_latency(&b1));
    }

    // -- select best backend ---------------------------------------------

    #[test]
    fn test_select_best_picks_lowest_latency() {
        let mut router =
            create_request_router(RouterConfig { prefer_gpu: false, ..default_config() });
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, cpu_status(true, 10.0));
        let req = large_batch_request();
        let best = cpu_select_best_backend(&router, &req);
        assert!(matches!(best, Backend::OpenClGpu { .. }));
    }

    #[test]
    fn test_select_best_cpu_when_gpu_loaded() {
        let mut router =
            create_request_router(RouterConfig { prefer_gpu: false, ..default_config() });
        cpu_register_backend(&mut router, gpu_status(0, true, 90.0));
        cpu_register_backend(&mut router, cpu_status(true, 10.0));
        let req = small_request();
        let best = cpu_select_best_backend(&router, &req);
        assert!(matches!(best, Backend::CpuSimd));
    }

    // -- fallback ---------------------------------------------------------

    #[test]
    fn test_fallback_gpu_unavailable_to_cpu() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, false, 0.0));
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let req = large_batch_request();
        let decision = cpu_route_request(&mut router, &req).unwrap();
        assert!(matches!(decision.backend, Backend::CpuSimd));
    }

    #[test]
    fn test_fallback_gpu_busy_to_cpu() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 96.0));
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let req = large_batch_request();
        let decision = cpu_route_request(&mut router, &req).unwrap();
        assert!(matches!(decision.backend, Backend::CpuSimd));
    }

    #[test]
    fn test_no_fallback_when_disabled() {
        let router =
            create_request_router(RouterConfig { fallback_to_cpu: false, ..default_config() });
        let primary = Backend::OpenClGpu { device_id: 0 };
        assert!(!cpu_should_fallback(&router, &primary));
    }

    #[test]
    fn test_should_fallback_unregistered_backend() {
        let router = create_request_router(default_config());
        let primary = Backend::OpenClGpu { device_id: 99 };
        assert!(cpu_should_fallback(&router, &primary));
    }

    // -- load balancing ---------------------------------------------------

    #[test]
    fn test_load_balance_empty() {
        let router = create_request_router(default_config());
        let balance = cpu_get_backend_load_balance(&router);
        assert!(balance.is_empty());
    }

    #[test]
    fn test_load_balance_after_routing() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, cpu_status(true, 10.0));
        let req = large_batch_request();
        let _ = cpu_route_request(&mut router, &req);
        let balance = cpu_get_backend_load_balance(&router);
        assert_eq!(balance.len(), 2);
        let total_frac: f32 = balance.iter().map(|(_, f)| f).sum();
        assert!((total_frac - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_load_balance_spreads_across_backends() {
        let mut router =
            create_request_router(RouterConfig { prefer_gpu: false, ..default_config() });
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, cpu_status(true, 10.0));
        // Route several requests to populate stats.
        for _ in 0..10 {
            let _ = cpu_route_request(&mut router, &large_batch_request());
            let _ = cpu_route_request(&mut router, &small_request());
        }
        let balance = cpu_get_backend_load_balance(&router);
        // Both backends should have received some share.
        for (_, frac) in &balance {
            assert!(*frac > 0.0, "each backend should have > 0 share");
        }
    }

    // -- hybrid routing ---------------------------------------------------

    #[test]
    fn test_hybrid_routing_splits_layers() {
        let mut router =
            create_request_router(RouterConfig { prefer_gpu: false, ..default_config() });
        cpu_register_backend(&mut router, hybrid_status(20, 12, 10.0));
        let req = large_batch_request();
        let decision = cpu_route_request(&mut router, &req).unwrap();
        assert!(matches!(decision.backend, Backend::Hybrid { .. }));
    }

    #[test]
    fn test_hybrid_latency_between_gpu_and_cpu() {
        let req = large_batch_request();
        let gpu_lat = cpu_estimate_gpu_latency(&req);
        let cpu_lat = cpu_estimate_cpu_latency(&req);
        let hybrid_lat = cpu_estimate_hybrid_latency(&req, 24);
        // Hybrid should sit between the two extremes (roughly).
        assert!(hybrid_lat > gpu_lat, "hybrid {hybrid_lat} > gpu {gpu_lat}");
        assert!(hybrid_lat < cpu_lat + 10.0, "hybrid {hybrid_lat} < cpu {cpu_lat}+10");
    }

    // -- edge: no backends → error ----------------------------------------

    #[test]
    fn test_no_backends_error() {
        let mut router = create_request_router(default_config());
        let req = small_request();
        let err = cpu_route_request(&mut router, &req).unwrap_err();
        assert_eq!(err, RouterError::NoBackendAvailable);
    }

    #[test]
    fn test_no_backends_error_display() {
        assert_eq!(RouterError::NoBackendAvailable.to_string(), "no backend available");
    }

    // -- edge: all backends busy → error ----------------------------------

    #[test]
    fn test_all_backends_busy_error() {
        let mut router =
            create_request_router(RouterConfig { fallback_to_cpu: false, ..default_config() });
        cpu_register_backend(&mut router, gpu_status(0, true, 99.0));
        let req = small_request();
        let err = cpu_route_request(&mut router, &req).unwrap_err();
        assert_eq!(err, RouterError::AllBackendsBusy);
    }

    #[test]
    fn test_all_backends_unavailable_error() {
        let mut router =
            create_request_router(RouterConfig { fallback_to_cpu: false, ..default_config() });
        cpu_register_backend(&mut router, gpu_status(0, false, 0.0));
        let req = small_request();
        let err = cpu_route_request(&mut router, &req).unwrap_err();
        assert_eq!(err, RouterError::AllBackendsBusy);
    }

    // -- edge: request too large ------------------------------------------

    #[test]
    fn test_request_too_large() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let req = RequestCharacteristics { model_size_mb: 100_000, ..small_request() };
        let err = cpu_route_request(&mut router, &req).unwrap_err();
        assert!(matches!(err, RouterError::RequestTooLarge { .. }));
    }

    // -- stats: correct counts --------------------------------------------

    #[test]
    fn test_stats_initial_zeros() {
        let router = create_request_router(default_config());
        let stats = cpu_get_stats(&router);
        assert_eq!(stats, RouterStats::default());
    }

    #[test]
    fn test_stats_incremented_on_route() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, cpu_status(true, 10.0));
        let _ = cpu_route_request(&mut router, &large_batch_request());
        let stats = cpu_get_stats(&router);
        assert_eq!(stats.total_routed, 1);
    }

    #[test]
    fn test_stats_gpu_count() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        let _ = cpu_route_request(&mut router, &large_batch_request());
        let stats = cpu_get_stats(&router);
        assert_eq!(stats.gpu_routed, 1);
    }

    #[test]
    fn test_stats_cpu_count() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let _ = cpu_route_request(&mut router, &small_request());
        let stats = cpu_get_stats(&router);
        assert_eq!(stats.cpu_routed, 1);
    }

    #[test]
    fn test_stats_fallback_counted() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, false, 0.0));
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let _ = cpu_route_request(&mut router, &large_batch_request());
        let stats = cpu_get_stats(&router);
        assert!(stats.fallbacks >= 1);
    }

    // -- property: routed total = gpu + cpu + hybrid ----------------------

    #[test]
    fn test_property_total_equals_sum() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, cpu_status(true, 10.0));
        for _ in 0..20 {
            let _ = cpu_route_request(&mut router, &large_batch_request());
            let _ = cpu_route_request(&mut router, &small_request());
        }
        let stats = cpu_get_stats(&router);
        assert_eq!(stats.total_routed, stats.gpu_routed + stats.cpu_routed + stats.hybrid_routed);
    }

    #[test]
    fn test_property_total_equals_sum_with_hybrid() {
        let mut router =
            create_request_router(RouterConfig { prefer_gpu: false, ..default_config() });
        cpu_register_backend(&mut router, gpu_status(0, true, 10.0));
        cpu_register_backend(&mut router, cpu_status(true, 10.0));
        cpu_register_backend(&mut router, hybrid_status(20, 12, 5.0));
        for _ in 0..10 {
            let _ = cpu_route_request(&mut router, &large_batch_request());
            let _ = cpu_route_request(&mut router, &small_request());
        }
        let stats = cpu_get_stats(&router);
        assert_eq!(stats.total_routed, stats.gpu_routed + stats.cpu_routed + stats.hybrid_routed);
    }

    // -- property: latency estimates > 0 ----------------------------------

    #[test]
    fn test_property_all_latencies_positive() {
        let requests = [small_request(), large_batch_request(), realtime_request()];
        for req in &requests {
            assert!(cpu_estimate_gpu_latency(req) > 0.0);
            assert!(cpu_estimate_cpu_latency(req) > 0.0);
            assert!(cpu_estimate_hybrid_latency(req, 16) > 0.0);
        }
    }

    #[test]
    fn test_property_throughput_positive_in_decision() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let decision = cpu_route_request(&mut router, &small_request()).unwrap();
        assert!(decision.estimated_throughput_tps > 0.0);
        assert!(decision.estimated_latency_ms > 0.0);
    }

    // -- format_routing_decision ------------------------------------------

    #[test]
    fn test_format_routing_decision() {
        let decision = RoutingDecision {
            backend: Backend::CpuSimd,
            reason: "test".to_string(),
            estimated_latency_ms: 10.5,
            estimated_throughput_tps: 42.0,
        };
        let formatted = format_routing_decision(&decision);
        assert!(formatted.contains("CPU SIMD"));
        assert!(formatted.contains("10.5"));
        assert!(formatted.contains("42.0"));
    }

    #[test]
    fn test_format_routing_decision_gpu() {
        let decision = RoutingDecision {
            backend: Backend::OpenClGpu { device_id: 1 },
            reason: "preferred".to_string(),
            estimated_latency_ms: 3.0,
            estimated_throughput_tps: 100.0,
        };
        let formatted = format_routing_decision(&decision);
        assert!(formatted.contains("OpenCL GPU"));
        assert!(formatted.contains("device 1"));
    }

    // -- update_backend_status --------------------------------------------

    #[test]
    fn test_update_backend_status_ema() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        let initial = router.backends[0].avg_latency_ms;
        cpu_update_backend_status(&mut router, &Backend::CpuSimd, 20.0);
        let updated = router.backends[0].avg_latency_ms;
        // EMA: 10*0.7 + 20*0.3 = 13.0
        assert!((updated - (initial * 0.7 + 20.0 * 0.3)).abs() < 0.01);
    }

    #[test]
    fn test_update_backend_status_noop_for_unknown() {
        let mut router = create_request_router(default_config());
        cpu_register_backend(&mut router, cpu_status(true, 5.0));
        cpu_update_backend_status(&mut router, &Backend::OpenClGpu { device_id: 99 }, 20.0);
        // Should not panic, and CPU latency unchanged.
        assert!((router.backends[0].avg_latency_ms - 10.0).abs() < 0.01);
    }

    // -- Backend Display --------------------------------------------------

    #[test]
    fn test_backend_display() {
        assert_eq!(Backend::CpuSimd.to_string(), "CPU SIMD");
        assert_eq!(Backend::OpenClGpu { device_id: 0 }.to_string(), "OpenCL GPU (device 0)");
        assert_eq!(
            Backend::Hybrid { gpu_layers: 20, cpu_layers: 12 }.to_string(),
            "Hybrid (GPU:20 CPU:12)"
        );
    }

    // -- RouterError Display ----------------------------------------------

    #[test]
    fn test_router_error_display() {
        assert_eq!(RouterError::AllBackendsBusy.to_string(), "all backends busy");
        let err = RouterError::RequestTooLarge { size_mb: 100, max_mb: 50 };
        assert!(err.to_string().contains("100 MB"));
    }
}

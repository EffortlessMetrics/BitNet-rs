//! Graceful shutdown implementation for BitNet server

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use tracing::{debug, info, warn};

use crate::concurrency::ConcurrencyManager;

/// Shutdown coordinator for graceful server termination
pub struct ShutdownCoordinator {
    shutdown_flag: Arc<AtomicBool>,
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    pub fn new() -> Self {
        Self { shutdown_flag: Arc::new(AtomicBool::new(false)) }
    }

    /// Get the shutdown flag for middleware
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.shutdown_flag)
    }

    /// Check if shutdown has been initiated
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_flag.load(AtomicOrdering::SeqCst)
    }

    /// Initiate graceful shutdown
    pub fn initiate_shutdown(&self) {
        self.shutdown_flag.store(true, AtomicOrdering::SeqCst);
        info!("Shutdown flag set - new requests will be rejected");
    }

    /// Wait for active requests to drain
    pub async fn drain_requests(
        &self,
        concurrency_manager: &ConcurrencyManager,
        timeout: Duration,
    ) -> Duration {
        let drain_start = Instant::now();
        let poll_interval = Duration::from_millis(100);

        info!("Waiting for active requests to drain (timeout: {:?})", timeout);

        loop {
            let stats = concurrency_manager.get_stats().await;
            let active_requests = stats.active_requests;

            if active_requests == 0 {
                info!("All active requests completed");
                break;
            }

            if drain_start.elapsed() >= timeout {
                warn!(
                    active_requests = active_requests,
                    "Shutdown timeout exceeded - proceeding with {} active requests",
                    active_requests
                );
                break;
            }

            debug!(
                active_requests = active_requests,
                elapsed = ?drain_start.elapsed(),
                "Waiting for active requests to complete"
            );

            tokio::time::sleep(poll_interval).await;
        }

        drain_start.elapsed()
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Shutdown check middleware - reject new requests during shutdown
pub async fn shutdown_check_middleware(
    State(shutdown_flag): State<Arc<AtomicBool>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check if server is shutting down
    if shutdown_flag.load(AtomicOrdering::SeqCst) {
        warn!(
            path = %request.uri().path(),
            "Request rejected - server is shutting down"
        );
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    Ok(next.run(request).await)
}

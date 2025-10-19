//! Example demonstrating health endpoint usage
//!
//! This example shows how to integrate health endpoints into a BitNet server
//! and query them for monitoring purposes.
//!
//! Run with:
//! ```bash
//! cargo run -p bitnet-server --no-default-features --features cpu --example health_endpoints
//! ```

use anyhow::Result;
use bitnet_server::monitoring::{
    MonitoringConfig,
    health::{HealthChecker, create_health_routes},
    metrics::MetricsCollector,
};
use std::sync::Arc;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("Starting BitNet health endpoints example...\n");

    // Create monitoring configuration
    let config = MonitoringConfig::default();

    // Initialize metrics collector
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    println!("✓ Metrics collector initialized");

    // Initialize health checker
    let health_checker = Arc::new(HealthChecker::new(metrics));
    println!("✓ Health checker initialized");

    // Create health routes
    let app = create_health_routes(health_checker);
    println!("✓ Health routes created");

    // Start server
    let addr = "127.0.0.1:8080";
    let listener = TcpListener::bind(addr).await?;
    println!("\n🚀 Server running on http://{}", addr);
    println!("\nAvailable endpoints:");
    println!("  - http://{}/health       - Comprehensive health check", addr);
    println!("  - http://{}/health/live  - Liveness probe (fast)", addr);
    println!("  - http://{}/health/ready - Readiness probe", addr);

    println!("\nExample queries:");
    println!("  curl http://{}/health | jq", addr);
    println!("  curl http://{}/health/live", addr);
    println!("  curl http://{}/health/ready", addr);

    println!("\nPress Ctrl+C to stop\n");

    // Serve
    axum::serve(listener, app).await?;

    Ok(())
}

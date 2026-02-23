//! BitNet server binary with comprehensive monitoring

use anyhow::Result;
use bitnet_server::{BitNetServer, ServerConfig};
use bitnet_startup_contract_guard::{ContractPolicy, RuntimeComponent, evaluate_and_emit};
use clap::Parser;
use tracing::info;

#[derive(Parser)]
#[command(name = "bitnet-server")]
#[command(about = "BitNet inference server with monitoring")]
struct Args {
    /// Server host address
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Server port
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Model path
    #[arg(long, required = true)]
    model: String,

    /// Tokenizer path
    #[arg(long)]
    tokenizer: Option<String>,

    /// Device to use (cpu, cuda)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Enable Prometheus metrics
    #[arg(long, default_value = "true")]
    prometheus: bool,

    /// Prometheus metrics endpoint path
    #[arg(long, default_value = "/metrics")]
    prometheus_path: String,

    /// Enable OpenTelemetry tracing
    #[arg(long)]
    opentelemetry: bool,

    /// OpenTelemetry endpoint URL
    #[arg(long)]
    opentelemetry_endpoint: Option<String>,

    /// Health check endpoint path
    #[arg(long, default_value = "/health")]
    health_path: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Log format (json, pretty, compact)
    #[arg(long, default_value = "json")]
    log_format: String,

    /// Metrics collection interval in seconds
    #[arg(long, default_value = "10")]
    metrics_interval: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let startup_contract_report =
        evaluate_and_emit(RuntimeComponent::Server, ContractPolicy::Observe)?;
    if !startup_contract_report.is_compatible() {
        tracing::warn!(component = ?RuntimeComponent::Server, "Server startup contract reported issues");
    }

    // Create server configuration
    let mut config = ServerConfig::default();

    // Override server settings
    config.server.host = args.host;
    config.server.port = args.port;
    config.server.default_model_path = Some(args.model);
    config.server.default_tokenizer_path = args.tokenizer;

    // Override monitoring settings
    config.monitoring.prometheus_enabled = args.prometheus;
    config.monitoring.prometheus_path = args.prometheus_path;
    config.monitoring.opentelemetry_enabled = args.opentelemetry;
    config.monitoring.opentelemetry_endpoint = args.opentelemetry_endpoint.clone();
    config.monitoring.otlp_endpoint = args.opentelemetry_endpoint;
    config.monitoring.health_path = args.health_path;
    config.monitoring.metrics_interval = args.metrics_interval;
    config.monitoring.log_level = args.log_level;
    config.monitoring.log_format = args.log_format;

    // Create and start server
    let server = BitNetServer::new(config).await?;

    // Set up graceful shutdown
    let server_handle = tokio::spawn(async move {
        if let Err(e) = server.start().await {
            tracing::error!("Server error: {}", e);
        }
    });

    // Wait for shutdown signal
    wait_for_shutdown().await;

    info!("Shutdown signal received, stopping server...");
    server_handle.abort();

    info!("Server stopped");
    Ok(())
}

/// Wait for shutdown signals
async fn wait_for_shutdown() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        signal(SignalKind::terminate()).expect("Failed to install signal handler").recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C");
        },
        _ = terminate => {
            info!("Received SIGTERM");
        },
    }
}

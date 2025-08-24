//! Demonstration of BitNet server monitoring capabilities

#[cfg(all(feature = "examples", feature = "server"))]
use anyhow::Result;
#[cfg(all(feature = "examples", feature = "server"))]
use bitnet_server::monitoring::MonitoringConfig;
#[cfg(all(feature = "examples", feature = "server"))]
use bitnet_server::{BitNetServer, ServerConfig};
#[cfg(all(feature = "examples", feature = "server"))]
use reqwest;
#[cfg(all(feature = "examples", feature = "server"))]
use serde_json::json;
#[cfg(all(feature = "examples", feature = "server"))]
use std::time::Duration;
#[cfg(all(feature = "examples", feature = "server"))]
use tokio::time::sleep;

#[cfg(all(feature = "examples", feature = "server"))]
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for the demo
    tracing_subscriber::fmt().with_env_filter("info").init();

    println!("🚀 Starting BitNet Server Monitoring Demo");

    // Create server configuration with monitoring enabled
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
        monitoring: MonitoringConfig {
            prometheus_enabled: true,
            prometheus_path: "/metrics".to_string(),
            opentelemetry_enabled: false, // Disabled for demo simplicity
            opentelemetry_endpoint: None,
            health_path: "/health".to_string(),
            metrics_interval: 5,
            structured_logging: true,
            log_level: "info".to_string(),
            log_format: "pretty".to_string(),
        },
    };

    // Start server in background
    let server = BitNetServer::new(config).await?;
    let server_handle = tokio::spawn(async move {
        if let Err(e) = server.start().await {
            eprintln!("Server error: {}", e);
        }
    });

    // Wait for server to start
    sleep(Duration::from_secs(2)).await;
    println!("✅ Server started on http://127.0.0.1:8080");

    // Create HTTP client for testing
    let client = reqwest::Client::new();
    let base_url = "http://127.0.0.1:8080";

    // Demonstrate monitoring endpoints
    println!("\n📊 Testing Monitoring Endpoints:");

    // 1. Health check
    println!("1. Health Check:");
    match client.get(&format!("{}/health", base_url)).send().await {
        Ok(response) => {
            println!("   Status: {}", response.status());
            if let Ok(body) = response.text().await {
                println!("   Response: {}", body);
            }
        }
        Err(e) => println!("   Error: {}", e),
    }

    // 2. Liveness probe
    println!("\n2. Liveness Probe:");
    match client.get(&format!("{}/health/live", base_url)).send().await {
        Ok(response) => {
            println!("   Status: {}", response.status());
        }
        Err(e) => println!("   Error: {}", e),
    }

    // 3. Readiness probe
    println!("\n3. Readiness Probe:");
    match client.get(&format!("{}/health/ready", base_url)).send().await {
        Ok(response) => {
            println!("   Status: {}", response.status());
        }
        Err(e) => println!("   Error: {}", e),
    }

    // 4. Prometheus metrics (before load)
    println!("\n4. Prometheus Metrics (baseline):");
    match client.get(&format!("{}/metrics", base_url)).send().await {
        Ok(response) => {
            println!("   Status: {}", response.status());
            if let Ok(body) = response.text().await {
                let lines: Vec<&str> = body.lines().take(10).collect();
                println!("   Sample metrics:");
                for line in lines {
                    if !line.starts_with('#') && !line.is_empty() {
                        println!("     {}", line);
                    }
                }
                println!("     ... (truncated)");
            }
        }
        Err(e) => println!("   Error: {}", e),
    }

    // 5. Generate some load to demonstrate metrics
    println!("\n🔄 Generating Load for Metrics Demo:");
    for i in 1..=5 {
        println!("   Request {}/5", i);

        let request_body = json!({
            "prompt": format!("Test prompt number {}", i),
            "max_tokens": 20 + i * 10,
            "model": "bitnet-test",
            "temperature": 0.7
        });

        match client.post(&format!("{}/inference", base_url)).json(&request_body).send().await {
            Ok(response) => {
                println!("     Status: {}", response.status());
                if let Ok(body) = response.text().await {
                    if let Ok(json_response) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let Some(tokens) = json_response.get("tokens_generated") {
                            println!("     Tokens generated: {}", tokens);
                        }
                        if let Some(tps) = json_response.get("tokens_per_second") {
                            println!("     Tokens/sec: {:.2}", tps);
                        }
                    }
                }
            }
            Err(e) => println!("     Error: {}", e),
        }

        sleep(Duration::from_millis(500)).await;
    }

    // 6. Check metrics after load
    println!("\n📈 Prometheus Metrics (after load):");
    match client.get(&format!("{}/metrics", base_url)).send().await {
        Ok(response) => {
            if let Ok(body) = response.text().await {
                println!("   Looking for BitNet-specific metrics:");
                for line in body.lines() {
                    if line.contains("bitnet_") && !line.starts_with('#') {
                        println!("     {}", line);
                    }
                }
            }
        }
        Err(e) => println!("   Error: {}", e),
    }

    // 7. Final health check
    println!("\n🏥 Final Health Check:");
    match client.get(&format!("{}/health", base_url)).send().await {
        Ok(response) => {
            if let Ok(body) = response.text().await {
                if let Ok(health) = serde_json::from_str::<serde_json::Value>(&body) {
                    println!(
                        "   Overall Status: {}",
                        health.get("status").unwrap_or(&json!("unknown"))
                    );
                    println!(
                        "   Uptime: {} seconds",
                        health.get("uptime_seconds").unwrap_or(&json!(0))
                    );
                    if let Some(metrics) = health.get("metrics") {
                        println!(
                            "   Active Requests: {}",
                            metrics.get("active_requests").unwrap_or(&json!(0))
                        );
                        println!(
                            "   Total Requests: {}",
                            metrics.get("total_requests").unwrap_or(&json!(0))
                        );
                    }
                }
            }
        }
        Err(e) => println!("   Error: {}", e),
    }

    println!("\n✨ Monitoring Demo Complete!");
    println!("\n📋 Summary of Monitoring Features Demonstrated:");
    println!("   ✅ Comprehensive health checks (/health, /health/live, /health/ready)");
    println!("   ✅ Prometheus metrics endpoint (/metrics)");
    println!("   ✅ Request tracking and performance metrics");
    println!("   ✅ Structured logging with request correlation");
    println!("   ✅ Real-time metrics collection during inference");

    println!("\n🔧 Production Features Available:");
    println!("   • OpenTelemetry distributed tracing");
    println!("   • Automatic performance regression detection");
    println!("   • Resource usage monitoring");
    println!("   • Error rate tracking and alerting");
    println!("   • Kubernetes-compatible health probes");
    println!("   • Configurable log formats (JSON, pretty, compact)");

    // Cleanup
    server_handle.abort();
    println!("\n🛑 Server stopped");

    Ok(())
}

#[cfg(not(all(feature = "examples", feature = "server")))]
fn main() {}

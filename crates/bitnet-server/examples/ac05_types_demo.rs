//! AC05 health check types demonstration
//!
//! This example demonstrates the AC05 health check data structures
//! and their JSON serialization/deserialization capabilities.

use bitnet_server::monitoring::ac05_types::{
    Ac05HealthResponse, LivenessResponse, PerformanceIndicators, ReadinessChecks,
    ReadinessResponse, SystemMetrics,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("AC05 Health Check Types Demonstration\n");

    // 1. Liveness Response
    println!("1. Liveness Response:");
    let liveness = LivenessResponse {
        status: "healthy".to_string(),
        timestamp: "2023-12-01T10:30:00Z".to_string(),
    };
    let liveness_json = serde_json::to_string_pretty(&liveness)?;
    println!("{}\n", liveness_json);

    // 2. Readiness Response
    println!("2. Readiness Response:");
    let readiness = ReadinessResponse {
        status: "ready".to_string(),
        timestamp: "2023-12-01T10:30:00Z".to_string(),
        checks: ReadinessChecks {
            model_loaded: true,
            inference_engine_ready: true,
            device_available: true,
            resources_available: true,
        },
    };
    let readiness_json = serde_json::to_string_pretty(&readiness)?;
    println!("{}\n", readiness_json);

    // 3. System Metrics
    println!("3. System Metrics:");
    let system_metrics = SystemMetrics {
        cpu_utilization: 0.65,
        gpu_utilization: 0.78,
        memory_usage_bytes: 6442450944,
        gpu_memory_usage_bytes: Some(2147483648),
        active_requests: 23,
        queue_depth: 5,
    };
    let metrics_json = serde_json::to_string_pretty(&system_metrics)?;
    println!("{}\n", metrics_json);

    // 4. Performance Indicators
    println!("4. Performance Indicators:");
    let performance = PerformanceIndicators {
        avg_response_time_ms: 1245.0,
        requests_per_second: 15.2,
        error_rate: 0.0035,
        sla_compliance: 0.995,
    };
    let performance_json = serde_json::to_string_pretty(&performance)?;
    println!("{}\n", performance_json);

    // 5. Full AC05 Health Response
    println!("5. Complete AC05 Health Response:");
    let health_response = Ac05HealthResponse {
        status: "healthy".to_string(),
        timestamp: "2023-12-01T10:30:00Z".to_string(),
        components: [
            ("model_manager".to_string(), "healthy".to_string()),
            ("execution_router".to_string(), "healthy".to_string()),
            ("batch_engine".to_string(), "healthy".to_string()),
            ("device_monitor".to_string(), "healthy".to_string()),
            ("quantization_engine".to_string(), "healthy".to_string()),
        ]
        .into_iter()
        .collect(),
        system_metrics: system_metrics.clone(),
        performance_indicators: performance.clone(),
    };
    let health_json = serde_json::to_string_pretty(&health_response)?;
    println!("{}\n", health_json);

    // 6. Verify deserialization round-trip
    println!("6. Verifying deserialization round-trip...");
    let deserialized: Ac05HealthResponse = serde_json::from_str(&health_json)?;
    assert_eq!(deserialized.status, "healthy");
    assert_eq!(deserialized.components.len(), 5);
    assert_eq!(deserialized.system_metrics.cpu_utilization, 0.65);
    assert_eq!(deserialized.performance_indicators.sla_compliance, 0.995);
    println!("✓ Round-trip serialization successful!\n");

    println!("All AC05 types demonstrated successfully!");

    Ok(())
}

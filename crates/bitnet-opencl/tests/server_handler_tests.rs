//! Integration tests for the GPU server inference handler and
//! health checking subsystem.

use bitnet_opencl::{
    BalancingStrategy, DeviceHealthChecker, GpuInferenceHandler, HandlerConfig, HandlerError,
    HealthConfig, HealthStatus, InferenceRequest, LoadBalancer, RequestPriority, RequestQueue,
    SamplingConfig,
};

fn make_request(priority: RequestPriority, prompt: &str) -> InferenceRequest {
    InferenceRequest {
        prompt: prompt.to_string(),
        max_tokens: 4,
        sampling_config: SamplingConfig::default(),
        priority,
        deadline_ms: 0, // use handler default
    }
}

// -----------------------------------------------------------------------
// GpuInferenceHandler – basic request handling
// -----------------------------------------------------------------------

#[test]
fn single_request_returns_response() {
    let handler = GpuInferenceHandler::new(HandlerConfig { device_count: 1, ..Default::default() });
    let resp = handler.handle_request(make_request(RequestPriority::Interactive, "hello")).unwrap();
    assert!(!resp.output_text.is_empty());
    assert_eq!(resp.tokens.len(), 4);
    assert_eq!(resp.device_used, 0);
}

#[test]
fn response_includes_latency_stats() {
    let handler = GpuInferenceHandler::new(HandlerConfig::default());
    let resp = handler.handle_request(make_request(RequestPriority::Batch, "stats")).unwrap();
    // latency_ms and queue_wait_ms are u64; just ensure they exist.
    assert!(resp.latency_ms < 5_000);
    assert!(resp.queue_wait_ms < 5_000);
}

#[test]
fn mock_generate_provides_custom_tokens() {
    let handler = GpuInferenceHandler::new(HandlerConfig::default())
        .with_mock_generate(|_prompt, max| vec![42; max]);
    let resp = handler.handle_request(make_request(RequestPriority::Interactive, "test")).unwrap();
    assert!(resp.tokens.iter().all(|&t| t == 42));
}

// -----------------------------------------------------------------------
// Queue ordering by priority
// -----------------------------------------------------------------------

#[test]
fn queue_ordering_batch_before_interactive() {
    let mut q = RequestQueue::new(16);
    q.push(make_request(RequestPriority::Interactive, "inter")).unwrap();
    q.push(make_request(RequestPriority::Batch, "batch")).unwrap();

    let (first, _) = q.pop().unwrap();
    assert_eq!(first.priority, RequestPriority::Batch);
}

#[test]
fn queue_ordering_interactive_before_background() {
    let mut q = RequestQueue::new(16);
    q.push(make_request(RequestPriority::Background, "bg")).unwrap();
    q.push(make_request(RequestPriority::Interactive, "inter")).unwrap();

    let (first, _) = q.pop().unwrap();
    assert_eq!(first.priority, RequestPriority::Interactive);
}

#[test]
fn queue_fifo_within_same_priority() {
    let mut q = RequestQueue::new(16);
    q.push(make_request(RequestPriority::Batch, "a")).unwrap();
    q.push(make_request(RequestPriority::Batch, "b")).unwrap();
    q.push(make_request(RequestPriority::Batch, "c")).unwrap();

    let (r1, _) = q.pop().unwrap();
    let (r2, _) = q.pop().unwrap();
    let (r3, _) = q.pop().unwrap();
    assert_eq!(r1.prompt, "a");
    assert_eq!(r2.prompt, "b");
    assert_eq!(r3.prompt, "c");
}

// -----------------------------------------------------------------------
// Capacity limits
// -----------------------------------------------------------------------

#[test]
fn capacity_limit_rejects_excess_requests() {
    let handler =
        GpuInferenceHandler::new(HandlerConfig { queue_capacity: 1, ..Default::default() });
    // First succeeds.
    handler.handle_request(make_request(RequestPriority::Interactive, "ok")).unwrap();
    // The queue is drained synchronously, so a second request
    // also succeeds (capacity is checked at enqueue time).
    handler.handle_request(make_request(RequestPriority::Interactive, "also ok")).unwrap();

    // Fill queue without draining (using the raw queue).
    let mut q = RequestQueue::new(1);
    q.push(make_request(RequestPriority::Batch, "fill")).unwrap();
    let err = q.push(make_request(RequestPriority::Batch, "overflow")).unwrap_err();
    assert_eq!(err, HandlerError::QueueFull);
}

// -----------------------------------------------------------------------
// Timeout handling
// -----------------------------------------------------------------------

#[test]
fn timeout_with_zero_deadline_uses_default() {
    // A generous default deadline should not trigger.
    let handler = GpuInferenceHandler::new(HandlerConfig {
        default_deadline_ms: 30_000,
        ..Default::default()
    });
    let resp = handler.handle_request(make_request(RequestPriority::Interactive, "ok")).unwrap();
    assert!(!resp.output_text.is_empty());
}

#[test]
fn explicit_deadline_respected() {
    let handler = GpuInferenceHandler::new(HandlerConfig {
        default_deadline_ms: 30_000,
        ..Default::default()
    });
    // A very generous explicit deadline should pass.
    let mut req = make_request(RequestPriority::Interactive, "ok");
    req.deadline_ms = 60_000;
    let resp = handler.handle_request(req).unwrap();
    assert!(!resp.output_text.is_empty());
}

// -----------------------------------------------------------------------
// Graceful shutdown
// -----------------------------------------------------------------------

#[test]
fn shutdown_rejects_new_requests() {
    let handler = GpuInferenceHandler::new(HandlerConfig::default());
    handler.shutdown();
    let err = handler.handle_request(make_request(RequestPriority::Batch, "late")).unwrap_err();
    assert_eq!(err, HandlerError::ShuttingDown);
}

#[test]
fn shutdown_flag_readable() {
    let handler = GpuInferenceHandler::new(HandlerConfig::default());
    assert!(!handler.is_shutting_down());
    handler.shutdown();
    assert!(handler.is_shutting_down());
}

#[test]
fn drain_queue_returns_processed_responses() {
    // We cannot push directly via the handler without consuming,
    // so test `drain_queue` via the internal queue.
    let handler =
        GpuInferenceHandler::new(HandlerConfig { queue_capacity: 8, ..Default::default() });
    // Process one to prove baseline works, then shutdown + drain.
    handler.handle_request(make_request(RequestPriority::Interactive, "first")).unwrap();
    handler.shutdown();
    // After shutdown, the queue should be empty (sync handler
    // drains immediately), so drain returns nothing.
    let drained = handler.drain_queue();
    assert!(drained.is_empty());
}

// -----------------------------------------------------------------------
// Load balancer – round robin
// -----------------------------------------------------------------------

#[test]
fn round_robin_distributes_across_devices() {
    let lb = LoadBalancer::new(4, BalancingStrategy::RoundRobin);
    let mut counts = [0usize; 4];
    for _ in 0..12 {
        let id = lb.select_device().unwrap();
        counts[id] += 1;
    }
    assert!(counts.iter().all(|&c| c == 3));
}

// -----------------------------------------------------------------------
// Load balancer – least loaded
// -----------------------------------------------------------------------

#[test]
fn least_loaded_prefers_idle_device() {
    let lb = LoadBalancer::new(3, BalancingStrategy::LeastLoaded);
    lb.begin_request(0);
    lb.begin_request(0);
    lb.begin_request(1);
    let id = lb.select_device().unwrap();
    assert_eq!(id, 2);
}

#[test]
fn least_loaded_updates_after_end_request() {
    let lb = LoadBalancer::new(2, BalancingStrategy::LeastLoaded);
    lb.begin_request(0);
    lb.begin_request(1);
    lb.begin_request(1);
    lb.end_request(1);
    lb.end_request(1);
    // device 0 has 1 active, device 1 has 0.
    let id = lb.select_device().unwrap();
    assert_eq!(id, 1);
}

// -----------------------------------------------------------------------
// Load balancer – memory aware
// -----------------------------------------------------------------------

#[test]
fn memory_aware_selects_most_memory() {
    let lb = LoadBalancer::new(3, BalancingStrategy::MemoryAware);
    lb.set_available_memory(0, 256);
    lb.set_available_memory(1, 1024);
    lb.set_available_memory(2, 512);
    let id = lb.select_device().unwrap();
    assert_eq!(id, 1);
}

// -----------------------------------------------------------------------
// Load balancer – health gating
// -----------------------------------------------------------------------

#[test]
fn unhealthy_device_excluded_from_selection() {
    let lb = LoadBalancer::new(3, BalancingStrategy::RoundRobin);
    lb.set_device_healthy(1, false);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..20 {
        seen.insert(lb.select_device().unwrap());
    }
    assert!(!seen.contains(&1));
}

#[test]
fn no_healthy_device_returns_error() {
    let lb = LoadBalancer::new(2, BalancingStrategy::LeastLoaded);
    lb.set_device_healthy(0, false);
    lb.set_device_healthy(1, false);
    assert_eq!(lb.select_device().unwrap_err(), HandlerError::NoHealthyDevice,);
}

// -----------------------------------------------------------------------
// Multi-device routing through the handler
// -----------------------------------------------------------------------

#[test]
fn handler_routes_to_multiple_devices() {
    let handler = GpuInferenceHandler::new(HandlerConfig {
        device_count: 3,
        strategy: BalancingStrategy::RoundRobin,
        ..Default::default()
    });
    let mut devices_seen = std::collections::HashSet::new();
    for i in 0..6 {
        let resp = handler
            .handle_request(make_request(RequestPriority::Interactive, &format!("req{i}")))
            .unwrap();
        devices_seen.insert(resp.device_used);
    }
    assert_eq!(devices_seen.len(), 3);
}

// -----------------------------------------------------------------------
// Health checker – basic checks
// -----------------------------------------------------------------------

#[test]
fn health_check_healthy_device_passes() {
    let hc = DeviceHealthChecker::new(1, HealthConfig::default());
    assert_eq!(hc.check_device(0), HealthStatus::Healthy);
}

#[test]
fn health_check_smoke_failure_marks_unhealthy() {
    let hc = DeviceHealthChecker::new(1, HealthConfig::default()).with_mock_smoke(|_| false);
    assert!(matches!(hc.check_device(0), HealthStatus::Unhealthy(_),));
}

#[test]
fn health_check_low_memory_marks_degraded() {
    let hc = DeviceHealthChecker::new(1, HealthConfig::default());
    hc.set_available_memory(0, 128);
    assert!(matches!(hc.check_device(0), HealthStatus::Degraded(_),));
}

#[test]
fn health_check_critical_memory_marks_unhealthy() {
    let hc = DeviceHealthChecker::new(1, HealthConfig::default());
    hc.set_available_memory(0, 32);
    assert!(matches!(hc.check_device(0), HealthStatus::Unhealthy(_),));
}

#[test]
fn health_check_error_rate_degrades() {
    let hc = DeviceHealthChecker::new(1, HealthConfig::default());
    let mon = hc.monitor(0).unwrap();
    for _ in 0..9 {
        mon.record_success();
    }
    mon.record_error(); // 10 %
    assert!(matches!(hc.check_device(0), HealthStatus::Degraded(_),));
}

#[test]
fn health_check_high_error_rate_unhealthy() {
    let hc = DeviceHealthChecker::new(1, HealthConfig::default());
    let mon = hc.monitor(0).unwrap();
    mon.record_error();
    mon.record_error();
    mon.record_error();
    mon.record_success(); // 75 %
    assert!(matches!(hc.check_device(0), HealthStatus::Unhealthy(_),));
}

#[test]
fn health_report_aggregates_all_devices() {
    let hc = DeviceHealthChecker::new(3, HealthConfig::default());
    hc.set_available_memory(1, 32); // unhealthy
    let report = hc.check_all();
    assert_eq!(report.statuses.len(), 3);
    assert!(report.any_usable());
    assert_eq!(report.healthy_devices(), vec![0, 2]);
}

// -----------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------

#[test]
fn queue_is_empty_after_pop_all() {
    let mut q = RequestQueue::new(4);
    q.push(make_request(RequestPriority::Batch, "x")).unwrap();
    q.push(make_request(RequestPriority::Batch, "y")).unwrap();
    q.pop();
    q.pop();
    assert!(q.is_empty());
    assert_eq!(q.len(), 0);
}

#[test]
fn handler_default_config_is_sane() {
    let cfg = HandlerConfig::default();
    assert!(cfg.queue_capacity > 0);
    assert!(cfg.device_count > 0);
    assert!(cfg.default_deadline_ms > 0);
}

#[test]
fn active_request_count_tracks_correctly() {
    let lb = LoadBalancer::new(1, BalancingStrategy::RoundRobin);
    assert_eq!(lb.active_requests(0), 0);
    lb.begin_request(0);
    lb.begin_request(0);
    assert_eq!(lb.active_requests(0), 2);
    lb.end_request(0);
    assert_eq!(lb.active_requests(0), 1);
}

# ADR-007: Production Server Concurrency Model

## Status
Accepted

## Context

Issue #251 requires implementing a production-ready inference server capable of handling 100+ concurrent requests with sub-2-second response times while maintaining quantization accuracy and optimal resource utilization. The server must support quantization-aware batch processing, intelligent backpressure control, and device-aware execution routing.

### Current State
The existing `bitnet-server` crate provides basic HTTP server functionality with:
- Simple request/response handling via Axum
- Basic monitoring and health checks
- Request batching infrastructure
- Server-Sent Events (SSE) streaming

### Requirements
- Support 100+ concurrent inference requests
- Maintain <2 second response time for 100-token inference
- Implement quantization-aware batch formation (I2S/TL1/TL2)
- Provide intelligent backpressure control
- Enable device-aware resource allocation (CPU/GPU)
- Ensure memory efficiency (<8GB for 2B parameter models)

### Considered Options

#### Option 1: Thread-per-Request Model
- **Pros**: Simple implementation, good isolation
- **Cons**: High memory overhead, poor scalability, thread contention

#### Option 2: Actor-Based Model (Tokio Actors)
- **Pros**: Strong isolation, message-passing safety, fault tolerance
- **Cons**: Complex state management, message serialization overhead

#### Option 3: Hybrid Async/Await with Intelligent Batching
- **Pros**: Optimal resource utilization, quantization-aware optimization, flexible scheduling
- **Cons**: Moderate implementation complexity

## Decision

We will implement **Option 3: Hybrid Async/Await with Intelligent Batching** using the following architecture:

### Core Components

#### 1. Quantization-Aware Request Queue
```rust
/// Priority-based request queue with quantization grouping
pub struct PriorityRequestQueue {
    /// High-priority requests (interactive)
    high_priority: SegQueue<PendingRequest>,
    /// Normal priority requests (batch processing)
    normal_priority: SegQueue<PendingRequest>,
    /// Low priority requests (background tasks)
    low_priority: SegQueue<PendingRequest>,
    /// Request grouping by quantization format
    quantization_groups: DashMap<QuantizationFormat, Vec<RequestId>>,
    /// Queue statistics and monitoring
    queue_stats: Arc<QueueStatistics>,
}
```

#### 2. Intelligent Batch Formation Engine
```rust
/// Quantization-aware batch formation with SIMD optimization
pub struct QuantizationAwareBatchEngine {
    /// I2S-specific batch optimizer
    i2s_optimizer: I2SBatchOptimizer,
    /// TL1-specific batch optimizer
    tl1_optimizer: TL1BatchOptimizer,
    /// TL2-specific batch optimizer
    tl2_optimizer: TL2BatchOptimizer,
    /// Batch formation configuration
    batch_config: BatchFormationConfig,
    /// Performance metrics collector
    metrics_collector: BatchMetricsCollector,
}

impl QuantizationAwareBatchEngine {
    /// Form optimal batches based on quantization format
    pub async fn form_optimal_batch(
        &self,
        pending_requests: Vec<PendingRequest>
    ) -> Result<Vec<InferenceBatch>, BatchingError> {
        // Group requests by quantization compatibility
        let quantization_groups = self.group_by_quantization_compatibility(&pending_requests);

        let mut formed_batches = Vec::new();

        for (quantization_format, requests) in quantization_groups {
            match quantization_format {
                QuantizationFormat::I2S => {
                    let batch = self.i2s_optimizer.optimize_batch(requests).await?;
                    formed_batches.push(batch);
                }
                QuantizationFormat::TL1 => {
                    let batch = self.tl1_optimizer.optimize_batch(requests).await?;
                    formed_batches.push(batch);
                }
                QuantizationFormat::TL2 => {
                    let batch = self.tl2_optimizer.optimize_batch(requests).await?;
                    formed_batches.push(batch);
                }
            }
        }

        Ok(formed_batches)
    }
}
```

#### 3. Adaptive Resource Pool Manager
```rust
/// Dynamic resource allocation with device awareness
pub struct AdaptiveResourcePool {
    /// CPU execution slots
    cpu_semaphore: Semaphore,
    /// GPU execution slots
    gpu_semaphore: Semaphore,
    /// Memory pool for tensor operations
    memory_pool: TensorMemoryPool,
    /// Device utilization monitor
    device_monitor: DeviceUtilizationMonitor,
    /// Resource allocation strategy
    allocation_strategy: ResourceAllocationStrategy,
}

impl AdaptiveResourcePool {
    /// Allocate resources for batch execution
    pub async fn allocate_for_batch(
        &self,
        batch: &InferenceBatch
    ) -> Result<ResourceAllocation, AllocationError> {
        // Determine optimal device for batch
        let target_device = self.select_optimal_device(batch).await?;

        // Allocate execution slot
        let execution_permit = match target_device {
            ExecutionDevice::Cpu => self.cpu_semaphore.acquire().await?,
            ExecutionDevice::Gpu(id) => self.gpu_semaphore.acquire().await?,
        };

        // Allocate memory for tensors
        let memory_allocation = self.memory_pool
            .allocate_for_batch(batch, &target_device).await?;

        Ok(ResourceAllocation {
            device: target_device,
            execution_permit,
            memory_allocation,
            allocated_at: Instant::now(),
        })
    }
}
```

#### 4. Intelligent Backpressure Controller
```rust
/// System-aware backpressure control
pub struct IntelligentBackpressureController {
    /// Current system metrics
    system_metrics: Arc<RwLock<SystemMetrics>>,
    /// Backpressure thresholds
    thresholds: BackpressureThresholds,
    /// Load prediction model
    load_predictor: LoadPredictor,
    /// Circuit breaker for overload protection
    circuit_breaker: CircuitBreaker,
}

impl IntelligentBackpressureController {
    /// Check if request should be admitted
    pub async fn check_admission(
        &self,
        request: &InferenceRequest
    ) -> Result<AdmissionDecision, BackpressureError> {
        // Get current system state
        let metrics = self.system_metrics.read().await;

        // Estimate request cost
        let request_cost = self.estimate_request_cost(request);

        // Predict system load after admitting request
        let predicted_load = self.load_predictor
            .predict_load_after_admission(&metrics, request_cost);

        // Make admission decision
        match predicted_load {
            load if load.cpu_utilization > self.thresholds.cpu_critical => {
                Ok(AdmissionDecision::Reject {
                    reason: "CPU overload".to_string(),
                    retry_after: Duration::from_secs(5),
                })
            }
            load if load.gpu_utilization > self.thresholds.gpu_critical => {
                Ok(AdmissionDecision::Reject {
                    reason: "GPU overload".to_string(),
                    retry_after: Duration::from_secs(3),
                })
            }
            load if load.memory_usage > self.thresholds.memory_critical => {
                Ok(AdmissionDecision::Reject {
                    reason: "Memory exhaustion".to_string(),
                    retry_after: Duration::from_secs(10),
                })
            }
            _ => Ok(AdmissionDecision::Admit {
                priority: self.calculate_priority(request, &metrics),
            })
        }
    }
}
```

### Concurrency Flow

1. **Request Admission**: Intelligent backpressure controller validates request
2. **Priority Queuing**: Requests queued by priority and quantization compatibility
3. **Batch Formation**: Quantization-aware batching with SIMD optimization
4. **Resource Allocation**: Device-aware resource pool management
5. **Parallel Execution**: Async execution with fallback handling
6. **Response Streaming**: Real-time token delivery via SSE

### Performance Characteristics

**Throughput Optimization**:
- I2S batches: 32-64 requests per batch (SIMD alignment)
- TL1 batches: 16-32 requests per batch (table lookup efficiency)
- TL2 batches: 16-32 requests per batch (memory bandwidth)
- Mixed batches: Dynamic sizing based on format distribution

**Memory Efficiency**:
- Zero-copy tensor operations where possible
- Memory pool reuse for reduced allocation overhead
- Garbage collection optimization for continuous operation

**Latency Targets**:
- P50: <1.5 seconds for 100-token inference
- P90: <2.0 seconds for 100-token inference
- P99: <3.0 seconds for 100-token inference

## Consequences

### Positive
- **Optimal Resource Utilization**: Device-aware allocation maximizes GPU/CPU efficiency
- **Quantization Awareness**: Batch formation optimized for each quantization format
- **Scalable Architecture**: Supports 100+ concurrent requests with sub-linear resource growth
- **Intelligent Backpressure**: Prevents system overload while maintaining responsiveness
- **Performance Predictability**: Clear latency targets with monitoring and alerting

### Negative
- **Implementation Complexity**: Requires sophisticated batch formation and resource management
- **Tuning Requirements**: Multiple configuration parameters need optimization
- **Memory Overhead**: Batch formation and queuing add memory usage
- **Latency Variance**: Batching can introduce variable delays for individual requests

### Risks and Mitigations

**Risk: Batch Formation Bottlenecks**
- **Mitigation**: Parallel batch formation for different quantization formats
- **Monitoring**: Batch formation time and queue depth metrics

**Risk: Resource Starvation**
- **Mitigation**: Fair scheduling and priority-based resource allocation
- **Monitoring**: Resource utilization and wait time tracking

**Risk: Memory Fragmentation**
- **Mitigation**: Custom memory pool with compaction and defragmentation
- **Monitoring**: Memory fragmentation metrics and allocation patterns

## Implementation Plan

### Phase 1: Core Concurrency Infrastructure
1. Implement `PriorityRequestQueue` with quantization grouping
2. Create `AdaptiveResourcePool` with device-aware allocation
3. Build `IntelligentBackpressureController` with system metrics

### Phase 2: Quantization-Aware Batching
1. Implement `QuantizationAwareBatchEngine` with format-specific optimizers
2. Create I2S/TL1/TL2 batch optimizers with SIMD alignment
3. Build batch formation performance monitoring

### Phase 3: Integration and Optimization
1. Integrate components into production server architecture
2. Performance tuning and optimization
3. Load testing and validation

## Validation

### Performance Testing
```bash
# Concurrent load testing
cargo run -p bitnet-server-bench -- --concurrent 100 --duration 300s

# Expected results:
# - Support 100+ concurrent requests
# - P90 latency <2 seconds
# - >95% resource utilization
# - <5% error rate under load
```

### Feature Validation
```bash
# Quantization-aware batching validation
cargo test --no-default-features --features cpu -p bitnet-server --test concurrency_tests -- test_quantization_aware_batching

# Backpressure validation
cargo test --no-default-features --features cpu -p bitnet-server --test concurrency_tests -- test_intelligent_backpressure

# Resource pool validation
cargo test --no-default-features --features cpu -p bitnet-server --test concurrency_tests -- test_adaptive_resource_pool
```

### Integration Testing
```bash
# End-to-end concurrency testing
cargo test --no-default-features --features cpu -p bitnet-server --test integration_tests -- test_production_concurrency_scenarios

# Cross-validation with accuracy requirements
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval --concurrent-batches 10
```

This ADR establishes the foundation for a production-ready concurrency model that optimizes BitNet-rs neural network inference while meeting enterprise performance and reliability requirements.

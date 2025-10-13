# Production Inference Server Quickstart

Learn how to set up and run the BitNet.rs production-ready inference server for high-throughput neural network inference with 1-bit quantization.

## Overview

The BitNet.rs production inference server provides enterprise-grade capabilities for deploying BitNet neural network models at scale. It features:

- **Quantization-Aware Processing**: Automatic I2S/TL1/TL2 quantization support with ≥99%/≥98% accuracy
- **Device-Aware Routing**: Intelligent CPU/GPU selection with automatic fallback
- **High Concurrency**: 100+ concurrent request handling with <2 second response times
- **Hot Model Swapping**: Zero-downtime model updates with rollback capabilities
- **Production Monitoring**: Prometheus metrics, health checks, and observability

## Prerequisites

Before starting, ensure you have:

1. **Rust Environment**: MSRV 1.90.0 or higher
2. **BitNet.rs Built**: With appropriate feature flags
3. **Model Files**: GGUF format BitNet models
4. **System Resources**: Minimum 8GB RAM, recommended GPU for acceleration

## Quick Setup

### 1. Build the Production Server

Choose your deployment configuration:

```bash
# CPU-optimized production server
cargo build --no-default-features --release --no-default-features --features "cpu,prometheus"

# GPU-accelerated production server (requires CUDA)
cargo build --no-default-features --release --no-default-features --features "gpu,prometheus"

# Full-featured server with all capabilities
cargo build --no-default-features --release --no-default-features --features "cpu,gpu,prometheus,opentelemetry"
```

### 2. Download a BitNet Model

```bash
# Download a compatible BitNet model
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

# Verify model compatibility
cargo run -p bitnet-cli -- compat-check models/bitnet/model.gguf
```

### 3. Start the Server

```bash
# Start with default configuration
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p bitnet-server -- \
    --host 0.0.0.0 \
    --port 8080 \
    --model-path models/bitnet/model.gguf \
    --tokenizer-path models/bitnet/tokenizer.json
```

The server will start on `http://localhost:8080` with comprehensive monitoring.

## Basic Usage Examples

### Single Inference Request

```bash
curl -X POST http://localhost:8080/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0.7,
    "quantization_preference": "i2s"
  }'
```

**Example Response**:
```json
{
  "text": "The future of AI is bright with advances in neural network quantization enabling efficient deployment.",
  "tokens_generated": 17,
  "inference_time_ms": 890,
  "tokens_per_second": 19.1,
  "device_used": "Cpu",
  "quantization_type": "i2s",
  "batch_size": 1,
  "queue_time_ms": 5
}
```

### Streaming Inference

```bash
curl -X POST http://localhost:8080/v1/inference/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "prompt": "Explain 1-bit neural networks:",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

**Example Streaming Response**:
```
data: {"type": "token", "text": "1-bit", "position": 0}
data: {"type": "token", "text": " neural", "position": 1}
data: {"type": "token", "text": " networks", "position": 2}
data: {"type": "metrics", "tokens_per_second": 25.3}
data: {"type": "complete", "total_tokens": 45, "inference_time_ms": 1780}
```

### Model Management

```bash
# Load additional model
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/another/model.gguf",
    "model_id": "bitnet-large",
    "device": "gpu"
  }'

# List loaded models
curl http://localhost:8080/v1/models

# Get model details
curl http://localhost:8080/v1/models/bitnet-large
```

## Health and Monitoring

### Health Checks

```bash
# Overall server health
curl http://localhost:8080/health

# Kubernetes liveness probe
curl http://localhost:8080/health/live

# Kubernetes readiness probe
curl http://localhost:8080/health/ready
```

### Server Statistics

```bash
# Comprehensive server statistics
curl http://localhost:8080/v1/stats

# Device status and utilization
curl http://localhost:8080/v1/devices
```

**Example Stats Response**:
```json
{
  "server_stats": {
    "uptime_seconds": 3600,
    "total_requests": 1524,
    "active_requests": 3,
    "error_rate": 0.002
  },
  "inference_stats": {
    "avg_tokens_per_second": 28.4,
    "quantization_distribution": {
      "i2s": 0.75,
      "tl1": 0.20,
      "tl2": 0.05
    }
  }
}
```

### Prometheus Metrics

Access metrics at `http://localhost:8080/metrics` for monitoring dashboards:

```
# HELP bitnet_inference_duration_seconds Time spent processing inference requests
# TYPE bitnet_inference_duration_seconds histogram
bitnet_inference_duration_seconds_bucket{quantization_type="i2s",device="cpu",le="0.5"} 245
bitnet_inference_duration_seconds_bucket{quantization_type="i2s",device="cpu",le="1.0"} 892
bitnet_inference_duration_seconds_bucket{quantization_type="i2s",device="cpu",le="2.0"} 1456

# HELP bitnet_tokens_per_second Current token generation rate
# TYPE bitnet_tokens_per_second gauge
bitnet_tokens_per_second{device="cpu",quantization_type="i2s"} 28.4

# HELP bitnet_quantization_accuracy_ratio Current quantization accuracy vs reference
# TYPE bitnet_quantization_accuracy_ratio gauge
bitnet_quantization_accuracy_ratio{quantization_type="i2s"} 0.995
```

## Configuration Options

### Environment Variables

```bash
# Deterministic inference for testing
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

# Default model path
export BITNET_GGUF=/path/to/model.gguf

# Performance tuning
export RAYON_NUM_THREADS=4
```

### Command Line Options

```bash
bitnet-server \
    --host 0.0.0.0 \              # Server bind address
    --port 8080 \                 # Server port
    --model-path model.gguf \     # Default model path
    --tokenizer-path tokenizer.json \  # Tokenizer path
    --max-concurrent-requests 100 \    # Concurrency limit
    --max-batch-size 8 \          # Batch processing limit
    --device-preference auto \    # Device selection: auto|cpu|gpu
    --enable-prometheus \         # Enable metrics export
    --enable-auth                 # Enable JWT authentication
```

## Performance Optimization

### Device Selection

The server automatically detects and optimizes for available hardware:

```bash
# Force CPU inference
curl -X POST http://localhost:8080/v1/inference \
  -d '{"prompt": "Hello", "device_preference": "cpu"}'

# Request GPU acceleration
curl -X POST http://localhost:8080/v1/inference \
  -d '{"prompt": "Hello", "device_preference": "gpu"}'

# Automatic selection (recommended)
curl -X POST http://localhost:8080/v1/inference \
  -d '{"prompt": "Hello", "device_preference": "auto"}'
```

### Batch Processing

Submit multiple requests to leverage automatic batching:

```bash
# High priority request
curl -X POST http://localhost:8080/v1/inference \
  -d '{
    "prompt": "Urgent query",
    "priority": "high",
    "timeout_ms": 1000
  }'

# Normal priority request
curl -X POST http://localhost:8080/v1/inference \
  -d '{
    "prompt": "Standard query",
    "priority": "normal"
  }'
```

### Memory Management

Monitor memory usage for optimal performance:

```bash
# Check memory statistics
curl http://localhost:8080/v1/stats | jq '.device_stats'

# View model memory footprint
curl http://localhost:8080/v1/models | jq '.[].performance_metrics'
```

## Testing Quantization Accuracy

Verify quantization accuracy against reference implementations:

```bash
# Cross-validation test
export BITNET_GGUF=models/bitnet/model.gguf
cargo run -p xtask -- crossval --samples 100

# Quantization accuracy validation
cargo test --no-default-features -p bitnet-quantization --no-default-features --features cpu test_i2s_simd_scalar_parity

# Feature flag validation
cargo run -p xtask -- test-matrix --features "cpu gpu"
```

Expected results:
- **I2S quantization**: ≥99% accuracy vs FP32 reference
- **TL1/TL2 quantization**: ≥98% accuracy vs reference
- **Cross-validation**: Statistical significance with p < 0.01

## Next Steps

Now that you have the production server running:

1. **[Deploy with Docker](../how-to/production-server-docker-deployment.md)** - Containerize for production
2. **[Kubernetes Setup](../how-to/production-server-kubernetes-deployment.md)** - Orchestrated deployment
3. **[Performance Tuning](../performance-tuning.md)** - Optimize for your workload
4. **[Monitor and Scale](../performance-tracking.md)** - Production observability

## Troubleshooting

### Common Issues

**Server won't start**:
```bash
# Check feature flags
cargo build --no-default-features --features cpu

# Verify model compatibility
cargo run -p bitnet-cli -- compat-check model.gguf
```

**High latency responses**:
```bash
# Check device utilization
curl http://localhost:8080/v1/devices

# Monitor batch processing
curl http://localhost:8080/v1/stats | jq '.batch_engine_stats'
```

**GPU not detected**:
```bash
# Verify CUDA installation
nvcc --version

# Check GPU feature build
cargo build --no-default-features --features gpu

# Test GPU functionality
cargo test --no-default-features -p bitnet-kernels --features gpu test_gpu_info_summary
```

For comprehensive troubleshooting, see the [Troubleshooting Guide](../troubleshooting/troubleshooting.md).

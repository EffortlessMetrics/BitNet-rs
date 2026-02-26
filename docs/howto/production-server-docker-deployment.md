# Deploy BitNet-rs Production Inference Server with Docker

This guide shows how to deploy the BitNet-rs production inference server using Docker containers for both CPU and GPU environments.

## Prerequisites

Before deploying, ensure you have:

- **Docker** 20.10 or later installed
- **Docker Compose** 2.0 or later (for multi-service deployments)
- **NVIDIA Container Toolkit** (GPU deployments only)
- **BitNet GGUF model files** downloaded locally
- **8GB RAM minimum** for 2B parameter models
- **CUDA 12.2+** for GPU deployments

## Quick Start

### CPU Deployment

```bash
# Clone repository
git clone https://github.com/EffortlessMetrics/BitNet-rs
cd BitNet-rs

# Download model
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

# Build CPU container
docker build -f infra/docker/Dockerfile.cpu -t bitnet-cpu:latest .

# Run container
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  -e RUST_LOG=info \
  -e BITNET_MODEL_PATH=/app/models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --name bitnet-server \
  bitnet-cpu:latest
```

### GPU Deployment

```bash
# Build GPU container
docker build -f infra/docker/Dockerfile.gpu -t bitnet-gpu:latest .

# Run container with GPU
docker run -d \
  --runtime=nvidia \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  -e RUST_LOG=info \
  -e BITNET_MODEL_PATH=/app/models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  -e CUDA_VISIBLE_DEVICES=0 \
  --name bitnet-server-gpu \
  bitnet-gpu:latest
```

## Building Containers

### CPU Container

The CPU container uses multi-stage builds for optimal size and security:

```bash
# Build with default settings
docker build -f infra/docker/Dockerfile.cpu -t bitnet-cpu:latest .

# Build with build-time optimizations
docker build \
  -f infra/docker/Dockerfile.cpu \
  --build-arg RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
  -t bitnet-cpu:optimized .

# Build specific version
docker build \
  -f infra/docker/Dockerfile.cpu \
  -t bitnet-cpu:1.0.0 \
  .
```

**Container Architecture:**
- **Builder Stage**: Rust 1.75+ with build dependencies
- **Runtime Stage**: Debian Bookworm slim with minimal dependencies
- **Features**: `--no-default-features --features cpu`
- **Binary**: `bitnet-cli` built in release mode
- **Security**: Non-root user, minimal attack surface

### GPU Container

The GPU container builds on NVIDIA CUDA base images:

```bash
# Build with CUDA 12.2
docker build -f infra/docker/Dockerfile.gpu -t bitnet-gpu:latest .

# Build with specific CUDA version
docker build \
  -f infra/docker/Dockerfile.gpu \
  --build-arg CUDA_VERSION=12.4 \
  -t bitnet-gpu:cuda12.4 \
  .

# Build with both CPU and GPU features
docker build \
  -f infra/docker/Dockerfile.gpu \
  -t bitnet-gpu:hybrid .
```

**Container Architecture:**
- **Builder Stage**: `nvidia/cuda:12.2-devel-ubuntu22.04`
- **Runtime Stage**: `nvidia/cuda:12.2-runtime-ubuntu22.04`
- **Features**: `--no-default-features --features cpu,gpu`
- **CUDA Support**: Automatic fallback to CPU if GPU unavailable
- **Binary**: `bitnet-cli` with GPU acceleration

## Configuration

### Environment Variables

BitNet-rs server supports extensive configuration through environment variables:

#### Core Configuration

```bash
# Model and inference configuration
BITNET_MODEL_PATH=/app/models/model.gguf          # Path to GGUF model file
BITNET_TOKENIZER_PATH=/app/models/tokenizer.json  # Optional tokenizer path
BITNET_STRICT_MODE=1                               # Prevent mock inference fallbacks
BITNET_DETERMINISTIC=1                             # Enable deterministic inference
BITNET_SEED=42                                     # Random seed for reproducibility

# Server configuration
RUST_LOG=info                                      # Logging level (error|warn|info|debug|trace)
BITNET_HOST=0.0.0.0                               # Server bind address
BITNET_PORT=8080                                  # Server port
```

#### Performance Tuning

```bash
# CPU configuration
RAYON_NUM_THREADS=4                               # Number of CPU threads
RUSTFLAGS="-C target-cpu=native"                  # CPU-specific optimizations

# GPU configuration
CUDA_VISIBLE_DEVICES=0                            # GPU device selection
BITNET_GPU_MEMORY_FRACTION=0.9                    # GPU memory allocation limit
```

#### Quantization Configuration

```bash
# Quantization format selection
BITNET_QUANTIZATION=i2s                           # i2s (default), tl1, tl2
BITNET_DEVICE=auto                                # auto, cpu, cuda
```

#### Monitoring Configuration

```bash
# Health check configuration
BITNET_HEALTH_CHECK_INTERVAL=30                   # Health check interval (seconds)
BITNET_METRICS_ENABLED=1                          # Enable Prometheus metrics
BITNET_METRICS_PORT=9090                          # Metrics endpoint port
```

### Volume Mounts

Mount model files and configuration as read-only volumes:

```bash
docker run -d \
  -v /path/to/models:/app/models:ro \              # Model files (read-only)
  -v /path/to/config:/app/config:ro \              # Configuration files (read-only)
  -v /path/to/logs:/app/logs \                     # Log output (read-write)
  -v /path/to/cache:/app/cache \                   # Model cache (read-write)
  bitnet-cpu:latest
```

**Volume Purposes:**
- `/app/models`: GGUF model files and tokenizer configurations
- `/app/config`: Server configuration TOML files
- `/app/logs`: Application logs and audit trails
- `/app/cache`: Compiled CUDA kernels and temporary files

## Running Containers

### Production Mode

Run with proper resource limits and health checks:

```bash
docker run -d \
  --name bitnet-production \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  -e BITNET_MODEL_PATH=/app/models/model.gguf \
  -e BITNET_STRICT_MODE=1 \
  -e BITNET_DETERMINISTIC=1 \
  -e RUST_LOG=info \
  --restart unless-stopped \
  --memory 8g \
  --cpus 4.0 \
  --health-cmd="curl -f http://localhost:8080/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  --health-start-period=40s \
  bitnet-cpu:latest
```

### Development Mode

Run with debug logging and interactive access:

```bash
docker run -it \
  --name bitnet-dev \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  -e BITNET_MODEL_PATH=/app/models/model.gguf \
  -e RUST_LOG=debug \
  bitnet-cpu:latest
```

### GPU Mode with Resource Limits

```bash
docker run -d \
  --name bitnet-gpu-production \
  --runtime=nvidia \
  --gpus '"device=0"' \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  -e BITNET_MODEL_PATH=/app/models/model.gguf \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e BITNET_STRICT_MODE=1 \
  -e RUST_LOG=info \
  --restart unless-stopped \
  --memory 16g \
  --cpus 8.0 \
  --health-cmd="curl -f http://localhost:8080/health || exit 1" \
  --health-interval=30s \
  bitnet-gpu:latest
```

## Using Docker Compose

### Basic Deployment

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  bitnet-cpu:
    build:
      context: .
      dockerfile: infra/docker/Dockerfile.cpu
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro
    environment:
      - RUST_LOG=info
      - BITNET_MODEL_PATH=/app/models/model.gguf
      - BITNET_STRICT_MODE=1
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Deploy with:

```bash
docker-compose up -d
```

### Multi-Service Deployment with Monitoring

Use the provided `infra/docker/docker-compose.yml`:

```bash
cd infra/docker

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f bitnet-cpu

# Scale CPU instances
docker-compose up -d --scale bitnet-cpu=3

# Stop all services
docker-compose down
```

**Services Included:**
- `bitnet-cpu`: CPU inference server
- `bitnet-gpu`: GPU inference server (optional)
- `prometheus`: Metrics collection
- `grafana`: Metrics visualization

## Health Checks and Monitoring

### Health Endpoints

The BitNet server provides three health check endpoints:

```bash
# Overall health status
curl http://localhost:8080/health

# Liveness probe (server running)
curl http://localhost:8080/health/live

# Readiness probe (ready for inference)
curl http://localhost:8080/health/ready
```

**Health Check Semantics:**
- `/health`: Overall system health with detailed component status
- `/health/live`: Basic liveness check (200 if server is running)
- `/health/ready`: Readiness check (200 only when ready for inference)

See [docs/health-endpoints.md](/home/steven/code/Rust/BitNet-rs/docs/health-endpoints.md) for detailed health check documentation.

### Prometheus Metrics

Access Prometheus metrics at:

```bash
curl http://localhost:8080/metrics
```

**Key Metrics:**
- `bitnet_inference_requests_total`: Total inference requests
- `bitnet_inference_duration_seconds`: Inference latency histogram
- `bitnet_model_memory_bytes`: Model memory usage
- `bitnet_tokens_generated_total`: Total tokens generated
- `system_cpu_usage_percent`: CPU utilization
- `system_memory_used_bytes`: Memory consumption

### Container Health Monitoring

Monitor container health:

```bash
# Check container health status
docker inspect --format='{{.State.Health.Status}}' bitnet-server

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' bitnet-server

# Monitor resource usage
docker stats bitnet-server
```

## Troubleshooting

### Container Won't Start

**Symptoms**: Container exits immediately after starting

**Solutions**:

```bash
# Check container logs
docker logs bitnet-server

# Check for missing model files
docker run --rm -v $(pwd)/models:/app/models:ro bitnet-cpu:latest ls -la /app/models

# Verify model path
docker run --rm -e BITNET_MODEL_PATH=/app/models/model.gguf bitnet-cpu:latest ls -la $BITNET_MODEL_PATH

# Run interactively to debug
docker run -it --entrypoint /bin/bash bitnet-cpu:latest
```

### Model Loading Failures

**Symptoms**: "Failed to load model" errors in logs

**Solutions**:

```bash
# Verify model file integrity
cargo run -p bitnet-cli -- compat-check models/model.gguf

# Check model file permissions
ls -la models/model.gguf

# Test model loading outside Docker
cargo run -p xtask -- verify --model models/model.gguf

# Enable debug logging
docker run -e RUST_LOG=debug bitnet-cpu:latest
```

### GPU Not Detected

**Symptoms**: Container falls back to CPU despite GPU availability

**Solutions**:

```bash
# Verify NVIDIA Container Toolkit installed
docker run --rm --gpus all nvidia/cuda:12.2-runtime nvidia-smi

# Check GPU visibility
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 bitnet-gpu:latest nvidia-smi

# Verify CUDA version compatibility
docker run --rm --gpus all bitnet-gpu:latest nvcc --version

# Check runtime configuration
docker info | grep -i runtime
```

### High Memory Usage

**Symptoms**: Container using excessive memory or OOM killed

**Solutions**:

```bash
# Set memory limits
docker run --memory 8g --memory-swap 8g bitnet-cpu:latest

# Monitor memory usage
docker stats bitnet-server

# Use smaller model or quantization format
# I2S: 2-bit (smallest), TL1/TL2: 2-bit with table lookup

# Reduce batch size
-e BITNET_BATCH_SIZE=1
```

### Performance Issues

**Symptoms**: Slow inference, low throughput

**Solutions**:

```bash
# Enable CPU optimizations
docker build --build-arg RUSTFLAGS="-C target-cpu=native" -f infra/docker/Dockerfile.cpu .

# Increase CPU allocation
docker run --cpus 8.0 bitnet-cpu:latest

# Use GPU acceleration
docker run --runtime=nvidia --gpus all bitnet-gpu:latest

# Check SIMD support
docker run bitnet-cpu:latest lscpu | grep -i simd

# Enable deterministic mode for consistent performance
-e BITNET_DETERMINISTIC=1 -e BITNET_SEED=42
```

### Health Check Failures

**Symptoms**: Container marked unhealthy by Docker

**Solutions**:

```bash
# Increase startup period
docker run --health-start-period=60s bitnet-cpu:latest

# Check health endpoint manually
docker exec bitnet-server curl -f http://localhost:8080/health

# Verify port binding
docker port bitnet-server

# Check for application errors
docker logs bitnet-server | grep -i error
```

## Security Considerations

### Container Security

**Best Practices**:

1. **Run as non-root user**: Containers use `bitnet` user by default
2. **Read-only model volumes**: Mount models with `:ro` flag
3. **Minimal base images**: Use slim/alpine variants
4. **No privileged mode**: Never use `--privileged`
5. **Network isolation**: Use Docker networks, not host networking
6. **Resource limits**: Always set memory and CPU limits
7. **Security scanning**: Scan images with `docker scan` or Trivy

```bash
# Scan container for vulnerabilities
docker scan bitnet-cpu:latest

# Run with security options
docker run \
  --security-opt=no-new-privileges:true \
  --cap-drop=ALL \
  --read-only \
  --tmpfs /tmp \
  bitnet-cpu:latest
```

### Model Security

**Best Practices**:

1. **Verify model checksums**: Validate GGUF file integrity
2. **Use trusted sources**: Download models from official repositories
3. **Read-only mounts**: Never allow container to modify models
4. **Secure storage**: Encrypt model files at rest
5. **Access control**: Restrict model file permissions

```bash
# Verify model checksum
sha256sum models/model.gguf

# Set strict permissions
chmod 444 models/model.gguf

# Mount as read-only
-v $(pwd)/models:/app/models:ro
```

### Network Security

**Best Practices**:

1. **Use reverse proxy**: Don't expose container ports directly
2. **Enable TLS**: Use HTTPS for production
3. **Rate limiting**: Implement request rate limits
4. **Authentication**: Require API keys or JWT tokens
5. **Firewall rules**: Restrict inbound traffic

```bash
# Use Docker network
docker network create bitnet-net
docker run --network bitnet-net bitnet-cpu:latest

# Bind to localhost only
docker run -p 127.0.0.1:8080:8080 bitnet-cpu:latest
```

## Advanced Configuration

### Multi-Model Deployment

Run multiple models on different ports:

```bash
# Model 1: 2B parameters
docker run -d \
  --name bitnet-2b \
  -p 8080:8080 \
  -e BITNET_MODEL_PATH=/app/models/model-2b.gguf \
  bitnet-cpu:latest

# Model 2: 7B parameters
docker run -d \
  --name bitnet-7b \
  -p 8081:8080 \
  -e BITNET_MODEL_PATH=/app/models/model-7b.gguf \
  --memory 16g \
  bitnet-cpu:latest
```

### Hot Model Swapping

The server supports hot-swapping models without restart:

```bash
# Send model reload signal (requires future API endpoint)
curl -X POST http://localhost:8080/admin/reload-model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/app/models/new-model.gguf"}'
```

### Custom Quantization Formats

Test different quantization formats:

```bash
# I2S (default, best accuracy)
docker run -e BITNET_QUANTIZATION=i2s bitnet-cpu:latest

# TL1 (optimized for ARM/NEON)
docker run -e BITNET_QUANTIZATION=tl1 bitnet-cpu:latest

# TL2 (optimized for x86/AVX2)
docker run -e BITNET_QUANTIZATION=tl2 bitnet-cpu:latest
```

## Integration Examples

### Load Balancer Integration

Example NGINX configuration:

```nginx
upstream bitnet_backend {
    least_conn;
    server localhost:8080 max_fails=3 fail_timeout=30s;
    server localhost:8081 max_fails=3 fail_timeout=30s;
    server localhost:8082 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name bitnet.example.com;

    location / {
        proxy_pass http://bitnet_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://bitnet_backend/health/ready;
        access_log off;
    }
}
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Build and Deploy BitNet Docker

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -f infra/docker/Dockerfile.cpu -t bitnet-cpu:${{ github.sha }} .

      - name: Test container
        run: |
          docker run -d --name test -p 8080:8080 bitnet-cpu:${{ github.sha }}
          sleep 30
          curl -f http://localhost:8080/health || exit 1
          docker stop test

      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag bitnet-cpu:${{ github.sha }} bitnet/bitnet-cpu:latest
          docker push bitnet/bitnet-cpu:latest
```

## Next Steps

- **Kubernetes Deployment**: See [production-server-kubernetes-deployment.md](production-server-kubernetes-deployment.md)
- **Performance Tuning**: See [docs/performance-benchmarking.md](/home/steven/code/Rust/BitNet-rs/docs/performance-benchmarking.md)
- **API Reference**: See [docs/reference/real-model-api-contracts.md](/home/steven/code/Rust/BitNet-rs/docs/reference/real-model-api-contracts.md)
- **GPU Setup**: See [docs/GPU_SETUP.md](/home/steven/code/Rust/BitNet-rs/docs/GPU_SETUP.md)
- **Environment Variables**: See [docs/environment-variables.md](/home/steven/code/Rust/BitNet-rs/docs/environment-variables.md)

## Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **BitNet-rs Repository**: https://github.com/EffortlessMetrics/BitNet-rs
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

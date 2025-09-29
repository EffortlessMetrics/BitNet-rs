# Production Server Docker Deployment

Deploy BitNet.rs inference server using Docker containers for production environments with optimal performance and security.

## Overview

This guide covers containerizing the BitNet.rs production inference server with multi-stage builds, security hardening, and production optimizations.

## Container Strategy

### Multi-Stage Docker Build

Create an optimized production container with minimal runtime dependencies:

```dockerfile
# Multi-stage build for production optimization
FROM rust:1.90-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /workspace
WORKDIR /workspace

# Build with production features
RUN cargo build --release --no-default-features --features "cpu,gpu,prometheus,opentelemetry"

# Production runtime stage
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 bitnet
USER bitnet

# Copy binary and models
COPY --from=builder /workspace/target/release/bitnet-server /usr/local/bin/
COPY --from=builder /workspace/models/ /app/models/

# Set working directory
WORKDIR /app

# Expose server port
EXPOSE 8080

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health/live || exit 1

# Default command
CMD ["bitnet-server", "--host", "0.0.0.0", "--port", "8080"]
```

### CPU-Only Production Image

For CPU-only deployments without GPU dependencies:

```dockerfile
FROM rust:1.90-slim as builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace
WORKDIR /workspace

# CPU-optimized build
RUN cargo build --release --no-default-features --features "cpu,prometheus"

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 bitnet
USER bitnet

COPY --from=builder /workspace/target/release/bitnet-server /usr/local/bin/
COPY --from=builder /workspace/models/ /app/models/

WORKDIR /app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health/live || exit 1

CMD ["bitnet-server"]
```

## Building Container Images

### Build Production Image

```bash
# Build GPU-enabled production image
docker build -f Dockerfile.production -t bitnet/inference-server:latest .

# Build CPU-only image
docker build -f Dockerfile.cpu -t bitnet/inference-server:cpu-latest .

# Build with specific version tag
docker build -f Dockerfile.production -t bitnet/inference-server:v1.0.0 .
```

### Build Arguments and Customization

```dockerfile
FROM rust:1.90-slim as builder

# Build arguments for customization
ARG FEATURES="cpu,gpu,prometheus,opentelemetry"
ARG PROFILE=release
ARG RUST_VERSION=1.90

# Use build arguments
RUN cargo build --${PROFILE} --no-default-features --features "${FEATURES}"
```

Build with custom features:

```bash
# Custom feature build
docker build \
  --build-arg FEATURES="cpu,prometheus" \
  --build-arg PROFILE=release \
  -t bitnet/inference-server:cpu-prod .
```

## Container Configuration

### Environment Variables

Configure the server using environment variables:

```bash
# Start container with configuration
docker run -d \
  --name bitnet-server \
  -p 8080:8080 \
  -e BITNET_DETERMINISTIC=1 \
  -e BITNET_SEED=42 \
  -e RUST_LOG=bitnet_server=info \
  -e RAYON_NUM_THREADS=4 \
  -v /path/to/models:/app/models:ro \
  -v /path/to/config:/app/config:ro \
  bitnet/inference-server:latest
```

### Configuration File Mount

Create a production configuration file:

```yaml
# config/production.yaml
server:
  host: "0.0.0.0"
  port: 8080
  default_model_path: "/app/models/bitnet-2b.gguf"
  default_tokenizer_path: "/app/models/tokenizer.json"

monitoring:
  prometheus_enabled: true
  opentelemetry_enabled: true
  metrics_interval: 10

concurrency:
  max_concurrent_requests: 100
  request_timeout_seconds: 30

batch_engine:
  max_batch_size: 8
  batch_timeout_ms: 50

security:
  require_authentication: false
  max_prompt_length: 4096
  rate_limit_requests_per_minute: 100
```

Mount the configuration:

```bash
docker run -d \
  --name bitnet-server \
  -p 8080:8080 \
  -v /path/to/config/production.yaml:/app/config/server.yaml:ro \
  -v /path/to/models:/app/models:ro \
  bitnet/inference-server:latest \
  --config /app/config/server.yaml
```

## GPU Support Configuration

### NVIDIA GPU Container

For GPU-accelerated inference:

```bash
# Install nvidia-container-toolkit first
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Run with GPU support
docker run -d \
  --name bitnet-gpu-server \
  --gpus all \
  -p 8080:8080 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /path/to/models:/app/models:ro \
  bitnet/inference-server:latest
```

### GPU Memory Limits

Control GPU memory allocation:

```bash
docker run -d \
  --name bitnet-server \
  --gpus '"device=0"' \
  --shm-size=1g \
  -p 8080:8080 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e BITNET_GPU_MEMORY_LIMIT=6GB \
  bitnet/inference-server:latest
```

## Production Deployment Patterns

### Docker Compose Setup

Create a complete production stack:

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  bitnet-server:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8080:8080"
    environment:
      - BITNET_DETERMINISTIC=1
      - BITNET_SEED=42
      - RUST_LOG=bitnet_server=info
      - RAYON_NUM_THREADS=4
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    depends_on:
      - prometheus
      - grafana
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/datasources:/etc/grafana/provisioning/datasources:ro
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - bitnet-server
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

### Load Balancer Configuration

Configure NGINX for load balancing:

```nginx
# nginx/nginx.conf
upstream bitnet_servers {
    server bitnet-server-1:8080 weight=1 max_fails=3 fail_timeout=30s;
    server bitnet-server-2:8080 weight=1 max_fails=3 fail_timeout=30s;
    server bitnet-server-3:8080 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name bitnet.example.com;

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://bitnet_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Inference endpoints
    location /v1/ {
        proxy_pass http://bitnet_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeout for inference
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 10s;

        # Handle streaming responses
        proxy_buffering off;
        proxy_cache off;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        proxy_pass http://bitnet_servers;
    }
}
```

### Horizontal Scaling

Scale with Docker Compose:

```bash
# Scale up to 3 instances
docker-compose -f docker-compose.production.yml up -d --scale bitnet-server=3

# Scale with specific resource limits
docker-compose -f docker-compose.production.yml up -d \
  --scale bitnet-server=5
```

## Security Hardening

### Container Security Best Practices

1. **Non-root User**:
```dockerfile
# Create dedicated user
RUN useradd -r -s /bin/false -u 1000 bitnet
USER bitnet
```

2. **Minimal Base Image**:
```dockerfile
# Use distroless for minimal attack surface
FROM gcr.io/distroless/cc-debian11
COPY --from=builder /workspace/target/release/bitnet-server /usr/local/bin/
```

3. **Read-only Filesystem**:
```bash
docker run -d \
  --name bitnet-server \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  -p 8080:8080 \
  bitnet/inference-server:latest
```

4. **Security Scanning**:
```bash
# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image bitnet/inference-server:latest
```

### Access Control

Configure authentication and authorization:

```bash
# Start with JWT authentication
docker run -d \
  --name bitnet-server \
  -p 8080:8080 \
  -e BITNET_AUTH_ENABLED=true \
  -e BITNET_JWT_SECRET="your-secret-key" \
  -e BITNET_CORS_ALLOWED_ORIGINS="https://your-domain.com" \
  bitnet/inference-server:latest
```

## Monitoring Integration

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'bitnet-server'
    static_configs:
      - targets: ['bitnet-server:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "BitNet Production Server",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          "rate(bitnet_requests_total[5m])"
        ]
      },
      {
        "title": "Inference Latency",
        "targets": [
          "histogram_quantile(0.95, rate(bitnet_inference_duration_seconds_bucket[5m]))"
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          "bitnet_gpu_utilization_ratio * 100"
        ]
      }
    ]
  }
}
```

## Performance Optimization

### Resource Limits

Set appropriate container limits:

```yaml
services:
  bitnet-server:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

### Volume Optimization

Optimize model loading with volumes:

```bash
# Pre-populate model cache
docker volume create bitnet-models
docker run --rm \
  -v bitnet-models:/models \
  -v /host/models:/source:ro \
  alpine cp -r /source/* /models/

# Use cached volume
docker run -d \
  --name bitnet-server \
  -v bitnet-models:/app/models:ro \
  bitnet/inference-server:latest
```

### Memory Management

Configure memory settings:

```bash
docker run -d \
  --name bitnet-server \
  --memory=8g \
  --memory-swap=8g \
  --oom-kill-disable=false \
  -e MALLOC_CONF="dirty_decay_ms:1000,muzzy_decay_ms:1000" \
  bitnet/inference-server:latest
```

## Deployment Verification

### Health Check Validation

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test health endpoints
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready
curl http://localhost:8080/health
```

### Load Testing

```bash
# Install hey for load testing
go install github.com/rakyll/hey@latest

# Test concurrent requests
hey -n 1000 -c 50 -m POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Test inference","max_tokens":10}' \
  http://localhost:8080/v1/inference
```

### Performance Validation

```bash
# Check performance metrics
curl http://localhost:8080/v1/stats | jq '.server_stats'

# Monitor resource usage
docker stats bitnet-server

# Check logs
docker logs -f bitnet-server
```

Expected performance benchmarks:
- **Response Time**: <2 seconds for 100-token inference
- **Throughput**: >20 tokens/second per request
- **Concurrency**: 100+ concurrent requests
- **Memory**: <8GB for 2B parameter models

## Troubleshooting

### Common Issues

**Container fails to start**:
```bash
# Check logs
docker logs bitnet-server

# Verify feature flags
docker run --rm bitnet/inference-server:latest bitnet-server --version
```

**GPU not detected in container**:
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check CUDA in BitNet container
docker run --rm --gpus all bitnet/inference-server:latest \
  bash -c "nvidia-smi && bitnet-server --help"
```

**High memory usage**:
```bash
# Monitor memory
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Adjust memory limits
docker update --memory=6g bitnet-server
```

For comprehensive troubleshooting, see the [Container Troubleshooting Guide](../troubleshooting/docker-deployment-issues.md).
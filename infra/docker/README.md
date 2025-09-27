# BitNet.rs Docker Deployment

This directory contains Docker configurations for deploying BitNet.rs in various environments.

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- 8GB+ RAM recommended
- Models in GGUF format

### Basic Deployment

```bash
# Deploy production environment
./deploy.sh deploy

# Deploy with GPU support
./deploy.sh deploy-gpu

# Deploy cross-validation environment
./deploy.sh crossval
```

## Docker Images

### Primary Images

| Image | Description | Use Case |
|-------|-------------|----------|
| `bitnet-rs:latest` | Production Rust implementation | Primary deployment |
| `bitnet-rs:gpu` | GPU-enabled Rust implementation | High-performance inference |
| `bitnet-legacy:crossval` | Legacy C++ implementation | Cross-validation only |

### Image Features

#### BitNet.rs Primary (`Dockerfile.rust-primary`)
- **Multi-stage build** for minimal image size
- **Security hardened** with non-root user
- **Health checks** built-in
- **Optimized** for production workloads
- **Size**: ~50MB (compressed)

#### BitNet.rs GPU (`Dockerfile.rust-gpu`)
- **CUDA 12.2** runtime support
- **GPU acceleration** for inference
- **Automatic GPU detection**
- **Memory optimization** for GPU workloads
- **Size**: ~800MB (with CUDA runtime)

#### Legacy Cross-validation (`Dockerfile.legacy-crossval`)
- **C++ implementation** for comparison
- **Isolated build** environment
- **Testing purposes only**
- **Not for production** use

## Deployment Configurations

### Production Deployment (`docker-compose.rust-primary.yml`)

Complete production stack with:
- **BitNet.rs server** (primary)
- **Nginx** load balancer with SSL
- **Prometheus** metrics collection
- **Grafana** visualization
- **Redis** caching
- **Jaeger** distributed tracing (optional)
- **Loki + Promtail** log aggregation (optional)

```bash
# Full production deployment
./deploy.sh deploy

# With distributed tracing
./deploy.sh deploy --profile tracing

# With log aggregation
./deploy.sh deploy --profile logging

# GPU-enabled deployment
./deploy.sh deploy --profile gpu
```

### Cross-validation Deployment (`docker-compose.crossval.yml`)

Development and testing stack with:
- **BitNet.rs** implementation
- **Legacy C++** implementation
- **Cross-validation runner**
- **Performance benchmarks**
- **Results visualization**

```bash
# Deploy cross-validation environment
./deploy.sh crossval

# View results
open http://localhost:8082
```

## Configuration

### Directory Structure

```
docker/
├── Dockerfile.rust-primary      # Primary Rust image
├── Dockerfile.rust-gpu          # GPU-enabled image
├── Dockerfile.legacy-crossval   # Legacy comparison image
├── docker-compose.rust-primary.yml  # Production stack
├── docker-compose.crossval.yml      # Cross-validation stack
├── deploy.sh                    # Deployment script
├── config/                      # Configuration files
│   ├── nginx/
│   │   └── nginx.conf          # Nginx configuration
│   ├── prometheus/
│   │   └── prometheus.yml      # Prometheus configuration
│   └── grafana/                # Grafana dashboards
├── models/                     # Model files (GGUF format)
└── logs/                       # Application logs
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BITNET_MODELS_DIR` | Model files directory | `./models` |
| `BITNET_CONFIG_DIR` | Configuration directory | `./config` |
| `GRAFANA_PASSWORD` | Grafana admin password | `admin123` |
| `RUST_LOG` | Rust logging level | `info` |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | `0` |

### Model Configuration

Place your GGUF model files in the `models/` directory:

```bash
models/
├── bitnet-1.58b.gguf           # Small model
├── bitnet-3b.gguf              # Medium model
└── bitnet-7b.gguf              # Large model
```

Update `config/server.toml`:

```toml
[model]
path = "/app/models/bitnet-3b.gguf"
max_tokens = 2048
batch_size = 512
```

## Service URLs

After deployment, services are available at:

| Service | URL | Credentials |
|---------|-----|-------------|
| BitNet API | http://localhost:8080 | - |
| Grafana | http://localhost:3000 | admin/admin123 |
| Prometheus | http://localhost:9090 | - |
| Cross-validation Results | http://localhost:8082 | - |
| Jaeger UI | http://localhost:16686 | - |

## Monitoring and Observability

### Metrics

BitNet.rs exposes Prometheus metrics at `/metrics`:

- **Request metrics**: Rate, latency, errors
- **Inference metrics**: Tokens/second, model performance
- **System metrics**: Memory, CPU, GPU utilization
- **Business metrics**: Model usage, user patterns

### Dashboards

Pre-configured Grafana dashboards:

1. **BitNet Overview** - High-level service metrics
2. **Inference Performance** - Model-specific metrics
3. **System Resources** - Infrastructure monitoring
4. **Cross-validation** - Comparison between implementations

### Alerting

Prometheus alerting rules for:

- **High error rate** (>5%)
- **High latency** (>2s p95)
- **Service downtime**
- **Resource exhaustion**
- **GPU issues** (if applicable)

## Performance Tuning

### CPU Optimization

```toml
# config/server.toml
[performance]
threads = 0  # Auto-detect
async_inference = true
batch_processing = true
memory_mapping = true
```

### GPU Optimization

```toml
# config/server-gpu.toml
[performance]
use_gpu = true
gpu_layers = 32
memory_pool = "8GB"
batch_size = 1024
```

### Container Resources

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

## Scaling

### Horizontal Scaling

Add more BitNet.rs instances:

```yaml
# docker-compose.yml
services:
  bitnet-server-1:
    # ... configuration
  bitnet-server-2:
    # ... configuration
  bitnet-server-3:
    # ... configuration
```

Update Nginx upstream:

```nginx
upstream bitnet_backend {
    server bitnet-server-1:8080;
    server bitnet-server-2:8080;
    server bitnet-server-3:8080;
}
```

### Load Testing

```bash
# Install hey load testing tool
go install github.com/rakyll/hey@latest

# Test inference endpoint
hey -n 1000 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 100}' \
  http://localhost:8080/generate
```

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check logs
./deploy.sh logs bitnet-server

# Check service status
./deploy.sh status

# Restart services
./deploy.sh restart
```

#### Model Loading Issues

```bash
# Verify model files
ls -la models/

# Check model format
file models/your-model.gguf

# Verify configuration
cat config/server.toml
```

#### GPU Not Detected

```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.2-runtime-ubuntu22.04 nvidia-smi
```

#### High Memory Usage

```bash
# Monitor memory usage
docker stats

# Adjust batch size
# Edit config/server.toml
[model]
batch_size = 256  # Reduce from 512
```

### Performance Issues

#### Slow Inference

1. **Check GPU utilization**:
   ```bash
   nvidia-smi
   ```

2. **Optimize batch size**:
   ```toml
   [model]
   batch_size = 1024  # Increase for GPU
   ```

3. **Enable async processing**:
   ```toml
   [performance]
   async_inference = true
   ```

#### High Latency

1. **Check network configuration**
2. **Optimize Nginx settings**
3. **Increase worker processes**
4. **Use connection pooling**

## Security

### Production Security Checklist

- [ ] **Use HTTPS** with valid certificates
- [ ] **Enable API authentication**
- [ ] **Configure firewall rules**
- [ ] **Regular security updates**
- [ ] **Monitor access logs**
- [ ] **Limit resource usage**

### Security Configuration

```toml
# config/server.toml
[security]
api_key_required = true
cors_enabled = false
max_request_size = "10MB"
rate_limiting = true
```

## Backup and Recovery

### Data Backup

```bash
# Backup volumes
docker run --rm -v bitnet-logs:/data -v $(pwd):/backup alpine \
  tar czf /backup/bitnet-logs-$(date +%Y%m%d).tar.gz -C /data .

# Backup configuration
tar czf config-backup-$(date +%Y%m%d).tar.gz config/
```

### Disaster Recovery

```bash
# Stop services
./deploy.sh stop

# Restore data
docker run --rm -v bitnet-logs:/data -v $(pwd):/backup alpine \
  tar xzf /backup/bitnet-logs-20240101.tar.gz -C /data

# Restart services
./deploy.sh deploy
```

## Development

### Building Custom Images

```bash
# Build with custom features
docker build -f Dockerfile.rust-primary \
  --build-arg FEATURES="gpu,metrics,tracing" \
  -t bitnet-rs:custom .
```

### Local Development

```bash
# Mount source code for development
docker run -it --rm \
  -v $(pwd):/app \
  -p 8080:8080 \
  bitnet-rs:latest \
  cargo run --bin bitnet-server
```

## Support

### Getting Help

1. **Check logs**: `./deploy.sh logs`
2. **Review documentation**: This README
3. **Check GitHub issues**: [BitNet Issues](https://github.com/microsoft/BitNet/issues)
4. **Community Discord**: [BitNet Discord](https://discord.gg/bitnet)

### Reporting Issues

When reporting issues, include:

- **Docker version**: `docker --version`
- **Compose version**: `docker-compose --version`
- **System info**: `uname -a`
- **Service logs**: `./deploy.sh logs`
- **Configuration files**: Sanitized config files

---

**Ready to deploy?** Start with `./deploy.sh deploy` for a production-ready BitNet.rs deployment!
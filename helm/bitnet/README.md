# BitNet.rs Helm Chart

This Helm chart deploys BitNet.rs, the high-performance Rust implementation of 1-bit LLM inference engine.

## Overview

BitNet.rs is the **primary, production-ready implementation** of BitNet, offering:

- **Memory Safety**: Rust's ownership system prevents memory leaks and buffer overflows
- **High Performance**: Zero-cost abstractions and SIMD optimizations
- **Reliability**: Comprehensive error handling and graceful degradation
- **Scalability**: Efficient resource utilization and horizontal scaling

## Quick Start

### Install with CPU-only deployment (recommended for most users):

```bash
helm install bitnet ./helm/bitnet
```

### Install with GPU support:

```bash
helm install bitnet ./helm/bitnet --set gpu.enabled=true
```

## Configuration

### Primary Rust Implementation (Default)

The chart deploys the Rust implementation by default, which is recommended for all production workloads.

Key configuration options:

```yaml
# CPU deployment (enabled by default)
cpu:
  enabled: true
  replicaCount: 3
  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 4000m
      memory: 8Gi

# GPU deployment (optional)
gpu:
  enabled: false
  replicaCount: 2
  resources:
    requests:
      nvidia.com/gpu: 1
    limits:
      nvidia.com/gpu: 1
```

### Legacy C++ Support (Not Recommended)

⚠️ **WARNING**: The legacy C++ deployment is provided only for cross-validation and migration purposes. It is **NOT recommended for production use**.

To enable legacy deployment for comparison:

```yaml
legacy:
  enabled: true  # Only for cross-validation
  replicaCount: 1  # Minimal replicas
  service:
    enabled: false  # Disabled by default
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BitNet.rs Deployment                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   CPU Pods  │  │   GPU Pods  │  │  Legacy (Optional)  │  │
│  │   (Rust)    │  │   (Rust)    │  │      (C++)          │  │
│  │             │  │             │  │                     │  │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │   ┌─────────────┐   │  │
│  │ │ bitnet  │ │  │ │ bitnet  │ │  │   │ bitnet-cpp  │   │  │
│  │ │ -rust   │ │  │ │ -rust   │ │  │   │ (legacy)    │   │  │
│  │ └─────────┘ │  │ └─────────┘ │  │   └─────────────┘   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Load Balancer   │
                    │    (Service)      │
                    └───────────────────┘
```

## Values Reference

### Global Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.implementation` | Primary implementation (always rust) | `rust` |
| `global.imageRegistry` | Global image registry override | `""` |

### Image Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.registry` | Image registry | `docker.io` |
| `image.repository` | Image repository | `bitnet/bitnet-rust` |
| `image.tag` | Image tag | `1.0.0` |

### CPU Deployment

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cpu.enabled` | Enable CPU deployment | `true` |
| `cpu.replicaCount` | Number of CPU replicas | `3` |
| `cpu.resources.requests.cpu` | CPU request | `1000m` |
| `cpu.resources.requests.memory` | Memory request | `2Gi` |

### GPU Deployment

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gpu.enabled` | Enable GPU deployment | `false` |
| `gpu.replicaCount` | Number of GPU replicas | `2` |
| `gpu.resources.requests.nvidia.com/gpu` | GPU request | `1` |

### Legacy Deployment (Not Recommended)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `legacy.enabled` | Enable legacy C++ deployment | `false` |
| `legacy.service.enabled` | Enable legacy service | `false` |

## Migration from C++ Implementation

If you're migrating from the legacy C++ implementation:

1. **Start with Rust-only deployment** (recommended):
   ```bash
   helm install bitnet ./helm/bitnet
   ```

2. **For gradual migration** (enable both temporarily):
   ```bash
   helm install bitnet ./helm/bitnet --set legacy.enabled=true
   ```

3. **Compare performance** using the cross-validation tools
4. **Switch traffic** to Rust services
5. **Disable legacy** deployment:
   ```bash
   helm upgrade bitnet ./helm/bitnet --set legacy.enabled=false
   ```

## Monitoring

The chart includes Prometheus metrics endpoints:

- **Rust services**: `/metrics` on port 9090
- **Health checks**: `/health` and `/ready` endpoints
- **Legacy services**: Metrics disabled by default

## Security

- **Non-root containers**: All containers run as non-root user (1000)
- **Read-only root filesystem**: Enhanced security posture
- **Resource limits**: Prevent resource exhaustion
- **Network policies**: Optional network isolation

## Troubleshooting

### Common Issues

1. **Image pull errors**: Ensure the Rust images are available
2. **Resource constraints**: Adjust CPU/memory limits
3. **GPU not available**: Check node selectors and tolerations

### Performance Optimization

1. **Enable SIMD**: Rust implementation includes SIMD optimizations
2. **Tune thread pools**: Configure `config.server.workers`
3. **Memory pools**: Adjust `rust.memory_pool_size`

## Support

- **Primary support**: Rust implementation issues and questions
- **Legacy support**: Limited to migration assistance only
- **Documentation**: See `/docs/` directory for detailed guides

## License

This chart is part of the BitNet.rs project and follows the same license terms.
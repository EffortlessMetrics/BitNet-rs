# bitnet-rs Monitoring and Observability

This directory contains the monitoring and observability stack for bitnet-rs, the primary Rust implementation of 1-bit LLM inference.

## Overview

The monitoring stack is designed with a **Rust-first approach**, providing comprehensive observability for the primary Rust implementation while supporting optional cross-validation monitoring with the legacy C++ implementation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    bitnet-rs Monitoring Stack               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Prometheus  │  │   Grafana   │  │   AlertManager      │  │
│  │ (Metrics)   │  │(Dashboards) │  │   (Alerts)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Node Exporter│  │  cAdvisor   │  │ NVIDIA DCGM (GPU)   │  │
│  │ (System)    │  │(Containers) │  │   (Optional)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   bitnet-rs       │
                    │ (Primary Rust)    │
                    └───────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ BitNet.cpp Legacy │
                    │ (Cross-validation)│
                    └───────────────────┘
```

## Quick Start

### 1. Start the Monitoring Stack

For CPU-only monitoring:
```bash
cd monitoring
docker-compose up -d
```

For GPU monitoring (includes NVIDIA DCGM):
```bash
cd monitoring
docker-compose --profile gpu up -d
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

### 3. Default Dashboards

1. **bitnet-rs Overview** - Primary Rust implementation metrics
2. **Performance Comparison** - Rust vs C++ legacy comparison
3. **Detailed Rust Metrics** - Rust-specific performance insights

## Configuration Files

### Prometheus (`prometheus.yml`)
- **Primary focus**: Rust implementation metrics
- **Secondary**: Legacy C++ metrics for cross-validation
- **Scrape intervals**: 10s for Rust, 30s for legacy
- **Labels**: Automatic implementation tagging

### Alert Rules (`rules/bitnet-rust-alerts.yml`)
- **Rust-focused alerts**: Service health, performance, resources
- **Cross-validation alerts**: Accuracy drift, test failures
- **Business logic alerts**: Model loading, inference queues
- **Severity levels**: Critical, Warning, Info

### AlertManager (`alertmanager.yml`)
- **Routing**: Implementation-based alert routing
- **Receivers**: Separate channels for Rust vs legacy
- **Inhibition**: Rust alerts take priority over legacy
- **Notifications**: Email, Slack, webhooks

## Dashboards

### 1. bitnet-rs Overview
**Purpose**: Primary operational dashboard for Rust implementation
**Metrics**:
- Request rate and latency (Rust-focused)
- Resource utilization (CPU, Memory, GPU)
- System-level metrics (CPU usage, memory usage, disk usage, network I/O)
- Service health indicators and uptime tracking
- Error rates and active connections
- Real-time performance monitoring with sysinfo-based system metrics

### 2. Performance Comparison
**Purpose**: Cross-validation and migration support
**Metrics**:
- Throughput comparison (Rust vs C++)
- Latency percentiles comparison
- Resource efficiency comparison
- Cross-validation test results

### 3. Detailed Rust Metrics
**Purpose**: Deep dive into Rust-specific performance
**Metrics**:
- Memory management (allocations/deallocations)
- SIMD vs scalar operations
- Thread pool utilization
- Garbage collection pauses
- Error handling (panics, Result errors)

## Alert Categories

### Critical Alerts (Immediate Action Required)
- **BitNetRustServiceDown**: Primary service unavailable
- **BitNetCrossValidationFailure**: Compatibility issues detected
- **BitNetRustModelLoadFailure**: Model loading failures

### Warning Alerts (Investigation Needed)
- **BitNetRustHighErrorRate**: Error rate > 10%
- **BitNetRustHighLatency**: P95 latency > 1s
- **BitNetRustHighCPUUsage**: CPU usage > 80%
- **BitNetRustHighMemoryUsage**: Memory usage > 85%
- **BitNetSystemHighCPU**: System CPU usage > 90%
- **BitNetSystemHighMemory**: System memory usage > 95%
- **BitNetSystemHighDisk**: System disk usage > 90%
- **BitNetSystemNetworkSaturated**: Network bytes/sec > threshold

### Performance Alerts
- **BitNetRustLowThroughput**: Throughput < 100 tokens/sec
- **BitNetCrossValidationPerformanceRegression**: Rust < 95% of C++
- **BitNetRustGPUHighUtilization**: GPU usage > 95%

## Metrics Reference

### Rust Implementation Metrics
```
# Request metrics
bitnet_requests_total{implementation="rust"}
bitnet_request_duration_seconds{implementation="rust"}
bitnet_tokens_per_second{implementation="rust"}
bitnet_tokens_generated_total

# Resource metrics
bitnet_rust_allocations_total
bitnet_rust_simd_operations_total
bitnet_rust_thread_pool_active_threads
bitnet_memory_usage_bytes
bitnet_gpu_memory_usage_bytes

# System-level metrics (NEW)
system_cpu_usage_percent
system_memory_usage_percent
system_disk_usage_percent
system_network_bytes_received_total
system_network_bytes_sent_total
system_uptime_seconds

# Model and inference metrics
bitnet_model_load_duration_seconds
bitnet_queue_depth
bitnet_cache_hit_rate

# Error metrics
bitnet_rust_panic_total
bitnet_rust_result_errors_total
bitnet_errors_total
```

### Cross-Validation Metrics
```
# Comparison metrics
bitnet_crossval_test_failures_total
bitnet_crossval_accuracy_difference
bitnet_crossval_rust_throughput
bitnet_crossval_cpp_throughput
```

## Deployment Options

### Docker Compose (Development)
```bash
# Start monitoring stack
docker-compose up -d

# View logs
docker-compose logs -f prometheus grafana

# Stop stack
docker-compose down
```

### Kubernetes (Production)
```bash
# Deploy monitoring namespace
kubectl create namespace bitnet-monitoring

# Deploy Prometheus
kubectl apply -f k8s/prometheus/

# Deploy Grafana
kubectl apply -f k8s/grafana/

# Deploy AlertManager
kubectl apply -f k8s/alertmanager/
```

### Helm Chart (Recommended)
```bash
# Add monitoring to bitnet-rs Helm chart
helm install bitnet ./helm/bitnet --set monitoring.enabled=true
```

## Customization

### Adding Custom Metrics
1. **Instrument Rust code** with prometheus crate
2. **Update Prometheus config** to scrape new endpoints
3. **Create dashboard panels** in Grafana
4. **Add alert rules** if needed

### Modifying Alert Thresholds
1. **Edit alert rules** in `rules/bitnet-rust-alerts.yml`
2. **Reload Prometheus** configuration
3. **Test alerts** using AlertManager

### Custom Dashboards
1. **Create dashboard** in Grafana UI
2. **Export JSON** configuration
3. **Save to** `grafana/dashboards/`
4. **Version control** the changes

## Troubleshooting

### Common Issues

1. **Metrics not appearing**
   - Check service discovery in Prometheus targets
   - Verify network connectivity
   - Confirm metric endpoints are accessible

2. **Alerts not firing**
   - Verify alert rule syntax
   - Check AlertManager routing configuration
   - Test notification channels

3. **Dashboard not loading**
   - Check Grafana datasource configuration
   - Verify dashboard JSON syntax
   - Confirm Prometheus connectivity

### Performance Optimization

1. **Reduce scrape intervals** for non-critical metrics
2. **Use recording rules** for complex queries
3. **Configure retention policies** for long-term storage
4. **Enable metric filtering** to reduce cardinality

## Security Considerations

### Authentication
- **Grafana**: Change default admin password
- **Prometheus**: Enable basic auth if exposed
- **AlertManager**: Configure webhook authentication

### Network Security
- **Firewall rules**: Restrict access to monitoring ports
- **TLS encryption**: Enable HTTPS for external access
- **Network policies**: Isolate monitoring namespace

### Data Privacy
- **Metric scrubbing**: Remove sensitive labels
- **Access control**: Role-based dashboard access
- **Audit logging**: Track configuration changes

## Migration from Legacy Monitoring

### Phase 1: Parallel Monitoring
1. **Deploy Rust-focused stack** alongside existing
2. **Compare metrics** between implementations
3. **Validate alert accuracy** and timing

### Phase 2: Gradual Migration
1. **Migrate critical alerts** to Rust-focused rules
2. **Update dashboards** to prioritize Rust metrics
3. **Train team** on new monitoring approach

### Phase 3: Legacy Deprecation
1. **Disable legacy alerts** (keep for reference)
2. **Archive old dashboards**
3. **Remove legacy monitoring** components

## Support and Maintenance

### Regular Tasks
- **Review alert thresholds** monthly
- **Update dashboards** based on team feedback
- **Clean up old metrics** and dashboards
- **Test disaster recovery** procedures

### Monitoring the Monitoring
- **Monitor Prometheus** itself for health
- **Set up AlertManager** redundancy
- **Backup Grafana** configurations
- **Document runbooks** for common issues

## Contributing

When adding new monitoring features:

1. **Focus on Rust implementation** first
2. **Add cross-validation** metrics if relevant
3. **Include appropriate alerts** and thresholds
4. **Update documentation** and runbooks
5. **Test in development** environment first

## License

This monitoring configuration is part of the bitnet-rs project and follows the same license terms.

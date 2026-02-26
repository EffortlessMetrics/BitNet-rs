#![allow(unused)]
#![allow(dead_code)]

//! Deployment Configuration Fixtures for BitNet-rs Inference Server
//!
//! This module provides comprehensive deployment fixtures for testing production
//! scenarios including Docker, Kubernetes, environment configurations, and
//! performance benchmarks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerConfig {
    pub name: &'static str,
    pub dockerfile_content: &'static str,
    pub docker_compose_content: &'static str,
    pub environment_variables: HashMap<&'static str, &'static str>,
    pub exposed_ports: Vec<u16>,
    pub resource_limits: ResourceLimits,
    pub health_check: HealthCheckConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    pub name: &'static str,
    pub deployment_yaml: &'static str,
    pub service_yaml: &'static str,
    pub configmap_yaml: &'static str,
    pub hpa_yaml: Option<&'static str>,
    pub ingress_yaml: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_cores: f32,
    pub memory_mb: u64,
    pub gpu_memory_mb: Option<u64>,
    pub disk_space_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub endpoint: &'static str,
    pub interval_seconds: u32,
    pub timeout_seconds: u32,
    pub retries: u32,
    pub start_period_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub name: &'static str,
    pub description: &'static str,
    pub variables: HashMap<&'static str, &'static str>,
    pub deployment_type: &'static str,
    pub resource_profile: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    pub name: &'static str,
    pub scenario: &'static str,
    pub concurrent_requests: u32,
    pub request_rate_per_second: u32,
    pub duration_seconds: u32,
    pub expected_throughput_rps: f32,
    pub expected_latency_p95_ms: u64,
    pub expected_accuracy: f32,
    pub resource_requirements: ResourceLimits,
}

/// Docker configuration fixtures
pub static DOCKER_CONFIGS: LazyLock<HashMap<&'static str, DockerConfig>> = LazyLock::new(|| {
    let mut configs = HashMap::new();

    configs.insert(
        "cpu_production",
        DockerConfig {
            name: "cpu_production",
            dockerfile_content: r#"
FROM rust:1.90-slim as builder

WORKDIR /app
COPY . .

# Install dependencies for BitNet-rs
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build with CPU features only
RUN cargo build --release --no-default-features --features cpu -p bitnet-server

FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false bitnet

WORKDIR /app
COPY --from=builder /app/target/release/bitnet-server /usr/local/bin/
COPY --from=builder /app/models/ /app/models/

# Set ownership and permissions
RUN chown -R bitnet:bitnet /app
USER bitnet

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["bitnet-server", "--config", "/app/config.toml"]
"#,
            docker_compose_content: r#"
version: '3.8'

services:
  bitnet-server:
    build: .
    ports:
      - "8080:8080"
    environment:
      BITNET_LOG_LEVEL: info
      BITNET_DEVICE_PREFERENCE: cpu
      BITNET_MAX_CONCURRENT_REQUESTS: 50
      BITNET_MODEL_PATH: /app/models/small_i2s_model.gguf
      RUST_LOG: bitnet_server=info
    volumes:
      - ./models:/app/models:ro
      - ./config.toml:/app/config.toml:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
"#,
            environment_variables: {
                let mut env = HashMap::new();
                env.insert("BITNET_LOG_LEVEL", "info");
                env.insert("BITNET_DEVICE_PREFERENCE", "cpu");
                env.insert("BITNET_MAX_CONCURRENT_REQUESTS", "50");
                env.insert("BITNET_DETERMINISTIC", "0");
                env.insert("RAYON_NUM_THREADS", "4");
                env
            },
            exposed_ports: vec![8080],
            resource_limits: ResourceLimits {
                cpu_cores: 2.0,
                memory_mb: 4096,
                gpu_memory_mb: None,
                disk_space_mb: 1024,
            },
            health_check: HealthCheckConfig {
                endpoint: "/health",
                interval_seconds: 30,
                timeout_seconds: 10,
                retries: 3,
                start_period_seconds: 60,
            },
        },
    );

    configs.insert(
        "gpu_production",
        DockerConfig {
            name: "gpu_production",
            dockerfile_content: r#"
FROM nvidia/cuda:12.0-devel-ubuntu22.04 as builder

WORKDIR /app
COPY . .

# Install Rust and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . /root/.cargo/env \
    && rustup default 1.90.0

ENV PATH="/root/.cargo/bin:${PATH}"

# Build with GPU features
RUN cargo build --release --no-default-features --features gpu -p bitnet-server

FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false bitnet

WORKDIR /app
COPY --from=builder /app/target/release/bitnet-server /usr/local/bin/
COPY --from=builder /app/models/ /app/models/

# Set ownership and permissions
RUN chown -R bitnet:bitnet /app
USER bitnet

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["bitnet-server", "--config", "/app/config.toml"]
"#,
            docker_compose_content: r#"
version: '3.8'

services:
  bitnet-server-gpu:
    build: .
    runtime: nvidia
    ports:
      - "8080:8080"
    environment:
      BITNET_LOG_LEVEL: info
      BITNET_DEVICE_PREFERENCE: gpu
      BITNET_MAX_CONCURRENT_REQUESTS: 100
      BITNET_MODEL_PATH: /app/models/large_tl2_model.gguf
      NVIDIA_VISIBLE_DEVICES: all
      RUST_LOG: bitnet_server=info
    volumes:
      - ./models:/app/models:ro
      - ./config.toml:/app/config.toml:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
        reservations:
          cpus: '2.0'
          memory: 8G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
"#,
            environment_variables: {
                let mut env = HashMap::new();
                env.insert("BITNET_LOG_LEVEL", "info");
                env.insert("BITNET_DEVICE_PREFERENCE", "gpu");
                env.insert("BITNET_MAX_CONCURRENT_REQUESTS", "100");
                env.insert("NVIDIA_VISIBLE_DEVICES", "all");
                env.insert("CUDA_VISIBLE_DEVICES", "0");
                env
            },
            exposed_ports: vec![8080],
            resource_limits: ResourceLimits {
                cpu_cores: 4.0,
                memory_mb: 16384,
                gpu_memory_mb: Some(8192),
                disk_space_mb: 2048,
            },
            health_check: HealthCheckConfig {
                endpoint: "/health",
                interval_seconds: 30,
                timeout_seconds: 10,
                retries: 3,
                start_period_seconds: 120,
            },
        },
    );

    configs
});

/// Kubernetes configuration fixtures
pub static KUBERNETES_CONFIGS: LazyLock<HashMap<&'static str, KubernetesConfig>> =
    LazyLock::new(|| {
        let mut configs = HashMap::new();

        configs.insert(
            "production_cluster",
            KubernetesConfig {
                name: "production_cluster",
                deployment_yaml: r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bitnet-server
  namespace: bitnet
  labels:
    app: bitnet-server
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bitnet-server
  template:
    metadata:
      labels:
        app: bitnet-server
    spec:
      containers:
      - name: bitnet-server
        image: bitnet/server:1.0.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: BITNET_LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: bitnet-config
              key: log_level
        - name: BITNET_DEVICE_PREFERENCE
          valueFrom:
            configMapKeyRef:
              name: bitnet-config
              key: device_preference
        - name: BITNET_MAX_CONCURRENT_REQUESTS
          valueFrom:
            configMapKeyRef:
              name: bitnet-config
              key: max_concurrent_requests
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
        - name: config
          mountPath: /app/config.toml
          subPath: config.toml
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: bitnet-models-pvc
      - name: config
        configMap:
          name: bitnet-config
      nodeSelector:
        workload-type: ai-inference
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
"#,
                service_yaml: r#"
apiVersion: v1
kind: Service
metadata:
  name: bitnet-server-service
  namespace: bitnet
  labels:
    app: bitnet-server
spec:
  selector:
    app: bitnet-server
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: bitnet-server-metrics
  namespace: bitnet
  labels:
    app: bitnet-server
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
spec:
  selector:
    app: bitnet-server
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
    name: metrics
  type: ClusterIP
"#,
                configmap_yaml: r#"
apiVersion: v1
kind: ConfigMap
metadata:
  name: bitnet-config
  namespace: bitnet
data:
  log_level: "info"
  device_preference: "auto"
  max_concurrent_requests: "100"
  config.toml: |
    [server]
    host = "0.0.0.0"
    port = 8080
    max_concurrent_requests = 100
    request_timeout_seconds = 30

    [model]
    default_model_path = "/app/models/medium_tl1_model.gguf"
    model_cache_size = 3

    [device]
    preference = "auto"
    cpu_threads = 0  # Auto-detect
    gpu_memory_fraction = 0.8

    [logging]
    level = "info"
    format = "json"

    [metrics]
    enabled = true
    endpoint = "/metrics"

    [health]
    readiness_timeout_seconds = 30
    liveness_timeout_seconds = 10

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bitnet-models-pvc
  namespace: bitnet
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
"#,
                hpa_yaml: Some(
                    r#"
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bitnet-server-hpa
  namespace: bitnet
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bitnet-server
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: bitnet_active_requests
      target:
        type: AverageValue
        averageValue: "30"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
"#,
                ),
                ingress_yaml: Some(
                    r#"
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bitnet-server-ingress
  namespace: bitnet
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.bitnet.example.com
    secretName: bitnet-tls
  rules:
  - host: api.bitnet.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bitnet-server-service
            port:
              number: 80
"#,
                ),
            },
        );

        configs
    });

/// Environment configuration fixtures
pub static ENVIRONMENT_CONFIGS: LazyLock<HashMap<&'static str, EnvironmentConfig>> =
    LazyLock::new(|| {
        let mut configs = HashMap::new();

        configs.insert(
            "development",
            EnvironmentConfig {
                name: "development",
                description: "Development environment for local testing",
                variables: {
                    let mut env = HashMap::new();
                    env.insert("BITNET_LOG_LEVEL", "debug");
                    env.insert("BITNET_DEVICE_PREFERENCE", "cpu");
                    env.insert("BITNET_MAX_CONCURRENT_REQUESTS", "10");
                    env.insert("BITNET_DETERMINISTIC", "1");
                    env.insert("BITNET_SEED", "42");
                    env.insert("RUST_LOG", "bitnet_server=debug,bitnet_inference=debug");
                    env.insert("RAYON_NUM_THREADS", "2");
                    env
                },
                deployment_type: "local",
                resource_profile: "minimal",
            },
        );

        configs.insert(
            "staging",
            EnvironmentConfig {
                name: "staging",
                description: "Staging environment for pre-production testing",
                variables: {
                    let mut env = HashMap::new();
                    env.insert("BITNET_LOG_LEVEL", "info");
                    env.insert("BITNET_DEVICE_PREFERENCE", "auto");
                    env.insert("BITNET_MAX_CONCURRENT_REQUESTS", "50");
                    env.insert("BITNET_DETERMINISTIC", "0");
                    env.insert("RUST_LOG", "bitnet_server=info");
                    env.insert("RAYON_NUM_THREADS", "4");
                    env.insert("BITNET_MODEL_CACHE_SIZE", "2");
                    env
                },
                deployment_type: "kubernetes",
                resource_profile: "medium",
            },
        );

        configs.insert(
            "production",
            EnvironmentConfig {
                name: "production",
                description: "Production environment for live traffic",
                variables: {
                    let mut env = HashMap::new();
                    env.insert("BITNET_LOG_LEVEL", "warn");
                    env.insert("BITNET_DEVICE_PREFERENCE", "gpu");
                    env.insert("BITNET_MAX_CONCURRENT_REQUESTS", "200");
                    env.insert("BITNET_DETERMINISTIC", "0");
                    env.insert("RUST_LOG", "bitnet_server=warn,error");
                    env.insert("RAYON_NUM_THREADS", "8");
                    env.insert("BITNET_MODEL_CACHE_SIZE", "5");
                    env.insert("BITNET_ENABLE_METRICS", "1");
                    env.insert("BITNET_ENABLE_TRACING", "1");
                    env
                },
                deployment_type: "kubernetes",
                resource_profile: "high",
            },
        );

        configs.insert(
            "benchmark",
            EnvironmentConfig {
                name: "benchmark",
                description: "Performance benchmarking environment",
                variables: {
                    let mut env = HashMap::new();
                    env.insert("BITNET_LOG_LEVEL", "error");
                    env.insert("BITNET_DEVICE_PREFERENCE", "gpu");
                    env.insert("BITNET_MAX_CONCURRENT_REQUESTS", "1000");
                    env.insert("BITNET_DETERMINISTIC", "1");
                    env.insert("BITNET_SEED", "12345");
                    env.insert("RUST_LOG", "error");
                    env.insert("RAYON_NUM_THREADS", "16");
                    env.insert("BITNET_DISABLE_LOGGING", "1");
                    env.insert("BITNET_ENABLE_DETAILED_METRICS", "1");
                    env
                },
                deployment_type: "bare_metal",
                resource_profile: "maximum",
            },
        );

        configs
    });

/// Performance benchmark fixtures
pub static PERFORMANCE_BENCHMARKS: LazyLock<HashMap<&'static str, PerformanceBenchmark>> =
    LazyLock::new(|| {
        let mut benchmarks = HashMap::new();

        benchmarks.insert(
            "baseline_cpu",
            PerformanceBenchmark {
                name: "baseline_cpu",
                scenario: "Single model CPU inference baseline",
                concurrent_requests: 10,
                request_rate_per_second: 5,
                duration_seconds: 300,
                expected_throughput_rps: 4.5,
                expected_latency_p95_ms: 2000,
                expected_accuracy: 0.99,
                resource_requirements: ResourceLimits {
                    cpu_cores: 2.0,
                    memory_mb: 4096,
                    gpu_memory_mb: None,
                    disk_space_mb: 1024,
                },
            },
        );

        benchmarks.insert(
            "high_load_gpu",
            PerformanceBenchmark {
                name: "high_load_gpu",
                scenario: "High concurrency GPU inference stress test",
                concurrent_requests: 100,
                request_rate_per_second: 50,
                duration_seconds: 600,
                expected_throughput_rps: 45.0,
                expected_latency_p95_ms: 1500,
                expected_accuracy: 0.98,
                resource_requirements: ResourceLimits {
                    cpu_cores: 8.0,
                    memory_mb: 16384,
                    gpu_memory_mb: Some(8192),
                    disk_space_mb: 4096,
                },
            },
        );

        benchmarks.insert(
            "burst_traffic",
            PerformanceBenchmark {
                name: "burst_traffic",
                scenario: "Traffic burst simulation with auto-scaling",
                concurrent_requests: 500,
                request_rate_per_second: 200,
                duration_seconds: 180,
                expected_throughput_rps: 180.0,
                expected_latency_p95_ms: 3000,
                expected_accuracy: 0.97,
                resource_requirements: ResourceLimits {
                    cpu_cores: 16.0,
                    memory_mb: 32768,
                    gpu_memory_mb: Some(16384),
                    disk_space_mb: 8192,
                },
            },
        );

        benchmarks.insert(
            "sustained_load",
            PerformanceBenchmark {
                name: "sustained_load",
                scenario: "24-hour sustained load test",
                concurrent_requests: 50,
                request_rate_per_second: 25,
                duration_seconds: 86400,
                expected_throughput_rps: 24.0,
                expected_latency_p95_ms: 1800,
                expected_accuracy: 0.985,
                resource_requirements: ResourceLimits {
                    cpu_cores: 4.0,
                    memory_mb: 8192,
                    gpu_memory_mb: Some(4096),
                    disk_space_mb: 2048,
                },
            },
        );

        benchmarks
    });

/// Get configuration fixtures by type
pub fn get_docker_config(name: &str) -> Option<&'static DockerConfig> {
    DOCKER_CONFIGS.get(name)
}

pub fn get_kubernetes_config(name: &str) -> Option<&'static KubernetesConfig> {
    KUBERNETES_CONFIGS.get(name)
}

pub fn get_environment_config(name: &str) -> Option<&'static EnvironmentConfig> {
    ENVIRONMENT_CONFIGS.get(name)
}

pub fn get_performance_benchmark(name: &str) -> Option<&'static PerformanceBenchmark> {
    PERFORMANCE_BENCHMARKS.get(name)
}

/// Get all configuration names by type
pub fn get_all_docker_configs() -> Vec<&'static str> {
    DOCKER_CONFIGS.keys().copied().collect()
}

pub fn get_all_kubernetes_configs() -> Vec<&'static str> {
    KUBERNETES_CONFIGS.keys().copied().collect()
}

pub fn get_all_environment_configs() -> Vec<&'static str> {
    ENVIRONMENT_CONFIGS.keys().copied().collect()
}

pub fn get_all_performance_benchmarks() -> Vec<&'static str> {
    PERFORMANCE_BENCHMARKS.keys().copied().collect()
}

/// Generate load test scenarios
pub fn generate_load_test_scenario(
    base_rps: u32,
    duration_seconds: u32,
    ramp_up_seconds: u32,
) -> Vec<(u32, u32)> {
    let mut scenario = Vec::new();

    // Ramp up phase
    for second in 0..ramp_up_seconds {
        let current_rps = (base_rps * second) / ramp_up_seconds;
        scenario.push((second, current_rps));
    }

    // Sustained load phase
    for second in ramp_up_seconds..duration_seconds {
        scenario.push((second, base_rps));
    }

    scenario
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docker_configs() {
        let cpu_config = get_docker_config("cpu_production").unwrap();
        assert!(cpu_config.dockerfile_content.contains("--features cpu"));
        assert_eq!(cpu_config.exposed_ports, vec![8080]);
        assert!(cpu_config.resource_limits.gpu_memory_mb.is_none());

        let gpu_config = get_docker_config("gpu_production").unwrap();
        assert!(gpu_config.dockerfile_content.contains("--features gpu"));
        assert!(gpu_config.resource_limits.gpu_memory_mb.is_some());
    }

    #[test]
    fn test_kubernetes_configs() {
        let k8s_config = get_kubernetes_config("production_cluster").unwrap();
        assert!(k8s_config.deployment_yaml.contains("bitnet-server"));
        assert!(k8s_config.service_yaml.contains("Service"));
        assert!(k8s_config.hpa_yaml.is_some());
        assert!(k8s_config.ingress_yaml.is_some());
    }

    #[test]
    fn test_environment_configs() {
        let dev_config = get_environment_config("development").unwrap();
        assert_eq!(dev_config.variables.get("BITNET_LOG_LEVEL"), Some(&"debug"));
        assert_eq!(dev_config.deployment_type, "local");

        let prod_config = get_environment_config("production").unwrap();
        assert_eq!(prod_config.variables.get("BITNET_LOG_LEVEL"), Some(&"warn"));
        assert_eq!(prod_config.deployment_type, "kubernetes");
    }

    #[test]
    fn test_performance_benchmarks() {
        let cpu_benchmark = get_performance_benchmark("baseline_cpu").unwrap();
        assert_eq!(cpu_benchmark.concurrent_requests, 10);
        assert!(cpu_benchmark.resource_requirements.gpu_memory_mb.is_none());

        let gpu_benchmark = get_performance_benchmark("high_load_gpu").unwrap();
        assert_eq!(gpu_benchmark.concurrent_requests, 100);
        assert!(gpu_benchmark.resource_requirements.gpu_memory_mb.is_some());
    }

    #[test]
    fn test_load_test_scenario_generation() {
        let scenario = generate_load_test_scenario(100, 300, 60);
        assert_eq!(scenario.len(), 300);

        // Check ramp up
        assert!(scenario[0].1 < scenario[30].1);
        assert!(scenario[30].1 < scenario[59].1);

        // Check sustained load
        assert_eq!(scenario[60].1, 100);
        assert_eq!(scenario[299].1, 100);
    }

    #[test]
    fn test_config_completeness() {
        // Verify all config types have entries
        assert!(!get_all_docker_configs().is_empty());
        assert!(!get_all_kubernetes_configs().is_empty());
        assert!(!get_all_environment_configs().is_empty());
        assert!(!get_all_performance_benchmarks().is_empty());
    }
}

# Deploy BitNet.rs Production Inference Server on Kubernetes

This guide shows how to deploy the BitNet.rs production inference server on Kubernetes using Helm charts for scalable, production-ready deployments.

## Prerequisites

Before deploying, ensure you have:

- **kubectl** 1.27+ configured with cluster access
- **Helm** 3.12+ installed
- **Kubernetes cluster** 1.27+ (GKE, EKS, AKS, or self-hosted)
- **NVIDIA GPU Operator** installed (for GPU deployments)
- **Persistent storage** provisioner (for model storage)
- **BitNet GGUF model files** available
- **8GB RAM per pod minimum** for 2B parameter models

## Quick Start

### Install via Helm Chart

```bash
# Clone repository
git clone https://github.com/microsoft/BitNet-rs
cd BitNet-rs

# Add Helm repository (future)
# helm repo add bitnet https://bitnet-rs.github.io/helm-charts
# helm repo update

# Install with default values (CPU only)
helm install bitnet infra/helm/bitnet \
  --namespace bitnet-system \
  --create-namespace

# Check deployment status
kubectl get pods -n bitnet-system

# Access service
kubectl port-forward -n bitnet-system svc/bitnet 8080:8080
curl http://localhost:8080/health
```

### GPU Deployment

```bash
# Install with GPU support
helm install bitnet infra/helm/bitnet \
  --namespace bitnet-system \
  --create-namespace \
  --set cpu.enabled=false \
  --set gpu.enabled=true \
  --set gpu.replicaCount=2

# Verify GPU pods
kubectl get pods -n bitnet-system -l app.kubernetes.io/variant=gpu
```

## Helm Chart Configuration

### Chart Structure

The BitNet Helm chart is located at `/home/steven/code/Rust/BitNet-rs/infra/helm/bitnet/`:

```
infra/helm/bitnet/
├── Chart.yaml                 # Chart metadata
├── values.yaml                # Default configuration
└── templates/
    ├── deployment.yaml        # CPU and GPU deployments
    ├── service.yaml           # ClusterIP service
    ├── configmap.yaml         # Configuration
    ├── serviceaccount.yaml    # RBAC service account
    └── pvc.yaml               # Persistent volume claims
```

### Configuration Options

#### Basic Configuration

```yaml
# values.yaml
global:
  implementation: rust          # Always rust for production

image:
  registry: docker.io
  repository: bitnet/bitnet-rust
  tag: "1.0.0"
  pullPolicy: IfNotPresent

cpu:
  enabled: true
  replicaCount: 3
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 1000m
      memory: 2Gi

gpu:
  enabled: false
  replicaCount: 2
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: 16Gi
    requests:
      nvidia.com/gpu: 1
      memory: 4Gi
```

#### Install with Custom Values

```bash
# Create custom values file
cat > custom-values.yaml <<EOF
cpu:
  replicaCount: 5
  resources:
    limits:
      cpu: 8000m
      memory: 16Gi

config:
  logging:
    level: debug

persistence:
  models:
    size: 200Gi
EOF

# Install with custom values
helm install bitnet infra/helm/bitnet \
  --namespace bitnet-system \
  --create-namespace \
  -f custom-values.yaml
```

## Model Storage Configuration

### Using Persistent Volumes

The chart creates PersistentVolumeClaims for model storage:

```yaml
# values.yaml
persistence:
  models:
    enabled: true
    storageClass: ""           # Use default storage class
    accessMode: ReadOnlyMany   # Multiple pods, read-only
    size: 100Gi
    annotations: {}

  cache:
    enabled: true
    storageClass: ""
    accessMode: ReadWriteMany  # Multiple pods, read-write
    size: 50Gi
```

**Storage Class Requirements:**
- **Models PVC**: `ReadOnlyMany` access mode (NFS, EFS, GlusterFS)
- **Cache PVC**: `ReadWriteMany` access mode (NFS, EFS, Ceph)

### Pre-populate Model Storage

```bash
# Create PVC manually
kubectl create -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bitnet-models
  namespace: bitnet-system
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs-client
EOF

# Copy models to PVC (using temporary pod)
kubectl run -n bitnet-system model-loader \
  --image=busybox \
  --restart=Never \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "model-loader",
      "image": "busybox",
      "command": ["sleep", "3600"],
      "volumeMounts": [{
        "name": "models",
        "mountPath": "/models"
      }]
    }],
    "volumes": [{
      "name": "models",
      "persistentVolumeClaim": {
        "claimName": "bitnet-models"
      }
    }]
  }
}'

# Copy model files
kubectl cp -n bitnet-system models/microsoft-bitnet-b1.58-2B-4T-gguf model-loader:/models/

# Cleanup
kubectl delete pod -n bitnet-system model-loader
```

### Using ConfigMaps for Small Models

For models <1MB (testing only):

```bash
# Create ConfigMap from model file
kubectl create configmap bitnet-small-model \
  --from-file=model.gguf=models/small-model.gguf \
  --namespace bitnet-system

# Mount in deployment
kubectl patch deployment bitnet-cpu -n bitnet-system --type=json -p='[{
  "op": "add",
  "path": "/spec/template/spec/volumes/-",
  "value": {
    "name": "model",
    "configMap": {"name": "bitnet-small-model"}
  }
}]'
```

## Resource Requirements and Limits

### CPU Deployment Resources

**Recommended Settings for 2B Parameter Models:**

```yaml
cpu:
  resources:
    limits:
      cpu: 4000m              # 4 CPU cores maximum
      memory: 8Gi             # 8GB RAM maximum
    requests:
      cpu: 1000m              # 1 CPU core minimum
      memory: 2Gi             # 2GB RAM minimum
```

**Resource Calculation:**
- **Memory**: Model size + 2GB overhead (e.g., 2B model = 2GB + 2GB = 4GB minimum)
- **CPU**: 1 core per 500M parameters as baseline
- **Storage**: Model size × 1.5 for cache and temporary files

### GPU Deployment Resources

**Recommended Settings for 2B Parameter Models:**

```yaml
gpu:
  resources:
    limits:
      nvidia.com/gpu: 1       # 1 GPU per pod
      cpu: 8000m              # 8 CPU cores for host operations
      memory: 16Gi            # 16GB host RAM
    requests:
      nvidia.com/gpu: 1       # GPU is required, not optional
      cpu: 2000m              # 2 CPU cores minimum
      memory: 4Gi             # 4GB RAM minimum
```

**GPU Requirements:**
- **GPU Memory**: 8GB minimum for 2B models, 24GB for 7B models
- **CUDA Compute Capability**: 7.0+ (Volta architecture or newer)
- **Recommended GPUs**: V100, T4, A10, A100, H100

### Autoscaling Configuration

```yaml
cpu:
  autoscaling:
    enabled: true
    minReplicas: 2            # Minimum pods for HA
    maxReplicas: 10           # Maximum pods for load
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

gpu:
  autoscaling:
    enabled: true
    minReplicas: 1            # GPUs are expensive
    maxReplicas: 5            # Scale based on workload
    targetCPUUtilizationPercentage: 60
```

**HPA Best Practices:**
- **CPU pods**: Scale based on CPU and memory utilization
- **GPU pods**: Scale conservatively due to cost
- **Cooldown**: Set 5-minute cooldown periods to prevent flapping
- **Metrics**: Use custom metrics for tokens/second throughput

## Health Probes Configuration

### Liveness, Readiness, and Startup Probes

The chart configures comprehensive health checks:

```yaml
healthCheck:
  enabled: true
  path: /health
  initialDelaySeconds: 30     # Wait 30s before first check
  periodSeconds: 10           # Check every 10 seconds
  timeoutSeconds: 5           # 5 second timeout
  failureThreshold: 3         # 3 failures = unhealthy

readinessCheck:
  enabled: true
  path: /ready                # Use readiness endpoint
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3

startupCheck:
  enabled: true
  path: /health
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 30        # Allow 300s startup time
```

**Health Check Semantics:**
- **Liveness**: Restarts pod if server is deadlocked
- **Readiness**: Removes pod from load balancing if not ready
- **Startup**: Allows slow model loading without false failures

**GPU Adjustments:**
- GPU pods use **2x initialDelaySeconds** (model loading to GPU is slower)
- GPU pods use **3x startupFailureThreshold** (CUDA initialization time)

### Testing Health Probes

```bash
# Test liveness probe
kubectl exec -n bitnet-system deployment/bitnet-cpu -- curl -f http://localhost:8080/health

# Test readiness probe
kubectl exec -n bitnet-system deployment/bitnet-cpu -- curl -f http://localhost:8080/ready

# Check probe status
kubectl describe pod -n bitnet-system -l app.kubernetes.io/component=inference

# View probe failures
kubectl get events -n bitnet-system --field-selector involvedObject.kind=Pod
```

## Horizontal Scaling

### Manual Scaling

```bash
# Scale CPU deployment
kubectl scale deployment bitnet-cpu \
  --replicas=5 \
  --namespace bitnet-system

# Scale GPU deployment
kubectl scale deployment bitnet-gpu \
  --replicas=3 \
  --namespace bitnet-system

# Check scaling status
kubectl get deployments -n bitnet-system
kubectl get pods -n bitnet-system -w
```

### Horizontal Pod Autoscaler

The chart creates HPA resources automatically:

```bash
# View HPA status
kubectl get hpa -n bitnet-system

# Describe HPA
kubectl describe hpa bitnet-cpu-hpa -n bitnet-system

# View HPA metrics
kubectl top pods -n bitnet-system
```

### Custom Metrics Autoscaling

Scale based on inference throughput:

```yaml
# Install Prometheus Adapter
helm install prometheus-adapter prometheus-community/prometheus-adapter

# Create custom HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bitnet-cpu-custom
  namespace: bitnet-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bitnet-cpu
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: bitnet_inference_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

## Monitoring Integration

### Prometheus Metrics

Enable Prometheus ServiceMonitor:

```yaml
monitoring:
  serviceMonitor:
    enabled: true
    namespace: monitoring      # Prometheus namespace
    interval: 30s
    scrapeTimeout: 10s
    labels:
      prometheus: kube-prometheus
```

**Exposed Metrics:**
- `bitnet_inference_requests_total`: Total inference requests
- `bitnet_inference_duration_seconds`: Inference latency histogram
- `bitnet_tokens_generated_total`: Total tokens generated
- `bitnet_model_memory_bytes`: Model memory usage
- `system_cpu_usage_percent`: CPU utilization
- `system_memory_used_bytes`: Memory consumption

### Grafana Dashboards

```bash
# Port-forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Import BitNet dashboard (JSON)
# Navigate to http://localhost:3000
# Dashboard > Import > Upload JSON
```

**Key Dashboard Panels:**
- Requests per second
- Latency percentiles (p50, p90, p99)
- Token throughput
- Resource utilization (CPU, memory, GPU)
- Error rates

### Logs Collection

```bash
# View pod logs
kubectl logs -n bitnet-system deployment/bitnet-cpu -f

# View logs from all replicas
kubectl logs -n bitnet-system -l app.kubernetes.io/component=inference --tail=100

# Export logs to file
kubectl logs -n bitnet-system deployment/bitnet-cpu > logs.txt
```

**Log Aggregation Integration:**
- **Fluentd**: Deploy DaemonSet for log collection
- **ELK Stack**: Forward logs to Elasticsearch
- **Loki**: Grafana Loki for log aggregation
- **CloudWatch/Stackdriver**: Cloud provider logging

## GPU Node Affinity and Tolerations

### Node Selection

Configure GPU pods to run on GPU-enabled nodes:

```yaml
gpu:
  nodeSelector:
    accelerator: nvidia-tesla-v100  # Node label for GPU type

  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"

  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: accelerator
            operator: In
            values:
            - nvidia-tesla-v100
            - nvidia-tesla-a100
```

### Label GPU Nodes

```bash
# Label nodes with GPU type
kubectl label nodes gpu-node-1 accelerator=nvidia-tesla-v100

# Taint GPU nodes to prevent non-GPU workloads
kubectl taint nodes gpu-node-1 nvidia.com/gpu=:NoSchedule

# Verify labels
kubectl get nodes --show-labels | grep accelerator
```

### GPU Operator Installation

```bash
# Install NVIDIA GPU Operator
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator-system \
  --create-namespace \
  --set driver.enabled=true

# Verify GPU operator
kubectl get pods -n gpu-operator-system
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable.'nvidia\.com/gpu'
```

## Rolling Updates and Rollbacks

### Perform Rolling Update

```bash
# Update image version
helm upgrade bitnet infra/helm/bitnet \
  --namespace bitnet-system \
  --set image.tag=1.1.0 \
  --reuse-values

# Watch rollout status
kubectl rollout status deployment/bitnet-cpu -n bitnet-system

# Check rollout history
kubectl rollout history deployment/bitnet-cpu -n bitnet-system
```

### Rollback Deployment

```bash
# Rollback to previous version
kubectl rollout undo deployment/bitnet-cpu -n bitnet-system

# Rollback to specific revision
kubectl rollout undo deployment/bitnet-cpu -n bitnet-system --to-revision=2

# Verify rollback
kubectl get deployments -n bitnet-system
```

### Deployment Strategy

```yaml
# values.yaml - configure in deployment template
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1              # 1 extra pod during update
      maxUnavailable: 0        # No pods unavailable during update
```

**Best Practices:**
- **CPU deployments**: `maxUnavailable: 0` for zero-downtime
- **GPU deployments**: `maxSurge: 0, maxUnavailable: 1` to save GPU resources
- **Pre-stop hooks**: Gracefully drain requests before termination
- **PodDisruptionBudget**: Ensure minimum availability

## Pod Disruption Budget

The chart includes PDB for high availability:

```yaml
podDisruptionBudget:
  enabled: true
  minAvailable: 1             # At least 1 pod must be available
  # OR
  # maxUnavailable: "30%"     # Maximum 30% pods unavailable
```

**Use Cases:**
- **Node maintenance**: Prevents all pods from being drained
- **Voluntary disruptions**: Protects against kubectl drain
- **Cluster upgrades**: Ensures service availability

## Troubleshooting

### Pods Not Starting

**Check pod status:**

```bash
# View pod status
kubectl get pods -n bitnet-system

# Describe pod for events
kubectl describe pod -n bitnet-system bitnet-cpu-xxx

# Check pod logs
kubectl logs -n bitnet-system bitnet-cpu-xxx
```

**Common Issues:**
- **ImagePullBackOff**: Image not found or registry authentication failed
- **CrashLoopBackOff**: Application crashes on startup (check logs)
- **Pending**: Insufficient resources or PVC not bound
- **Init:Error**: Init container failed (model loading issue)

### PVC Not Binding

```bash
# Check PVC status
kubectl get pvc -n bitnet-system

# Describe PVC for events
kubectl describe pvc bitnet-models-pvc -n bitnet-system

# Check storage class
kubectl get storageclass
```

**Solutions:**
- Verify storage class supports `ReadOnlyMany` access mode
- Check PV provisioner is running
- Manually create PV if using static provisioning
- Verify storage quota and capacity

### Model Loading Failures

```bash
# Check model path configuration
kubectl get configmap -n bitnet-system bitnet-config -o yaml

# Verify model files exist
kubectl exec -n bitnet-system deployment/bitnet-cpu -- ls -la /app/models

# Test model loading manually
kubectl exec -it -n bitnet-system deployment/bitnet-cpu -- /bin/sh
# Inside pod:
bitnet compat-check /app/models/model.gguf
```

### GPU Not Detected

```bash
# Check GPU resources
kubectl describe node gpu-node-1 | grep -A 10 "Allocatable"

# Verify GPU operator
kubectl get pods -n gpu-operator-system

# Test GPU in pod
kubectl exec -n bitnet-system deployment/bitnet-gpu -- nvidia-smi

# Check CUDA visibility
kubectl exec -n bitnet-system deployment/bitnet-gpu -- env | grep CUDA
```

### High Memory Usage / OOM Kills

```bash
# Check memory usage
kubectl top pods -n bitnet-system

# View OOM events
kubectl get events -n bitnet-system --field-selector reason=OOMKilled

# Increase memory limits
helm upgrade bitnet infra/helm/bitnet \
  --namespace bitnet-system \
  --set cpu.resources.limits.memory=16Gi \
  --reuse-values
```

### Health Check Failures

```bash
# View health check events
kubectl get events -n bitnet-system --field-selector reason=Unhealthy

# Test health endpoint manually
kubectl exec -n bitnet-system deployment/bitnet-cpu -- curl -v http://localhost:8080/health

# Increase probe timeouts
helm upgrade bitnet infra/helm/bitnet \
  --namespace bitnet-system \
  --set healthCheck.initialDelaySeconds=60 \
  --set startupCheck.failureThreshold=60 \
  --reuse-values
```

### Network Issues

```bash
# Test service connectivity
kubectl run -n bitnet-system test-pod --rm -it --image=curlimages/curl -- curl http://bitnet:8080/health

# Check service endpoints
kubectl get endpoints -n bitnet-system bitnet

# Verify network policies
kubectl get networkpolicies -n bitnet-system
```

## Production Best Practices

### High Availability Configuration

```yaml
# Minimum 3 replicas for HA
cpu:
  replicaCount: 3

# Pod anti-affinity for different nodes
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/component
            operator: In
            values:
            - inference
        topologyKey: kubernetes.io/hostname

# PDB for disruption protection
podDisruptionBudget:
  enabled: true
  minAvailable: 2
```

### Resource Optimization

```yaml
# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: bitnet-cpu-vpa
  namespace: bitnet-system
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bitnet-cpu
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: bitnet
      minAllowed:
        cpu: 1000m
        memory: 2Gi
      maxAllowed:
        cpu: 8000m
        memory: 16Gi
```

### Security Hardening

```yaml
# Security context
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000

# Network policies
networkPolicy:
  enabled: true
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
```

### Monitoring and Alerting

```yaml
# PrometheusRule for alerting
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: bitnet-alerts
  namespace: bitnet-system
spec:
  groups:
  - name: bitnet
    interval: 30s
    rules:
    - alert: BitNetHighLatency
      expr: histogram_quantile(0.99, bitnet_inference_duration_seconds_bucket) > 5
      for: 5m
      annotations:
        summary: "BitNet inference latency is high"

    - alert: BitNetLowAvailability
      expr: up{job="bitnet"} < 0.9
      for: 5m
      annotations:
        summary: "BitNet availability is below 90%"

    - alert: BitNetOOMKills
      expr: rate(kube_pod_container_status_restarts_total{namespace="bitnet-system"}[5m]) > 0
      annotations:
        summary: "BitNet pods are being OOM killed"
```

## Advanced Deployment Patterns

### Blue-Green Deployment

```bash
# Deploy blue version
helm install bitnet-blue infra/helm/bitnet \
  --namespace bitnet-blue \
  --create-namespace \
  --set image.tag=1.0.0

# Deploy green version
helm install bitnet-green infra/helm/bitnet \
  --namespace bitnet-green \
  --create-namespace \
  --set image.tag=1.1.0

# Switch traffic (update ingress)
kubectl patch ingress bitnet -n bitnet-system -p '{"spec":{"rules":[{"host":"bitnet.example.com","http":{"paths":[{"path":"/","pathType":"Prefix","backend":{"service":{"name":"bitnet-green","port":{"number":8080}}}}]}}]}}'

# Cleanup old version
helm uninstall bitnet-blue -n bitnet-blue
```

### Canary Deployment

```bash
# Deploy canary with 10% traffic
helm install bitnet-canary infra/helm/bitnet \
  --namespace bitnet-system \
  --set cpu.replicaCount=1 \
  --set image.tag=1.1.0-canary

# Use Istio or Linkerd for traffic splitting
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: bitnet
  namespace: bitnet-system
spec:
  hosts:
  - bitnet
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: bitnet-canary
  - route:
    - destination:
        host: bitnet
      weight: 90
    - destination:
        host: bitnet-canary
      weight: 10
EOF
```

## Next Steps

- **Docker Deployment**: See [production-server-docker-deployment.md](production-server-docker-deployment.md) for container basics
- **Performance Tuning**: See [docs/performance-benchmarking.md](/home/steven/code/Rust/BitNet-rs/docs/performance-benchmarking.md)
- **API Reference**: See [docs/reference/real-model-api-contracts.md](/home/steven/code/Rust/BitNet-rs/docs/reference/real-model-api-contracts.md)
- **Health Monitoring**: See [docs/health-endpoints.md](/home/steven/code/Rust/BitNet-rs/docs/health-endpoints.md)
- **Environment Variables**: See [docs/environment-variables.md](/home/steven/code/Rust/BitNet-rs/docs/environment-variables.md)

## Additional Resources

- **Kubernetes Documentation**: https://kubernetes.io/docs/
- **Helm Documentation**: https://helm.sh/docs/
- **NVIDIA GPU Operator**: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
- **BitNet.rs Repository**: https://github.com/microsoft/BitNet-rs
- **Kubernetes Best Practices**: https://kubernetes.io/docs/concepts/configuration/overview/
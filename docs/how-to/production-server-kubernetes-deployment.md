# Production Server Kubernetes Deployment

Deploy BitNet.rs inference server on Kubernetes for enterprise-scale production environments with auto-scaling, high availability, and comprehensive monitoring.

## Overview

This guide covers deploying the BitNet.rs production inference server on Kubernetes with:

- **Auto-scaling**: Horizontal and vertical pod scaling based on load
- **High Availability**: Multi-replica deployment with rolling updates
- **Load Balancing**: Service discovery and traffic distribution
- **Monitoring**: Prometheus integration and Grafana dashboards
- **Security**: RBAC, network policies, and secret management

## Prerequisites

- **Kubernetes Cluster**: v1.24+ with GPU support (optional)
- **kubectl**: Configured to access your cluster
- **Helm**: v3.0+ for package management
- **Container Registry**: Access to push BitNet images
- **Storage**: Persistent storage for models and data

## Basic Deployment

### Namespace Configuration

Create a dedicated namespace:

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: bitnet-inference
  labels:
    app: bitnet-server
    environment: production
```

```bash
kubectl apply -f namespace.yaml
```

### ConfigMap for Server Configuration

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bitnet-server-config
  namespace: bitnet-inference
data:
  config.yaml: |
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

  prometheus.yaml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'bitnet-server'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - bitnet-inference
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
```

### Secret Management

Store sensitive configuration:

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: bitnet-server-secrets
  namespace: bitnet-inference
type: Opaque
stringData:
  jwt-secret: "your-jwt-secret-key"
  api-key: "your-api-key"
  opentelemetry-endpoint: "https://your-otel-collector:4317"
```

```bash
kubectl apply -f secrets.yaml
```

### Deployment Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bitnet-inference-server
  namespace: bitnet-inference
  labels:
    app: bitnet-server
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: bitnet-server
  template:
    metadata:
      labels:
        app: bitnet-server
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: bitnet-server
        image: bitnet/inference-server:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        env:
        - name: BITNET_DETERMINISTIC
          value: "1"
        - name: BITNET_SEED
          value: "42"
        - name: RUST_LOG
          value: "bitnet_server=info"
        - name: RAYON_NUM_THREADS
          value: "4"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: bitnet-server-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: tmp
          mountPath: /tmp
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 30
      volumes:
      - name: config
        configMap:
          name: bitnet-server-config
      - name: models
        persistentVolumeClaim:
          claimName: bitnet-models-pvc
      - name: tmp
        emptyDir:
          sizeLimit: 1Gi
      nodeSelector:
        kubernetes.io/os: linux
      tolerations:
      - key: "node.kubernetes.io/not-ready"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300
      - key: "node.kubernetes.io/unreachable"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300
```

### GPU-Enabled Deployment

For GPU acceleration:

```yaml
# deployment-gpu.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bitnet-inference-server-gpu
  namespace: bitnet-inference
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: bitnet-server
        image: bitnet/inference-server:gpu-v1.0.0
        resources:
          requests:
            memory: "6Gi"
            cpu: "3"
            nvidia.com/gpu: "1"
          limits:
            memory: "12Gi"
            cpu: "6"
            nvidia.com/gpu: "1"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"
      nodeSelector:
        accelerator: nvidia-tesla-v100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

## Service Configuration

### Service for Load Balancing

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: bitnet-server-service
  namespace: bitnet-inference
  labels:
    app: bitnet-server
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  selector:
    app: bitnet-server
```

### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bitnet-server-ingress
  namespace: bitnet-inference
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - bitnet-api.example.com
    secretName: bitnet-server-tls
  rules:
  - host: bitnet-api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bitnet-server-service
            port:
              number: 80
```

## Storage Configuration

### Persistent Volume for Models

```yaml
# pv-models.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: bitnet-models-pv
  labels:
    type: local
spec:
  storageClassName: fast-ssd
  capacity:
    storage: 50Gi
  accessModes:
    - ReadOnlyMany
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: /data/bitnet/models
```

### Persistent Volume Claim

```yaml
# pvc-models.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bitnet-models-pvc
  namespace: bitnet-inference
spec:
  storageClassName: fast-ssd
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
```

## Auto-scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bitnet-server-hpa
  namespace: bitnet-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bitnet-inference-server
  minReplicas: 2
  maxReplicas: 10
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
        averageValue: "50"
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
        periodSeconds: 60
```

### Vertical Pod Autoscaler

```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: bitnet-server-vpa
  namespace: bitnet-inference
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bitnet-inference-server
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: bitnet-server
      maxAllowed:
        cpu: 8
        memory: 16Gi
      minAllowed:
        cpu: 1
        memory: 2Gi
      mode: Auto
```

## Monitoring and Observability

### ServiceMonitor for Prometheus

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: bitnet-server-monitor
  namespace: bitnet-inference
  labels:
    app: bitnet-server
spec:
  selector:
    matchLabels:
      app: bitnet-server
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
```

### PrometheusRule for Alerting

```yaml
# prometheusrule.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: bitnet-server-alerts
  namespace: bitnet-inference
spec:
  groups:
  - name: bitnet-server.rules
    rules:
    - alert: BitNetServerDown
      expr: up{job="bitnet-server"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "BitNet server instance is down"
        description: "BitNet server {{ $labels.instance }} has been down for more than 1 minute."

    - alert: BitNetHighLatency
      expr: histogram_quantile(0.95, rate(bitnet_inference_duration_seconds_bucket[5m])) > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "BitNet server high latency"
        description: "95th percentile latency is {{ $value }}s for more than 5 minutes."

    - alert: BitNetHighErrorRate
      expr: rate(bitnet_requests_total{status=~"5.."}[5m]) / rate(bitnet_requests_total[5m]) > 0.05
      for: 3m
      labels:
        severity: critical
      annotations:
        summary: "BitNet server high error rate"
        description: "Error rate is {{ $value | humanizePercentage }} for more than 3 minutes."

    - alert: BitNetHighMemoryUsage
      expr: container_memory_usage_bytes{pod=~"bitnet-inference-server-.*"} / container_spec_memory_limit_bytes > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "BitNet server high memory usage"
        description: "Memory usage is {{ $value | humanizePercentage }} for more than 5 minutes."
```

## Security Configuration

### RBAC Configuration

```yaml
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: bitnet-server-sa
  namespace: bitnet-inference
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: bitnet-inference
  name: bitnet-server-role
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: bitnet-server-rolebinding
  namespace: bitnet-inference
subjects:
- kind: ServiceAccount
  name: bitnet-server-sa
  namespace: bitnet-inference
roleRef:
  kind: Role
  name: bitnet-server-role
  apiGroup: rbac.authorization.k8s.io
```

### Network Policy

```yaml
# networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: bitnet-server-netpol
  namespace: bitnet-inference
spec:
  podSelector:
    matchLabels:
      app: bitnet-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 443
```

### Pod Security Policy

```yaml
# podsecuritypolicy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: bitnet-server-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Helm Chart Deployment

### Chart Structure

```
charts/bitnet-inference/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── hpa.yaml
│   ├── servicemonitor.yaml
│   └── _helpers.tpl
└── README.md
```

### Values Configuration

```yaml
# values.yaml
replicaCount: 3

image:
  repository: bitnet/inference-server
  tag: "v1.0.0"
  pullPolicy: Always

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: nginx
  host: bitnet-api.example.com
  tls:
    enabled: true
    secretName: bitnet-server-tls

resources:
  limits:
    cpu: 4
    memory: 8Gi
  requests:
    cpu: 2
    memory: 4Gi

gpu:
  enabled: false
  type: nvidia-tesla-v100
  count: 1

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s

config:
  server:
    host: "0.0.0.0"
    port: 8080
  concurrency:
    max_concurrent_requests: 100
  batch_engine:
    max_batch_size: 8
  monitoring:
    prometheus_enabled: true

storage:
  models:
    size: 50Gi
    storageClass: fast-ssd
    accessMode: ReadOnlyMany
```

### Deploy with Helm

```bash
# Add BitNet Helm repository (if available)
helm repo add bitnet https://charts.bitnet.rs
helm repo update

# Install with custom values
helm install bitnet-inference bitnet/bitnet-inference \
  --namespace bitnet-inference \
  --create-namespace \
  --values values.production.yaml

# Upgrade deployment
helm upgrade bitnet-inference bitnet/bitnet-inference \
  --namespace bitnet-inference \
  --values values.production.yaml

# Rollback if needed
helm rollback bitnet-inference 1 --namespace bitnet-inference
```

## Deployment Verification

### Health and Status Checks

```bash
# Check deployment status
kubectl get deployment -n bitnet-inference
kubectl get pods -n bitnet-inference
kubectl get services -n bitnet-inference

# Check pod logs
kubectl logs -f deployment/bitnet-inference-server -n bitnet-inference

# Check resource usage
kubectl top pods -n bitnet-inference
kubectl top nodes
```

### Connectivity Tests

```bash
# Port forward for local testing
kubectl port-forward service/bitnet-server-service 8080:80 -n bitnet-inference

# Test health endpoints
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready

# Test inference
curl -X POST http://localhost:8080/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Test deployment","max_tokens":10}'
```

### Load Testing

```bash
# Install load testing tool
kubectl create namespace loadtest
kubectl run loadtest --image=williamyeh/hey -n loadtest -- \
  -n 1000 -c 50 -m POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Load test","max_tokens":5}' \
  http://bitnet-server-service.bitnet-inference.svc.cluster.local/v1/inference

# Monitor during load test
kubectl get hpa -n bitnet-inference -w
kubectl top pods -n bitnet-inference
```

## Production Operations

### Rolling Updates

```bash
# Update image version
kubectl set image deployment/bitnet-inference-server \
  bitnet-server=bitnet/inference-server:v1.1.0 \
  -n bitnet-inference

# Check rollout status
kubectl rollout status deployment/bitnet-inference-server -n bitnet-inference

# Rollback if needed
kubectl rollout undo deployment/bitnet-inference-server -n bitnet-inference
```

### Configuration Updates

```bash
# Update ConfigMap
kubectl apply -f configmap.yaml

# Restart deployment to pick up changes
kubectl rollout restart deployment/bitnet-inference-server -n bitnet-inference
```

### Scaling Operations

```bash
# Manual scaling
kubectl scale deployment bitnet-inference-server --replicas=5 -n bitnet-inference

# Check HPA status
kubectl describe hpa bitnet-server-hpa -n bitnet-inference

# Update HPA limits
kubectl patch hpa bitnet-server-hpa -n bitnet-inference -p '{"spec":{"maxReplicas":15}}'
```

## Troubleshooting

### Common Issues

**Pods stuck in Pending state**:
```bash
kubectl describe pod <pod-name> -n bitnet-inference
kubectl get events -n bitnet-inference --sort-by=.metadata.creationTimestamp
```

**High resource usage**:
```bash
kubectl top pods -n bitnet-inference
kubectl describe vpa bitnet-server-vpa -n bitnet-inference
```

**Ingress not accessible**:
```bash
kubectl describe ingress bitnet-server-ingress -n bitnet-inference
kubectl logs -f deployment/nginx-ingress-controller -n ingress-nginx
```

For comprehensive troubleshooting, see the [Kubernetes Troubleshooting Guide](../troubleshooting/kubernetes-deployment-issues.md).
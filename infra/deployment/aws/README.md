# AWS Deployment Guide

This guide covers deploying BitNet on Amazon Web Services (AWS) using various services including EKS, ECS, and EC2.

## Prerequisites

- AWS CLI configured with appropriate permissions
- kubectl installed and configured
- Docker installed
- Helm 3.x installed

## EKS Deployment

### 1. Create EKS Cluster

```bash
# Create EKS cluster with GPU node groups
eksctl create cluster \
  --name bitnet-cluster \
  --version 1.28 \
  --region us-west-2 \
  --nodegroup-name cpu-nodes \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Add GPU node group
eksctl create nodegroup \
  --cluster bitnet-cluster \
  --region us-west-2 \
  --name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 5 \
  --node-ami-family AmazonLinux2 \
  --node-labels accelerator=nvidia-tesla-v100
```

### 2. Install NVIDIA Device Plugin

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

### 3. Deploy BitNet using Helm

```bash
# Add BitNet Helm repository
helm repo add bitnet https://charts.bitnet.rs
helm repo update

# Install BitNet with GPU support
helm install bitnet bitnet/bitnet \
  --namespace bitnet \
  --create-namespace \
  --set gpu.enabled=true \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=bitnet.yourdomain.com \
  --set persistence.models.storageClass=gp3 \
  --set persistence.cache.storageClass=gp3
```

### 4. Configure Load Balancer

```bash
# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=bitnet-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

## ECS Deployment

### 1. Create ECS Cluster

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name bitnet-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

### 2. ECS Task Definition

Create `ecs-task-definition.json`:

```json
{
  "family": "bitnet-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "bitnet",
      "image": "bitnet:cpu-latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "RUST_LOG",
          "value": "info"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bitnet",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### 3. Create ECS Service

```bash
aws ecs create-service \
  --cluster bitnet-cluster \
  --service-name bitnet-service \
  --task-definition bitnet-task \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abcdef],assignPublicIp=ENABLED}"
```

## EC2 Deployment

### 1. Launch EC2 Instances

```bash
# Launch CPU instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type m5.2xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --user-data file://user-data-cpu.sh

# Launch GPU instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type p3.2xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --user-data file://user-data-gpu.sh
```

### 2. User Data Scripts

Create `user-data-cpu.sh`:

```bash
#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Pull and run BitNet
docker pull bitnet:cpu-latest
docker run -d \
  --name bitnet \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /opt/bitnet/models:/app/models:ro \
  -v /opt/bitnet/config:/app/config:ro \
  bitnet:cpu-latest
```

Create `user-data-gpu.sh`:

```bash
#!/bin/bash
yum update -y
yum install -y docker

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
yum install -y nvidia-docker2
systemctl restart docker

# Pull and run BitNet with GPU
docker pull bitnet:gpu-latest
docker run -d \
  --name bitnet \
  --restart unless-stopped \
  --gpus all \
  -p 8080:8080 \
  -v /opt/bitnet/models:/app/models:ro \
  -v /opt/bitnet/config:/app/config:ro \
  bitnet:gpu-latest
```

## Auto Scaling Configuration

### 1. EKS Auto Scaling

```yaml
# cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/bitnet-cluster
```

### 2. Application Load Balancer

```yaml
# alb-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bitnet-alb
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/load-balancer-attributes: idle_timeout.timeout_seconds=300
spec:
  rules:
  - host: bitnet.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bitnet-service
            port:
              number: 8080
```

## Monitoring and Logging

### 1. CloudWatch Integration

```bash
# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cloudwatch-namespace.yaml

kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-daemonset.yaml
```

### 2. Prometheus and Grafana

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123
```

## Security Best Practices

### 1. IAM Roles and Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::bitnet-models/*",
        "arn:aws:s3:::bitnet-models"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

### 2. Network Security

```bash
# Create security group for BitNet
aws ec2 create-security-group \
  --group-name bitnet-sg \
  --description "Security group for BitNet inference service"

# Allow HTTP traffic
aws ec2 authorize-security-group-ingress \
  --group-id sg-12345678 \
  --protocol tcp \
  --port 8080 \
  --cidr 0.0.0.0/0

# Allow metrics traffic
aws ec2 authorize-security-group-ingress \
  --group-id sg-12345678 \
  --protocol tcp \
  --port 9090 \
  --source-group sg-87654321
```

## Cost Optimization

### 1. Spot Instances

```bash
# Create spot instance node group
eksctl create nodegroup \
  --cluster bitnet-cluster \
  --name spot-nodes \
  --node-type m5.large,m5.xlarge,m4.large \
  --nodes 2 \
  --nodes-min 0 \
  --nodes-max 10 \
  --spot
```

### 2. Reserved Instances

Consider purchasing Reserved Instances for predictable workloads to reduce costs by up to 75%.

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure NVIDIA device plugin is installed
2. **Out of memory**: Increase instance memory or reduce model size
3. **Slow startup**: Use model pre-warming or increase startup probe timeout
4. **Network timeouts**: Adjust load balancer timeout settings

### Debugging Commands

```bash
# Check pod logs
kubectl logs -f deployment/bitnet-cpu -n bitnet

# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check resource usage
kubectl top pods -n bitnet
```
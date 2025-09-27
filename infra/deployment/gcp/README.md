# Google Cloud Platform (GCP) Deployment Guide

This guide covers deploying BitNet on Google Cloud Platform using GKE, Cloud Run, and Compute Engine.

## Prerequisites

- Google Cloud SDK (gcloud) installed and configured
- kubectl installed
- Docker installed
- Helm 3.x installed
- Project with billing enabled

## GKE Deployment

### 1. Create GKE Cluster

```bash
# Set project and region
export PROJECT_ID=your-project-id
export REGION=us-central1
export CLUSTER_NAME=bitnet-cluster

# Create GKE cluster with GPU support
gcloud container clusters create $CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$REGION-a \
  --machine-type=n1-standard-4 \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade \
  --addons=HorizontalPodAutoscaling,HttpLoadBalancing

# Create GPU node pool
gcloud container node-pools create gpu-pool \
  --project=$PROJECT_ID \
  --cluster=$CLUSTER_NAME \
  --zone=$REGION-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=1 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=5 \
  --node-labels=accelerator=nvidia-tesla-t4
```

### 2. Install NVIDIA GPU Drivers

```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### 3. Deploy BitNet using Helm

```bash
# Get cluster credentials
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$REGION-a

# Deploy BitNet
helm install bitnet ./helm/bitnet \
  --namespace bitnet \
  --create-namespace \
  --set gpu.enabled=true \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=bitnet.yourdomain.com \
  --set persistence.models.storageClass=standard-rwo \
  --set persistence.cache.storageClass=standard-rwo
```

### 4. Configure Load Balancer

```bash
# Create static IP
gcloud compute addresses create bitnet-ip --global

# Get the IP address
gcloud compute addresses describe bitnet-ip --global --format="value(address)"

# Update ingress with static IP
kubectl annotate ingress bitnet-ingress \
  kubernetes.io/ingress.global-static-ip-name=bitnet-ip \
  -n bitnet
```

## Cloud Run Deployment

### 1. Build and Push Container

```bash
# Build container
docker build -f docker/Dockerfile.cpu -t gcr.io/$PROJECT_ID/bitnet:cpu-latest .

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/bitnet:cpu-latest
```

### 2. Deploy to Cloud Run

```bash
# Deploy CPU version
gcloud run deploy bitnet-cpu \
  --image=gcr.io/$PROJECT_ID/bitnet:cpu-latest \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=8Gi \
  --cpu=4 \
  --concurrency=10 \
  --timeout=300 \
  --set-env-vars="RUST_LOG=info,BITNET_MODEL_PATH=/app/models"

# Get service URL
gcloud run services describe bitnet-cpu --region=$REGION --format="value(status.url)"
```

### 3. Cloud Run with GPU (Preview)

```bash
# Deploy GPU version (requires allowlisting)
gcloud run deploy bitnet-gpu \
  --image=gcr.io/$PROJECT_ID/bitnet:gpu-latest \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=16Gi \
  --cpu=8 \
  --gpu=1 \
  --gpu-type=nvidia-t4 \
  --concurrency=5 \
  --timeout=600
```

## Compute Engine Deployment

### 1. Create VM Instances

```bash
# Create CPU instance
gcloud compute instances create bitnet-cpu \
  --project=$PROJECT_ID \
  --zone=$REGION-a \
  --machine-type=n1-standard-4 \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=50GB \
  --metadata-from-file startup-script=startup-cpu.sh \
  --tags=bitnet-server

# Create GPU instance
gcloud compute instances create bitnet-gpu \
  --project=$PROJECT_ID \
  --zone=$REGION-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=100GB \
  --metadata-from-file startup-script=startup-gpu.sh \
  --maintenance-policy=TERMINATE \
  --tags=bitnet-server
```

### 2. Startup Scripts

Create `startup-cpu.sh`:

```bash
#!/bin/bash

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Pull and run BitNet
docker pull gcr.io/PROJECT_ID/bitnet:cpu-latest
docker run -d \
  --name bitnet \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /opt/bitnet/models:/app/models:ro \
  -v /opt/bitnet/config:/app/config:ro \
  gcr.io/PROJECT_ID/bitnet:cpu-latest
```

Create `startup-gpu.sh`:

```bash
#!/bin/bash

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# Pull and run BitNet with GPU
docker pull gcr.io/PROJECT_ID/bitnet:gpu-latest
docker run -d \
  --name bitnet \
  --restart unless-stopped \
  --gpus all \
  -p 8080:8080 \
  -v /opt/bitnet/models:/app/models:ro \
  -v /opt/bitnet/config:/app/config:ro \
  gcr.io/PROJECT_ID/bitnet:gpu-latest
```

### 3. Create Firewall Rules

```bash
# Allow HTTP traffic
gcloud compute firewall-rules create allow-bitnet-http \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --target-tags bitnet-server \
  --description "Allow HTTP traffic to BitNet servers"

# Allow metrics traffic
gcloud compute firewall-rules create allow-bitnet-metrics \
  --allow tcp:9090 \
  --source-ranges 10.0.0.0/8 \
  --target-tags bitnet-server \
  --description "Allow metrics traffic to BitNet servers"
```

## Auto Scaling

### 1. GKE Cluster Autoscaler

```bash
# Enable cluster autoscaler
gcloud container clusters update $CLUSTER_NAME \
  --zone=$REGION-a \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10
```

### 2. Managed Instance Groups

```bash
# Create instance template
gcloud compute instance-templates create bitnet-template \
  --machine-type=n1-standard-4 \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --metadata-from-file startup-script=startup-cpu.sh \
  --tags=bitnet-server

# Create managed instance group
gcloud compute instance-groups managed create bitnet-group \
  --template=bitnet-template \
  --size=3 \
  --zone=$REGION-a

# Set up autoscaling
gcloud compute instance-groups managed set-autoscaling bitnet-group \
  --zone=$REGION-a \
  --max-num-replicas=10 \
  --min-num-replicas=2 \
  --target-cpu-utilization=0.7
```

## Load Balancing

### 1. HTTP(S) Load Balancer

```bash
# Create health check
gcloud compute health-checks create http bitnet-health-check \
  --port=8080 \
  --request-path=/health

# Create backend service
gcloud compute backend-services create bitnet-backend \
  --protocol=HTTP \
  --health-checks=bitnet-health-check \
  --global

# Add instance group to backend
gcloud compute backend-services add-backend bitnet-backend \
  --instance-group=bitnet-group \
  --instance-group-zone=$REGION-a \
  --global

# Create URL map
gcloud compute url-maps create bitnet-map \
  --default-service=bitnet-backend

# Create HTTP proxy
gcloud compute target-http-proxies create bitnet-proxy \
  --url-map=bitnet-map

# Create forwarding rule
gcloud compute forwarding-rules create bitnet-rule \
  --global \
  --target-http-proxy=bitnet-proxy \
  --ports=80
```

## Monitoring and Logging

### 1. Cloud Monitoring

```bash
# Enable monitoring API
gcloud services enable monitoring.googleapis.com

# Create notification channel
gcloud alpha monitoring channels create \
  --display-name="BitNet Alerts" \
  --type=email \
  --channel-labels=email_address=admin@yourdomain.com
```

### 2. Cloud Logging

```bash
# Enable logging API
gcloud services enable logging.googleapis.com

# Create log sink
gcloud logging sinks create bitnet-sink \
  bigquery.googleapis.com/projects/$PROJECT_ID/datasets/bitnet_logs \
  --log-filter='resource.type="gce_instance" AND labels."bitnet"'
```

### 3. Prometheus and Grafana on GKE

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123 \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=standard-rwo
```

## Security

### 1. IAM and Service Accounts

```bash
# Create service account
gcloud iam service-accounts create bitnet-sa \
  --display-name="BitNet Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:bitnet-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:bitnet-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/monitoring.metricWriter"
```

### 2. VPC and Network Security

```bash
# Create VPC
gcloud compute networks create bitnet-vpc --subnet-mode=custom

# Create subnet
gcloud compute networks subnets create bitnet-subnet \
  --network=bitnet-vpc \
  --range=10.1.0.0/16 \
  --region=$REGION

# Create firewall rules
gcloud compute firewall-rules create bitnet-internal \
  --network=bitnet-vpc \
  --allow=tcp,udp,icmp \
  --source-ranges=10.1.0.0/16

gcloud compute firewall-rules create bitnet-ssh \
  --network=bitnet-vpc \
  --allow=tcp:22 \
  --source-ranges=0.0.0.0/0
```

## Cost Optimization

### 1. Preemptible Instances

```bash
# Create preemptible node pool
gcloud container node-pools create preemptible-pool \
  --cluster=$CLUSTER_NAME \
  --zone=$REGION-a \
  --machine-type=n1-standard-2 \
  --preemptible \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=5
```

### 2. Committed Use Discounts

Consider purchasing Committed Use Discounts for predictable workloads to save up to 57%.

### 3. Cloud Storage for Models

```bash
# Create bucket for models
gsutil mb gs://$PROJECT_ID-bitnet-models

# Upload models
gsutil -m cp -r ./models/* gs://$PROJECT_ID-bitnet-models/

# Set up model loading from GCS
kubectl create secret generic gcs-key \
  --from-file=key.json=path/to/service-account-key.json \
  -n bitnet
```

## Disaster Recovery

### 1. Multi-Region Deployment

```bash
# Create cluster in secondary region
export SECONDARY_REGION=us-east1

gcloud container clusters create $CLUSTER_NAME-dr \
  --project=$PROJECT_ID \
  --zone=$SECONDARY_REGION-a \
  --machine-type=n1-standard-4 \
  --num-nodes=2
```

### 2. Backup and Restore

```bash
# Backup cluster configuration
kubectl get all -n bitnet -o yaml > bitnet-backup.yaml

# Create scheduled backups
gcloud compute disks snapshot bitnet-models-disk \
  --snapshot-names=bitnet-models-$(date +%Y%m%d) \
  --zone=$REGION-a
```

## Troubleshooting

### Common Issues

1. **GPU quota exceeded**: Request quota increase in IAM & Admin
2. **Image pull errors**: Check Container Registry permissions
3. **Load balancer timeout**: Increase backend timeout settings
4. **Out of disk space**: Increase persistent disk size

### Debugging Commands

```bash
# Check cluster status
gcloud container clusters describe $CLUSTER_NAME --zone=$REGION-a

# Check node status
kubectl describe nodes

# Check pod logs
kubectl logs -f deployment/bitnet-cpu -n bitnet

# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu
```
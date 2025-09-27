# Microsoft Azure Deployment Guide

This guide covers deploying BitNet on Microsoft Azure using AKS, Container Instances, and Virtual Machines.

## Prerequisites

- Azure CLI installed and configured
- kubectl installed
- Docker installed
- Helm 3.x installed
- Azure subscription with appropriate permissions

## AKS Deployment

### 1. Create AKS Cluster

```bash
# Set variables
export RESOURCE_GROUP=bitnet-rg
export CLUSTER_NAME=bitnet-cluster
export LOCATION=eastus

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create AKS cluster
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpupool \
  --node-count 1 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 0 \
  --max-count 3 \
  --node-taints sku=gpu:NoSchedule
```

### 2. Install NVIDIA Device Plugin

```bash
# Get cluster credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

### 3. Deploy BitNet using Helm

```bash
# Deploy BitNet
helm install bitnet ./helm/bitnet \
  --namespace bitnet \
  --create-namespace \
  --set gpu.enabled=true \
  --set gpu.tolerations[0].key=sku \
  --set gpu.tolerations[0].value=gpu \
  --set gpu.tolerations[0].effect=NoSchedule \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=bitnet.yourdomain.com \
  --set persistence.models.storageClass=managed-premium \
  --set persistence.cache.storageClass=managed-premium
```

### 4. Configure Application Gateway

```bash
# Create Application Gateway
az network application-gateway create \
  --name bitnet-appgw \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_v2 \
  --public-ip-address bitnet-pip \
  --vnet-name bitnet-vnet \
  --subnet appgw-subnet \
  --capacity 2 \
  --http-settings-cookie-based-affinity Disabled \
  --frontend-port 80 \
  --http-settings-port 8080 \
  --http-settings-protocol Http

# Install Application Gateway Ingress Controller
helm repo add application-gateway-kubernetes-ingress https://appgwingress.blob.core.windows.net/ingress-azure-helm-package/
helm install ingress-azure application-gateway-kubernetes-ingress/ingress-azure \
  --namespace default \
  --set appgw.name=bitnet-appgw \
  --set appgw.resourceGroup=$RESOURCE_GROUP \
  --set appgw.subscriptionId=$(az account show --query id -o tsv) \
  --set armAuth.type=servicePrincipal \
  --set armAuth.secretJSON=$(az ad sp create-for-rbac --sdk-auth | base64 -w0)
```

## Container Instances Deployment

### 1. Create Container Registry

```bash
# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name bitnetregistry \
  --sku Basic \
  --admin-enabled true

# Login to registry
az acr login --name bitnetregistry

# Build and push images
docker build -f docker/Dockerfile.cpu -t bitnetregistry.azurecr.io/bitnet:cpu-latest .
docker push bitnetregistry.azurecr.io/bitnet:cpu-latest

docker build -f docker/Dockerfile.gpu -t bitnetregistry.azurecr.io/bitnet:gpu-latest .
docker push bitnetregistry.azurecr.io/bitnet:gpu-latest
```

### 2. Deploy Container Instances

```bash
# Get registry credentials
ACR_USERNAME=$(az acr credential show --name bitnetregistry --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name bitnetregistry --query passwords[0].value -o tsv)

# Create CPU container instance
az container create \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-cpu \
  --image bitnetregistry.azurecr.io/bitnet:cpu-latest \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --cpu 4 \
  --memory 8 \
  --ports 8080 9090 \
  --dns-name-label bitnet-cpu \
  --environment-variables RUST_LOG=info BITNET_MODEL_PATH=/app/models \
  --restart-policy Always

# Create GPU container instance (requires quota)
az container create \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-gpu \
  --image bitnetregistry.azurecr.io/bitnet:gpu-latest \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --cpu 8 \
  --memory 16 \
  --gpu-count 1 \
  --gpu-sku V100 \
  --ports 8080 9090 \
  --dns-name-label bitnet-gpu \
  --environment-variables RUST_LOG=info BITNET_MODEL_PATH=/app/models CUDA_VISIBLE_DEVICES=0 \
  --restart-policy Always
```

## Virtual Machine Deployment

### 1. Create Virtual Machines

```bash
# Create CPU VM
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-cpu-vm \
  --image UbuntuLTS \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --custom-data startup-cpu.sh \
  --public-ip-sku Standard

# Create GPU VM
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-gpu-vm \
  --image microsoft-dsvm:ubuntu-1804:1804:latest \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --custom-data startup-gpu.sh \
  --public-ip-sku Standard
```

### 2. Startup Scripts

Create `startup-cpu.sh`:

```bash
#!/bin/bash
apt-get update
apt-get install -y docker.io
systemctl start docker
systemctl enable docker
usermod -aG docker azureuser

# Login to ACR
echo "ACR_PASSWORD" | docker login bitnetregistry.azurecr.io -u "ACR_USERNAME" --password-stdin

# Pull and run BitNet
docker pull bitnetregistry.azurecr.io/bitnet:cpu-latest
docker run -d \
  --name bitnet \
  --restart unless-stopped \
  -p 8080:8080 \
  -p 9090:9090 \
  -v /opt/bitnet/models:/app/models:ro \
  -v /opt/bitnet/config:/app/config:ro \
  bitnetregistry.azurecr.io/bitnet:cpu-latest
```

Create `startup-gpu.sh`:

```bash
#!/bin/bash
# NVIDIA drivers are pre-installed on DSVM

# Install Docker
apt-get update
apt-get install -y docker.io
systemctl start docker
systemctl enable docker
usermod -aG docker azureuser

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# Login to ACR
echo "ACR_PASSWORD" | docker login bitnetregistry.azurecr.io -u "ACR_USERNAME" --password-stdin

# Pull and run BitNet with GPU
docker pull bitnetregistry.azurecr.io/bitnet:gpu-latest
docker run -d \
  --name bitnet \
  --restart unless-stopped \
  --gpus all \
  -p 8080:8080 \
  -p 9090:9090 \
  -v /opt/bitnet/models:/app/models:ro \
  -v /opt/bitnet/config:/app/config:ro \
  bitnetregistry.azurecr.io/bitnet:gpu-latest
```

### 3. Configure Network Security

```bash
# Create network security group
az network nsg create \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-nsg

# Allow HTTP traffic
az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name bitnet-nsg \
  --name allow-http \
  --protocol tcp \
  --priority 1000 \
  --destination-port-range 8080 \
  --access allow

# Allow metrics traffic
az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name bitnet-nsg \
  --name allow-metrics \
  --protocol tcp \
  --priority 1001 \
  --destination-port-range 9090 \
  --source-address-prefix VirtualNetwork \
  --access allow
```

## Auto Scaling

### 1. Virtual Machine Scale Sets

```bash
# Create scale set
az vmss create \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-vmss \
  --image UbuntuLTS \
  --vm-sku Standard_D2s_v3 \
  --instance-count 3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --custom-data startup-cpu.sh \
  --load-balancer bitnet-lb \
  --backend-pool-name bitnet-pool

# Configure autoscaling
az monitor autoscale create \
  --resource-group $RESOURCE_GROUP \
  --resource bitnet-vmss \
  --resource-type Microsoft.Compute/virtualMachineScaleSets \
  --name bitnet-autoscale \
  --min-count 2 \
  --max-count 10 \
  --count 3

# Add scale-out rule
az monitor autoscale rule create \
  --resource-group $RESOURCE_GROUP \
  --autoscale-name bitnet-autoscale \
  --condition "Percentage CPU > 70 avg 5m" \
  --scale out 2

# Add scale-in rule
az monitor autoscale rule create \
  --resource-group $RESOURCE_GROUP \
  --autoscale-name bitnet-autoscale \
  --condition "Percentage CPU < 30 avg 5m" \
  --scale in 1
```

### 2. Load Balancer Configuration

```bash
# Create load balancer
az network lb create \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-lb \
  --sku Standard \
  --public-ip-address bitnet-pip \
  --frontend-ip-name bitnet-frontend \
  --backend-pool-name bitnet-pool

# Create health probe
az network lb probe create \
  --resource-group $RESOURCE_GROUP \
  --lb-name bitnet-lb \
  --name bitnet-health \
  --protocol http \
  --port 8080 \
  --path /health

# Create load balancing rule
az network lb rule create \
  --resource-group $RESOURCE_GROUP \
  --lb-name bitnet-lb \
  --name bitnet-rule \
  --protocol tcp \
  --frontend-port 80 \
  --backend-port 8080 \
  --frontend-ip-name bitnet-frontend \
  --backend-pool-name bitnet-pool \
  --probe-name bitnet-health
```

## Monitoring and Logging

### 1. Azure Monitor

```bash
# Enable monitoring for AKS
az aks enable-addons \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --addons monitoring

# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name bitnet-workspace \
  --location $LOCATION
```

### 2. Application Insights

```bash
# Create Application Insights
az extension add -n application-insights
az monitor app-insights component create \
  --app bitnet-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --application-type web

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app bitnet-insights \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey -o tsv)
```

### 3. Prometheus and Grafana

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123 \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=managed-premium
```

## Security

### 1. Azure Active Directory Integration

```bash
# Enable AAD integration for AKS
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --enable-aad \
  --aad-admin-group-object-ids $(az ad group show --group "AKS Admins" --query objectId -o tsv)
```

### 2. Key Vault Integration

```bash
# Create Key Vault
az keyvault create \
  --name bitnet-kv \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Store secrets
az keyvault secret set \
  --vault-name bitnet-kv \
  --name model-api-key \
  --value "your-secret-key"

# Install CSI driver
helm repo add csi-secrets-store-provider-azure https://azure.github.io/secrets-store-csi-driver-provider-azure/charts
helm install csi csi-secrets-store-provider-azure/csi-secrets-store-provider-azure \
  --namespace kube-system
```

### 3. Network Policies

```bash
# Enable network policy
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --network-policy azure

# Apply network policy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: bitnet-netpol
  namespace: bitnet
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: bitnet
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - {}
EOF
```

## Cost Optimization

### 1. Spot Instances

```bash
# Create spot node pool
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name spotpool \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1 \
  --node-count 2 \
  --node-vm-size Standard_D2s_v3 \
  --enable-cluster-autoscaler \
  --min-count 0 \
  --max-count 5
```

### 2. Reserved Instances

Consider purchasing Reserved VM Instances for predictable workloads to save up to 72%.

### 3. Azure Storage for Models

```bash
# Create storage account
az storage account create \
  --name bitnetmodels \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

# Create container
az storage container create \
  --name models \
  --account-name bitnetmodels

# Upload models
az storage blob upload-batch \
  --destination models \
  --source ./models \
  --account-name bitnetmodels
```

## Disaster Recovery

### 1. Multi-Region Deployment

```bash
# Create secondary region resources
export SECONDARY_REGION=westus2
export SECONDARY_RG=bitnet-dr-rg

az group create --name $SECONDARY_RG --location $SECONDARY_REGION

az aks create \
  --resource-group $SECONDARY_RG \
  --name $CLUSTER_NAME-dr \
  --location $SECONDARY_REGION \
  --node-count 2 \
  --node-vm-size Standard_D2s_v3
```

### 2. Backup and Restore

```bash
# Create backup vault
az backup vault create \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-vault \
  --location $LOCATION

# Enable backup for VMs
az backup protection enable-for-vm \
  --resource-group $RESOURCE_GROUP \
  --vault-name bitnet-vault \
  --vm bitnet-cpu-vm \
  --policy-name DefaultPolicy
```

## Troubleshooting

### Common Issues

1. **GPU quota exceeded**: Request quota increase in Azure portal
2. **Image pull errors**: Check ACR authentication and permissions
3. **Load balancer timeout**: Increase backend timeout settings
4. **Out of disk space**: Increase managed disk size

### Debugging Commands

```bash
# Check cluster status
az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

# Check node status
kubectl describe nodes

# Check pod logs
kubectl logs -f deployment/bitnet-cpu -n bitnet

# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check container instance logs
az container logs --resource-group $RESOURCE_GROUP --name bitnet-cpu
```

### Performance Tuning

```bash
# Enable accelerated networking
az vm update \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-cpu-vm \
  --set networkProfile.networkInterfaces[0].enableAcceleratedNetworking=true

# Optimize disk performance
az disk update \
  --resource-group $RESOURCE_GROUP \
  --name bitnet-disk \
  --sku Premium_LRS
```
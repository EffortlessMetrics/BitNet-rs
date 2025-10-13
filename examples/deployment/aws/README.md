# AWS Deployment Guide for BitNet.rs

This guide covers deploying BitNet.rs on Amazon Web Services using various services.

## Deployment Options

1. **ECS (Elastic Container Service)** - Managed container orchestration
2. **EKS (Elastic Kubernetes Service)** - Managed Kubernetes
3. **Lambda** - Serverless deployment (for smaller models)
4. **EC2** - Direct virtual machine deployment

## Prerequisites

- AWS CLI configured with appropriate permissions
- Docker installed
- kubectl (for EKS deployment)

## Quick Deploy with ECS

```bash
# Build and push to ECR
./scripts/deploy-ecs.sh

# Deploy CloudFormation stack
aws cloudformation deploy \
  --template-file cloudformation/bitnet-ecs.yaml \
  --stack-name bitnet-rs-stack \
  --capabilities CAPABILITY_IAM
```

## Quick Deploy with EKS

```bash
# Create EKS cluster
eksctl create cluster --config-file eks/cluster.yaml

# Deploy application
kubectl apply -f kubernetes/
```

## Monitoring

- CloudWatch for logs and metrics
- X-Ray for distributed tracing
- Application Load Balancer health checks

## Cost Optimization

- Use Spot instances for non-critical workloads
- Configure auto-scaling policies
- Consider Reserved Instances for production

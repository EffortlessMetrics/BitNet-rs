# BitNet Deployment PowerShell Script
# Cross-platform deployment script for Windows environments

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("aws", "gcp", "azure", "local")]
    [string]$Platform = "local",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("cpu", "gpu", "both")]
    [string]$Variant = "both",
    
    [Parameter(Mandatory=$false)]
    [string]$Registry = "bitnet",
    
    [Parameter(Mandatory=$false)]
    [string]$Tag = "latest",
    
    [Parameter(Mandatory=$false)]
    [string]$Namespace = "bitnet",
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun,
    
    [Parameter(Mandatory=$false)]
    [switch]$Help
)

# Colors for output
$Red = "`e[31m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-Info {
    param([string]$Message)
    Write-Host "${Blue}[INFO]${Reset} $Message"
}

function Write-Success {
    param([string]$Message)
    Write-Host "${Green}[SUCCESS]${Reset} $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "${Yellow}[WARNING]${Reset} $Message"
}

function Write-Error {
    param([string]$Message)
    Write-Host "${Red}[ERROR]${Reset} $Message"
}

function Show-Help {
    @"
BitNet Deployment Script

USAGE:
    .\deploy.ps1 [OPTIONS]

OPTIONS:
    -Platform <aws|gcp|azure|local>    Target deployment platform (default: local)
    -Variant <cpu|gpu|both>            Deployment variant (default: both)
    -Registry <string>                 Docker registry (default: bitnet)
    -Tag <string>                      Image tag (default: latest)
    -Namespace <string>                Kubernetes namespace (default: bitnet)
    -DryRun                           Show what would be deployed without executing
    -Help                             Show this help message

EXAMPLES:
    # Deploy locally with Docker Compose
    .\deploy.ps1 -Platform local

    # Deploy to AWS EKS
    .\deploy.ps1 -Platform aws -Variant gpu

    # Dry run deployment to GCP
    .\deploy.ps1 -Platform gcp -DryRun

    # Deploy with custom registry
    .\deploy.ps1 -Registry myregistry.com/bitnet -Tag v1.0.0
"@
}

function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    $missing = @()
    
    # Check Docker
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        $missing += "docker"
    }
    
    # Check kubectl for Kubernetes deployments
    if ($Platform -ne "local" -and -not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        $missing += "kubectl"
    }
    
    # Check Helm for Kubernetes deployments
    if ($Platform -ne "local" -and -not (Get-Command helm -ErrorAction SilentlyContinue)) {
        $missing += "helm"
    }
    
    # Platform-specific checks
    switch ($Platform) {
        "aws" {
            if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
                $missing += "aws-cli"
            }
        }
        "gcp" {
            if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
                $missing += "gcloud"
            }
        }
        "azure" {
            if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
                $missing += "azure-cli"
            }
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Error "Missing prerequisites: $($missing -join ', ')"
        Write-Info "Please install the missing tools and try again."
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

function Deploy-Local {
    Write-Info "Deploying BitNet locally with Docker Compose..."
    
    $composeFile = "docker/docker-compose.yml"
    
    if (-not (Test-Path $composeFile)) {
        Write-Error "Docker Compose file not found: $composeFile"
        exit 1
    }
    
    # Set environment variables
    $env:BITNET_REGISTRY = $Registry
    $env:BITNET_TAG = $Tag
    
    if ($DryRun) {
        Write-Info "Dry run - would execute: docker-compose -f $composeFile up -d"
        return
    }
    
    try {
        # Pull images
        Write-Info "Pulling Docker images..."
        docker-compose -f $composeFile pull
        
        # Start services
        Write-Info "Starting services..."
        docker-compose -f $composeFile up -d
        
        # Wait for services to be ready
        Write-Info "Waiting for services to be ready..."
        Start-Sleep -Seconds 30
        
        # Check service health
        $cpuHealth = docker-compose -f $composeFile exec -T bitnet-cpu curl -f http://localhost:8080/health 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "CPU service is healthy"
        } else {
            Write-Warning "CPU service health check failed"
        }
        
        Write-Success "Local deployment completed"
        Write-Info "Services available at:"
        Write-Info "  CPU: http://localhost:8080"
        Write-Info "  GPU: http://localhost:8081"
        Write-Info "  Prometheus: http://localhost:9090"
        Write-Info "  Grafana: http://localhost:3000"
        
    } catch {
        Write-Error "Local deployment failed: $_"
        exit 1
    }
}

function Deploy-Kubernetes {
    param([string]$Platform)
    
    Write-Info "Deploying BitNet to $Platform with Kubernetes..."
    
    # Check if kubectl is configured
    try {
        kubectl cluster-info | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "kubectl is not configured or cluster is not accessible"
            exit 1
        }
    } catch {
        Write-Error "Failed to connect to Kubernetes cluster: $_"
        exit 1
    }
    
    # Create namespace
    if ($DryRun) {
        Write-Info "Dry run - would create namespace: $Namespace"
    } else {
        Write-Info "Creating namespace: $Namespace"
        kubectl create namespace $Namespace --dry-run=client -o yaml | kubectl apply -f -
    }
    
    # Deploy using Helm
    $helmArgs = @(
        "install", "bitnet", "./helm/bitnet"
        "--namespace", $Namespace
        "--set", "image.registry=$Registry"
        "--set", "image.tag=$Tag"
    )
    
    # Configure variant-specific settings
    switch ($Variant) {
        "cpu" {
            $helmArgs += "--set", "cpu.enabled=true"
            $helmArgs += "--set", "gpu.enabled=false"
        }
        "gpu" {
            $helmArgs += "--set", "cpu.enabled=false"
            $helmArgs += "--set", "gpu.enabled=true"
        }
        "both" {
            $helmArgs += "--set", "cpu.enabled=true"
            $helmArgs += "--set", "gpu.enabled=true"
        }
    }
    
    # Platform-specific configurations
    switch ($Platform) {
        "aws" {
            $helmArgs += "--set", "persistence.models.storageClass=gp3"
            $helmArgs += "--set", "ingress.className=alb"
        }
        "gcp" {
            $helmArgs += "--set", "persistence.models.storageClass=standard-rwo"
            $helmArgs += "--set", "ingress.className=gce"
        }
        "azure" {
            $helmArgs += "--set", "persistence.models.storageClass=managed-premium"
            $helmArgs += "--set", "ingress.className=azure/application-gateway"
        }
    }
    
    if ($DryRun) {
        $helmArgs += "--dry-run"
        Write-Info "Dry run - would execute: helm $($helmArgs -join ' ')"
    } else {
        Write-Info "Deploying with Helm..."
        & helm $helmArgs
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Helm deployment failed"
            exit 1
        }
    }
    
    if (-not $DryRun) {
        # Wait for deployment to be ready
        Write-Info "Waiting for deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/bitnet-cpu -n $Namespace
        
        if ($Variant -eq "gpu" -or $Variant -eq "both") {
            kubectl wait --for=condition=available --timeout=300s deployment/bitnet-gpu -n $Namespace
        }
        
        Write-Success "Kubernetes deployment completed"
        
        # Show service information
        Write-Info "Service information:"
        kubectl get services -n $Namespace
        
        # Show ingress information if available
        $ingress = kubectl get ingress -n $Namespace -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}' 2>$null
        if ($ingress) {
            Write-Info "Ingress endpoint: http://$ingress"
        }
    }
}

function Deploy-AWS {
    Write-Info "Configuring AWS-specific settings..."
    
    # Check AWS CLI configuration
    try {
        aws sts get-caller-identity | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "AWS CLI is not configured"
            exit 1
        }
    } catch {
        Write-Error "Failed to verify AWS credentials: $_"
        exit 1
    }
    
    Deploy-Kubernetes -Platform "aws"
}

function Deploy-GCP {
    Write-Info "Configuring GCP-specific settings..."
    
    # Check gcloud configuration
    try {
        gcloud auth list --filter=status:ACTIVE --format="value(account)" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "gcloud is not authenticated"
            exit 1
        }
    } catch {
        Write-Error "Failed to verify GCP credentials: $_"
        exit 1
    }
    
    Deploy-Kubernetes -Platform "gcp"
}

function Deploy-Azure {
    Write-Info "Configuring Azure-specific settings..."
    
    # Check Azure CLI configuration
    try {
        az account show | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Azure CLI is not logged in"
            exit 1
        }
    } catch {
        Write-Error "Failed to verify Azure credentials: $_"
        exit 1
    }
    
    Deploy-Kubernetes -Platform "azure"
}

function Main {
    if ($Help) {
        Show-Help
        return
    }
    
    Write-Info "Starting BitNet deployment..."
    Write-Info "Platform: $Platform"
    Write-Info "Variant: $Variant"
    Write-Info "Registry: $Registry"
    Write-Info "Tag: $Tag"
    Write-Info "Namespace: $Namespace"
    Write-Info "Dry Run: $DryRun"
    
    Test-Prerequisites
    
    switch ($Platform) {
        "local" { Deploy-Local }
        "aws" { Deploy-AWS }
        "gcp" { Deploy-GCP }
        "azure" { Deploy-Azure }
        default {
            Write-Error "Unsupported platform: $Platform"
            exit 1
        }
    }
    
    Write-Success "Deployment process completed!"
}

# Run main function
Main
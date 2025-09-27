#!/bin/bash
# BitNet.rs Docker Deployment Script
# Provides easy deployment options for different environments

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_COMPOSE_FILE="docker-compose.rust-primary.yml"
CROSSVAL_COMPOSE_FILE="docker-compose.crossval.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
BitNet.rs Docker Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy          Deploy BitNet.rs production environment
    deploy-gpu      Deploy with GPU support
    crossval        Deploy cross-validation environment
    stop            Stop all services
    restart         Restart all services
    logs            Show service logs
    status          Show service status
    cleanup         Remove all containers and volumes
    build           Build Docker images
    update          Update and restart services

Options:
    --profile PROFILE   Use specific docker-compose profile (gpu, tracing, logging)
    --env ENV          Environment (dev, staging, prod) [default: prod]
    --models-dir DIR   Directory containing model files [default: ./models]
    --config-dir DIR   Directory containing config files [default: ./config]
    --no-monitoring    Skip monitoring stack deployment
    --help             Show this help message

Examples:
    $0 deploy                    # Deploy production environment
    $0 deploy-gpu                # Deploy with GPU support
    $0 deploy --profile tracing  # Deploy with distributed tracing
    $0 crossval                  # Deploy cross-validation environment
    $0 logs bitnet-server        # Show BitNet server logs
    $0 status                    # Show all service status

Environment Variables:
    BITNET_MODELS_DIR     Directory containing model files
    BITNET_CONFIG_DIR     Directory containing configuration files
    GRAFANA_PASSWORD      Grafana admin password [default: admin123]
    DOCKER_REGISTRY       Docker registry for custom images
EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup directories
setup_directories() {
    local models_dir="${BITNET_MODELS_DIR:-${SCRIPT_DIR}/models}"
    local config_dir="${BITNET_CONFIG_DIR:-${SCRIPT_DIR}/config}"
    
    log_info "Setting up directories..."
    
    # Create directories if they don't exist
    mkdir -p "$models_dir"
    mkdir -p "$config_dir"
    mkdir -p "${SCRIPT_DIR}/logs"
    
    # Check if model files exist
    if [ ! "$(ls -A "$models_dir" 2>/dev/null)" ]; then
        log_warning "No model files found in $models_dir"
        log_info "Please place your GGUF model files in the models directory"
    fi
    
    # Create default configuration if it doesn't exist
    if [ ! -f "$config_dir/server.toml" ]; then
        log_info "Creating default server configuration..."
        cat > "$config_dir/server.toml" << 'EOF'
[server]
host = "0.0.0.0"
port = 8080
max_connections = 1000

[model]
path = "/app/models/model.gguf"
max_tokens = 2048

[inference]
temperature = 0.7
top_p = 0.9

[performance]
threads = 0
use_gpu = false

[logging]
level = "info"
format = "json"

[metrics]
enabled = true
prometheus_port = 9090
EOF
        log_success "Created default configuration at $config_dir/server.toml"
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$SCRIPT_DIR"
    
    # Build primary Rust image
    docker build -f Dockerfile.rust-primary -t bitnet-rs:latest "$PROJECT_ROOT"
    
    # Build GPU image if requested
    if [[ "$1" == *"gpu"* ]]; then
        log_info "Building GPU-enabled image..."
        docker build -f Dockerfile.rust-gpu -t bitnet-rs:gpu "$PROJECT_ROOT"
    fi
    
    # Build legacy image for cross-validation if requested
    if [[ "$1" == *"crossval"* ]]; then
        log_info "Building legacy cross-validation image..."
        docker build -f Dockerfile.legacy-crossval -t bitnet-legacy:crossval "$PROJECT_ROOT"
    fi
    
    log_success "Docker images built successfully"
}

# Deploy production environment
deploy_production() {
    local profile="${1:-}"
    local no_monitoring="${2:-false}"
    
    log_info "Deploying BitNet.rs production environment..."
    
    cd "$SCRIPT_DIR"
    
    # Set environment variables
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin123}"
    export COMPOSE_PROJECT_NAME="bitnet-rs"
    
    # Deploy services
    if [ "$no_monitoring" = "true" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d bitnet-server nginx redis
    else
        if [ -n "$profile" ]; then
            docker-compose -f "$DOCKER_COMPOSE_FILE" --profile "$profile" up -d
        else
            docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
        fi
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "BitNet.rs production environment deployed successfully"
    show_service_urls
}

# Deploy GPU environment
deploy_gpu() {
    log_info "Deploying BitNet.rs with GPU support..."
    
    # Check for NVIDIA Docker runtime
    if ! docker info | grep -q nvidia; then
        log_warning "NVIDIA Docker runtime not detected. GPU features may not work."
    fi
    
    cd "$SCRIPT_DIR"
    export COMPOSE_PROJECT_NAME="bitnet-rs-gpu"
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" --profile gpu up -d
    
    log_success "BitNet.rs GPU environment deployed successfully"
}

# Deploy cross-validation environment
deploy_crossval() {
    log_info "Deploying cross-validation environment..."
    
    cd "$SCRIPT_DIR"
    export COMPOSE_PROJECT_NAME="bitnet-crossval"
    
    docker-compose -f "$CROSSVAL_COMPOSE_FILE" up -d
    
    log_info "Running cross-validation tests..."
    docker-compose -f "$CROSSVAL_COMPOSE_FILE" up crossval-runner benchmark-runner
    
    log_success "Cross-validation environment deployed successfully"
    log_info "Results available at http://localhost:8082"
}

# Check service health
check_service_health() {
    local max_attempts=30
    local attempt=1
    
    log_info "Checking service health..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8080/health > /dev/null; then
            log_success "BitNet server is healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: Waiting for BitNet server..."
        sleep 5
        ((attempt++))
    done
    
    log_error "BitNet server health check failed after $max_attempts attempts"
    return 1
}

# Show service URLs
show_service_urls() {
    log_info "Service URLs:"
    echo "  BitNet API:      http://localhost:8080"
    echo "  Grafana:         http://localhost:3000 (admin/admin123)"
    echo "  Prometheus:      http://localhost:9090"
    echo "  Nginx Status:    http://localhost:8080/nginx_status"
}

# Show service status
show_status() {
    cd "$SCRIPT_DIR"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
}

# Show service logs
show_logs() {
    local service="${1:-}"
    cd "$SCRIPT_DIR"
    
    if [ -n "$service" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f "$service"
    else
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
    fi
}

# Stop services
stop_services() {
    log_info "Stopping BitNet.rs services..."
    cd "$SCRIPT_DIR"
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    docker-compose -f "$CROSSVAL_COMPOSE_FILE" down 2>/dev/null || true
    
    log_success "Services stopped"
}

# Restart services
restart_services() {
    log_info "Restarting BitNet.rs services..."
    stop_services
    sleep 5
    deploy_production
}

# Cleanup everything
cleanup() {
    log_warning "This will remove all containers, volumes, and data. Are you sure? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Cleaning up BitNet.rs deployment..."
        cd "$SCRIPT_DIR"
        
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v --remove-orphans
        docker-compose -f "$CROSSVAL_COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true
        
        # Remove images
        docker rmi bitnet-rs:latest bitnet-rs:gpu bitnet-legacy:crossval 2>/dev/null || true
        
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Update services
update_services() {
    log_info "Updating BitNet.rs services..."
    
    # Pull latest images
    cd "$SCRIPT_DIR"
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull
    
    # Rebuild custom images
    build_images
    
    # Restart services
    restart_services
    
    log_success "Services updated successfully"
}

# Main script logic
main() {
    local command="${1:-}"
    local profile=""
    local env="prod"
    local no_monitoring="false"
    
    # Parse arguments
    shift || true
    while [[ $# -gt 0 ]]; do
        case $1 in
            --profile)
                profile="$2"
                shift 2
                ;;
            --env)
                env="$2"
                shift 2
                ;;
            --no-monitoring)
                no_monitoring="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Execute command
    case $command in
        deploy)
            check_prerequisites
            setup_directories
            build_images
            deploy_production "$profile" "$no_monitoring"
            ;;
        deploy-gpu)
            check_prerequisites
            setup_directories
            build_images "gpu"
            deploy_gpu
            ;;
        crossval)
            check_prerequisites
            setup_directories
            build_images "crossval"
            deploy_crossval
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs "$1"
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup
            ;;
        build)
            build_images "$1"
            ;;
        update)
            update_services
            ;;
        help|--help)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
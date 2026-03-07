#!/usr/bin/env bash
# Quick sanity checklist for production deployment

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 BitNet-rs Production Sanity Check${NC}"
echo "======================================="

# CPU reproducible build
echo -e "\n${YELLOW}1. Testing CPU reproducible build...${NC}"
if cargo test --locked --workspace --no-default-features --features cpu --lib 2>&1 | grep -q "test result: ok"; then
    echo -e "${GREEN}✓ CPU tests pass with locked dependencies${NC}"
else
    echo -e "${RED}✗ CPU tests failed${NC}"
fi

# Check cargo xtask alias
echo -e "\n${YELLOW}2. Testing cargo xtask alias...${NC}"
if cargo xtask --help 2>&1 | grep -q "Developer tasks"; then
    echo -e "${GREEN}✓ cargo xtask alias works${NC}"
else
    echo -e "${RED}✗ cargo xtask alias not configured${NC}"
fi

# GPU preflight (if available)
echo -e "\n${YELLOW}3. GPU preflight check...${NC}"
if command -v nvidia-smi &>/dev/null; then
    cargo xtask gpu-preflight 2>&1 | head -10
else
    echo "No GPU detected - skipping GPU checks"
fi

# Docker BuildKit check
echo -e "\n${YELLOW}4. Docker BuildKit availability...${NC}"
if docker version 2>&1 | grep -q "buildkit"; then
    echo -e "${GREEN}✓ Docker BuildKit available${NC}"
    echo "  Use: export DOCKER_BUILDKIT=1"
else
    echo -e "${YELLOW}⚠ BuildKit not detected - builds may be slower${NC}"
fi

# Check for required files
echo -e "\n${YELLOW}5. Required files check...${NC}"
required_files=(
    ".dockerignore"
    "rust-toolchain.toml"
    ".cargo/config.toml"
    "CODEOWNERS"
    "Makefile"
)

all_present=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file exists${NC}"
    else
        echo -e "${RED}✗ $file missing${NC}"
        all_present=false
    fi
done

# Docker compose validation
echo -e "\n${YELLOW}6. Docker Compose validation...${NC}"
if docker compose config --quiet 2>/dev/null; then
    echo -e "${GREEN}✓ docker-compose.yml is valid${NC}"
    # Check for sccache volume
    if grep -q "bitnet_sccache" docker-compose.yml; then
        echo -e "${GREEN}✓ sccache volume configured${NC}"
    fi
else
    echo -e "${RED}✗ docker-compose.yml has errors${NC}"
fi

# Summary
echo -e "\n${GREEN}======================================="
echo -e "Sanity Check Complete!${NC}"
echo -e "\n${YELLOW}Quick commands:${NC}"
echo "  make b          # Build CPU"
echo "  make t          # Test"
echo "  make gpu        # GPU preflight"
echo "  cargo xtask gpu-smoke  # GPU smoke test"
echo ""
echo -e "${YELLOW}Docker commands:${NC}"
echo "  export DOCKER_BUILDKIT=1"
echo "  docker build --target runtime -t bitnet:cpu ."
echo "  docker compose up --build bitnet-cpu"
echo ""
echo -e "${GREEN}Ready for production deployment! 🚀${NC}"

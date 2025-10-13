#!/usr/bin/env bash
# The ULTIMATE one-click script
# Just run: ./start.sh

set -e

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 BitNet-rs Ultimate One-Click Start 🚀${NC}"
echo

# Check if this is first run
if [ ! -f ".initialized" ]; then
    echo -e "${GREEN}First run detected. Setting up everything...${NC}"
    ./deploy.sh quick
    touch .initialized
    echo
    echo -e "${GREEN}✨ Setup complete! BitNet-rs is ready to use.${NC}"
else
    echo -e "${GREEN}Starting BitNet-rs...${NC}"
    make run
fi

echo
echo -e "${BLUE}Quick commands:${NC}"
echo "  make run   - Run CLI"
echo "  make serve - Start server"
echo "  make test  - Run tests"
echo "  make help  - See all commands"
echo

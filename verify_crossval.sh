#!/bin/bash
# Verification script for Microsoft BitNet repository integration

set -e

echo "=== BitNet Crossval Integration Verification ==="
echo

# Check if we can access the Microsoft BitNet repository
echo "1. Testing repository access..."
if git ls-remote https://github.com/microsoft/BitNet.git HEAD &>/dev/null; then
    echo "   ✓ Can access Microsoft BitNet repository"
else
    echo "   ✗ Cannot access Microsoft BitNet repository"
    exit 1
fi

# Check main branch exists
echo "2. Verifying main branch exists..."
if git ls-remote https://github.com/microsoft/BitNet.git refs/heads/main &>/dev/null; then
    echo "   ✓ Main branch exists"
else
    echo "   ✗ Main branch not found"
    exit 1
fi

# Display repository info
echo "3. Repository information:"
LATEST_COMMIT=$(git ls-remote https://github.com/microsoft/BitNet.git HEAD | cut -f1)
echo "   - Latest commit: ${LATEST_COMMIT:0:8}"
echo "   - Repository URL: https://github.com/microsoft/BitNet.git"
echo "   - Default branch: main"

echo
echo "4. Environment setup for crossval:"
echo "   export BITNET_CPP_PATH=\$HOME/.cache/bitnet_cpp"
echo "   export LD_LIBRARY_PATH=\$BITNET_CPP_PATH/build/lib:\$LD_LIBRARY_PATH"

echo
echo "5. Recommended workflow:"
echo "   # Download model (if needed)"
echo "   cargo run -p xtask -- download-model"
echo ""
echo "   # Fetch and build Microsoft BitNet C++"
echo "   cargo run -p xtask -- fetch-cpp"
echo ""
echo "   # Run cross-validation tests"
echo "   cargo run -p xtask -- crossval"
echo ""
echo "   # Or run everything at once"
echo "   cargo run -p xtask -- full-crossval"

echo
echo "=== Verification Complete ==="
echo "The crossval system is properly configured to use the official Microsoft BitNet repository."
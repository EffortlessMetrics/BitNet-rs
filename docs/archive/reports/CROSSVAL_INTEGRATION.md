> **ARCHIVED DOCUMENT** (Archived: 2025-10-23)
>
> This is a historical Validation Report from active development (Sept-Oct 2025).
> **This document is no longer maintained and may contain outdated information.**
>
> **For current information, see:**
> - [Current Test Suite Documentation](../development/test-suite.md)
> - [CLAUDE.md](../../CLAUDE.md) — Project reference and status
> - [PR #475 Final Report](../../PR_475_FINAL_SUCCESS_REPORT.md) — Comprehensive implementation summary
> - [Current CI Documentation](../development/validation-ci.md) — Test suite and validation
>
> **Archive Note**: This report was archived during documentation cleanup to establish
> current docs as the single source of truth. The content below is preserved for
> historical reference and audit purposes.

---
# Cross-Validation Integration with Microsoft BitNet

## Summary

The BitNet-rs cross-validation system has been successfully integrated with the official Microsoft BitNet repository at `https://github.com/microsoft/BitNet.git`.

## Key Changes Made

### 1. Repository Configuration
- **Old**: Used non-existent tag `b1-65-ggml`
- **New**: Uses official Microsoft repository with `main` branch as default
- **URL**: `https://github.com/microsoft/BitNet.git`

### 2. Build System Updates
- Updated CMake flags from `LLAMA_*` to `BITNET_*` prefixes
- Made binary validation more flexible to handle various build artifacts
- Improved header discovery with multiple search paths
- Added better error reporting and fallback mechanisms

### 3. Environment Variables
- Supports both `BITNET_CPP_DIR` and legacy `BITNET_CPP_PATH`
- Automatic library path configuration
- Flexible include path detection

### 4. Improved Robustness
- Handles varying repository structures
- Validates any build artifacts (libraries or executables)
- Non-fatal warnings for missing optional components
- Clear error messages with troubleshooting guidance

## Usage

### Quick Start
```bash
# Complete cross-validation workflow
cargo run -p xtask -- full-crossval
```

### Step by Step
```bash
# 1. Download a model (if needed)
cargo run -p xtask -- download-model

# 2. Fetch and build Microsoft BitNet C++
cargo run -p xtask -- fetch-cpp

# 3. Run cross-validation tests
cargo run -p xtask -- crossval
```

### Advanced Usage
```bash
# Use a specific commit for reproducible builds
cargo run -p xtask -- fetch-cpp --tag <commit-hash>

# Force rebuild
cargo run -p xtask -- fetch-cpp --force --clean

# Use a specific model
cargo run -p xtask -- crossval --model path/to/model.gguf
```

## Environment Setup

```bash
# Required environment variables (set automatically by xtask)
export BITNET_CPP_PATH=$HOME/.cache/bitnet_cpp
export LD_LIBRARY_PATH=$BITNET_CPP_PATH/build/lib:$LD_LIBRARY_PATH
```

## Verification

Run the verification script to test the integration:
```bash
./verify_crossval.sh
```

## Files Modified

- `ci/fetch_bitnet_cpp.sh` - Updated repository URL and build process
- `xtask/src/main.rs` - Changed defaults and improved validation
- `crates/bitnet-sys/build.rs` - Flexible header discovery
- `crossval/build.rs` - Enhanced library linking
- `ci/bitnet_cpp_version.txt` - Updated to use main branch
- `ci/bitnet_cpp_checksums.txt` - Documentation for branch-based approach

## Benefits

1. **Official Support**: Uses the official Microsoft repository
2. **Always Up-to-date**: Tracks main branch for latest features
3. **Flexible**: Adapts to repository structure changes
4. **Robust**: Better error handling and recovery
5. **Production Ready**: Can pin to specific commits for stability

## Next Steps

For production deployments:
1. Pin to a specific commit hash instead of using `main`
2. Document the tested commit in your release notes
3. Consider caching the built C++ artifacts in CI/CD

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Setup BitNet Crossval
  run: |
    cargo run -p xtask -- fetch-cpp --tag ${{ env.BITNET_CPP_COMMIT }}
    cargo run -p xtask -- crossval
```

## Support

If you encounter issues:
1. Check that you can access https://github.com/microsoft/BitNet
2. Ensure you have CMake and a C++ compiler installed
3. Run `./verify_crossval.sh` to test the setup
4. Check the build logs in `$HOME/.cache/bitnet_cpp/build`

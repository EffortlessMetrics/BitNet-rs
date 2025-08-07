# BitNet.rs Patch Policy

This directory contains minimal patches applied to external dependencies, primarily BitNet.cpp for cross-validation purposes.

## 🎯 Policy Overview

**Our patch policy prioritizes upstream fixes over local patches:**

1. **Prefer upstream fixes**: Always try to fix issues upstream first
2. **Minimal patches**: Keep patches as small and focused as possible  
3. **Required metadata**: All patches must include upstream issue links
4. **Regular review**: Patches are reviewed weekly for cleanup opportunities
5. **Temporary nature**: Most patches should be temporary until upstream fixes

## 📋 Patch Requirements

Every patch file must include the following metadata in its header:

```patch
# Upstream-Issue: https://github.com/microsoft/BitNet/issues/123
# Reason: Fix compilation error on ARM64 platforms
# Status: temporary
# Created: 2024-01-15
# Review-By: 2024-02-15
# Author: BitNet.rs Team <team@bitnet.rs>

--- a/src/example.cpp
+++ b/src/example.cpp
@@ -10,7 +10,7 @@
 // Your patch content here
```

### Required Metadata Fields

- **Upstream-Issue**: Link to the upstream GitHub issue tracking this problem
- **Reason**: Brief explanation of why this patch is needed
- **Status**: One of `temporary`, `permanent`, `under-review`
- **Created**: Date when the patch was created (YYYY-MM-DD)
- **Review-By**: Date when the patch should be reviewed (YYYY-MM-DD)
- **Author**: Who created the patch

### Status Values

- **temporary**: Patch is temporary until upstream fix is available
- **permanent**: Patch addresses a design difference that won't be upstreamed
- **under-review**: Patch is being reviewed for upstream submission

## 🛠️ Creating Patches

### 1. Check if Upstream Fix Exists

Before creating a patch, always check:

```bash
# Search upstream issues
curl -s "https://api.github.com/repos/microsoft/BitNet/issues?q=your-search-terms"

# Check if fix is already in main branch
git log --oneline --grep="your-search-terms"
```

### 2. Create Upstream Issue

If no upstream issue exists, create one:

1. Go to https://github.com/microsoft/BitNet/issues
2. Create a detailed issue describing the problem
3. Include reproduction steps and proposed solution
4. Wait for upstream response before creating local patch

### 3. Create Minimal Patch

```bash
# Make your changes in a clean working directory
cd /path/to/bitnet_cpp

# Create the patch
git diff > ../patches/001-fix-arm64-compilation.patch

# Add required metadata to the patch header
cat > ../patches/001-fix-arm64-compilation.patch << 'EOF'
# Upstream-Issue: https://github.com/microsoft/BitNet/issues/123
# Reason: Fix compilation error on ARM64 platforms
# Status: temporary
# Created: 2024-01-15
# Review-By: 2024-02-15
# Author: BitNet.rs Team <team@bitnet.rs>

EOF

# Append the actual diff
git diff >> ../patches/001-fix-arm64-compilation.patch
```

### 4. Test the Patch

```bash
# Test patch application
cd /path/to/clean/bitnet_cpp
patch -p1 --dry-run < ../patches/001-fix-arm64-compilation.patch

# Test that it fixes the issue
make test
```

### 5. Submit for Review

Create a PR with your patch and include:

- Link to upstream issue
- Explanation of why the patch is needed
- Test results showing the fix works
- Timeline for upstream resolution

## 🔄 Patch Lifecycle

### Weekly Review Process

Every Monday, our automated system:

1. **Reviews patch age**: Identifies patches older than 30 days
2. **Checks upstream status**: Verifies if upstream fixes are available
3. **Creates review issues**: Generates tasks for patch cleanup
4. **Updates tracking**: Maintains patch lifecycle documentation

### Patch Cleanup

Patches should be removed when:

- Upstream fix is available and tested
- Issue is resolved in a different way
- Patch is no longer needed
- Upstream explicitly rejects the change

### Review Schedule

- **Daily**: Automated policy compliance checks
- **Weekly**: Patch lifecycle review and cleanup
- **Monthly**: Comprehensive patch audit and upstream sync

## 🚨 Policy Enforcement

### Automated Checks

Our CI system automatically:

- ✅ Validates patch format and metadata
- ✅ Checks upstream issue links are valid
- ✅ Ensures all required fields are present
- ✅ Tracks patch age and review dates
- ❌ Fails builds if patches violate policy

### Manual Review

All patches require:

- Code review from at least one maintainer
- Verification that upstream issue exists
- Confirmation that minimal approach is used
- Timeline for upstream resolution

## 📊 Current Status

<!-- This section is automatically updated by CI -->

**Last Updated**: Auto-updated by CI
**Total Patches**: 0
**Policy Compliance**: ✅ Perfect

## 🎯 Goals

Our patch policy aims to:

1. **Minimize maintenance burden**: Fewer patches = less maintenance
2. **Improve upstream relationship**: Contribute fixes back to the community
3. **Ensure reproducibility**: Clear tracking of all modifications
4. **Maintain quality**: High standards for patch creation and review

## 📚 Examples

### Good Patch Example

```patch
# Upstream-Issue: https://github.com/microsoft/BitNet/issues/456
# Reason: Fix memory leak in tensor allocation
# Status: temporary
# Created: 2024-01-15
# Review-By: 2024-02-15
# Author: BitNet.rs Team <team@bitnet.rs>

--- a/src/tensor.cpp
+++ b/src/tensor.cpp
@@ -45,6 +45,7 @@ Tensor::~Tensor() {
     if (data_) {
         free(data_);
+        data_ = nullptr;
     }
 }
```

### Bad Patch Example

```patch
# This patch is missing required metadata
# and makes unnecessary changes

--- a/src/tensor.cpp
+++ b/src/tensor.cpp
@@ -1,10 +1,15 @@
+// BitNet.rs modifications
+// TODO: Fix this properly
+
 #include "tensor.h"
 
 // Large, unfocused changes that could be upstreamed
```

## 🤝 Contributing

When contributing patches:

1. **Follow the policy**: Ensure all requirements are met
2. **Engage upstream**: Create issues and propose fixes upstream first
3. **Keep it minimal**: Make the smallest change that fixes the issue
4. **Document thoroughly**: Include clear metadata and reasoning
5. **Test extensively**: Verify the patch works and doesn't break anything

## 📞 Support

For questions about patch policy:

- **GitHub Issues**: https://github.com/microsoft/BitNet/issues
- **Discussions**: https://github.com/microsoft/BitNet/discussions
- **Documentation**: See `/docs/contributing.md`

---

**Remember**: The best patch is no patch. Always prefer upstream fixes!
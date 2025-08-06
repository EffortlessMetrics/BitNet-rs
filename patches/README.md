# BitNet C++ Patches

This directory contains minimal patches applied to the external BitNet C++ implementation for compatibility with the Rust cross-validation framework.

## Patch Policy

**Our strong preference is to avoid patches entirely.** Patches create maintenance burden and can break when the upstream C++ implementation is updated.

### When Patches Are Acceptable

Patches should only be used as a **last resort** when:

1. **FFI Compatibility**: The C++ code needs minimal changes to expose a C-compatible API
2. **Build System Integration**: Minor build script modifications for our specific use case
3. **Critical Bug Fixes**: Upstream bugs that block cross-validation and haven't been fixed upstream

### When Patches Are NOT Acceptable

- **Feature additions**: Contribute features upstream instead
- **Performance optimizations**: These belong in the upstream project
- **API changes**: Work with upstream to design better APIs
- **Coding style**: Accept upstream coding conventions

## Patch Requirements

Every patch in this directory MUST:

1. **Have an upstream issue**: Link to a GitHub issue in the upstream BitNet.cpp repository
2. **Be minimal**: Change only what's absolutely necessary
3. **Be documented**: Include clear rationale and expected upstream resolution
4. **Be temporary**: Have a plan for removal when upstream is fixed

## Patch Format

Patches should be in standard unified diff format with descriptive names:

```
001-expose-c-api.patch          # Exposes C API for FFI
002-fix-memory-leak-issue-123.patch  # References upstream issue #123
```

## Patch Application

Patches are automatically applied by `ci/apply_patches.sh`:

```bash
# Apply all patches in order
./ci/apply_patches.sh

# Apply specific patch
./ci/apply_patches.sh patches/001-expose-c-api.patch
```

## CI Enforcement

Our CI system enforces the patch policy:

- **Fails if patches exist without upstream issues**: Every patch must reference an upstream issue
- **Warns about patch age**: Old patches (>90 days) trigger warnings
- **Tracks patch lifecycle**: Automated reminders to check if upstream fixes are available

## Creating a New Patch

If you absolutely must create a patch:

1. **Create upstream issue first**: File an issue in the BitNet.cpp repository
2. **Make minimal changes**: Only change what's necessary for your specific need
3. **Generate the patch**:
   ```bash
   cd ~/.cache/bitnet_cpp
   git diff > ../../path/to/bitnet/patches/NNN-description-issue-XXX.patch
   ```
4. **Document the patch**: Add entry to this README with issue link and rationale
5. **Set up tracking**: Add the patch to our automated tracking system

## Current Patches

*This directory is intentionally empty.* 

If patches are added, they will be documented here with:
- Patch filename
- Upstream issue link
- Rationale for the patch
- Expected resolution timeline
- Maintenance contact

## Patch Lifecycle

1. **Creation**: Patch created with upstream issue reference
2. **Review**: Patch reviewed for minimality and necessity
3. **Application**: Patch automatically applied during C++ build
4. **Tracking**: Automated monitoring of upstream issue status
5. **Removal**: Patch removed when upstream fix is available

## Maintenance

Patches require ongoing maintenance:

- **Weekly**: Check upstream issue status for resolution
- **Monthly**: Verify patches still apply cleanly to latest upstream
- **Quarterly**: Review patch necessity and explore alternatives

## Alternative Approaches

Before creating a patch, consider these alternatives:

1. **Wrapper functions**: Create C wrapper functions in a separate file
2. **Build-time configuration**: Use preprocessor flags to enable needed functionality
3. **Runtime adaptation**: Adapt to the existing C++ API in Rust code
4. **Upstream contribution**: Contribute the needed changes to upstream first

## Contact

For questions about patch policy or specific patches:
- Create an issue in this repository
- Tag the cross-validation team
- Reference this document in discussions

Remember: **The best patch is no patch.** Always explore alternatives first.
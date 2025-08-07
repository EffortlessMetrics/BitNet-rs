# BitNet C++ Version Management

This document describes the version management system for the external BitNet C++ dependency used in cross-validation.

## Overview

The BitNet Rust implementation uses the original Microsoft BitNet C++ implementation for cross-validation purposes. To ensure reproducible builds and testing, we pin to specific versions of the C++ implementation.

## Files

- `bitnet_cpp_version.txt` - Current pinned version
- `bitnet_cpp_checksums.txt` - SHA256 checksums for verification
- `bump_bitnet_tag.sh` / `bump_bitnet_tag.ps1` - Version management scripts
- `fetch_bitnet_cpp.sh` / `fetch_bitnet_cpp.ps1` - Download and build scripts

## Version Management Commands

### Check Current Version

```bash
# Unix/Linux/macOS
./ci/bump_bitnet_tag.sh current

# Windows
.\ci\bump_bitnet_tag.ps1 current
```

### List Available Versions

```bash
# Unix/Linux/macOS
./ci/bump_bitnet_tag.sh list

# Windows
.\ci\bump_bitnet_tag.ps1 list
```

### Update to Specific Version

```bash
# Unix/Linux/macOS
./ci/bump_bitnet_tag.sh update v1.2.0

# Windows
.\ci\bump_bitnet_tag.ps1 update v1.2.0
```

### Update to Latest Release

```bash
# Unix/Linux/macOS
./ci/bump_bitnet_tag.sh latest

# Windows
.\ci\bump_bitnet_tag.ps1 latest
```

### Check for Updates

```bash
# Unix/Linux/macOS
./ci/bump_bitnet_tag.sh check

# Windows
.\ci\bump_bitnet_tag.ps1 check
```

### Validate Current Setup

```bash
# Unix/Linux/macOS
./ci/bump_bitnet_tag.sh validate

# Windows
.\ci\bump_bitnet_tag.ps1 validate
```

### Generate Checksums

```bash
# Unix/Linux/macOS
./ci/bump_bitnet_tag.sh generate-checksums

# Windows
.\ci\bump_bitnet_tag.ps1 generate-checksums
```

## Update Process

When updating to a new version:

1. **Check available versions**: `./ci/bump_bitnet_tag.sh list`
2. **Update to new version**: `./ci/bump_bitnet_tag.sh update v1.2.0`
3. **Test cross-validation**: `cargo test --features crossval`
4. **Commit changes**: Commit the updated version and checksum files

### Automated Steps

The update process automatically:

1. Validates the new version exists upstream
2. Updates the version file
3. Cleans the existing cache
4. Downloads and builds the new version
5. Generates new checksums
6. Validates the installation

## Version Selection Strategy

### Semantic Versioning

Prefer semantic version tags (e.g., `v1.2.0`) over other tags:

- **Major versions** (v1.0.0 → v2.0.0): May have breaking changes
- **Minor versions** (v1.0.0 → v1.1.0): New features, should be compatible
- **Patch versions** (v1.0.0 → v1.0.1): Bug fixes, should be safe

### Update Frequency

- **Patch updates**: Apply promptly for bug fixes
- **Minor updates**: Apply after testing for new features
- **Major updates**: Apply carefully after thorough testing

## Checksum Verification

Checksums provide integrity verification:

### Generation

Checksums are automatically generated for:
- All `.cpp`, `.h`, `.hpp` source files
- `CMakeLists.txt` build files
- Other critical configuration files

### Validation

Checksums are validated during:
- Version updates
- Build processes (if enabled)
- Manual validation commands

### Security

Checksums help detect:
- Corrupted downloads
- Unexpected changes in upstream code
- Supply chain attacks

## Troubleshooting

### Version Not Found

```
ERROR: Version 'v1.2.0' does not exist upstream
```

**Solution**: Check available versions with `list` command

### Cache Issues

```
ERROR: Cache directory not found
```

**Solution**: Run the fetch script to download the C++ implementation

### Checksum Failures

```
ERROR: Checksum validation failed
```

**Solutions**:
1. Re-download: `./ci/fetch_bitnet_cpp.sh --force`
2. Regenerate checksums: `./ci/bump_bitnet_tag.sh generate-checksums`
3. Check for upstream changes

### Build Failures

```
ERROR: Build failed
```

**Solutions**:
1. Check system dependencies (cmake, compiler)
2. Try clean rebuild: `./ci/fetch_bitnet_cpp.sh --clean --force`
3. Check version compatibility

## CI Integration

### Automated Checks

CI should regularly:

```yaml
# Check for updates (weekly)
- name: Check for C++ updates
  run: ./ci/bump_bitnet_tag.sh check

# Validate current version
- name: Validate C++ version
  run: ./ci/bump_bitnet_tag.sh validate
```

### Update Notifications

Set up notifications for:
- New upstream releases
- Failed checksum validations
- Build failures with new versions

## Best Practices

### Before Updating

1. **Check release notes** for breaking changes
2. **Test locally** before committing
3. **Run full test suite** including cross-validation
4. **Check performance** for regressions

### After Updating

1. **Commit version files** (`bitnet_cpp_version.txt`, `bitnet_cpp_checksums.txt`)
2. **Update documentation** if needed
3. **Notify team** of the update
4. **Monitor CI** for any issues

### Version Pinning

- **Always pin to specific versions** (not branches or `latest`)
- **Use semantic versions** when available
- **Document reasons** for version choices
- **Test thoroughly** before production use

## Emergency Procedures

### Rollback

If a new version causes issues:

```bash
# Rollback to previous version
./ci/bump_bitnet_tag.sh update v1.1.0 --force --yes

# Test the rollback
cargo test --features crossval
```

### Hotfix

For critical security updates:

1. **Update immediately**: `./ci/bump_bitnet_tag.sh update v1.2.1 --yes`
2. **Test quickly**: Focus on security-related tests
3. **Deploy rapidly**: Skip non-critical validation
4. **Follow up**: Complete full testing post-deployment

## Maintenance

### Regular Tasks

- **Weekly**: Check for updates
- **Monthly**: Validate checksums
- **Quarterly**: Review version strategy
- **Annually**: Audit dependency security

### Monitoring

Monitor for:
- New upstream releases
- Security advisories
- Performance regressions
- Compatibility issues

## Contact

For questions about version management:
- Create an issue in this repository
- Tag the cross-validation team
- Reference this document in discussions
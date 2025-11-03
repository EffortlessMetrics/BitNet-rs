# Security Policy

## Supported Versions

We actively support the following versions for security updates:

| Version | Status | Support Level |
|---------|--------|---------------|
| `main` branch | ✅ Active Development | Full support with rapid response |
| `v0.1.x` (MVP) | ✅ Maintained | Security fixes and critical bugs |
| Older versions | ⚠️ Best Effort | May receive fixes at maintainer discretion |

## Reporting a Vulnerability

**Please do not file public GitHub issues for security vulnerabilities.**

### How to Report

Send security reports to: **security@effortlessmetrics.com**

Include the following information:
1. **Affected version/commit** and configuration (OS, Rust version, feature flags)
2. **Detailed reproduction steps** with minimal example
3. **Impact assessment**: What could an attacker achieve?
4. **Proof-of-concept** (if applicable) or logs demonstrating the issue
5. **Suggested fix** (optional but appreciated)

### Response Timeline

- **72-hour acknowledgment**: We will confirm receipt within 3 business days
- **14-day initial assessment**: Preliminary analysis and severity classification
- **30-day resolution target**: For critical vulnerabilities (best effort)
- **Coordinated disclosure**: We will work with you on an appropriate disclosure timeline

### PGP Encryption (Optional)

If you prefer to encrypt your report, you may request our PGP public key via the security email address.

## Security Scope

We consider the following areas in scope for security reports:

### High Priority

1. **Runtime Memory Safety**
   - Unsafe Rust usage leading to undefined behavior
   - Memory corruption or use-after-free
   - Buffer overflows in kernels or quantization code

2. **Honest Compute Verification**
   - Receipt forgery or tampering
   - Mock compute paths bypassing validation
   - Kernel ID manipulation in receipts

3. **Inference Correctness**
   - Quantization algorithm vulnerabilities
   - Numerical instability leading to incorrect outputs
   - GPU/CPU kernel parity violations

4. **Supply Chain Security**
   - GitHub Actions workflow vulnerabilities
   - Dependency confusion or malicious packages
   - Build reproducibility issues

### Medium Priority

5. **Model Loading Security**
   - GGUF/SafeTensors parser vulnerabilities
   - Malformed model file crashes or hangs
   - Resource exhaustion (memory/CPU)

6. **CLI Security**
   - Command injection vulnerabilities
   - Path traversal issues
   - Unsafe file operations

### Out of Scope

- **Vulnerabilities in dependencies**: Report to the upstream project first, then notify us if we need a coordinated patch
- **Denial of Service via large models**: Expected behavior; use resource limits
- **Model quality issues**: Not a security concern (report as bug)
- **Theoretical attacks without PoC**: Low priority without demonstrated impact

## Security Best Practices

### For Users

1. **Use official releases**: Download from GitHub releases or crates.io
2. **Verify checksums**: Check release signatures when available
3. **Pin dependencies**: Use `Cargo.lock` for reproducible builds
4. **Enable strict mode**: Set `BITNET_STRICT_MODE=1` for production
5. **Review receipts**: Validate compute paths are `"real"`, not `"mocked"`

### For Contributors

1. **Minimize unsafe code**: Justify all `unsafe` blocks with safety comments
2. **Validate inputs**: Check model files, prompts, and configuration
3. **Test edge cases**: Fuzzing, property-based testing, malformed inputs
4. **Review CI workflows**: Avoid secrets in logs, pin action SHAs
5. **Document security assumptions**: Explain trust boundaries

## Known Security Considerations

### Receipt Verification

BitNet.rs implements "honest compute" verification through receipts. Receipts prove that real computation occurred by including kernel IDs and validation gates.

**Security Properties**:
- `compute_path` must be `"real"` (strict mode enforces this)
- Non-empty `kernels` array with valid kernel IDs
- Schema v1.0.0 compliance

**Limitations**:
- Receipts are not cryptographically signed (planned for v0.2.0)
- Kernel IDs are self-reported (future: hash-based verification)

### Quantization Accuracy

BitNet.rs validates quantization accuracy through cross-validation with C++ reference implementations.

**Security Implications**:
- Incorrect quantization could leak model weights or produce biased outputs
- Validation gates enforce ≥99% accuracy for I2_S/TL1/TL2
- GPU/CPU parity checked via cosine similarity (≥0.999 threshold)

### Unsafe Rust Usage

The codebase uses `unsafe` in performance-critical paths (SIMD kernels, FFI). All unsafe blocks are:
- Documented with safety comments
- Tested with Miri (undefined behavior detection)
- Reviewed for memory safety

## Security Updates

We will publish security advisories through:
1. **GitHub Security Advisories**: https://github.com/EffortlessMetrics/BitNet-rs/security/advisories
2. **Release Notes**: CVE references in CHANGELOG.md
3. **Crates.io**: Security advisories (if published)

## Contact

For non-security questions, use GitHub issues or discussions.

For security-related inquiries: **security@effortlessmetrics.com**

---

**Last Updated**: 2025-11-03
**Policy Version**: 1.0

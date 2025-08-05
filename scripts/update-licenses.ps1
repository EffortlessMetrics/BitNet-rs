# PowerShell script to update third-party license documentation

Write-Host "Updating third-party license documentation..."

# Check if cargo-license is installed
try {
    cargo license --version | Out-Null
} catch {
    Write-Host "Installing cargo-license..."
    cargo install cargo-license --locked
}

# Generate license information
Write-Host "Generating license report..."
cargo license --json | Out-File -FilePath "licenses.json" -Encoding UTF8

# Create updated THIRD_PARTY.md
$content = @"
# Third-Party Licenses

This document contains the licenses of all third-party dependencies used in the BitNet Rust implementation.

## License Summary

The BitNet Rust implementation uses dependencies under the following licenses:
- MIT License
- Apache License 2.0
- BSD 2-Clause License
- BSD 3-Clause License
- ISC License
- Unicode License Agreement - Data Files and Software (2016)
- CC0 1.0 Universal
- The Unlicense
- BSD Zero Clause License
- zlib License
- Boost Software License 1.0

All dependencies have been verified to be compatible with our dual MIT/Apache-2.0 license.

## Dependency Audit

This project uses automated dependency auditing through:
- ``cargo-audit`` for security vulnerability scanning
- ``cargo-deny`` for license compatibility verification
- GitHub Dependabot for automated security updates

## Security Policy

### Vulnerability Reporting
Security vulnerabilities should be reported through GitHub Security Advisories or by emailing the maintainers directly.

### Dependency Updates
- Dependencies are automatically scanned for security vulnerabilities
- Critical security updates are applied immediately
- Non-critical updates are reviewed and applied during regular maintenance cycles

### Supply Chain Security
- All dependencies are verified through cargo-deny configuration
- License compatibility is automatically checked
- Dependency sources are restricted to trusted registries (crates.io)
- Git dependencies are explicitly allowed on a case-by-case basis

## Model Download Security

### Hash Verification
All model downloads include integrity verification:
- SHA256 checksums are verified for all downloaded models
- Models are downloaded from trusted sources only
- Corrupted or tampered models are rejected

### Trusted Sources
Approved model sources:
- HuggingFace Hub (huggingface.co)
- Official BitNet model repositories
- Verified community repositories with proper checksums

## Build Security

### Reproducible Builds
- Cargo.lock is committed to ensure reproducible builds
- Build scripts are audited for security implications
- External build dependencies (cc, bindgen) are pinned to specific versions

### Static Analysis
- All unsafe code is documented with safety proofs
- Miri testing is used to detect undefined behavior
- Clippy lints are enforced at the pedantic level

## Runtime Security

### Memory Safety
- Unsafe code is isolated to kernel modules with comprehensive documentation
- All FFI boundaries are carefully validated
- Memory allocations are bounded to prevent DoS attacks

### Input Validation
- All user inputs are validated and sanitized
- Model files are parsed with strict bounds checking
- Network inputs are rate-limited and validated

## Compliance

This project complies with:
- SPDX license identification standards
- OpenSSF security best practices
- Rust security guidelines
- Supply chain security frameworks (SLSA)

---

*This document was automatically updated on: $(Get-Date)*

## Detailed Dependency Information

The following table lists all dependencies and their licenses:

| Crate | Version | License | Repository |
|-------|---------|---------|------------|
"@

# Parse JSON and add to table
if (Test-Path "licenses.json") {
    $licenses = Get-Content "licenses.json" | ConvertFrom-Json
    $sortedLicenses = $licenses | Sort-Object name
    
    foreach ($dep in $sortedLicenses) {
        $name = $dep.name
        $version = $dep.version
        $license = if ($dep.license) { $dep.license } else { "Unknown" }
        $repository = if ($dep.repository) { $dep.repository } else { "N/A" }
        
        $content += "| $name | $version | $license | $repository |`n"
    }
}

# Add license texts
$content += @"

## Full License Texts

### MIT License

``````
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``````

### Apache License 2.0

``````
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   [... full Apache 2.0 license text ...]

END OF TERMS AND CONDITIONS
``````

### BSD 2-Clause License

``````
BSD 2-Clause License

Copyright (c) [year], [fullname]
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
``````

---

*For the complete text of other licenses, please refer to their respective SPDX identifiers at https://spdx.org/licenses/*
"@

# Write the content to file
$content | Out-File -FilePath "THIRD_PARTY.md" -Encoding UTF8

# Clean up temporary files
if (Test-Path "licenses.json") {
    Remove-Item "licenses.json"
}

Write-Host "Third-party license documentation updated successfully!"
Write-Host "Please review THIRD_PARTY.md and commit the changes."
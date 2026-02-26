# Build.rs Documentation Index

This index guides you to comprehensive documentation about BitNet-rs library discovery and linking patterns.

## Documents

### 1. BUILD_RS_QUICK_REFERENCE.md (5.2 KB, 165 lines)

**Purpose**: Fast reference for developers  
**Read Time**: 5-10 minutes  
**Difficulty**: Beginner

Essential information in tabular format:
- Library search chain (7 tiers)
- Key files and their error strategies
- Environment variable mapping
- Linking pattern (5-step sequence)
- The critical gap (backend-aware discovery)
- Error handling strategies
- Device detection code examples
- Backend detection pseudocode template
- Troubleshooting table

**When to read**: You need to understand build.rs quickly
**Best for**: Code reviews, quick diagnostics, onboarding

---

### 2. BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md (29 KB, 890 lines)

**Purpose**: Complete technical reference  
**Read Time**: 45-60 minutes  
**Difficulty**: Intermediate

14 major sections with comprehensive analysis:

1. **Executive Summary**: Overview of findings
2. **Library Discovery Logic**: Multi-tier algorithm with code snippets
3. **Search Path Priority**: Detailed environment variable hierarchy
4. **Linking Directives**: All cargo:rustc-link directives documented
5. **Backend Selection Analysis**: Current state and gaps
6. **Error Reporting Patterns**: All three error strategies
7. **Preflight Check Patterns**: Device detection and bootstrap
8. **The Gap**: Backend-specific discovery missing (critical finding)
9. **Environment Variable Summary**: Complete reference table
10. **Implementation Documentation**: Algorithm walkthroughs
11. **Integration Blueprint**: File-by-file patterns
12. **Root Cause Analysis**: Why backend selection missing
13. **Observations and Recommendations**: Strengths/weaknesses
14. **Appendix**: Complete code snippets

**When to read**: Implementing changes, architecture review, deep understanding
**Best for**: Backend detection implementation, build system refactoring

---

## How to Use These Documents

### For Quick Understanding (5-10 minutes)
Start with **BUILD_RS_QUICK_REFERENCE.md**
- Read the one-sentence summary
- Review the library search chain diagram
- Check the critical gap section
- See backend detection template

### For Implementation (45-60 minutes)
Read **BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md** sections:
- Section 4: Backend Selection Analysis (current state)
- Section 8: The Gap (detailed problem statement)
- Section 11: Integration Blueprint (architectural patterns)
- Section 11.2: Solution Architecture (pseudocode template)
- Appendix A.1: Complete code reference

### For Code Review (30 minutes)
Use both documents:
1. Quick Reference: Share with reviewers unfamiliar with build.rs
2. Full Analysis: Reference specific sections during review
3. Cross-reference: Compare proposed changes to documented patterns

### For Documentation Updates
Link to these from:
- CLAUDE.md: Build system section
- Architecture documentation
- Contributing guide
- Developer onboarding
- Pull request templates

---

## Key Topics Covered

### Library Discovery
- Multi-tier priority system
- Environment variable precedence
- CMake path patterns
- Fallback strategies
- Pattern matching for library names

### Linking Directives
- Dynamic vs static linking
- RPATH integration
- Platform-specific dependencies
- Framework linking (macOS)
- CUDA library linking

### Error Handling
- Hard panic patterns (required dependencies)
- Soft warning patterns (optional features)
- Conditional fallback patterns
- Error message best practices

### Backend Detection (The Gap)
- Current CUDA-only implementation
- Missing ROCm support
- Missing Metal support
- Missing oneAPI support
- Solution architecture provided

### Environment Variables
- BITNET_CPP_DIR usage
- BITNET_CROSSVAL_LIBDIR usage
- BITNET_GPU_FAKE (runtime only)
- Default fallback chains
- Legacy variable support

---

## File Map

```
docs/reference/
├── BUILD_RS_DOCUMENTATION_INDEX.md (this file)
├── BUILD_RS_QUICK_REFERENCE.md (start here!)
└── BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md (complete reference)

Implementation References:
├── ../../crossval/build.rs (canonical example)
├── ../../crates/bitnet-sys/build.rs (complex example)
├── ../../crates/bitnet-kernels/build.rs (GPU example)
├── ../../xtask-build-helper/src/lib.rs (patterns library)
└── ../../xtask/src/cpp_setup_auto.rs (bootstrap logic)
```

---

## Quick Links

### By Use Case

**I need to understand library discovery in 5 minutes:**
→ Quick Reference: "Library Search Chain" section

**I'm implementing backend detection:**
→ Full Analysis: Section 11.2 "Solution Architecture"

**I'm debugging a linker error:**
→ Quick Reference: "Troubleshooting" section

**I'm reviewing a build.rs change:**
→ Full Analysis: Section 10 "Complete Integration Blueprint"

**I'm refactoring for DRY:**
→ Full Analysis: Section 12.3 "Recommended Improvements"

**I need complete algorithm walkthrough:**
→ Full Analysis: Section 9 "Document the Current Implementation"

---

## Integration Points

### For CLAUDE.md
Add a section linking to these documents:
```
## Build System Architecture

For comprehensive documentation on library discovery and linking:
- Quick overview: docs/reference/BUILD_RS_QUICK_REFERENCE.md
- Complete reference: docs/reference/BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md

Key pattern: Multi-tier priority-based directory scanning with environment variable overrides.
Critical gap: No backend-aware GPU library discovery (CUDA only, no ROCm/Metal auto-detection).
```

### For Contributing Guide
Add to build-related sections:
```
## Understanding build.rs

BitNet-rs uses sophisticated build script patterns. See:
- BUILD_RS_QUICK_REFERENCE.md for patterns
- BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md for algorithms
```

### For Architecture Documentation
Reference when discussing:
- Linking strategies (section 3)
- Device detection (section 6)
- Platform support (section 3.3)
- Backend flexibility (section 11.2)

---

## Document Statistics

| Metric | Value |
|--------|-------|
| Total documentation | 1,055 lines |
| Quick Reference | 165 lines (5.2 KB) |
| Full Analysis | 890 lines (29 KB) |
| Code snippets included | 50+ |
| Build.rs files analyzed | 8 |
| Environment variables documented | 9 |
| Search path tiers identified | 7 |
| Linking patterns catalogued | 25+ |
| Error strategies identified | 3 |
| Recommendations provided | 5+ |

---

## Version Information

- **Report Date**: 2025-10-25
- **Repository**: BitNet-rs (Rust/BitNet-rs)
- **Scope**: Comprehensive build.rs analysis
- **Status**: Complete, ready for implementation

---

## Next Steps

### For Immediate Use
1. Read BUILD_RS_QUICK_REFERENCE.md (5-10 min)
2. Share with team members unfamiliar with build.rs
3. Link from CLAUDE.md

### For Implementation
1. Review Full Analysis Section 11.2 (solution architecture)
2. Implement backend detection in new crate or xtask-build-helper
3. Cross-reference patterns from Section 10 (integration blueprint)
4. Test on multiple GPU platforms (CUDA, ROCm, Metal)

### For Documentation
1. Link both reports from CLAUDE.md
2. Update Contributing guide
3. Add to architecture documentation
4. Reference in pull request templates for build.rs changes

---

**Last Updated**: 2025-10-25  
**Maintained By**: BitNet-rs Documentation Team

# ADR-003: Receipt Schema v1.0.0 Stability

**Status**: ACCEPTED
**Date**: 2025-10-15
**Context**: Issue #465 (v0.1.0-mvp Release Polish)
**Related**: AC3/AC4 (CPU Baseline Receipt Generation and Verification)

---

## Context

Issue #465 requires pinned CPU baseline receipt with schema v1.0.0 for v0.1.0-mvp release. The receipt schema defines validation rules for honest compute evidence.

Two schema versioning approaches exist:

### Option 1: Extend Schema v1.0.0 (Breaking Change)
- **Method**: Add new required fields to existing schema
- **Fields**: `architecture`, `quantization_method`, `model_hash`
- **Pros**: Richer metadata for diagnostics and analysis
- **Cons**: Breaking change, requires migration path, invalidates existing receipts

### Option 2: Keep Schema v1.0.0 Unchanged (Backward Compatible)
- **Method**: No changes to existing schema for MVP
- **Fields**: Existing fields sufficient for honest compute validation
- **Pros**: Backward compatible, stable baseline, no migration needed
- **Cons**: Limited metadata for advanced analysis (deferred to v1.1.0)

---

## Decision

**Keep existing schema v1.0.0 unchanged (Option 2).**

---

## Rationale

### 1. MVP Stability Focus
- **Release Priority**: v0.1.0-mvp focuses on stability, not feature expansion
- **Baseline Consistency**: Pinned CPU baseline should not require schema migration
- **User Experience**: Existing receipts remain valid across v0.1.0-mvp lifecycle
- **Breaking Changes**: Initial MVP release should avoid unnecessary churn

### 2. Sufficient Validation Metadata
Current schema v1.0.0 provides adequate fields for honest compute verification:
- ✅ `version`: Schema compatibility (1.0.0 or 1.0)
- ✅ `compute_path`: Honest compute flag ("real" vs "mock")
- ✅ `kernels`: Kernel execution evidence (non-empty array)
- ✅ `performance`: Measured throughput (tokens_per_sec)
- ✅ `success`: Inference completion status
- ✅ `timing`: Detailed timing breakdown (warmup, prefill, decode)

**Missing Fields** (deferred to v1.1.0):
- ⏭️ `architecture`: Model architecture name (e.g., "BitNet-2B-4T")
- ⏭️ `quantization_method`: Quantization type (e.g., "I2_S", "TL1", "TL2")
- ⏭️ `model_hash`: SHA256 fingerprint of GGUF file
- ⏭️ `rust_version`: Rust compiler version for reproducibility
- ⏭️ `bitnet_version`: BitNet-rs crate version

### 3. Schema Evolution Strategy
- **Backward Compatibility**: v1.1.0 can add optional fields without breaking v1.0.0 receipts
- **Migration Guide**: Future schema versions will document migration path
- **Deprecation Policy**: v1.0.0 support maintained for at least 6 months after v1.1.0 release
- **Validation Modes**: Receipt verifier supports multiple schema versions concurrently

### 4. Current Schema Adequacy
Receipt verification gates implemented in PR #462 validate:
1. Schema version compatibility (1.0.0 or 1.0)
2. Compute path requirement (`compute_path: "real"`)
3. Kernel hygiene (non-empty, length ≤128, count ≤10,000)
4. Type safety (all kernels are strings, not numbers/objects)
5. GPU enforcement (backend="cuda" requires GPU kernels automatically)

**No additional fields needed for MVP honest compute validation.**

### 5. Post-MVP Schema Enhancement
v1.1.0 schema can add optional fields without breaking changes:
```json
{
  "version": "1.1.0",
  "compute_path": "real",
  "kernels": [...],
  // NEW OPTIONAL FIELDS (v1.1.0)
  "architecture": "BitNet-2B-4T",           // Optional: Model architecture
  "quantization_method": "I2_S",            // Optional: Quantization type
  "model_hash": "sha256:abc123...",         // Optional: GGUF fingerprint
  "rust_version": "1.90.0",                 // Optional: Rust compiler version
  "bitnet_version": "0.1.0",                // Optional: BitNet-rs version
  // EXISTING FIELDS (backward compatible)
  "performance": {...},
  "success": true
}
```

**Migration Path**: v1.0.0 receipts remain valid, v1.1.0 fields optional.

---

## Consequences

### Positive
- ✅ **Backward Compatibility**: All existing receipts remain valid across v0.1.0-mvp
- ✅ **Stable Baseline**: CPU baseline receipt does not require schema migration
- ✅ **No Breaking Changes**: Initial MVP release avoids unnecessary complexity
- ✅ **Clear Evolution Path**: v1.1.0 can add optional fields with migration guide
- ✅ **User Confidence**: Schema stability signals production readiness

### Negative
- ⚠️ **Limited Metadata**: Advanced diagnostics (architecture, quantization method) deferred
- ⚠️ **Future Migration**: v1.1.0 will require migration guide for new fields
- ⚠️ **Fingerprinting**: Model hash validation requires external tooling (not in receipt)

### Mitigation Strategies
1. **External Metadata**: Use baseline README for model architecture, quantization method
2. **Fingerprint Validation**: Document GGUF SHA256 fingerprinting in baseline documentation
3. **Schema Versioning**: Plan v1.1.0 schema with optional fields for post-MVP
4. **Migration Guide**: Provide clear migration path when v1.1.0 is released

---

## Alternatives Considered

### Alternative 1: Schema v1.1.0 with New Required Fields
**Rejected**: Breaking change not justified for MVP. Advanced metadata can be added post-MVP as optional fields.

**Proposed v1.1.0 Fields** (future work):
- `architecture`: Model architecture name (e.g., "BitNet-2B-4T")
- `quantization_method`: Quantization type (e.g., "I2_S", "TL1", "TL2")
- `model_hash`: SHA256 fingerprint of GGUF file
- `rust_version`: Rust compiler version (reproducibility)
- `bitnet_version`: BitNet-rs crate version

### Alternative 2: Schema v1.0.1 with Optional Fields
**Deferred**: Patch version bump not needed for MVP. Optional fields can be added in v1.1.0 minor version.

### Alternative 3: Multiple Schema Versions (v1.0.0 + v1.1.0)
**Rejected**: Concurrent schema support adds complexity without immediate value. v1.1.0 planned for v0.2.0 release.

---

## Implementation Details

### Current Schema v1.0.0

**Required Fields**:
- `version`: "1.0.0" or "1.0"
- `compute_path`: "real" (not "mock")
- `kernels`: Non-empty array of strings
- `success`: Boolean

**Optional Fields**:
- `model_path`: Path to GGUF model
- `device`: "cpu" or "cuda"
- `backend`: "cpu" or "cuda"
- `performance`: { tokens_per_sec, ms_per_token }
- `timing`: { warmup_ms, prefill_ms, decode_ms, total_ms }
- `error`: Error message (if success=false)

### Validation Rules

1. **Schema Version**: Must be "1.0.0" or "1.0" (backward compatible)
2. **Compute Path**: Must be "real" (blocks mock inference)
3. **Kernels Hygiene**:
   - Non-empty array (at least 1 kernel)
   - All items are strings (not numbers/objects)
   - No empty strings
   - Length ≤128 characters per kernel
   - Count ≤10,000 total kernels
4. **Success Flag**: Must be `true` for valid receipts
5. **Performance**: `tokens_per_sec > 0` (if present)

### Verification Command

```bash
# Verify receipt against v1.0.0 schema
cargo run -p xtask -- verify-receipt --path ci/inference.json

# Expected output:
# ✅ Receipt schema valid (v1.0.0)
# ✅ Compute path: real
# ✅ Kernels: 8 CPU kernels detected
# ✅ Performance: 15.3 tok/s measured
# ✅ Success: true
#
# Receipt verification PASSED
```

---

## Future Schema Evolution (v1.1.0)

### Planned Enhancements (Post-MVP)

**New Optional Fields** (v1.1.0):
```json
{
  "version": "1.1.0",
  "architecture": "BitNet-2B-4T",           // Model architecture
  "quantization_method": "I2_S",            // Quantization type
  "model_hash": "sha256:abc123...",         // GGUF fingerprint
  "rust_version": "1.90.0",                 // Rust compiler
  "bitnet_version": "0.1.0",                // BitNet-rs version
  // ... existing fields unchanged
}
```

**Migration Guide** (v1.1.0):
- v1.0.0 receipts remain valid (no migration required)
- v1.1.0 fields are optional (backward compatible)
- Receipt verifier supports both v1.0.0 and v1.1.0 concurrently

**Timeline**: Planned for v0.2.0 release (post-MVP)

---

## References

- **Issue #465**: CPU Path Followup (v0.1.0-mvp Release Polish)
- **AC3**: Generate Pinned CPU Baseline Receipt
- **AC4**: Verify Baseline Receipt
- **PR #462**: Receipt Validation Implementation
- **Receipt Schema**: `docs/reference/receipt-schema-v1.0.md`

---

## Changelog

- **2025-10-15**: Initial decision for v0.1.0-mvp schema stability

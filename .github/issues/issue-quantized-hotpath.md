# Enforce quantized projections in attention mechanism (no FP32 staging)

**Labels:** `enhancement`, `quantization`, `validation`

**Priority:** High

**Depends on:** PR #452 (receipt verification)

## Summary

Add runtime assertions to ensure attention mechanisms use quantized projections without unnecessary FP32 staging, preventing silent performance degradation.

## Problem

Currently, the attention mechanism could theoretically fall back to FP32 staging without detection, defeating the purpose of quantization and causing silent performance regression.

## Acceptance Criteria

- [ ] Add `debug_assert!(self.qkv_proj.is_quantized(), "Attention must use quantized projections")` in attention forward pass
- [ ] Confirm `QuantizedLinear::forward(..)` dispatches to `i2s/tl1/tl2` kernels
- [ ] Verify output is FP32 without full-weight dequant staging
- [ ] Add tests for attention Q/K/V/O via quantized linears
- [ ] Remove remaining placeholder `#[ignore]` on TL1/TL2 tests when tables are wired

## Implementation Notes

- Should be debug assertions (no runtime overhead in release builds)
- Focus on `bitnet-inference/src/attention.rs`
- Update corresponding tests in `tests/attention_quantized.rs`

## Files to Modify

- `crates/bitnet-inference/src/attention.rs` - Add assertions
- `tests/attention_quantized.rs` - Add test coverage

## Related

- Blocks: Full quantization validation
- Related: Receipt verification infrastructure (PR #452)

## Estimated Effort

~1 day (quick win with high value)

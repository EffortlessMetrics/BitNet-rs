# Performance Smoke Test Added to CI

**Date**: 2025-10-22
**Implementation**: Non-gating performance observability in CI
**Reference**: `ci/exploration/PR3_perf_receipts_plan.md`

## Summary

Added a non-gating performance smoke test step to `.github/workflows/ci.yml` that:

1. ✅ Sets deterministic environment variables (`BITNET_DETERMINISTIC=1`, `RAYON_NUM_THREADS=1`)
2. ✅ Uses the release binary (already built with optimizations)
3. ✅ Runs a 4-token inference with `/usr/bin/time -v` for detailed timing
4. ✅ Uses `|| true` pattern to make it non-gating (observability only)
5. ✅ Logs elapsed time without failing builds

## Implementation Details

### Step Added: "Perf smoke (non-gating)"

```yaml
- name: Perf smoke (non-gating)
  # Non-gating: observability only for elapsed time tracking
  # Based on ci/exploration/PR3_perf_receipts_plan.md
  continue-on-error: true
  env:
    BITNET_DETERMINISTIC: "1"
    RAYON_NUM_THREADS: "1"
    RUST_LOG: warn
  run: |
    echo "::group::Perf smoke test with /usr/bin/time (non-gating)"
    echo "Running deterministic 4-token inference with timing..."

    # Run inference with /usr/bin/time to capture elapsed time
    /usr/bin/time -v target/release/bitnet-cli run \
      --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
      --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
      --prompt "2+2=" \
      --max-tokens 4 \
      --temperature 0 --greedy || {
        echo "::notice::Perf smoke test failed (non-gating, for observability only)"
        true
      }

    echo "::endgroup::"
```

### Key Features

1. **Determinism**:
   - `BITNET_DETERMINISTIC=1` ensures reproducible inference
   - `RAYON_NUM_THREADS=1` ensures single-threaded execution
   - `RUST_LOG=warn` reduces log noise

2. **Timing Measurement**:
   - Uses `/usr/bin/time -v` for detailed resource usage
   - Captures elapsed (wall-clock) time, CPU time, memory usage
   - Outputs timing data to CI logs for observability

3. **Non-Gating**:
   - `continue-on-error: true` ensures step doesn't fail the workflow
   - `|| { ... true }` ensures command failure doesn't stop execution
   - Logs `::notice::` instead of failing on errors

4. **Position in Workflow**:
   - Runs after "Build CLI (release)" step
   - Before "Performance smoke test (4-token inference)" receipt generation step
   - Part of `perf-smoke` job that depends on `test` job passing

## Success Criteria Met

- ✅ Smoke test runs in CI (`perf-smoke` job)
- ✅ Logs elapsed time (via `/usr/bin/time -v`)
- ✅ Does not fail builds (`continue-on-error: true`, `|| true` pattern)
- ✅ Uses deterministic settings (`BITNET_DETERMINISTIC=1`, `RAYON_NUM_THREADS=1`)

## Expected Output

When running in CI, the step will produce output like:

```
Perf smoke test with /usr/bin/time (non-gating)
Running deterministic 4-token inference with timing...

	Command being timed: "target/release/bitnet-cli run ..."
	User time (seconds): 35.42
	System time (seconds): 0.31
	Percent of CPU this job got: 99%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:35.87
	Maximum resident set size (kbytes): 4523456
	...
```

This provides:
- Wall-clock time (real elapsed time)
- CPU time breakdown (user/system)
- Memory usage (max RSS)
- Other resource metrics

## Integration with Existing Workflow

The new step integrates seamlessly with the existing `perf-smoke` job:

1. **Download test model** (existing)
2. **Build CLI (release)** (existing)
3. **Perf smoke (non-gating)** ← NEW STEP
4. **Performance smoke test (4-token inference)** (existing, with receipt generation)
5. **Verify positive receipt example** (existing)
6. **Verify negative receipt example** (existing)
7. **Verify generated receipt** (existing)
8. **Comment performance results (PR only)** (existing)

## Future Enhancements

Based on `ci/exploration/PR3_perf_receipts_plan.md`, potential future improvements:

1. **Receipt Generation**: Integrate with `xtask benchmark` for JSON receipts (already in next step)
2. **Baseline Comparison**: Store timing baselines and detect regressions
3. **Flamegraph Integration**: Optional profiling on demand (Phase 3)
4. **Gating Option**: Make gating after baseline establishment (2-3 months)

## Verification

The YAML syntax has been validated:

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
# ✅ YAML syntax is valid
```

## References

- **Exploration Plan**: `ci/exploration/PR3_perf_receipts_plan.md`
- **CI Workflow**: `.github/workflows/ci.yml` (line 160-183)
- **BitNet.rs CLAUDE.md**: Performance testing and CI integration sections

# integrative:gate:mutation Check Run

**Name:** `integrative:gate:mutation`
**Status:** completed
**Conclusion:** neutral
**Title:** integrative:gate:mutation - Bounded Policy Applied

## Summary

Mutation testing skipped due to bounded policy: 1,973 mutants detected in bitnet-inference would require ~32-52 hours unbounded execution (actual: 8m timeout). Bounded policy applied per agent instructions (max 8min execution). Status: NEUTRAL (not a failure).

**Evidence:**
- Total mutants: 1,973
- Baseline: 57.3s build + 38.4s test
- Theoretical time: ~32-52 hours unbounded
- Bounded timeout: 8 minutes (exceeded)
- Observed before timeout: 1 MISSED mutant in backends.rs (capability validation gap)

**Routing:** Proceed to safety-scanner (bounded skip does not block merge-readiness)

## Details

**Method:** primary
**Result:** skipped (bounded)
**Reason:** bounded by policy: 1,973 mutants would require ~32-52 hours; exceeded 8min threshold

## Commands for Manual Creation

```bash
SHA=$(git rev-parse HEAD)
gh api -X POST repos/EffortlessMetrics/BitNet-rs/check-runs \
  -f name="integrative:gate:mutation" \
  -f head_sha="$SHA" \
  -f status=completed \
  -f conclusion=neutral \
  -f output[title]="integrative:gate:mutation - Bounded Policy Applied" \
  -f output[summary]="Mutation testing skipped due to bounded policy: 1,973 mutants detected in bitnet-inference would require ~32-52 hours unbounded execution (actual: 8m timeout). Bounded policy applied per agent instructions (max 8min execution). Status: NEUTRAL (not a failure).

**Evidence:**
- Total mutants: 1,973
- Baseline: 57.3s build + 38.4s test
- Theoretical time: ~32-52 hours unbounded
- Bounded timeout: 8 minutes (exceeded)
- Observed before timeout: 1 MISSED mutant in backends.rs (capability validation gap)

**Routing:** Proceed to safety-scanner (bounded skip does not block merge-readiness)"
```

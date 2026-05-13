# CUDA Benchmark Qualification

Benchmark qualification turns timing receipts into narrow speed decisions. It
does not create model answer proof, hardware execution proof, or global CUDA
speedup by itself.

## Required Profiles

```text
one_token
short_decode_8
short_decode_32
warm_session_3_turns
warm_session_10_turns
```

## Required Receipt Fields

Every benchmark qualification receipt must record:

```text
model artifact identity
tokenizer authority
prompt template authority
requested backend
selected backend
runtime API
fallback_used
CPU mean / p50 / p95
CUDA mean / p50 / p95
prompt prefill time
first-token latency
steady decode time
kernel time
H2D timing source
D2H timing source
VRAM high-water
power/thermal context when available
speedup_accepted = true|false
reason
```

## Decision Rules

- A profile can pass while another profile fails.
- BitNet profile decisions do not apply to dense SLMs.
- Dense Qwen decisions do not apply to BitNet QK256.
- A benchmark baseline is evidence, not an accepted speed claim.
- Missing transfer, power, thermal, or variance context should usually reject
  speedup while preserving the measurement.

## Work item: BitNet Official I2_S Qualification

Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-PROD-010`

Proof:

```bash
cargo test --locked -p bitnet-bench-receipts --no-default-features
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --cuda-benchmark-receipt <receipt>
git diff --check
```

Receipt root:

```text
ci/hardware/windows-9950x3d-rtx5070ti/<date>/bitnet-i2s-*-benchmark.json
```

Rollback:

Remove the qualification receipt or demote the accepted profile. Do not edit
raw historical benchmark receipts by hand.

## Work item: Dense Qwen Qualification

Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-DENSE-054`

Proof:

```bash
python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/<date>/dense-qwen25-q8-*.json
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
git diff --check
```

Receipt root:

```text
ci/hardware/windows-9950x3d-rtx5070ti/<date>/dense-qwen25-q8-*-benchmark.json
```

Rollback:

Remove or demote the dense benchmark qualification. Keep
`bitnet_packed_i2s_qk256_proof=false`.

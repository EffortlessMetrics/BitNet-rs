# SLM CPU Artifact Export

`bitnet-rs-swarm` is the development surface for the SLM CPU runtime lane.
`BitNet-rs` remains the release and evidence surface. Runtime candidate work
must therefore leave this repository as a small reviewable artifact package,
not as an implicit claim that release evidence has already been accepted.

## Export Package

The first Qwen3 Q8_0 packed-sidecar export package must contain:

```text
candidate_summary.json
before_receipt.json
after_receipt.json
equivalence_report.json
timing_report.json
```

If timing is intentionally not claimed, replace `timing_report.json` with:

```text
timing_not_claimed.json
```

Every package also records:

```text
source_commit.txt
```

`source_commit.txt` must name the exact `bitnet-rs-swarm` commit that produced
the package. If the package comes from a dirty tree or untracked local files,
the package is not release-ready.

## Required Identity

The package must preserve the Qwen3 Q8_0 behavior oracle:

```text
model SHA
tokenizer.source = gguf_metadata
tokenizer.strict = true
prompt IDs
generated IDs
decoded text
selected CPU backend
selected CPU kernel
dense hook-selection identity
fallback_used = false
```

The candidate summary must also name the exact dense tensor path and opt-in
environment used for the candidate. For the current single-tensor sidecar lane,
that means recording whether these were set:

```text
BITNET_DENSE_Q8_PAYLOAD_ENABLE
BITNET_DENSE_Q8_PAYLOAD_TENSOR
```

## Claim Boundary

Exporting a package from swarm does not make the package accepted by the
release surface. The release repository must review the package against its
artifact intake gate before any release claim is made.

The package must not claim:

```text
speedup
sustained throughput
broad answer quality
Q4/Q5 runtime support
server inference
GPU, NPU, OpenVINO, or UHD 620 execution
Qwen3.5 or hybrid architecture support
BitNet QK256/I2_S changes
```

## First Export Target

`SLM-CPU-062` owns the first export package after `SLM-CPU-061`. It should use
the committed Qwen3 Q8_0 sidecar payload candidate as input and produce the
package for release-surface intake without widening runtime selection or making
new performance claims.

# Apple Silicon MacBook Lane

The MacBook lane is the Apple Silicon cross-reference and larger-artifact lane. It is not a replacement for the M4 Mac mini dense SLM product lane and it is not a shortcut to Apple BitNet claims.

See also [Reference Topology](../architecture/reference-topology.md) for the
scalar -> AVX2/NEON -> accelerator validation chain and the dense SLM vs BitNet
claim boundary.

## Roles

M4 Mac mini:

```text
stable Apple Silicon dense SLM product/performance lane
published Qwen2.5 Apple CPU/NEON envelope
resident warm-session UX
phase-scoped Metal evidence
```

MacBook:

```text
mobile Apple Silicon cross-reference
storage-aware larger-artifact exploration
thermal/mobile context receipts
Apple BitNet candidate sweeps before M4 strict proof
```

BitNet Apple lane:

```text
artifact-qualified 1-bit / 1.58-bit proof lane
requires reference-good model/tokenizer authority
requires strict backend receipts before local-answer claims
```

## Machine Profile Receipt

The first MacBook receipt should identify:

```text
machine_id
chip
cpu_brand
memory_bytes
macos_version
available_disk_bytes
model_cache_root
power_source
thermal_state when available
cpu_neon_available
metal_visible
mpsgraph_visible when available
```

This profile decides whether the MacBook should attempt larger artifacts. It should not run model inference or claim answer quality.

`MB-AS-001` defines the receipt contract at:

```text
ci/hardware/apple-silicon-macbook/receipt-contracts/machine-profile.schema.json
```

and a non-hardware example at:

```text
ci/hardware/apple-silicon-macbook/receipt-contracts/machine-profile.example.json
```

The example deliberately records `inference_run=false` and sets every model, BitNet, Metal, Neural Engine, QK256, and broad-performance claim flag to false. A real MacBook profile receipt should replace the placeholder machine and storage values, but it must keep the same claim boundary unless a later campaign item explicitly allows a broader proof.

## Dense SLM Cross-Check

After the machine profile exists, mirror the known-good dense Qwen Mac path:

```text
model_id = qwen2.5-0.5b-instruct-q8_0
model_sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
requested_backend = apple-m4-cpu-neon or the MacBook Apple CPU/NEON label defined by the item
selected_backend = recorded actual Apple CPU/NEON route
fallback_used = false
quality corpus = apple-m4-slm-quality-determinism-v1 shape
```

The MacBook receipt should be compared to the M4 Mac mini for behavior and timing context, but it must not replace the M4 performance envelope.

## BitNet Candidate Sweeps

MacBook is the right first Apple machine for larger BitNet candidate exploration. Candidate records should include:

```text
source repo
file
revision
size_bytes
sha256
model family
kernel routes: I2_S, TL1, TL2, diagnostic only, unsupported
tokenizer authority
reference command
prompt outputs
acceptance or rejection
cleanup status
```

Initial candidate priority:

```text
official Microsoft BitNet b1.58 2B / 2B4T I2_S with external tokenizer authority
1bitLLM/bitnet_b1_58-large 0.7B as the smaller Apple candidate
1bitLLM/bitnet_b1_58-3B only on supported TL1/TL2 diagnostic routes
Falcon-E candidates after Microsoft / 1bitLLM behavior is understood
```

`MB-AS-003` records the first machine-readable candidate matrix at:

```text
ci/hardware/apple-silicon-macbook/bitnet-candidate-matrix.toml
```

with the companion guide:

```text
docs/apple-silicon/bitnet-candidate-matrix.md
```

The dedicated Apple BitNet sweep control plane is documented at:

```text
docs/apple-silicon/apple-bitnet-artifact-sweep.md
```

No artifact becomes an Apple local-answer claim until a strict backend receipt produces coherent output with real model, tokenizer authority, selected backend, fallback status, generated text, token IDs, and timing.

## Claim Boundaries

Do not claim:

```text
BitNet local-answer quality from dense Qwen evidence
QK256 support on Apple Silicon
full Apple Metal inference from phase evidence
Neural Engine execution from MPSGraph visibility
broad Apple Silicon performance from one MacBook run
M4 Mac mini performance regression from a MacBook-only timing change
```

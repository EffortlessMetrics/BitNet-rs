# Apple M3 MacBook Air Dense SLM Synthesis

Date: 2026-05-15
Work item: `M3MBA-009`

## Result

The M3 Air dense Qwen SLM receipts are usable as MacBook Apple CPU/NEON
cross-check evidence. They are comparable to the M4 dense SLM receipts for
model identity, tokenizer metadata, backend label shape, fallback status,
profile IDs, prompt count, quality-pass status, and warm-session timing fields.

They are not a replacement for the M4 Mac mini performance envelope and they do
not prove BitNet behavior. The SLM CPU Qwen3 receipt is useful as strict CPU
warm-session and receipt-shape context only; it uses a different model,
tokenizer, host, prompt corpus, and token budget.

## Evidence Compared

| Lane | Receipt | Scope |
|---|---|---|
| M3 Air dense smoke | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-smoke.json` | M3 Air dense Qwen smoke, `apple-m3-air-cpu-neon`, no fallback |
| M3 Air dense operator | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-operator.json` | M3 Air `warm_16`, `warm_32`, `warm_64` operator profiles with allocation audit |
| M4 dense baseline | `ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/release-baseline.json` | M4 `warm_16`, `warm_32`, `warm_64`, `warm_128` release baseline |
| M4 allocation audit | `ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/allocation-audit.json` | M4 operator-profile allocation audit comparison |
| SLM CPU warm session | `ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-warm-session.json` | Qwen3 strict CPU warm-session behavior oracle, not a timing peer |

## Comparable Fields

| Field | M3 Air dense Qwen | M4 dense Qwen | SLM CPU Qwen3 |
|---|---|---|---|
| Model | `qwen2.5-0.5b-instruct-q8_0` | `qwen2.5-0.5b-instruct-q8_0` | `Qwen3-0.6B-Q8_0.gguf` |
| SHA-256 | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| Tokenizer | GGUF metadata, `tokenizer_model=gpt2`, `tokenizer_pre=qwen2` | GGUF metadata, `tokenizer_model=gpt2`, `tokenizer_pre=qwen2` | GGUF metadata, strict tokenizer |
| Backend | `apple-m3-air-cpu-neon` | `apple-m4-cpu-neon` | `cpu-rust` |
| Runtime API | `cpu` | `cpu` | `cpu` |
| Fallback used | false | false | false |
| Prompt shape | 3 prompts per warm profile | 3 prompts per warm profile | 3 prompts repeated twice in one strict CPU session |
| Timing peer | yes, M3 vs M4 dense Qwen only | yes, M4 vs M3 dense Qwen only | no, different model and host |

## Timing Snapshot

The M3 and M4 dense receipts share model hash, tokenizer metadata, profile
shape, prompt count, greedy deterministic settings, and no-fallback CPU runtime.
The timing comparison below is bounded to those receipts.

| Profile | M3 generated tokens | M3 decode tok/s | M3 first-token mean ms | M3 warm prompt tok/s | M4 decode tok/s | M4 first-token mean ms | M4 warm prompt tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `warm_16` | 34 | 13.125 | 2,132.333 | 3.926 | 14.962 | 1,885.000 | 4.435 |
| `warm_32` | 50 | 13.043 | 2,339.667 | 4.783 | 15.317 | 1,779.333 | 5.947 |
| `warm_64` | 82 | 13.592 | 2,196.000 | 6.656 | 15.269 | 1,763.000 | 7.840 |

M4 also has `warm_128` release and allocation-audit receipts. M3 Air does not
yet have a `warm_128` operator receipt, so `warm_128` is not used as a
MacBook-to-M4 comparison row.

## Allocation Shape

Both M3 and M4 allocation-audit receipts identify the same broad hot-path
components:

| Shared hotspot | M3 evidence | M4 evidence | Synthesis |
|---|---|---|---|
| `prompt_setup` | Top allocation component across M3 `warm_16`, `warm_32`, and `warm_64` | Top setup component in M4 allocation audit | Prompt setup remains the clearest cross-lane cleanup target. |
| `prompt_prefill` | Stable large allocation bucket across all M3 profiles | Stable large allocation bucket across all M4 profiles | Prefill allocation behavior is structurally comparable, but timing claims remain lane-local. |
| `model.forward` / `decode_total` | Decode allocations grow with token budget | Decode allocations grow with token budget | Token-budget scaling shape matches expectations across Apple CPU/NEON lanes. |
| `prompt_tokenize` | Large repeated allocation bucket | Large repeated allocation bucket | Tokenization/setup reuse remains useful to track after SLM-CPU-012-style cleanup. |

The M3 allocation audit is diagnostic. It does not claim an optimization,
speedup, or broad M3 performance result.

## SLM CPU Context

The i5-8250U Qwen3 warm-session receipt records useful control properties:

- strict CPU backend, `selected_backend=cpu-rust`
- GGUF tokenizer authority with strict tokenizer mode
- `fallback_used=false`
- model and tokenizer loaded once per warm session
- deterministic repeated prompt behavior
- no speedup, GPU, NPU, OpenVINO, server, QK256, or broad throughput claim

That receipt should be compared to M3 only for receipt discipline and failure
signatures. It should not be used as a timing baseline because the CPU lane uses
Qwen3-0.6B Q8_0, a different prompt corpus, a different host, and different
token budgets.

## Gaps

| Gap | Effect | Next owner |
|---|---|---|
| No M3 `warm_128` operator profile | M3 cannot be compared against the full M4 release envelope. | Future M3 dense SLM item only if the lane needs deeper timing. |
| M3 timing is diagnostic-only | M3 Air numbers must not replace M4 Mac mini product/performance claims. | Keep claim boundary in every M3 report. |
| SLM CPU model differs | CPU warm-session evidence is receipt-shape context, not a timing peer. | SLM CPU lane continues as strict CPU behavior oracle. |
| Allocation audit is diagnostic | Hotspots can guide cleanup, but not claim speedup. | Future SLM cleanup PRs with behavior-preserving tests. |

## Decision

`M3MBA-009` supports continuing to use the M3 Air as an Apple CPU/NEON dense SLM
and BitNet-screening host. The dense Qwen receipts show the MacBook lane can
preserve model identity, tokenizer authority, backend identity, fallback status,
quality-pass status, warm-session timing fields, and allocation-audit shape.

The next M3 Air work should remain split:

1. `M3MBA-006` may evaluate the smaller 0.7B BitNet control candidate only with
   serialized download, fresh free-space preflight, SHA-256 evidence, tokenizer
   authority, reference-runner output, and cleanup/retention status.
2. `M3MBA-007` remains diagnostic-only for 3B TL routes.
3. `M3MBA-008` must create separate M4 strict-proof work for accepted artifacts
   and must not convert M3 receipts into M4 evidence.

## Claim Boundary

This synthesis claims only that named M3 Air dense SLM evidence has been
compared against named M4 dense SLM and SLM CPU receipts where fields match.

It does not claim BitNet local-answer quality, M4 Mac mini performance, broad
Apple Silicon performance, full Metal inference, MPSGraph inference, Neural
Engine execution, QK256 support, or a speedup.

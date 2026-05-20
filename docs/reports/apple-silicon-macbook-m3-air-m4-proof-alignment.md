# Apple M3 MacBook Air M4 Proof Alignment

Date: 2026-05-20
Work item: `M3MBA-025`

## Result

`M3MBA-025` aligns the accepted M3 Air Microsoft 2B I2_S artifact metadata with
the separate M4 strict Apple CPU/NEON BitNet receipt lane. This is a checklist
and source-of-truth map only. It does not run M4 inference, validate a new M4
receipt, or convert M3 reference-runner output into M4 proof.

The only M3 Air BitNet candidate currently eligible for M4 handoff remains:

- repository: `microsoft/bitnet-b1.58-2B-4T-gguf`
- source revision: `a1f2f1c765812aa8af3f6eda4a313707064bba15`
- filename: `ggml-model-i2_s.gguf`
- format: GGUF
- quantization: `i2_s`
- size: 1,187,801,280 bytes
- SHA-256: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- M3 source report:
  `docs/reports/apple-silicon-macbook-m3-air-microsoft-2b-i2s.md`
- M3 handoff report:
  `docs/reports/apple-silicon-macbook-m3-air-m4-proof-handoff.md`

`1bitLLM/bitnet_b1_58-large` and `1bitLLM/bitnet_b1_58-3B` are not handoff
targets. `M3MBA-024` recorded that they remain blocked by official-artifact,
conversion or approval, and storage-gate requirements.

## Tokenizer Authority

The accepted M3 reference-runner path depends on external tokenizer and
pre-tokenizer authority:

| Field | Required value |
|---|---|
| External tokenizer source | `microsoft/bitnet-b1.58-2B-4T` |
| External tokenizer revision | `04c3b9ad9361b824064a1f25ea60a8be9599b127` |
| External tokenizer file | `tokenizer.json` |
| External tokenizer SHA-256 | `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7` |
| GGUF tokenizer model | `gpt2` |
| GGUF pre-tokenizer metadata | missing in GGUF |
| Required override | `tokenizer.ggml.pre=str:llama-bpe` or repository-native equivalent |

Any M4 proof item consuming this artifact must either use that same authority
path or record an equivalent repository-native tokenizer authority decision in
the fresh M4 receipt.

## M4 Receipt Checklist

The M4 lane may cite the M3 evidence only as artifact-selection input. A fresh
M4 Apple CPU/NEON BitNet receipt must record:

| Requirement | Checklist item |
|---|---|
| Host identity | M4 Mac mini host facts and `apple-m4-cpu-neon` selected backend |
| Artifact identity | Same repository, revision, filename, size, and SHA-256 as the M3 accepted artifact |
| Tokenizer authority | External tokenizer revision/SHA and required pre-tokenizer override or equivalent native authority |
| Runner route | Repository Rust route or explicitly named M4 reference route, not inherited M3 BitNet.cpp output |
| Fallback state | `fallback_used=false` for strict backend proof, or an explicit non-proof fallback reason |
| Quality evidence | Generated text, generated token IDs, prompt/corpus ID, scorer result, and failure taxonomy where applicable |
| Timing evidence | Load, tokenization, prefill/TTFT, decode, total wall time, timeout stage, and memory fields for benchmark claims |
| Claim boundary | No Metal, MPSGraph, Neural Engine, QK256-on-Apple, broad Apple Silicon, broad quality, or speedup claim unless a matching receipt family proves it |

Separate M4 campaign items already name the receipt families that consume the
accepted Microsoft I2_S artifact and external tokenizer identity, including
`M4-BITNET-EX-003`, `M4-BITNET-EX-011`, `M4-BITNET-EX-014`, `M4-BENCH-006`,
and `M4-SETUP-001`. This M3 alignment item does not reopen or validate those
M4 results; it only pins the metadata and checklist that prevents M3 evidence
from being treated as an M4 runtime substitute.

## Unsupported Claims

This report does not support:

- that M4 strict proof has passed because M3 reference-runner output passed;
- that M3 Air evidence is M4 Mac mini evidence;
- that the repository Rust Apple backend accepted the artifact because
  BitNet.cpp accepted the M3 reference-runner prompt suite;
- that Apple Metal, MPSGraph, Neural Engine, or QK256-on-Apple execution works;
- that any secondary BitNet candidate is accepted; or
- that a broad Apple Silicon quality or performance claim exists.

## Operator Handoff

When a future M4 item or replay bundle consumes this metadata, it should copy
the artifact and tokenizer rows above, cite the M3 source reports as selection
evidence, and attach fresh M4 receipt paths that satisfy the checklist. If the
M4 artifact hash, tokenizer revision, backend label, fallback state, or corpus
context differs, the M4 item must record the mismatch and avoid inheriting the
M3 acceptance decision.

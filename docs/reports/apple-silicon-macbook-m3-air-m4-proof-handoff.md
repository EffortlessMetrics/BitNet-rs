# Apple M3 MacBook Air M4 Proof Handoff

Date: 2026-05-17
Work item: `M3MBA-008`

## Result

`M3MBA-008` creates a handoff, not a proof. The M3 MacBook Air lane has one
artifact ready for a separate M4 Mac mini strict-proof item:

- `microsoft/bitnet-b1.58-2B-4T-gguf`
- revision `a1f2f1c765812aa8af3f6eda4a313707064bba15`
- file `ggml-model-i2_s.gguf`
- SHA-256 `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`

The artifact is accepted only for the recorded M3 Air BitNet.cpp
reference-runner context. It is not accepted as repository Rust Apple backend
evidence, M4 Mac mini evidence, Apple Metal proof, QK256 proof, or broad Apple
Silicon performance evidence.

## Source Evidence

| Evidence | Path |
|---|---|
| Identity/hash/storage | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-i2s-identity.json` |
| Tokenizer authority | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-i2s-tokenizer-authority.json` |
| Reference output | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-i2s-reference-output.json` |
| Narrative report | `docs/reports/apple-silicon-macbook-m3-air-microsoft-2b-i2s.md` |

The M3 reference-output receipt records:

- decision `accepted_for_m3_air_reference_context`,
- answer gate passed with 5 passed, 0 failed, and 0 not run,
- required tokenizer override `tokenizer.ggml.pre=str:llama-bpe`,
- Microsoft BitNet.cpp runner build `3962 (1f86f058)`,
- default M3 BitNet.cpp device context with `31/31` layers offloaded, and
- a failed forced `-ngl 0` diagnostic retained so the accepted result cannot be
  misread as CPU-only evidence.

## Handoff Target

A future M4 strict-proof item should start from the Microsoft 2B I2_S evidence
above and produce fresh M4 receipts. The handoff target should require:

- source repository, revision, filename, size, and SHA-256 equality with the M3
  evidence,
- external tokenizer authority from `microsoft/bitnet-b1.58-2B-4T` revision
  `04c3b9ad9361b824064a1f25ea60a8be9599b127`,
- the required `tokenizer.ggml.pre=str:llama-bpe` override or an equivalent
  repository-native tokenizer authority path,
- M4 Mac mini host identity and Apple CPU/NEON backend label in the new receipt,
- explicit fallback status,
- answer-gate output for the committed BitNet answer corpus, and
- a separate decision on whether any Rust backend path is accepted.

The M4 item may cite M3 evidence as artifact selection evidence only. It must
not reuse the M3 receipt as proof.

## Secondary Candidate State

The secondary candidates are not handoff targets:

| Candidate | Work item | State | Reason |
|---|---|---|---|
| `1bitLLM/bitnet_b1_58-large` | `M3MBA-006` | blocked | Official repository has no GGUF artifact at the recorded revision. |
| `1bitLLM/bitnet_b1_58-3B` | `M3MBA-007` | blocked | Official repository has no TL1/TL2 GGUF artifact, and downloading safetensors shards would violate the recorded free-space floor. |

Future unblocking work for those candidates needs an official GGUF,
reproducible conversion path, or explicitly approved third-party artifact plus
fresh storage preflight. Until then, the Microsoft 2B I2_S artifact is the only
handoff-ready M3 Air BitNet candidate.

## Claim Boundary

This report may claim that the official Microsoft 2B I2_S artifact is ready for
a separate M4 strict-proof work item. It must not claim that M4 proof has
passed, that M3 evidence is M4 evidence, that repository Rust Apple inference
works for this artifact, that Apple Metal BitNet inference works, that QK256 is
supported on Apple Silicon, or that broad Apple Silicon performance has been
measured.

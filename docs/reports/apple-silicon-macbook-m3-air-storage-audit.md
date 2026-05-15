# Apple M3 MacBook Air Storage Audit

Date: 2026-05-15
Work item: `M3MBA-010`

## Result

The M3 Air cache remains usable for the next bounded BitNet candidate work. The
local model cache retains the dense Qwen control artifact and the official
Microsoft 2B I2_S reference artifact, both with matching SHA-256 evidence, and
the volume remains above the preferred free-space floor.

Evidence receipt:

- `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/storage-audit.json`

## Storage

| Field | Value |
|---|---:|
| Volume | `/System/Volumes/Data` |
| Available | 49,507,532 KiB |
| Total | 482,797,652 KiB |
| Used | 384,533,348 KiB |
| Capacity | 89% |
| Hard floor | 8,388,608 KiB |
| Preferred floor | 26,214,400 KiB |

The lane is above the preferred floor after retaining the current artifacts, so
additional M3 candidate work is allowed if it remains serialized and records
before/after free space.

## Retained Artifacts

| Artifact | Role | Size | SHA-256 |
|---|---|---:|---|
| `qwen2.5-0.5b-instruct-q8_0.gguf` | Dense SLM control | 659,884 KiB | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `ggml-model-i2_s.gguf` | Microsoft 2B BitNet reference candidate | 1,159,964 KiB | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |

No model binaries are committed to the repository.

## Deleted Artifacts

No artifacts were deleted for this audit. The Microsoft 2B I2_S artifact remains
retained because `M3MBA-005C` accepted it for the M3 Air BitNet.cpp reference
context and follow-on candidate comparison.

## Decision

`M3MBA-006` and `M3MBA-007` may proceed, but only one large candidate download
should be active at a time. Each follow-on candidate must preflight disk space,
record cache root, record free space before and after, hash retained artifacts,
and keep model binaries out of git.

## Claim Boundary

This audit records cache and storage state only. It does not claim artifact
quality, Rust Apple backend support, M4 Mac mini proof, Apple Metal BitNet
inference, QK256 on Apple Silicon, or broad Apple Silicon performance.

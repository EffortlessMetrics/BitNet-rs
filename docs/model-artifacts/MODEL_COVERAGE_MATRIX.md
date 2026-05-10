# Model Coverage Matrix

`ci/model-artifacts/model-coverage-matrix.toml` is the cross-family claim
surface for local inference coverage. It complements the BitNet-family contract
registry and the dense Qwen capability summaries by showing where each model
lane sits in the proof ladder.

## Coverage Tiers

| Tier | Meaning |
|---|---|
| `registered` | The repo knows the model family and artifact class. |
| `structurally_valid` | The artifact parses and tensor roles are classified. |
| `reference_good` | A reference runner or accepted external evidence produced bounded coherent output. |
| `cpu_answer_ready` | The Rust CPU path has strict answer receipts. |
| `accelerator_answer_ready` | A strict accelerator path has fallback-free one-token, short-decode, or warm-session receipts. |
| `benchmark_qualified` | Exact profiles have governed same-artifact benchmark qualification receipts. |
| `product_cli_ready` | Normal user CLI paths exist for verified ask/chat/bench receipt surfaces; server readiness is still separate. |

Higher tiers do not erase the underlying claim boundary. For example, a model
can be CLI-ready for a bounded CUDA ask/chat path while still having
`speedup_claim=false`, `full_residency_claim=false`, and `server_ready=false`.

## Required Boundaries

- BitNet packed I2_S/QK256 proof and dense regular-LLM CUDA proof are separate
  claims.
- Unsupported upstream routes can be registered, but they cannot claim
  structural validity, answer-readiness, backend parity, speedup, or server
  readiness.
- Dense SLM and small-LLM entries must not claim BitNet packed proof.
- Speedup claims require benchmark qualification receipts for exact profiles.
- Product CLI readiness does not imply server readiness.

## Validation

Run:

```powershell
cargo run --release --locked -p xtask --no-default-features -- check-model-coverage
```

The validator parses the matrix, checks tier ordering, requires core lane
coverage, and rejects common claim leaks such as dense entries claiming BitNet
packed proof or unsupported entries claiming answer readiness.

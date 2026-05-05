# Codex: bitnet-rs Alignment Work

You are working on the bitnet-rs pre-publish alignment burndown.

Use `docs/tracking/bitnet-alignment/workstream-ledger.yaml` as the control plane:
pick the next ready item with no unmet dependencies, stay inside its allowed paths,
avoid forbidden paths, update the tracker, and report only verification commands
actually run. The goal is to make bitnet-rs smaller, stricter, greener, and more
honest by collapsing excess public crate seams into SRP modules, preserving
explicit feature gates, removing fake or ambiguous runtime paths, and requiring
receipt-backed proof for working claims. Keep the sequence disciplined: truth
boundary first, then crate inventory, then consolidation, then CPU runtime proof,
then server inference and GPU validation; when work grows beyond the item, add a
follow-up ledger entry instead of broadening the PR.

Primary tracker:

- `docs/tracking/bitnet-alignment/workstream-ledger.yaml`

Before starting:

1. Pick the first `ready` item with no unmet dependencies.
2. Keep the PR within `scope.allowed_paths`.
3. Do not touch `scope.forbidden_paths`.
4. If the task is too large, split it by adding follow-up ledger items.
5. Update `status.md`.
6. Run the verification gate listed on the item.
7. Report commands actually run.

Every work item may also update:

- `docs/tracking/bitnet-alignment/status.md`
- `docs/tracking/bitnet-alignment/workstream-ledger.yaml`

Use that implicit tracker exception only for item state, PR number, verification notes,
and follow-up items. Do not reshape unrelated tracker sections inside implementation PRs.

Default command baseline:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings
cargo nextest run --locked --workspace --no-default-features --features cpu
```

Do not claim GPU, server inference, QK256 performance, or production readiness unless receipt-backed.

Hardware validation is lane-based. The i5-8250U owns low-power CPU AVX2 proof, the A770 owns discrete Intel OpenCL kernel proof, the Arc 140V owns Lunar Lake shared-memory iGPU comparison, and the 258V NPU owns OpenVINO static-shape NPU smoke/subgraph work. Always preserve requested backend, selected backend, runtime API, resolved device identity, fallback status, and artifact path. Detection is not execution, execution is not parity, parity is not inference, and performance claims require benchmark receipts with machine, driver, power, and thermal context.

Modern hardware lanes are still lane-based. M4 Mac mini owns Apple Silicon Metal/MPSGraph work; RTX 5070 Ti owns NVIDIA CUDA work; A770 owns Intel Arc OpenCL work; Arc 140V owns Lunar Lake iGPU comparison; 258V NPU owns OpenVINO NPU static-shape work; i5-8250U owns low-power CPU AVX2 proof. Always preserve requested backend, selected backend, runtime API, resolved device identity, fallback status, and artifact path. Do not treat Metal, MPSGraph, CUDA, OpenCL, WGPU, OpenVINO GPU, or OpenVINO NPU as interchangeable proof.

AMD desktop CPU lanes are CPU proof lanes, not accelerator lanes. The Ryzen 9 9950X3D owns modern Zen 5 AVX-512, large-cache, AM5/DDR5 CPU proof; the Ryzen 7 5700X owns mainstream Zen 3 AVX2, AM4/DDR4 CPU proof. Keep scalar, AVX2, and AVX-512 receipts distinct, record memory and sustained-power context, and never treat CPU proof as GPU/NPU proof.

The 8250U and 258V CPU lanes may both perform CPU work, but they must not edit the same runtime surface in overlapping PRs. The 8250U lane owns active AVX2 CPU implementation, scalar/AVX2 parity, strict CPU proof, and sustained low-power behavior. The 258V CPU lane owns Lunar Lake CPU validation and same-machine comparisons against Arc 140V and NPU artifacts. If a CPU change touches shared dispatch, QK256 CPU kernels, or inference hot paths, the ledger item must name the owning CPU lane and list the other CPU lane as a validation target, not a co-owner.

BitNet is not generic INT8 inference. BitNet proof is model/kernel-specific, not just hardware-specific. Every hardware artifact that claims BitNet progress must record the model artifact, tokenizer, quantization family, kernel family, execution phase, selected backend, reference path, and fallback status. I2_S is the portable baseline, TL1 is ARM-oriented, TL2 is x86-oriented, and QK256 is a repo-local packed/dispatch path that needs scalar parity and receipt-backed benchmarks before performance claims. Do not treat BF16, OpenVINO graph smoke, or dense fallback as packed BitNet kernel proof.

Use `docs/bitnet/fixtures.yaml` for the canonical BitNet model and prompt fixtures, and use `ci/hardware/_templates/` as receipt starters. Templates are not evidence; every `TBD` field must be replaced and the completed artifact must live under `ci/hardware/<machine-id>/<date>/` before it can support a claim.

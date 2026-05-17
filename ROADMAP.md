# BitNet-rs Roadmap

BitNet-rs is a pre-alpha Rust-native local model runtime and validation
workspace for small efficient language models, including dense SLMs and BitNet /
1-bit model families. The roadmap below is intentionally proof-first: every
capability graduates only after the relevant model artifact, tokenizer, backend,
receipt, and campaign gates agree.

This file describes release direction and current limitations. It is not the
active work queue. Active branch names, allowed paths, reviewer policy, and exact
proof commands live in campaign trackers under
`docs/tracking/campaigns/<campaign>/active.toml`.

## Roadmap Authority Model

| Surface | Owns | Use it for |
| --- | --- | --- |
| `ROADMAP.md` | Release direction, sequencing, user-facing limitations | Understanding what the project is trying to become |
| `docs/tracking/campaigns/*/active.toml` | Live work-item state, branch names, allowed paths, merge policy | Executing PR-sized work |
| `docs/model-artifacts/ANSWER_ARTIFACT_GATE.md` | Artifact, tokenizer, prompt, and reference-runner authority | Deciding whether coherent answers can be claimed |
| `docs/model-artifacts/MODEL_COVERAGE_MATRIX.md` | Per-model coverage status | Choosing model targets and interpreting rejections |
| `docs/hardware/HARDWARE_MATRIX.md` | Hardware lane identity and proof labels | Deciding which machine/backend produced evidence |
| `docs/hardware/PROOF_STAGES.md` | Proof-stage ordering | Preventing detection-only evidence from becoming performance claims |
| `docs/claims.md` | Current claim ledger | Checking claim status and known blockers |
| `docs/status/` | User-facing claim tier summaries | Publishing supported, diagnostic, or unsupported status |

If these surfaces disagree, prefer the stricter proof gate or campaign tracker
and update the stale summary. Do not infer support from runnable code alone.

## Current Baseline

BitNet-rs already has substantial infrastructure, but supported coherent BitNet
local answers are still gated by artifact and backend proof.

### Available or useful today

- Strict GGUF loading, tokenizer metadata checks, and artifact diagnostics.
- Dense SLM local-answer work on selected Apple Silicon paths, with dense-model
  evidence kept separate from BitNet / 1-bit proof.
- I2_S / QK256 quantization and kernel infrastructure for scalar, SIMD, CUDA,
  OpenCL, OpenVINO, Metal, NPU, and WASM-oriented lanes.
- Honest-compute receipts that record selected backend, runtime API, fallback
  behavior, hardware identity, kernel coverage, and timing context.
- Cross-validation tooling for comparing Rust paths against reference runners
  and for preserving first-divergence evidence.
- Campaign-local trackers, generated dashboards, and xtask checks that make
  proof work reviewable and automergeable when green.

### Limitations that define the roadmap

- BitNet `run` and `chat` output remains diagnostic until the answer artifact
  gate, deterministic prompt suite, reference runner, and target backend receipts
  all pass for the specific model and backend.
- Structural GGUF validity is not answer readiness.
- Backend execution receipts are not proof of coherent answer quality.
- Hardware detection, smoke tests, and selected-device proofs are not speed or
  full-residency claims.
- Dense SLM success does not prove BitNet / 1-bit model quality.
- Server, WASM, and accelerator surfaces must return explicit unsupported or
  not-yet-implemented results until they are backed by real model execution and
  receipts.

## Graduation Ladder

Every major capability should move through the same ladder before it becomes a
supported claim:

1. **Identity** — exact model, tokenizer, pre-tokenizer, prompt template,
   hardware, runtime API, and selected backend are recorded.
2. **Probe** — the runtime can discover the target without fallback or label
   conflation.
3. **Smoke** — a narrow operation executes on the claimed backend and emits a
   strict receipt.
4. **Parity** — Rust output agrees with the accepted reference path under the
   lane tolerance.
5. **Answer gate** — deterministic prompt-suite output is coherent under the
   accepted artifact and reference runner.
6. **Operational path** — CLI, server, WASM, or batch APIs expose the path
   without simulated success or hidden fallback.
7. **Benchmark** — throughput, latency, memory, and sustained-power claims are
   qualified only after correctness and answer-readiness are proven.

## Milestone 0 — Keep Claim Boundaries Honest

**Goal:** Make the repo difficult to misread. Users and agents should be able to
see what is supported, diagnostic, planned, or unsupported without relying on
chat history or stale sprint notes.

**Primary workstreams**

- Maintain `README.md`, `docs/status/`, `docs/claims.md`, and generated tracking
  dashboards so they preserve the same claim tiers.
- Keep model artifact state and model/kernel compatibility explicit before
  hardware lanes use an artifact as evidence.
- Preserve campaign-local execution state instead of reviving global hidden goal
  files or hand-edited legacy trackers.
- Continue crate-boundary cleanup only where it does not obscure proof seams,
  feature gates, or public API intent.

**Exit criteria**

- Stale pages point to the maintained proof surfaces.
- Unsupported paths fail honestly rather than returning placeholder success.
- Work items include allowed paths, forbidden paths, proof commands, and
  `may_claim` / `must_not_claim` boundaries.

## Milestone 1 — Answer Artifact Authority

**Goal:** Qualify at least one BitNet-family artifact as answer-ready before any
backend claims coherent BitNet local answers.

**Scope**

- Official Microsoft BitNet b1.58 2B / 2B4T I2_S remains the main target for
  x86 CPU and CUDA answer lanes.
- Apple BitNet artifact sweeps may qualify candidates or reject them, but they
  do not by themselves prove M4 Mac mini, Metal, CPU/NEON, or Rust answer paths.
- Alternate quantizations and dense SLMs remain control lanes unless their own
  contracts explicitly say otherwise.

**Near-term work**

- Record exact source, revision, file name, size, SHA256, tokenizer authority,
  pre-tokenizer authority, prompt-template family, reference runner, command
  shape, prompt outputs, and cleanup state for each candidate.
- Preserve bad-artifact and missing-authority rejection evidence so future
  backend failures are not misdiagnosed as kernel bugs.
- Map legacy artifact statuses to the shared answer-gate states.

**Exit criteria**

- An artifact is marked `answer_ready` only after the deterministic prompt suite
  passes under the accepted reference runner.
- Rejected candidates explain whether the blocker is tokenizer authority,
  reference-runner failure, prompt-suite failure, unsupported upstream pairing,
  or campaign-level blocking.

## Milestone 2 — Rust CPU BitNet Correctness

**Goal:** Make Rust CPU BitNet inference strict, receipt-backed, and comparable
to the accepted reference path before treating GPU or server results as product
claims.

**Scope**

- Preserve separate scalar, AVX2, AVX-512, and NEON proof labels.
- Keep the Intel i5-8250U, AMD 5700X, AMD 9950X3D, Lunar Lake 258V, and Apple
  Silicon lanes distinct; do not transfer proof between machines.
- Continue QK256 and I2_S work as proof surfaces first, performance surfaces
  second.

**Near-term work**

- Finish remaining CPU dispatch and receipt proof for AMD 5700X scalar/AVX2 and
  AMD 9950X3D scalar/AVX2/AVX-512.
- Keep Kaby Lake 8250U and Lunar Lake 258V results as their own CPU baselines.
- Preserve layout, tokenizer, prompt, and first-divergence diagnostics so answer
  failures can be traced to artifact, tokenization, kernel, or generation state.
- Improve QK256 performance only after the correctness receipts show which path
  is being measured.

**Exit criteria**

- CPU lanes can explain exactly which SIMD path was selected and whether any
  fallback occurred.
- Reference parity failures produce actionable first-divergence evidence.
- CPU performance reports name the model, quantization, backend, machine,
  kernel family, and receipt artifact.

## Milestone 3 — Accelerator Proof Lanes

**Goal:** Validate accelerator execution without conflating detection, smoke,
parity, answer quality, residency, or throughput.

**Lane priorities**

| Lane | Near-term direction | Claim boundary |
| --- | --- | --- |
| NVIDIA RTX 5070 Ti CUDA | Maintain CUDA proof lane and qualify strict BitNet CUDA performance after answer/correctness gates. | CUDA execution and receipts are not coherent-answer or speed claims by themselves. |
| Intel Arc A770 | Continue OpenCL-first selected-device BitNet acceleration proof. | No full device residency, selected-attention, or Gemma-class support claim without the A770 claim ledger blockers resolved. |
| Lunar Lake 258V Arc 140V | Preserve integrated-GPU proof separate from CPU and NPU. | Same-machine evidence is comparison context, not interchangeable backend proof. |
| Lunar Lake NPU | Keep OpenVINO static-shape NPU smoke/parity/receipt evidence separate from GPU and CPU. | NPU detection is not dynamic-shape inference or speed proof. |
| Apple Metal / CPU NEON | Use Apple Silicon receipts for strict backend labels while artifact sweeps continue elsewhere. | Dense SLM or Metal smoke evidence is not broad BitNet Metal inference. |
| WASM CPU / SIMD | Establish compile, byte-loader, worker API, tiny-fixture receipt, and SIMD smoke stages. | WASM remains scaffolded until real generation receipts exist. |

**Exit criteria**

- Accelerator receipts include requested backend, selected backend, runtime API,
  resolved device identity, fallback state, and proof artifact path.
- Parity and answer-quality claims cite the same model artifact authority used
  by CPU proof.
- Performance claims are withheld until correctness and answer readiness pass on
  that backend.

## Milestone 4 — Local Answer UX

**Goal:** Turn proven model/backend paths into useful prompt-in,
answer-out behavior without hiding fallback or overstating support.

**Scope**

- Keep dense SLM local-answer UX useful where it is already proven, especially
  as an operator and regression surface on Apple M4.
- Promote BitNet local-answer UX only after answer-artifact and backend receipts
  pass for that exact path.
- Make failure modes explicit: missing tokenizer authority, unsupported model /
  kernel pairing, unavailable backend, fallback forbidden, or answer gate not
  passed.

**Exit criteria**

- CLI output and receipts agree on the selected model, backend, prompt template,
  and fallback behavior.
- Operator docs explain which commands produce diagnostic evidence and which
  commands produce supported local-answer evidence.
- Regressions can be detected with deterministic prompt and receipt checks.

## Milestone 5 — Server and API Productization

**Goal:** Expose only real engine execution or explicit unavailable responses
through server and API surfaces.

**Near-term work**

- Keep health and readiness endpoints aligned with actual model/backend state.
- Wire inference endpoints to real engine execution only after the target path
  has the same artifact, tokenizer, backend, and receipt proof expected of CLI
  execution.
- Preserve streaming, receipt export, model identity, and fallback reporting as
  first-class API behavior.

**Exit criteria**

- No server endpoint returns simulated success for inference.
- API responses include enough model, backend, and receipt context to audit the
  claim.
- Server status does not outrun the underlying CLI/backend proof lane.

## Milestone 6 — Performance, Benchmarks, and Release Readiness

**Goal:** Measure speed only after correctness is established, then turn proven
paths into reproducible release candidates.

**Scope**

- Separate one-shot correctness receipts, warm-session behavior, sustained
  benchmark profiles, and operator dashboards.
- Report throughput with hardware identity, power/thermal context where
  relevant, artifact hash, tokenizer authority, backend route, and fallback
  state.
- Keep CI economics explicit so expensive validation does not become accidental
  default work.

**Exit criteria**

- Benchmark reports are reproducible from documented commands and receipt
  artifacts.
- Performance dashboards never promote diagnostic or unsupported paths to
  supported claims.
- Release notes list supported paths, diagnostic paths, known blockers, and
  rollback guidance.

## Current Campaign Map

| Campaign | Roadmap role | Current emphasis |
| --- | --- | --- |
| `model-artifacts` | Milestone 1 | Shared answer-artifact authority and rejection evidence. |
| `apple-bitnet-artifact-sweep` | Milestone 1 | Apple Silicon candidate sweeps before M4 BitNet claims. |
| `cpu-proof` | Milestone 2 | Strict Rust CPU BitNet proof surface. |
| `amd-cpu-baselines` | Milestone 2 | 5700X and 9950X3D dispatch and benchmark context. |
| `cpu-qk256-performance` | Milestones 2 and 6 | QK256 performance after correctness boundaries. |
| `slm-cpu` | Milestones 2 and 4 | Small dense model CPU proof and local-answer baseline. |
| `nvidia-5070ti` | Milestones 3 and 6 | CUDA BitNet proof and performance qualification. |
| `intel-a770` | Milestone 3 | OpenCL-first A770 selected-device proof. |
| `intel-258v-platform` | Milestone 3 | CPU / Arc 140V / NPU comparison without label conflation. |
| `intel-npu` | Milestone 3 | OpenVINO NPU static-shape proof. |
| `apple-silicon-macbook` | Milestones 1 and 3 | MacBook cross-reference and larger artifact validation. |
| `apple-m4-*` | Milestones 3, 4, and 6 | Completed Apple M4 dense SLM, BitNet evidence, ops, server, and regression surfaces. |
| `server-real-inference` | Milestone 5 | Replace simulated server inference with real execution or honest unavailable responses. |
| `wasm-inference` | Milestone 3 and 5 | WASM compile, byte loaders, worker API, tiny-fixture proof, SIMD smoke. |
| `crate-collapse` | Milestone 0 | Reduce artificial microcrates without weakening proof seams. |
| `tracker-infra` | Milestone 0 | Campaign-local trackers and generated dashboards. |
| `ci-coverage` | Milestone 0 and 6 | Reliable coverage reporting without false failures from missing secrets. |

Use this table to orient yourself, then inspect the campaign `active.toml` before
editing code or docs for a specific work item.

## Non-goals for the Current Roadmap

- Claiming coherent BitNet answers from structurally valid artifacts alone.
- Treating dense SLM evidence as BitNet / 1-bit proof.
- Promoting hardware detection, smoke tests, or fallback-enabled runs to speed
  claims.
- Supporting broad general-purpose LLM serving before the strict local model
  proof surfaces are stable.
- Committing model binaries or other large generated artifacts.
- Creating hidden global goal files outside the campaign tracker model.

## How to Propose Roadmap Changes

1. Add or update a proposal when the change affects user-visible capability,
   proof policy, or major architecture.
2. Add or update specs, ADRs, plans, and campaign work items for executable
   scope.
3. Include allowed paths, forbidden paths, proof commands, claim boundaries, and
   rollback guidance in the campaign item.
4. Keep `ROADMAP.md` as a concise summary of direction, not as a duplicate of
   every generated dashboard row.

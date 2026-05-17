# bitnet-rs Roadmap

This roadmap translates the active campaign tracker into a public, contributor-friendly sequence of work. The tracker remains the executable source of truth for branch names, allowed paths, review mode, merge policy, and live work-item state.

- **Executable tracker:** `docs/tracking/campaigns/<campaign>/active.toml`
- **Planning conventions:** `plans/README.md`
- **Build and validation reference:** `CLAUDE.md`
- **Release state:** pre-alpha `v0.2.1-dev`

## Roadmap Rules

1. **Receipts before claims.** Hardware, backend, model-quality, answer-quality, and throughput claims require lane-specific receipts and proof commands.
2. **Answer quality is artifact-gated.** A structurally valid GGUF is not enough; answer claims require a reference-good artifact with tokenizer or pre-tokenizer authority and deterministic prompt evidence.
3. **Backend labels must stay separate.** CPU, CUDA, OpenCL, Metal, MPSGraph, NPU, WASM scalar, and WASM SIMD evidence must not be conflated.
4. **Fallbacks must be explicit.** Any CPU fallback, simulated response, placeholder generation, missing accelerator path, or malformed receipt must be recorded rather than hidden.
5. **No model binaries in git.** Roadmap items may add metadata, hashes, scripts, receipts, and docs, but not large model artifacts.
6. **No production claim yet.** bitnet-rs remains pre-alpha until strict proof, quality, performance, server, and packaging gates are all satisfied.

## Where We Are

bitnet-rs is a pre-alpha Rust inference engine for BitNet-family and small dense-model proof work. CPU inference works with SIMD dispatch for supported paths. CUDA, OpenCL, Apple, Intel NPU, and WASM lanes now have tracker scaffolding or partial evidence, but most accelerator and browser claims remain gated by receipts.

**What works today:**

- CPU inference surfaces for supported BitNet paths, including scalar and SIMD-dispatched kernel coverage.
- GGUF and SafeTensors loading surfaces with format and metadata validation work underway.
- Interactive CLI and prompt templating with broad chat-template coverage.
- Cross-validation and receipt workflows for proof-oriented inference runs.
- Apple M4 dense SLM local-answer, operational, Metal-phase, and productization evidence campaigns completed for their scoped claims.
- NVIDIA RTX 5070 Ti CUDA validation campaign largely completed, with a remaining SmolLM2 artifact-contract item before further model-specific claims.
- Intel 258V and Intel NPU campaigns completed or heavily populated with receipt-backed platform evidence under their scoped labels.
- WASM inference is explicitly tracked as a proof lane, with compile, byte-loader, worker API, tiny fixture, SIMD smoke, browser-worker, and canonical BitNet feasibility items queued.

**Known limitations:**

- QK256 is not yet a production-performance path; scalar and packed-layout work must continue before throughput claims.
- Full GPU BitNet inference is not broadly validated. CUDA is the furthest along; Metal, OpenCL, and other GPU lanes remain claim-gated.
- Model answer quality remains constrained by artifact authority. Some available BitNet artifacts can be structurally loadable but answer-bad.
- Server inference must continue replacing simulated or placeholder responses with real engine execution or explicit unavailable responses.
- WASM and browser inference are planned proof lanes, not product-ready browser inference.
- The workspace still carries a large scaffold surface and many campaign-specific proof artifacts.

## Milestone Map

| Milestone | Goal | Exit criteria | Primary campaigns |
| --- | --- | --- | --- |
| `v0.2.1` Proof hygiene | Keep CPU proof, receipts, docs, CI, and tracker gates honest while unblocking immediate proof lanes. | Campaign checks pass; no hidden fallback claims; public docs reflect tracker state. | `cpu-proof`, `model-artifacts`, `tracker-infra`, `ci-coverage` |
| `v0.2.x` CPU performance | Move QK256 and baseline CPU lanes from correctness into measured, receipt-backed performance envelopes. | QK256 scalar/packed/AVX2 evidence exists; AMD and Intel CPU baselines record dispatch and host context. | `cpu-qk256-performance`, `amd-cpu-baselines`, `intel-258v-platform` |
| `v0.2.x` Hardware breadth | Preserve platform-specific evidence without conflating CPU, GPU, NPU, Apple, CUDA, OpenCL, or dense-model claims. | Active accelerator lanes have selected-device receipts or explicit blockers. | `nvidia-5070ti`, `intel-a770`, `intel-npu`, Apple M-series campaigns |
| `v0.3.0` Real serving | Make server endpoints use the real inference engine or return explicit unavailable results. | Simulated inference surfaces are removed from production paths; server receipts and tests cover success and unavailable modes. | `server-real-inference` |
| `v0.3.x` Artifact authority | Establish at least one reference-good BitNet artifact with tokenizer authority and deterministic answer evidence. | A model artifact passes the answer prompt suite under reference authority and unblocks downstream hardware answer claims. | `model-artifacts`, `apple-bitnet-artifact-sweep`, `apple-m3-macbook-air` |
| `v0.4.0` Browser and sandbox proof | Turn WASM from scaffold into a strict, receipt-backed inference lane. | wasm32 compile succeeds; byte-backed loaders and worker API exist; tiny fixture and SIMD smoke produce strict receipts. | `wasm-inference` |
| Post-proof productization | Package validated lanes for contributors without weakening proof boundaries. | Quickstarts are tested, claims are receipt-linked, and unsupported paths fail honestly. | Cross-campaign closeouts |

## Active Workstreams

### 1. CPU proof and QK256 performance

**Goal:** Keep the CPU path as the reference proof surface while improving QK256 performance without exaggerating readiness.

**Next work:**

- Prove AMD Ryzen 7 5700X scalar and AVX2 dispatch without AVX-512 claims.
- Prove AMD Ryzen 9 9950X3D scalar, AVX2, and AVX-512 dispatch with cache-domain and scheduler context.
- Continue QK256 scalar, packed-layout, AVX2, and sustained benchmark evidence before claiming usable production throughput.
- Preserve Intel 258V CPU evidence separately from Arc GPU and Intel AI Boost NPU evidence.

**Exit gates:**

- Dispatch receipts identify selected CPU backend and fallback status.
- Performance receipts include host, compiler, feature flags, workload, token counts, and sustained-run context.
- 5700X material never claims AVX-512.

### 2. Model artifact and answer authority

**Goal:** Prevent hardware lanes from claiming coherent local answers using artifacts that are merely loadable.

**Next work:**

- Resolve the blocked reference-good BitNet artifact item by acquiring or regenerating a GGUF/tokenizer pair that passes deterministic prompt evidence under reference authority.
- Record SHA256, byte size, tokenizer/pre-tokenizer authority, chat template, context length, source, license, and cleanup status for accepted artifacts.
- Keep answer-quality claims separate from backend validation and throughput claims.

**Exit gates:**

- At least one BitNet-family artifact is accepted by deterministic reference prompt evidence.
- Rejected artifacts record why they failed and which downstream claims remain blocked.
- No hardware campaign promotes answer quality without linking to accepted artifact authority.

### 3. Apple Silicon lanes

**Goal:** Use Apple hardware for disciplined proof and artifact qualification without conflating dense SLM success with BitNet readiness.

**Completed foundations:**

- Apple M4 dense SLM local-answer, operational, productization, performance, eval, and durable-evidence campaigns are complete for their scoped claims.
- Apple M4 BitNet productization and eval/benchmark campaigns have completed their scoped tracker work.

**Next work:**

- Use MacBook lanes for larger Apple BitNet artifact sweeps before promoting M4 strict Apple CPU/NEON proof plans.
- Continue M3 MacBook Air proposed work for Metal/MPSGraph visibility and bounded preflight contracts.
- Keep superseded Apple Silicon MacBook umbrella items blocked unless proxy notes are needed.

**Exit gates:**

- Apple BitNet artifacts have source, hash, tokenizer authority, prompt output, runner, and cleanup evidence.
- M4 strict proof claims are created only after target backend receipts exist.
- Dense Qwen or other dense SLM evidence is not used as BitNet answer authority.

### 4. NVIDIA CUDA and Intel accelerator lanes

**Goal:** Validate accelerators with selected-device receipts while keeping accelerator, CPU fallback, and model-artifact claims separate.

**Next work:**

- Add the queued SmolLM2 360M artifact contract for the RTX 5070 Ti lane before further model-specific CUDA claims.
- Preserve Intel Arc A770 requested and selected backend identity before adding OpenCL inference claims.
- Maintain Intel NPU evidence as OpenVINO/static-shape scoped and separate from Intel GPU or CPU proof.

**Exit gates:**

- Receipts record requested backend, selected backend, device identity, driver/runtime, fallback status, and model/artifact hash.
- CUDA, OpenCL, and NPU material does not imply broad GPU support.
- Dense CUDA or dense SLM success is not used to claim BitNet CUDA answer quality.

### 5. Server real inference

**Goal:** Replace server-side simulated inference with real engine execution or explicit unavailable responses.

**Next work:**

- Keep removing fake generation from externally visible server routes.
- Wire real engine execution only where model, tokenizer, backend, and receipt boundaries are enforceable.
- Return explicit unavailable or not-implemented errors when proof prerequisites are absent.

**Exit gates:**

- Health endpoints remain lightweight and honest.
- Inference endpoints either produce real model-backed output with receipts or an explicit unavailable response.
- Tests cover both real execution and unavailable/failure modes.

### 6. WASM inference proof lane

**Goal:** Establish WASM as a real proof lane rather than a placeholder browser demo.

**Queued sequence:**

1. Compile `bitnet-wasm` for `wasm32-unknown-unknown` with browser inference features and explicit not-implemented errors.
2. Add byte-backed model and tokenizer loader APIs.
3. Expose a worker-safe JavaScript API for load, generate, streaming generate, unload, memory stats, and abort.
4. Prove a tiny committed fixture emits one real token with a strict WASM scalar receipt.
5. Add WASM SIMD packed-kernel smoke proof with scalar parity and nonzero SIMD invocation count.
6. Prove browser-worker short greedy decode with cached artifacts, streamed real output, memory high-water mark, and fallback status.
7. Attempt canonical BitNet GGUF feasibility with strict loader, tokenizer, packed weights, one greedy token, and native CPU parity.

**Exit gates:**

- No fake generation responses are exposed as success.
- WASM scalar and WASM SIMD claims are separated.
- Browser feasibility does not imply product speed unless separately benchmarked.

### 7. Workspace and CI maintainability

**Goal:** Reduce maintenance burden without breaking proof boundaries.

**Next work:**

- Continue crate-collapse only for low-risk public microcrates where behavior, feature gates, and API intent are preserved.
- Keep coverage upload/reporting reliable without failing unrelated forked PRs or missing-secret scenarios.
- Preserve campaign-local TOML manifests, append-only events, generated dashboards, and `xtask` gates.

**Exit gates:**

- `campaign check`, `campaign generate --check`, and `campaign doctor` stay green for touched campaigns.
- Public API movement is documented and tested.
- CI cost controls do not hide proof failures.

## Release Readiness Checklist

Before declaring any non-pre-alpha release, the project needs all of the following:

- [ ] A reference-good BitNet artifact with tokenizer authority and deterministic answer evidence.
- [ ] CPU proof receipts for correctness and an honest CPU performance envelope.
- [ ] At least one accelerator lane with strict selected-device receipts and no hidden CPU fallback.
- [ ] Server inference endpoints that avoid simulated success.
- [ ] Contributor quickstarts tested from a clean checkout.
- [ ] Receipt schema and verification commands documented for all public claims.
- [ ] Model-artifact acquisition, cache, hash, and cleanup procedures documented.
- [ ] CI and tracker gates that fail stale, malformed, or overclaiming evidence.

## Non-goals

These remain outside the roadmap unless a future proposal changes scope:

- General-purpose LLM inference beyond BitNet-family and proof-lane dense SLM support.
- SaaS or hosted inference service.
- Distributed inference across machines.
- Mobile product deployment.
- Committing model binaries or other large proprietary artifacts.
- Claiming production quality from scaffold, placeholder, or fallback-backed execution.

## How To Pick Up Work

1. Choose an active campaign in `docs/tracking/campaigns/<campaign>/active.toml`.
2. Select a `ready` work item unless the campaign explicitly authorizes proposed-item promotion.
3. Follow its `allowed_paths`, `forbidden_paths`, `commands`, `may_claim`, and `must_not_claim` fields.
4. Keep each PR scoped to one objective.
5. Update generated tracker outputs only through the documented `xtask` commands.
6. Commit proof artifacts, receipts, metadata, and docs; never commit model binaries.

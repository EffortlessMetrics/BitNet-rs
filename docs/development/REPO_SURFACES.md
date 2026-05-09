# Repository Surfaces

This document maps the intended BitNet-rs repository shape. It is a planning
document for reducing accidental Cargo package boundaries while preserving the
monorepo and single-responsibility design seams.

## Target model

The monorepo remains the home for:

- Runtime and inference orchestration.
- Kernels and hardware lanes.
- Model, tokenizer, prompt, and artifact contracts.
- Receipts and claim validation.
- CLI and server products.
- Bindings and installable sidecars.
- Tests, fixtures, benchmarks, CI, and docs.
- Campaign tracking and historical evidence.

Inside the monorepo, boundaries should be classified as:

- Public package surface.
- Internal module family.
- Dev/test/lab tooling.
- Historical evidence.

## Proposed public package set

The target public surface should be closer to 20-35 well-documented crates than
to every single SRP seam in the workspace.

### Product facades

| Crate | Role |
| --- | --- |
| `bitnet` | Primary user-facing facade, docs entry, install entry, and optional re-exports. |
| `bitnet-cli` | Command-line adapter and command UX. |
| `bitnet-server` | HTTP and OpenAI-compatible serving adapter. |

### Core SDK surfaces

| Crate | Role |
| --- | --- |
| `bitnet-inference` | Engine, session, prefill/decode, and generation orchestration API. |
| `bitnet-models` | Model artifact inspection, contracts, GGUF/SafeTensors authority, and compatibility checks. |
| `bitnet-tokenizers` | Tokenizer authority, prompt/token handling, discovery, and text normalization. |
| `bitnet-prompt-templates` | Prompt template contracts intended for application users. |
| `bitnet-sampling` | Sampling, logits policy, and decode choice configuration. |
| `bitnet-kernels` | CPU/CUDA/dense/BitNet kernel contracts and implementations. |
| `bitnet-receipts` | Receipt schemas, validators, and claim boundaries. |
| `bitnet-device-probe` | Hardware discovery when the external device-selection story is stable. |

### Optional backend, binding, and tool surfaces

These should be public only when each crate has a real external consumer story,
documented feature surface, stable claim boundary, and clean packaging dry run.

| Crate | Public condition |
| --- | --- |
| `bitnet-nvidia` | NVIDIA-specific backend or tuning API useful outside the CLI. |
| `bitnet-wgpu` | Cross-platform WebGPU backend with a standalone integration path. |
| `bitnet-metal` | Apple Metal backend with documented hardware targets. |
| `bitnet-opencl` | OpenCL backend with stable device support claims. |
| `bitnet-rocm` | AMD ROCm backend with stable packaging and support claims. |
| `bitnet-vulkan` | Vulkan backend with stable shader and dispatch contracts. |
| `bitnet-st2gguf` | Installable conversion tool. |
| `bitnet-ffi` | C ABI surface. |
| `bitnet-py` | Python binding package. |
| `bitnet-wasm` | WebAssembly binding package. |

### Maybe-public decisions

These need case-by-case boundary memos before they are treated as public
contracts:

- `bitnet-gguf`
- `bitnet-artifacts` if created for fetch, verify, cache, and artifact authority
- `bitnet-quantization`
- `bitnet-quantization-bits`
- `bitnet-bench`
- `bitnet-test-support`
- `bitnet-qk256-layout-core` or a renamed public QK256 layout crate, if external
  low-bit kernel authors genuinely need it

## Planned collapse waves

Collapse candidates should move into owner crates as module families unless a
boundary memo proves standalone value.

### CLI adapter internals

Owner crate: `bitnet-cli`

Candidate crates:

- `bitnet-cli-config-core`
- `bitnet-cli-sampling-core`
- `bitnet-build-info-core`

Target layout:

```text
crates/bitnet-cli/src/
  build_info/
  commands/
  config/
  sampling/
```

### Server adapter internals

Owner crate: `bitnet-server`

Candidate crates:

- `bitnet-client-ip-core`
- `bitnet-api-versioning-core`
- `bitnet-api-key-auth-core`
- `bitnet-http-auth-core`
- `bitnet-server-health-types-core`
- `bitnet-endpoint-registry-core`
- `bitnet-request-context-core`
- `bitnet-request-router-core`

Target layout:

```text
crates/bitnet-server/src/
  auth/
  endpoint_registry/
  health/
  request_context/
  routing/
  versioning/
```

### Inference, session, and generation internals

Owner crate: `bitnet-inference`

Candidate crates:

- `bitnet-generation-events-core`
- `bitnet-generation-stop-core`
- `bitnet-kv-cache-policy-core`
- `bitnet-engine-state-core`
- `bitnet-session-config-core`
- `bitnet-engine-core`
- `bitnet-repl-core`
- `bitnet-inference-metrics-core`

Target layout:

```text
crates/bitnet-inference/src/
  engine/
    lifecycle.rs
    ports.rs
  generation/
    events.rs
    stop.rs
  kv_cache/
    policy.rs
  metrics/
  session/
    config.rs
    state.rs
```

### Tokenizer internals

Owner crate: `bitnet-tokenizers`

Candidate crates:

- `bitnet-token-merge-core`
- `bitnet-tokenizer-model-core`
- `bitnet-tokenizer-discovery-core`
- `bitnet-tokenizer-text-core`

Target layout:

```text
crates/bitnet-tokenizers/src/
  authority/
  discovery/
  merge/
  model_detection/
  text/
```

### Model and artifact internals

Owner crate: `bitnet-models`, unless `bitnet-artifacts` is explicitly promoted
as a public artifact authority.

Candidate crates:

- `bitnet-compat-core`
- `bitnet-weight-name-core`
- `bitnet-layer-index-core`
- `bitnet-download-core`
- `bitnet-download`
- `bitnet-atomic-file-core`
- `bitnet-minimal-json-core`
- `bitnet-safetensors-ln`

Target layout:

```text
crates/bitnet-models/src/
  artifacts/
  compatibility/
  download/
  gguf/
  layer_index/
  naming/
  safetensors/
```

### Kernel implementation internals

Owner crate: `bitnet-kernels`

Candidate crates:

- `bitnet-cpu-activations`
- `bitnet-qk256-layout-core`
- `bitnet-dispatch-core`
- `bitnet-vulkan-shaders`
- `bitnet-wgpu-shaders-i2s`
- `bitnet-qk256-dispatch`

Target layout:

```text
crates/bitnet-kernels/src/
  cpu/
  cuda/
  dense/
  dispatch/
  qk256/
    dispatch.rs
    layout.rs
  shaders/
    vulkan/
    wgpu/
```

### Receipt and claim internals

Owner crate: `bitnet-receipts`

Candidate crates:

- `bitnet-receipts-core`
- `bitnet-bench-receipts`
- `bitnet-bench-regression-core`
- `bitnet-feature-contract`
- `bitnet-runtime-profile-contract`
- `bitnet-runtime-profile-contract-core`

Target layout:

```text
crates/bitnet-receipts/src/
  benchmark/
  explain/
  feature_contracts/
  schemas/
  validators/
```

### Dev and test support

Default state: workspace crates may remain separate when useful for CI or test
composition, but they should normally be `publish = false`.

Examples:

- `bitnet-test-support`
- `bitnet-test-fixtures-core`
- `bitnet-bdd-grid`
- `bitnet-bdd-grid-core`
- `bitnet-testing-policy`
- `bitnet-testing-policy-*`
- `bitnet-testing-scenarios-*`

## Clean architecture mapping

Use the package graph for public surfaces and the module graph for clean
architecture.

### `bitnet-inference`

```text
crates/bitnet-inference/src/
  adapters/
    cpu.rs
    cuda.rs
  domain/
    generation.rs
    kv_cache.rs
    session.rs
  engine/
    decode.rs
    prefill.rs
    warm_session.rs
  metrics/
  ports/
    kernel.rs
    model.rs
    receipts.rs
    tokenizer.rs
```

### `bitnet-server`

```text
crates/bitnet-server/src/
  adapters/
    axum/
    openai/
  auth/
  domain/
    request.rs
    response.rs
  health/
  ports/
    inference.rs
    receipt_sink.rs
  routing/
```

### `bitnet-cli`

```text
crates/bitnet-cli/src/
  commands/
  config/
  device_selection/
  model_aliases/
  output/
  receipts/
```

Ports and adapters remain explicit. They just do not need a separate package for
every port.

## Initial implementation roadmap

1. **Boundary doctrine**: keep this file and `CRATE_BOUNDARY_POLICY.md` as the
   baseline architectural agreement.
2. **Inventory and owner map**: add `ci/crate-boundaries/public-surfaces.toml`,
   `ci/crate-boundaries/collapse-candidates.toml`, and
   `ci/crate-boundaries/dev-only-crates.toml`.
3. **Advisory checker**: add `xtask crate-boundaries check` to report missing
   boundary memos, missing `publish = false`, missing public metadata, and
   publishable crates depending on internal path crates without migration notes.
4. **Server adapter collapse**: move obvious server `*-core` crates into
   `bitnet-server` module families.
5. **CLI adapter collapse**: move CLI config, sampling, and build-info internals
   into `bitnet-cli`.
6. **Tokenizer internals collapse**: move merge, model detection, discovery, and
   text internals into `bitnet-tokenizers`.
7. **Inference/session/generation collapse**: move generation, stop, KV policy,
   session, engine, REPL, and metrics seams into `bitnet-inference`.
8. **Receipts consolidation**: move receipt implementation shards into
   `bitnet-receipts` unless a memo proves a separate SDK surface.
9. **Kernel layout consolidation**: move QK256 layout/dispatch and shader catalog
   internals into `bitnet-kernels`, except for deliberately public seams.
10. **Publish surface dry run**: run package listing, publish dry run, and docs
    generation for every surviving public crate.

## Done criteria

The migration is complete when:

- Public crates each have a standalone user story.
- Internal SRP seams have module-family facades.
- Implementation-layer crates are not public by accident.
- Published crates do not depend on unpublished runtime path crates.
- `workspace.default-members` reflects normal developer workflow.
- `xtask` enforces crate-boundary policy.
- Docs explain which crate to use for each job.
- `cargo package`, `cargo publish --dry-run`, and `cargo doc --no-deps` pass for
  every public crate.

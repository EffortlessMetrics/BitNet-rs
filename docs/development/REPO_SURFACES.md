# Repository Surfaces

This document maps the desired BitNet-rs monorepo shape. It is a working target for crate-boundary cleanup, not a declaration that every listed crate already satisfies the public package quality bar.

The target is:

```text
large monorepo
clear public crate set
crate-grade internal module families
stable documented APIs where public
versioned schemas where external
strict layer checks replacing accidental crate boundaries
```

## Surface model

The repo should distinguish these layers:

| Layer | Meaning | Default package posture |
|---|---|---|
| Public package surface | Product, SDK, backend, binding, tool, or schema surfaces with external users. | Publishable only after a boundary memo and package dry-run. |
| Internal module families | SRP implementation seams owned by one public/product crate. | No separate crate; expose a local facade. |
| Dev/test/lab tooling | Test support, fixtures, fuzzing, benchmark harnesses, hardware experiments, migration tools. | Workspace-only and usually `publish = false`. |
| Historical evidence | Campaign events, receipts, archive notes, generated dashboards, baseline reports. | Docs/tracking/archive data, not package surface. |
| Forced package boundaries | Proc macros, sys/FFI, build helpers, fuzz, xtask, or feature-isolated packages. | Separate only because Cargo/distribution mechanics require it. |

## Target public surfaces

The desired public package set should be closer to 20-35 crates than 130+ crates. The categories below define the target vocabulary for future inventory work.

### Product facades

| Crate | Role |
|---|---|
| `bitnet` | Primary facade, user-facing crate name, install/docs entry, optional re-exports. |
| `bitnet-cli` | CLI adapter and command UX. |
| `bitnet-server` | HTTP/OpenAI-compatible serving adapter. |

### Core SDK surfaces

| Crate | Role |
|---|---|
| `bitnet-inference` | Engine, session, prefill/decode, and orchestration API. |
| `bitnet-models` | Artifact inspection, model contracts, GGUF/SafeTensors authority. |
| `bitnet-tokenizers` | Tokenizer authority, prompt/token handling, tokenizer discovery facade. |
| `bitnet-sampling` | Sampling, probability, logits, and logits policy surface. |
| `bitnet-receipts` | Receipt schemas, validators, claim boundaries, and explainability contracts. |
| `bitnet-kernels` | CPU/CUDA/dense/BitNet kernel contracts and implementations. |
| `bitnet-prompt-templates` | Prompt templates that are intentionally reusable outside the CLI/server. |

### Hardware/backend surfaces

Backend crates should be public only when they can stand alone. A backend needs a clear feature surface, documented hardware target, stable receipt/claim boundary, usefulness outside `bitnet-cli`, and clean package dry-runs.

Candidate public backend surfaces:

```text
bitnet-device-probe
bitnet-nvidia
bitnet-wgpu
bitnet-metal
bitnet-opencl
bitnet-rocm
bitnet-vulkan
```

Backends that do not yet pass the standalone test should live under `bitnet-kernels`, an owner backend module, or a hardware lab crate with `publish = false`.

### Tool, binding, and product sidecars

These are legitimate package surfaces when users install them directly or embed them through language/runtime-specific packaging:

```text
bitnet-st2gguf
bitnet-ffi
bitnet-py
bitnet-wasm
```

### Maybe-public, decide case by case

These require individual boundary memos before publication is assumed:

```text
bitnet-gguf
bitnet-artifacts
bitnet-quantization
bitnet-quantization-bits
bitnet-bench
bitnet-test-support
```

## Likely collapse waves

The lists below name seams that look like implementation layers today. They should be inventoried, assigned owners, and either collapsed into module families or explicitly justified with boundary memos.

### Server adapter internals -> `bitnet-server`

Likely owner layout:

```text
crates/bitnet-server/src/
  auth/
  routing/
  request_context/
  health/
  versioning/
  endpoint_registry/
```

Likely collapse candidates:

```text
bitnet-client-ip-core
bitnet-api-versioning-core
bitnet-api-key-auth-core
bitnet-server-health-types-core
bitnet-endpoint-registry-core
bitnet-request-context-core
bitnet-request-router-core
```

### CLI adapter internals -> `bitnet-cli`

Likely owner layout:

```text
crates/bitnet-cli/src/
  config/
  sampling/
  build_info/
  commands/
```

Likely collapse candidates:

```text
bitnet-cli-config-core
bitnet-cli-sampling-core
bitnet-build-info-core
```

### Inference/session/generation internals -> `bitnet-inference`

Likely owner layout:

```text
crates/bitnet-inference/src/
  generation/
    events.rs
    stop.rs
  session/
    config.rs
    state.rs
  kv_cache/
    policy.rs
  engine/
    ports.rs
    lifecycle.rs
  metrics/
```

Likely collapse candidates:

```text
bitnet-generation-events-core
bitnet-generation-stop-core
bitnet-kv-cache-policy-core
bitnet-engine-state-core
bitnet-session-config-core
bitnet-engine-core
bitnet-repl-core
bitnet-inference-metrics-core
```

### Tokenizer internals -> `bitnet-tokenizers`

Likely owner layout:

```text
crates/bitnet-tokenizers/src/
  merge/
  model_detection/
  discovery/
  text/
  authority/
```

Likely collapse candidates:

```text
bitnet-token-merge-core
bitnet-tokenizer-model-core
bitnet-tokenizer-discovery-core
bitnet-tokenizer-text-core
```

### Model/artifact support -> `bitnet-models` or `bitnet-artifacts`

If model fetch, verify, cache, and artifact authority are intended as a standalone SDK, introduce or preserve a public `bitnet-artifacts` surface. Otherwise, keep these under `bitnet-models`.

Likely owner layout under `bitnet-models`:

```text
crates/bitnet-models/src/
  artifacts/
  compatibility/
  download/
  gguf/
  safetensors/
  naming/
  layer_index/
```

Likely collapse candidates or case-by-case public surfaces:

```text
bitnet-compat-core
bitnet-weight-name-core
bitnet-layer-index-core
bitnet-download-core
bitnet-download
bitnet-atomic-file-core
bitnet-minimal-json-core
bitnet-safetensors-ln
```

### Kernel implementation seams -> `bitnet-kernels`

Likely owner layout:

```text
crates/bitnet-kernels/src/
  cpu/
  cuda/
  qk256/
    layout.rs
    dispatch.rs
  dense/
  dispatch/
  shaders/
    vulkan/
    wgpu/
```

Likely collapse candidates:

```text
bitnet-cpu-activations
bitnet-qk256-layout-core
bitnet-dispatch-core
bitnet-vulkan-shaders
bitnet-wgpu-shaders-i2s
bitnet-qk256-dispatch
```

`qk256-layout` is a watch item. If low-bit kernel authors outside the repo would choose it directly, it may deserve a public boundary. Otherwise it belongs under `bitnet-kernels::qk256`.

### Receipt and contract seams -> `bitnet-receipts`

The receipt system is a real public surface, but each implementation shard does not need a package identity.

Likely owner layout:

```text
crates/bitnet-receipts/src/
  schemas/
  validators/
  benchmark/
  feature_contracts/
  explain/
```

Likely collapse candidates:

```text
bitnet-receipts-core
bitnet-bench-receipts
bitnet-bench-regression-core
bitnet-feature-contract
bitnet-runtime-profile-contract
```

## Clean architecture module maps

The package graph should express public surfaces. The module graph should express clean architecture.

### `bitnet-inference`

```text
crates/bitnet-inference/src/
  domain/
    session.rs
    generation.rs
    kv_cache.rs
  ports/
    model.rs
    tokenizer.rs
    kernel.rs
    receipts.rs
  engine/
    prefill.rs
    decode.rs
    warm_session.rs
  adapters/
    cpu.rs
    cuda.rs
  metrics/
```

### `bitnet-server`

```text
crates/bitnet-server/src/
  domain/
    request.rs
    response.rs
  ports/
    inference.rs
    receipt_sink.rs
  adapters/
    axum/
    openai/
  auth/
  routing/
  health/
```

### `bitnet-cli`

```text
crates/bitnet-cli/src/
  commands/
  output/
  config/
  receipts/
  model_aliases/
  device_selection/
```

## Work sequence

Recommended order:

1. Add the boundary doctrine and repo surface docs.
2. Add crate inventory and owner-map files under `ci/crate-boundaries/`.
3. Add advisory `xtask crate-boundaries check` enforcement.
4. Collapse server adapter internals into `bitnet-server`.
5. Collapse CLI adapter internals into `bitnet-cli`.
6. Collapse tokenizer internals into `bitnet-tokenizers`.
7. Collapse inference/session/generation internals into `bitnet-inference`.
8. Consolidate receipts under `bitnet-receipts` unless a separate SDK reason survives.
9. Consolidate kernel layout/dispatch/shader catalog seams under `bitnet-kernels`, except deliberately public seams.
10. Run publish-surface dry-runs for every surviving public package.

## Inventory file targets

Follow-up inventory should live under:

```text
ci/crate-boundaries/public-surfaces.toml
ci/crate-boundaries/collapse-candidates.toml
ci/crate-boundaries/dev-only-crates.toml
```

Example entry:

```toml
[crate.bitnet-api-key-auth-core]
current_role = "internal_workspace_crate"
target_owner = "bitnet-server"
target_state = "module_family"
seam_type = "server_adapter"
public = false
collapse_wave = "server-adapters"
```

## Completion criteria

The migration is done when:

- public crates each have a standalone user story;
- internal SRP seams have module-family facades;
- no implementation-layer crate is public by accident;
- published crates do not depend on unpublished runtime path crates;
- `workspace.default-members` reflects normal developer workflow;
- `xtask` enforces crate-boundary policy;
- docs explain which crate to use for which job;
- `cargo package`, `cargo publish --dry-run`, and `cargo doc --no-deps` pass for every public crate.

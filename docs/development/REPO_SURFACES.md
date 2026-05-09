# Repository Surfaces

This document maps BitNet-rs workspace members into intended support surfaces.
It is the working companion to the crate boundary policy: the policy explains
how to decide; this map explains where the repository should move.

The target shape is:

```text
large monorepo
clear public crate set
crate-grade internal module families
stable documented APIs where public
versioned schemas where external
strict layer checks replacing accidental crate boundaries
```

## Layer model

The repository has four different kinds of things. They may all live in one
workspace, but they must not be treated as the same support boundary.

| Layer | Meaning | Default package stance |
| --- | --- | --- |
| Public package surface | Crates users install, embed, read docs for, or depend on directly. | Publishable only with a boundary memo and package metadata. |
| Internal module family | SRP seams owned by one public or product crate. | Implement under the owner crate as modules. |
| Dev/test/lab tooling | Fixtures, harnesses, benchmarks, hardware experiments, policy snapshots, CI helpers. | Workspace crate only when useful; usually `publish = false`. |
| Historical evidence | Tracking, campaign logs, archive notes, generated dashboards. | Documentation/data, not crates. |

## Target public package set

The public set is intentionally smaller than the workspace. It should be large
enough to support real users and small enough that each package has a clear
contract.

### Product facades

| Crate | Role |
| --- | --- |
| `bitnet` | Primary user-facing crate, install/docs entry point, and optional re-export facade. |
| `bitnet-cli` | Command-line adapter and user workflow surface. |
| `bitnet-server` | HTTP and OpenAI-compatible serving adapter. |

### Core SDK surfaces

| Crate | Role |
| --- | --- |
| `bitnet-inference` | Engine, session, prefill/decode, and orchestration API. |
| `bitnet-models` | Model artifact inspection, compatibility, GGUF/SafeTensors authority, and naming contracts. |
| `bitnet-tokenizers` | Tokenizer authority, prompt/token handling, model detection, and text conversion. |
| `bitnet-sampling` | Sampling, logits policy, filters, and probability behavior. |
| `bitnet-receipts` | Receipt schemas, validators, and claim boundaries. |
| `bitnet-kernels` | CPU/CUDA/dense/BitNet kernel contracts and implementations. |

### Backend, binding, and tool surfaces

These become public only when they have a clear standalone user story,
documented hardware or packaging target, stable claim boundary, and clean dry-run
packaging.

| Crate | Intended public story |
| --- | --- |
| `bitnet-device-probe` | Device and capability inspection usable outside the CLI. |
| `bitnet-nvidia` | NVIDIA backend/tuning surface when stable enough to use independently. |
| `bitnet-wgpu` | WebGPU backend surface when stable enough to embed directly. |
| `bitnet-metal` | Apple Metal backend surface. |
| `bitnet-opencl` | OpenCL backend surface, especially for Intel/portable GPU paths. |
| `bitnet-rocm` | AMD ROCm backend surface. |
| `bitnet-vulkan` | Vulkan backend surface. |
| `bitnet-st2gguf` | SafeTensors-to-GGUF conversion tool surface. |
| `bitnet-ffi` | C-compatible embedding boundary. |
| `bitnet-py` | Python package/binding boundary. |
| `bitnet-wasm` | WebAssembly package/binding boundary. |

### Maybe-public decisions

These require explicit boundary memos before publishing or long-term support.

| Crate | Decision question |
| --- | --- |
| `bitnet-gguf` | Is GGUF authority useful as a standalone SDK surface or only through `bitnet-models`? |
| `bitnet-artifacts` | Should fetch, verify, cache, and artifact authority stand alone from `bitnet-models`? |
| `bitnet-quantization` | Is quantization a user-facing API or a kernel/model implementation detail? |
| `bitnet-quantization-bits` | Do external low-bit authors need these contracts directly? |
| `bitnet-bench` | Is there an installable benchmark product or only internal performance tracking? |
| `bitnet-test-support` | Is another repository a real consumer, or should this remain dev-only? |

## Owner map for collapse candidates

The entries below are starting targets. They are not a command to delete crates
in one PR; they are migration lanes for inventory, advisory checks, and focused
collapse waves.

### Server adapter internals -> `bitnet-server`

| Current crate | Target module family | Target state |
| --- | --- | --- |
| `bitnet-client-ip-core` | `client_ip` or `request_context::client_ip` | Internal module family. |
| `bitnet-api-versioning-core` | `versioning` | Internal module family. |
| `bitnet-api-key-auth-core` | `auth::api_key` | Internal module family. |
| `bitnet-server-health-types-core` | `health` | Internal module family. |
| `bitnet-endpoint-registry-core` | `endpoint_registry` | Internal module family. |
| `bitnet-request-context-core` | `request_context` | Internal module family. |
| `bitnet-request-router-core` | `routing` | Internal module family. |

Target layout:

```text
crates/bitnet-server/src/
  auth/
  routing/
  request_context/
  health/
  versioning/
  endpoint_registry/
```

### CLI internals -> `bitnet-cli`

| Current crate | Target module family | Target state |
| --- | --- | --- |
| `bitnet-cli-config-core` | `config` | Internal module family. |
| `bitnet-cli-sampling-core` | `sampling` | Internal module family. |
| `bitnet-build-info-core` | `build_info` | Internal module family. |

Target layout:

```text
crates/bitnet-cli/src/
  config/
  sampling/
  build_info/
  commands/
```

### Inference/session/generation internals -> `bitnet-inference`

| Current crate | Target module family | Target state |
| --- | --- | --- |
| `bitnet-generation-events-core` | `generation::events` | Internal module family, with selected public re-exports if part of the SDK. |
| `bitnet-generation-stop-core` | `generation::stop` | Internal module family. |
| `bitnet-kv-cache-policy-core` | `kv_cache::policy` | Internal module family. |
| `bitnet-engine-state-core` | `session::state` or `engine::state` | Internal module family. |
| `bitnet-session-config-core` | `session::config` | Internal module family. |
| `bitnet-engine-core` | `engine` | Internal module family. |
| `bitnet-repl-core` | `repl` or CLI-owned module if only CLI uses it. | Internal module family. |
| `bitnet-inference-metrics-core` | `metrics` | Internal module family or public re-export from `bitnet-inference`. |

Target layout:

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

### Tokenizer internals -> `bitnet-tokenizers`

| Current crate | Target module family | Target state |
| --- | --- | --- |
| `bitnet-token-merge-core` | `merge` | Internal module family unless external tokenizer authors use it directly. |
| `bitnet-tokenizer-model-core` | `model_detection` or `model` | Internal module family. |
| `bitnet-tokenizer-discovery-core` | `discovery` | Internal module family. |
| `bitnet-tokenizer-text-core` | `text` | Internal module family. |

Target layout:

```text
crates/bitnet-tokenizers/src/
  merge/
  model_detection/
  discovery/
  text/
  authority/
```

### Model/artifact internals -> `bitnet-models` or `bitnet-artifacts`

| Current crate | Target module family | Target state |
| --- | --- | --- |
| `bitnet-compat-core` | `compatibility` | Internal module family unless compatibility diagnostics are public SDK. |
| `bitnet-weight-name-core` | `naming` | Internal module family. |
| `bitnet-layer-index-core` | `layer_index` | Internal module family. |
| `bitnet-download-core` | `artifacts::download` | Internal module family or part of `bitnet-artifacts`. |
| `bitnet-download` | `artifacts::download` | Maybe-public only if model download is a standalone tool/API. |
| `bitnet-atomic-file-core` | `artifacts::atomic_file` | Internal implementation detail. |
| `bitnet-minimal-json-core` | `artifacts::json` or private utility | Internal implementation detail. |
| `bitnet-safetensors-ln` | `safetensors::layer_norm` | Internal module family unless standalone conversion users exist. |

Target layout if owned by `bitnet-models`:

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

Create `bitnet-artifacts` only if fetch, verify, cache, and artifact authority
have a standalone SDK audience.

### Kernel internals -> `bitnet-kernels`

| Current crate | Target module family | Target state |
| --- | --- | --- |
| `bitnet-cpu-activations` | `cpu::activations` | Internal module family unless external CPU kernel authors depend on it. |
| `bitnet-qk256-layout-core` | `qk256::layout` | Watchlist: maybe public if external low-bit authors need it. |
| `bitnet-dispatch-core` | `dispatch` | Internal module family unless backend authors need a stable contract. |
| `bitnet-vulkan-shaders` | `shaders::vulkan` | Internal module family or backend-private artifact catalog. |
| `bitnet-wgpu-shaders-i2s` | `shaders::wgpu` | Internal module family or backend-private artifact catalog. |
| `bitnet-qk256-dispatch` | `qk256::dispatch` | Internal module family unless standalone dispatch tuning is public. |

Target layout:

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

### Receipts and contracts -> `bitnet-receipts`

| Current crate | Target module family | Target state |
| --- | --- | --- |
| `bitnet-receipts-core` | `schemas` and `validators` | Internal module family with public re-exports for schema types. |
| `bitnet-bench-receipts` | `benchmark` | Internal module family unless benchmark receipts are a standalone API. |
| `bitnet-bench-regression-core` | `benchmark::regression` | Internal module family. |
| `bitnet-feature-contract` | `feature_contracts` | Internal module family or public receipt contract if externally consumed. |
| `bitnet-runtime-profile-contract` | `runtime_profile` | Internal module family or public receipt contract if externally consumed. |

Target layout:

```text
crates/bitnet-receipts/src/
  schemas/
  validators/
  benchmark/
  feature_contracts/
  explain/
```

## Clean architecture examples

Use the package graph for support surfaces and the module graph for clean
architecture.

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

## Migration plan

### PR 1: boundary doctrine

Add the policy and surface map documents. This establishes vocabulary and stops
new accidental public package surfaces.

### PR 2: crate inventory and owner map

Add machine-readable inventories:

```text
ci/crate-boundaries/public-surfaces.toml
ci/crate-boundaries/collapse-candidates.toml
ci/crate-boundaries/dev-only-crates.toml
```

Each entry should record the current role, target owner, target state, seam type,
public flag, and collapse wave.

Example:

```toml
[crate.bitnet-api-key-auth-core]
current_role = "internal_workspace_crate"
target_owner = "bitnet-server"
target_state = "module_family"
seam_type = "server_adapter"
public = false
collapse_wave = "server-adapters"
```

### PR 3: `xtask crate-boundaries check`

Add advisory checks for the policy:

```text
no new public crate without boundary memo
publish=false for internal/dev-only crates
public crates have README/description/license/docs
internal collapse candidates are not dependencies of publishable crates without a migration note
```

### PRs 4-9: collapse waves

Collapse one owner family per PR, in the order listed in the policy. Prefer
compatibility shims over large breaking changes when a public surface temporarily
needs time to move.

### PR 10: publish surface dry-run

For each surviving public crate, run:

```text
cargo package --list -p <crate>
cargo publish --dry-run -p <crate>
cargo doc -p <crate> --no-deps
```

The migration is not complete until the published package graph is real.

## Completion criteria

The migration is complete when:

- public crates each have a standalone user story;
- internal SRP seams have module-family facades;
- no implementation-layer crate is public by accident;
- published crates do not depend on unpublished runtime path crates;
- `workspace.default-members` reflects normal developer workflow;
- `xtask` enforces the crate-boundary policy;
- docs explain which crate to use for which job;
- package listing, publish dry-runs, and docs generation pass for every public
  crate.

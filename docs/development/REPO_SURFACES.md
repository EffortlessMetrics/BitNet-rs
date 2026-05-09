# Repository Surfaces

This document is the working map for reducing accidental public package surface
while preserving the BitNet-rs monorepo and SRP architecture. Use it together
with the [Crate Boundary Policy](CRATE_BOUNDARY_POLICY.md).

## Target model

BitNet-rs remains one repository for:

- runtime,
- kernels,
- model contracts,
- receipts,
- CLI,
- server,
- bindings,
- hardware lanes,
- tests,
- docs,
- campaign tracking, and
- historical evidence.

Inside that monorepo, the workspace must distinguish:

- public package surface,
- internal module families,
- dev/test/lab tooling, and
- historical evidence.

## Target public surface

The target is roughly **20-35 public crates**, not a crate for every SRP seam.
This is a planning target, not an automatic deletion list. A crate survives by
passing the policy's boundary memo and survival tests.

### Core product set

| Crate | Role |
| --- | --- |
| `bitnet` | Primary facade, user-facing crate name, install/docs entry, optional re-exports. |
| `bitnet-cli` | CLI adapter and command UX. |
| `bitnet-server` | HTTP/OpenAI-compatible serving adapter. |
| `bitnet-inference` | Engine/session/decode orchestration API. |
| `bitnet-models` | Artifact inspection, contracts, GGUF/SafeTensors model authority. |
| `bitnet-tokenizers` | Tokenizer authority and prompt/token handling. |
| `bitnet-prompt-templates` | Prompt and chat-template contracts when useful outside tokenization. |
| `bitnet-sampling` | Sampling and logits policy. |
| `bitnet-kernels` | CPU/CUDA/dense/BitNet kernel contracts and implementations. |
| `bitnet-receipts` | Receipt schemas, validators, and claim boundaries. |
| `bitnet-device-probe` | Device detection and capability probing when useful outside product facades. |

### Optional backend, binding, and tool set

These crates should be public only when their packaging story is real and their
standalone dry-runs pass.

| Crate | Public only if... |
| --- | --- |
| `bitnet-nvidia` | NVIDIA/CUDA users can consume it directly with documented features and claims. |
| `bitnet-wgpu` | WGPU backend has a stable, documented backend surface. |
| `bitnet-metal` | Metal backend has a standalone Apple Silicon/iOS/macOS story. |
| `bitnet-opencl` | OpenCL backend has a stable hardware target and external use case. |
| `bitnet-rocm` | ROCm backend has a stable AMD GPU story. |
| `bitnet-vulkan` | Vulkan backend has a stable external backend API. |
| `bitnet-st2gguf` | Users install the converter directly. |
| `bitnet-ffi` | C ABI consumers embed BitNet-rs through the FFI package. |
| `bitnet-py` | Python users install or build the Python binding. |
| `bitnet-wasm` | Browser/JS users consume the WASM package. |

### Maybe-public, decide case by case

| Crate | Decision question |
| --- | --- |
| `bitnet-gguf` | Is it a stable GGUF parser users would choose directly, or a model-loader implementation detail? |
| `bitnet-artifacts` | Do fetch/verify/cache/artifact contracts deserve a standalone SDK surface outside `bitnet-models`? |
| `bitnet-quantization` | Do external low-bit model authors consume it directly? |
| `bitnet-quantization-bits` | Is the bit layout a stable external contract or an implementation detail? |
| `bitnet-bench` | Is it a user-installed benchmark product or repository-local validation tooling? |
| `bitnet-test-support` | Does another repository depend on it, or is it workspace-only support? |

### Likely internal module families

Most crates matching these patterns should move toward module families unless a
boundary memo says otherwise:

- `*-core`,
- `*-contract-core`,
- `*-policy-core`,
- `*-state-core`,
- `*-snapshot-core`, and
- `*-diagnostics-core`.

These names usually indicate a design seam rather than an independent package
product.

## Owner map

Use this owner map when creating inventory entries or planning collapse PRs.

| Owner crate | Internal module families to prefer |
| --- | --- |
| `bitnet-cli` | `config`, `sampling`, `build_info`, `commands`, `output`, `receipts`, `model_aliases`, `device_selection` |
| `bitnet-server` | `auth`, `routing`, `request_context`, `health`, `versioning`, `endpoint_registry`, `domain`, `ports`, `adapters` |
| `bitnet-inference` | `generation`, `session`, `kv_cache`, `engine`, `metrics`, `domain`, `ports`, `adapters` |
| `bitnet-tokenizers` | `merge`, `model_detection`, `discovery`, `text`, `authority` |
| `bitnet-models` | `artifacts`, `compatibility`, `download`, `gguf`, `safetensors`, `naming`, `layer_index` |
| `bitnet-kernels` | `cpu`, `cuda`, `qk256`, `dense`, `dispatch`, `shaders` |
| `bitnet-receipts` | `schemas`, `validators`, `benchmark`, `feature_contracts`, `runtime_profile`, `explain` |

## Clean architecture examples

Use the package graph for public surfaces and the module graph for clean
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

## Migration roadmap

### PR 1 — Boundary doctrine

Add this policy set:

- `docs/development/CRATE_BOUNDARY_POLICY.md`
- `docs/development/REPO_SURFACES.md`

### PR 2 — Crate inventory and owner map

Add:

- `ci/crate-boundaries/public-surfaces.toml`
- `ci/crate-boundaries/collapse-candidates.toml`
- `ci/crate-boundaries/dev-only-crates.toml`

Each entry should look like:

```toml
[crate.bitnet-api-key-auth-core]
current_role = "internal_workspace_crate"
target_owner = "bitnet-server"
target_state = "module_family"
seam_type = "server_adapter"
public = false
collapse_wave = "server-adapters"
```

### PR 3 — Advisory boundary check

Add `xtask crate-boundaries check` to report:

- new public crates without boundary memos,
- internal/dev-only crates missing `publish = false`,
- public crates missing README/description/license/docs metadata, and
- publishable crates depending on collapse candidates without a migration note.

The check should start advisory and become blocking after the inventory is
reviewed.

### PR 4 — Server adapter collapse

Collapse obvious server `*-core` crates into `bitnet-server` module families.

### PR 5 — CLI adapter collapse

Collapse CLI config, sampling, and build-info internals into `bitnet-cli`.

### PR 6 — Tokenizer internals collapse

Collapse token merge, model detection, discovery, and text internals into
`bitnet-tokenizers`.

### PR 7 — Inference/session/generation collapse

Collapse generation, stop, session, engine, KV-cache, REPL, and metrics internals
into `bitnet-inference`. This is larger than the server and CLI waves and should
come after the easier migrations.

### PR 8 — Receipts consolidation

Collapse `bitnet-receipts-core` into `bitnet-receipts` unless an independent SDK
reason is documented.

### PR 9 — Kernel layout consolidation

Move QK256 layout/dispatch and shader catalog crates under `bitnet-kernels`,
except for seams deliberately kept public.

### PR 10 — Public surface dry-run

For every surviving public crate, run:

```bash
cargo package --list -p <crate>
cargo publish --dry-run -p <crate>
cargo doc -p <crate> --no-deps
```

## Acceptance criteria

The migration is complete when:

- public crates each have a standalone user story,
- internal SRP seams have module-family facades,
- no implementation-layer crate is public by accident,
- published crates do not depend on unpublished runtime path crates,
- `workspace.default-members` reflects normal developer workflow,
- `xtask` enforces crate-boundary policy,
- docs explain which crate to use for which job, and
- packaging, publish dry-runs, and docs pass for every public crate.

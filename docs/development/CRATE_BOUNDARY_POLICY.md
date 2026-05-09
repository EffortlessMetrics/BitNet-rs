# Crate Boundary Policy

BitNet-rs is a monorepo, not a single-crate project. The correction this policy
makes is **not anti-monorepo** and **not anti-SRP**. It is
**anti-accidental-public-surface**: design seams may be as small as microcrates in
our architecture, but most of those seams should be implemented as module
families instead of permanently published package boundaries.

This policy applies to every new or existing `Cargo.toml` in the workspace.

## Boundary doctrine

> Design seams like microcrates, implement most seams as module families, and
> publish only the seams we are willing to support as real contracts.

The package graph is for stable external surfaces. The module graph is for clean
architecture. A small responsibility does not automatically deserve a crate.

## Repository layers

Every workspace component must be classified into exactly one of these layers:

| Layer | Definition | Default publishing stance |
| --- | --- | --- |
| Public package surface | A crate with a standalone user story and an API we are willing to version and support. | May publish after boundary memo approval. |
| Internal module family | A cohesive SRP seam owned by one public crate and reached through that crate's facade. | No separate `Cargo.toml`. |
| Dev-only crate | Test, fixture, grid, policy, benchmark, or local automation support. | `publish = false`. |
| Hardware lab crate | Experimental or bring-up lane for backend exploration before it has a stable external story. | `publish = false` until promoted. |
| Evidence/history | Receipts, generated baselines, migration notes, campaign tracking, and archived reports. | Not a package surface. |
| Forced package boundary | A separate crate required by build target, language binding, proc macro, feature isolation, publication, or binary packaging constraints. | Must document why a module family is not enough. |

## Public crate categories

A public crate should fit one of the following categories.

### Product facades

These are how users enter the system:

- `bitnet` — primary facade, user-facing crate name, documentation entry point,
  and optional re-export surface.
- `bitnet-cli` — CLI adapter and command UX.
- `bitnet-server` — HTTP/OpenAI-compatible serving adapter.

### Core SDK surfaces

These are direct programmatic surfaces:

- `bitnet-inference` — engine, session, prefill/decode, and generation
  orchestration.
- `bitnet-models` — artifact inspection, model contracts, GGUF/SafeTensors model
  authority.
- `bitnet-tokenizers` — tokenizer authority, prompt/token handling, discovery,
  and text normalization.
- `bitnet-sampling` — sampling and logits policy.
- `bitnet-receipts` — receipt schemas, validators, claim boundaries, and
  versioned evidence contracts.
- `bitnet-kernels` — CPU/CUDA/dense/BitNet kernel contracts and supported
  implementations.

### Hardware/backend surfaces

Backend crates are public only after they have a real external user story:

- documented hardware target,
- clear feature surface,
- stable receipt/claim boundary,
- usefulness outside `bitnet-cli`, and
- clean package and publish dry-runs.

Before that point, backend work belongs under `bitnet-kernels` or a
`publish = false` hardware-lab crate.

### Tool, binding, and sidecar surfaces

A separate public crate is appropriate when users install or embed it through a
different delivery path, for example `bitnet-st2gguf`, `bitnet-ffi`, `bitnet-py`,
or `bitnet-wasm`.

### Test and development support

Test fixtures, BDD grids, testing policy crates, and local support libraries
should default to `publish = false` unless another repository genuinely consumes
them as an external dependency.

## Boundary memo requirement

A new `Cargo.toml` is forbidden unless the proposed boundary has an approved
boundary memo. The memo must answer:

- crate name,
- owner,
- audience,
- public API,
- semver promise,
- standalone README/docs story,
- who consumes it outside the owner crate,
- what invariant it owns,
- why a module family is not enough, and
- what dependency closure it forces public.

For forced package boundaries, the memo must name the force explicitly: binary,
proc macro, build target, FFI/binding packaging, feature isolation, publication,
or another concrete cargo/build-system constraint.

## Survival tests

A crate may remain public when it passes at least one strong test:

- an outsider would choose it directly,
- multiple surviving public packages need it,
- it is a stable contract surface,
- it is a hardware/backend surface with standalone value, or
- it is a binding or tool users install directly.

A crate should collapse when any of these are true:

- it has a single obvious owner,
- its name describes an implementation layer,
- it has no standalone README/docs story,
- it has no independent semver meaning,
- it only exists to keep files small, or
- it is only used by one public crate.

Suffixes such as `-core`, `-contract-core`, `-policy-core`, `-state-core`,
`-snapshot-core`, and `-diagnostics-core` are warning signs. They often mean
"architecture seam" rather than "public package".

## Module-family discipline

Collapsing a crate must preserve architecture. A module family is not a junk
drawer. Each internal SRP module family must have:

- one owner crate,
- one responsibility,
- one public or `pub(crate)` facade,
- private internals,
- seam-focused tests,
- clear dependency direction, and
- no sibling deep imports.

A preferred shape is:

```text
crates/bitnet-inference/src/generation/
  mod.rs          # facade
  events.rs
  stop.rs
  policy.rs
  tests.rs
```

The facade exposes the seam and hides internals:

```rust
pub(crate) mod stop;
pub mod events;

pub use events::{GenerationEvent, GenerationStats};
pub(crate) use stop::{StopDecision, StopPolicy};
```

Other modules should import through the facade, not through paths such as:

```rust
crate::generation::stop::internal_detail
```

## Collapse waves

Use the following default migration waves unless a boundary memo justifies a
different target.

| Wave | Current pattern | Target owner | Target module family |
| --- | --- | --- | --- |
| Server adapters | `bitnet-client-ip-core`, `bitnet-api-versioning-core`, `bitnet-api-key-auth-core`, `bitnet-server-health-types-core`, `bitnet-endpoint-registry-core`, `bitnet-request-context-core`, `bitnet-request-router-core` | `bitnet-server` | `auth`, `routing`, `request_context`, `health`, `versioning`, `endpoint_registry` |
| CLI adapters | `bitnet-cli-config-core`, `bitnet-cli-sampling-core`, `bitnet-build-info-core` | `bitnet-cli` | `config`, `sampling`, `build_info`, `commands` |
| Tokenizer internals | `bitnet-token-merge-core`, `bitnet-tokenizer-model-core`, `bitnet-tokenizer-discovery-core`, `bitnet-tokenizer-text-core` | `bitnet-tokenizers` | `merge`, `model_detection`, `discovery`, `text`, `authority` |
| Inference/session/generation | `bitnet-generation-events-core`, `bitnet-generation-stop-core`, `bitnet-kv-cache-policy-core`, `bitnet-engine-state-core`, `bitnet-session-config-core`, `bitnet-engine-core`, `bitnet-repl-core`, `bitnet-inference-metrics-core` | `bitnet-inference` | `generation`, `session`, `kv_cache`, `engine`, `metrics` |
| Models/artifacts | `bitnet-compat-core`, `bitnet-weight-name-core`, `bitnet-layer-index-core`, `bitnet-download-core`, `bitnet-download`, `bitnet-atomic-file-core`, `bitnet-minimal-json-core`, `bitnet-safetensors-ln` | `bitnet-models` or approved `bitnet-artifacts` | `artifacts`, `compatibility`, `download`, `gguf`, `safetensors`, `naming`, `layer_index` |
| Kernels | `bitnet-cpu-activations`, `bitnet-qk256-layout-core`, `bitnet-dispatch-core`, `bitnet-vulkan-shaders`, `bitnet-wgpu-shaders-i2s`, `bitnet-qk256-dispatch` | `bitnet-kernels` unless explicitly public | `cpu`, `cuda`, `qk256`, `dense`, `dispatch`, `shaders` |
| Receipts/contracts | `bitnet-receipts-core`, `bitnet-bench-receipts`, `bitnet-bench-regression-core`, `bitnet-feature-contract`, `bitnet-runtime-profile-contract` | `bitnet-receipts` unless separately justified | `schemas`, `validators`, `benchmark`, `feature_contracts`, `explain` |

## Enforcement plan

1. Document the policy and target surfaces.
2. Add crate-boundary inventory files under `ci/crate-boundaries/`.
3. Add `xtask crate-boundaries check` in advisory mode.
4. Make obvious collapses in waves, beginning with server and CLI adapter
   internals.
5. Promote the check to blocking once the inventory is stable.

The check should eventually enforce:

- no new public crate without a boundary memo,
- `publish = false` for internal and dev-only crates,
- README/description/license/docs metadata for public crates,
- migration notes for publishable crates depending on collapse candidates, and
- clean package/doc dry-runs for the surviving public set.

## Done state

The boundary cleanup is done when:

- every public crate has a standalone user story,
- internal SRP seams have module-family facades,
- no implementation-layer crate is public by accident,
- published crates do not depend on unpublished runtime path crates,
- `workspace.default-members` reflects normal developer workflow,
- `xtask` enforces this policy,
- docs explain which crate to use for which job, and
- `cargo package`, `cargo publish --dry-run`, and `cargo doc --no-deps` pass for
  every public crate.

# Crate Boundary Policy

BitNet-rs is a monorepo with multiple legitimate public surfaces, but every
architecture seam does not need to be a Cargo package. This policy keeps the
monorepo and the single-responsibility design discipline while reducing
accidental public API, release choreography, and dependency blast radius.

The doctrine is:

> Design seams like microcrates, implement most seams as module families, and
> publish only the boundaries we are willing to support as real contracts.

## Goals

- Keep the monorepo as the container for runtime, kernels, model contracts,
  receipts, CLI, server, bindings, hardware lanes, tests, docs, campaign
  tracking, and historical evidence.
- Separate public package surface from internal module families, dev/test/lab
  tooling, and archived evidence.
- Make every surviving public crate explain its audience, invariant, API, and
  semver promise.
- Prevent new `Cargo.toml` files from appearing merely because a design seam is
  useful or because a file grew large.
- Preserve clean architecture inside owner crates with crate-grade module
  discipline.

## Boundary categories

### Public package

A public package is a Cargo crate that users may choose directly, read about as
an installable or embeddable unit, and reasonably expect to have a stable API or
schema boundary.

A public package must have:

- a standalone user story;
- an owner;
- a documented audience;
- a stable public API or explicit pre-stable contract;
- a semver promise appropriate for its maturity;
- a README or equivalent docs entry;
- package metadata suitable for `cargo package` and docs generation;
- a dependency closure we are willing to expose.

Public packages are the exception, not the default.

### Internal module family

An internal module family is an SRP seam implemented inside one owner crate. It
may be as carefully designed as a microcrate, but its support boundary is the
owner crate rather than a separate package.

A module family must have:

- one owner crate;
- one responsibility;
- one `pub` or `pub(crate)` facade;
- private internals;
- seam-focused tests;
- clear dependency direction;
- no sibling deep imports around the facade.

A good shape is:

```text
crates/bitnet-inference/src/generation/
  mod.rs
  events.rs
  stop.rs
  policy.rs
  tests.rs
```

The facade owns what other modules may touch:

```rust
pub(crate) mod stop;
pub mod events;

pub use events::{GenerationEvent, GenerationStats};
pub(crate) use stop::{StopDecision, StopPolicy};
```

Other modules should depend on `crate::generation` rather than reaching through
`crate::generation::stop::internal_detail`.

### Dev-only crate

A dev-only crate supports local development, tests, fuzzing, fixtures,
benchmarks, policy snapshots, generated checks, or CI lanes. It may remain a
workspace crate when a package boundary materially improves build times,
isolation, or test ergonomics, but it is not a product surface.

Default rule: dev-only crates use `publish = false` unless another repository
has a documented consumer and a boundary memo approves publishing.

### Hardware lab crate

A hardware lab crate explores or validates a backend, shader catalog, dispatch
strategy, device profile, or architecture-specific optimization. It becomes a
public backend crate only after it has a real external user story and stable
claim boundary.

Until then, prefer modules under `bitnet-kernels`, `bitnet-device-probe`, or a
clearly marked lab crate with `publish = false`.

### Evidence and history

Evidence, campaign tracking, migration notes, generated dashboards, and archived
implementation records document what happened. They are not API surfaces and
must not drive package boundaries by themselves.

### Forced package boundary

A forced package boundary is allowed when a module family cannot satisfy a hard
constraint without a crate. Examples include:

- a different crate type, such as `cdylib`, `staticlib`, or `proc-macro`;
- a separate binary or installed tool;
- platform-specific packaging that cannot be expressed cleanly with features;
- dependency isolation for heavyweight optional toolchains;
- fuzz, benchmark, or integration-test harnesses that need separate manifests;
- published bindings with a distinct installation path.

Forced package boundaries still need a boundary memo.

## New crate rule

New `Cargo.toml` files are forbidden unless the boundary memo passes review.

The memo must answer:

```text
crate name
owner
audience
public API or intentionally private API
semver promise or publish=false rationale
standalone README/docs plan
who consumes it outside the owner crate
what invariant it owns
why a module family is not enough
what dependency closure it forces public
whether it is public, dev-only, lab, evidence, or forced-boundary
```

If the crate is public, the memo must also identify at least one strong reason
it cannot be an internal module family.

## Survival tests for public crates

A crate can remain public if it passes at least one strong test:

- an outsider would choose it directly;
- multiple surviving public packages need it as a shared stable contract;
- it is a stable schema, receipt, model, tokenizer, or runtime contract surface;
- it is a hardware/backend surface with standalone value;
- it is a binding or tool users install directly;
- it has a distinct packaging or crate-type requirement.

A crate should collapse when most of these are true:

- it has a single obvious owner;
- its name reads like an implementation layer, policy, state object, or `*-core`
  seam;
- it has no standalone README story;
- it has no independent semver meaning;
- it exists mainly to keep files small;
- it is used only by one public crate;
- publishing it would expose dependencies we do not want to support.

## Default target surfaces

The expected public set should be small enough that every crate can be reviewed,
packaged, documented, and supported. A healthy target is roughly 20-35 public
crates, not every workspace member.

### Product facades

```text
bitnet
bitnet-cli
bitnet-server
```

### Core SDK surfaces

```text
bitnet-inference
bitnet-models
bitnet-tokenizers
bitnet-sampling
bitnet-receipts
bitnet-kernels
```

### Backend, binding, and tool surfaces

These are public only when their boundary memo proves standalone value:

```text
bitnet-device-probe
bitnet-nvidia
bitnet-wgpu
bitnet-metal
bitnet-opencl
bitnet-rocm
bitnet-vulkan
bitnet-st2gguf
bitnet-ffi
bitnet-py
bitnet-wasm
```

### Maybe-public surfaces

Decide case by case:

```text
bitnet-gguf
bitnet-artifacts
bitnet-quantization
bitnet-quantization-bits
bitnet-bench
bitnet-test-support
```

### Likely internal seams

Most crates with these suffixes should become module families unless a memo says
otherwise:

```text
*-core
*-contract-core
*-policy-core
*-state-core
*-snapshot-core
*-diagnostics-core
```

## Module-family discipline

Collapsing a crate boundary is not permission to create a junk drawer. The
owner crate must preserve clean architecture in its module graph.

Required practices:

- expose only the seam facade from `mod.rs`;
- keep implementation modules private or `pub(crate)`;
- colocate tests with the seam when practical;
- avoid sibling modules importing private leaf modules from each other;
- keep dependencies pointing inward to domain contracts and outward through
  ports/adapters;
- document any public re-export from the owner crate as part of that owner
  crate's API.

## Collapse waves

Use small migration waves. Each wave should preserve behavior, move one family
of seams into its owner crate, and leave compatibility shims only when necessary.

Suggested order:

1. server adapter internals into `bitnet-server`;
2. CLI config, sampling, and build-info internals into `bitnet-cli`;
3. tokenizer merge, model, discovery, and text internals into
   `bitnet-tokenizers`;
4. generation, session, engine, KV-cache, REPL, and metrics internals into
   `bitnet-inference`;
5. receipt implementation shards into `bitnet-receipts`;
6. QK256 layout, dispatch, and shader catalog internals into `bitnet-kernels`,
   except seams deliberately kept public.

## Enforcement roadmap

Start advisory, then make the policy blocking once the inventory is complete.

Advisory checks should report:

- new workspace crates without a boundary memo;
- internal/dev/lab crates missing `publish = false`;
- public crates missing README, description, license, docs, or package metadata;
- public crates depending on unpublished path-only runtime crates without a
  migration note;
- collapse candidates still exposed as public package surfaces;
- `workspace.default-members` drifting away from normal developer workflow.

Blocking checks should require:

- a boundary memo for every new `Cargo.toml`;
- `publish = false` for internal and dev-only crates;
- complete package metadata for public crates;
- a migration note for each public crate that temporarily depends on a collapse
  candidate.

## Review checklist

Use this checklist for PRs that add, remove, publish, or collapse crates:

- [ ] Does this change preserve the monorepo as the development container?
- [ ] Is the seam public, internal, dev-only, lab, evidence, or forced-boundary?
- [ ] If public, would an outside user choose this crate directly?
- [ ] If internal, does it have an owner crate and facade module?
- [ ] Does any new `Cargo.toml` have an approved boundary memo?
- [ ] Are internal/dev/lab crates marked `publish = false`?
- [ ] Are public package metadata and docs complete?
- [ ] Does the dependency graph avoid accidental public exposure?
- [ ] Does the migration keep tests and receipts stable?

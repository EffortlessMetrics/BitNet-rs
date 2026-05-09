# Crate Boundary Policy

BitNet-rs is a monorepo with multiple public surfaces, not a loose collection of publishable microcrates. Design seams may be small and SRP-focused, but only a subset of seams should become Cargo package boundaries. The default implementation shape for an internal seam is a module family inside an owning crate.

This policy turns the doctrine into working rules:

> Design seams like microcrates, implement most as module families, and publish only the seams we are willing to support as real contracts.

## Scope

This policy applies to every workspace package in the repository and every new `Cargo.toml` proposed under the repo. It governs whether a seam is a public package, an internal module family, a dev-only crate, a hardware lab crate, evidence/history, or a forced package boundary.

The monorepo remains the container for runtime code, kernels, model contracts, receipts, CLI, server, bindings, hardware lanes, tests, docs, campaign tracking, and historical evidence. The policy only narrows which seams become public package surface.

## Core terms

### Public package

A public package is a crate whose package boundary is part of the supported product or SDK surface. It has a standalone user story, a documented API, a semver promise, package metadata, and a path to `cargo package` / `cargo publish --dry-run` success.

Public packages are appropriate for:

- product facades users install or run directly;
- SDK surfaces programmatic users choose directly;
- hardware/backend surfaces with independent external value;
- bindings and sidecar tools with distinct installation paths;
- stable schema or contract surfaces that other public crates and external users rely on.

### Internal module family

An internal module family is an SRP seam implemented inside one owner crate. It has a local facade (`mod.rs` or equivalent), private internals, seam-focused tests, and explicit dependency direction, but it does not have its own package identity.

Module families are the default for implementation layers, single-owner seams, and code splits that exist primarily to keep files small or agent-editable.

### Dev-only crate

A dev-only crate is a workspace crate for tests, fixtures, fuzzing, benchmarks, policy harnesses, migration tooling, or CI support. It should normally use `publish = false` and must not become a dependency of a publishable package unless the boundary memo explains why that is safe.

### Hardware lab crate

A hardware lab crate is a workspace package used to explore, benchmark, or validate a backend before it has a stable standalone public story. Hardware lab crates should default to `publish = false` until they have a clear feature surface, documented hardware target, stable receipt/claim boundary, and clean packaging dry run.

### Evidence/history

Evidence/history is material retained to explain past decisions, validation results, campaigns, migration notes, and receipts. It belongs in docs, tracking, archive, benchmarks, or generated evidence locations, not in newly published crates.

### Forced package boundary

A forced package boundary is a crate that remains separate because Cargo or distribution mechanics require it, even if the design seam would otherwise be a module family. Examples include proc macros, build helpers that must compile independently, FFI/sys crates with native build semantics, fuzz targets, workspace tools, generated bindings, or packages with incompatible feature/dependency constraints.

Forced package boundaries still need a memo. The memo can cite the mechanic instead of an external user story.

## Boundary rule

New `Cargo.toml` files are forbidden unless a boundary memo passes review.

A change that adds or preserves a crate as public must include or update a boundary memo that answers the questions in this policy. A change that adds an internal/dev/lab package must explicitly mark why it cannot be a module family and whether it must be `publish = false`.

## Boundary memo template

Use this template in the PR description or in a colocated design document when adding a new crate, making a crate public, or deciding that an existing crate survives a collapse wave.

```text
crate name:
owner:
classification: public package | internal module family | dev-only crate | hardware lab crate | forced package boundary | evidence/history
intended audience:
public API or local facade:
semver promise:
standalone README/docs:
external consumers outside the owner crate/repo:
invariant owned by this boundary:
why a module family is not enough:
forced package mechanic, if any:
dependency closure made public by this boundary:
packaging/doc dry-run status:
migration/collapse note, if applicable:
```

## Survival tests for public crates

A crate may remain public when at least one strong test is true:

- an outsider would choose it directly by name;
- multiple surviving public packages need it as a stable shared contract;
- it owns a stable schema, receipt, API, or hardware claim boundary;
- it is a hardware/backend package with real standalone value;
- it is a binding, converter, CLI, server, or tool users install directly;
- Cargo mechanics force a separate package and the dependency closure is acceptable.

A crate should collapse when one or more of these are true and no survival test applies:

- it has a single obvious owner crate;
- the name reads like an implementation layer (`*-core`, `*-policy-core`, `*-state-core`, `*-snapshot-core`, `*-diagnostics-core`);
- it has no standalone README story or docs entry;
- it has no independent semver meaning;
- it exists mainly to keep files small;
- it is only consumed by one public crate;
- publishing it would expose accidental dependencies or release choreography.

## Module-family discipline

Collapsing a crate must preserve architecture. A collapsed seam must not become a junk drawer.

Each internal module family must have:

- one owner crate;
- one responsibility;
- one public or `pub(crate)` facade;
- private internals behind that facade;
- seam-focused tests;
- clear dependency direction;
- no sibling deep imports that bypass the facade.

Preferred shape:

```text
crates/bitnet-inference/src/generation/
  mod.rs
  events.rs
  stop.rs
  policy.rs
  tests.rs
```

Facade example:

```rust
pub(crate) mod stop;
pub mod events;

pub use events::{GenerationEvent, GenerationStats};
pub(crate) use stop::{StopDecision, StopPolicy};
```

Other modules should depend on `crate::generation` exports, not on private paths like `crate::generation::stop::internal_detail`.

## Public package quality bar

Every public package should have:

- complete package metadata: `description`, `license`, `repository`, `readme` where applicable, and docs metadata;
- a README or docs page that explains who should use it;
- an explicit feature story and default-feature policy;
- a documented semver/API contract;
- a clean or intentionally documented dependency closure;
- no dependency on unpublished runtime path crates unless the release plan explains how packaging works;
- `cargo package --list -p <crate>` coverage before publication;
- `cargo publish --dry-run -p <crate>` before actual publication;
- `cargo doc -p <crate> --no-deps` for public API review.

## Internal and dev-only package rules

Internal/dev/lab crates that remain as workspace packages should:

- set `publish = false` unless the boundary memo says otherwise;
- document their owner crate or owning subsystem;
- avoid being dependencies of publishable crates;
- have a migration note if they are collapse candidates;
- avoid public README language that implies external support.

## Promotion and demotion workflow

### Promote a module family to a public crate

1. Write the boundary memo.
2. Identify the audience, invariant, and semver promise.
3. Add package metadata and standalone docs.
4. Confirm no accidental dependency closure is exposed.
5. Run package/doc dry-runs.
6. Update the repo surface inventory.

### Demote a public/internal crate to a module family

1. Pick the owner crate.
2. Move code behind a local facade that preserves the seam.
3. Rewrite callers to use the facade.
4. Remove package dependencies and workspace membership when safe.
5. Add compatibility shims only when an existing public API requires them.
6. Update the collapse inventory and migration notes.

## Review checklist

Reviewers should ask:

- Who is the user of this crate boundary?
- What invariant does the boundary own?
- What breaks if this is a module family instead?
- Does the crate force internal dependencies into public packaging?
- Does it have a README/docs story independent of the owner crate?
- Does semver for this crate mean anything by itself?
- Is this a Cargo/distribution constraint rather than an architectural preference?
- If internal, is `publish = false` set or explicitly justified?

## Enforcement roadmap

Initial enforcement is documentation-first and advisory. The intended follow-up is an `xtask crate-boundaries check` command that can enforce:

- no new public crate without a boundary memo;
- `publish = false` for internal/dev/lab crates;
- public crates have README, description, license, repository, and docs metadata;
- collapse candidates are not dependencies of publishable crates without a migration note;
- repo surface inventory files stay in sync with the workspace manifest.

Once the inventory is stable, CI can make the check blocking.

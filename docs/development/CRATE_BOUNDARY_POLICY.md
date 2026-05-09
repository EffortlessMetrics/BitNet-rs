# Crate Boundary Policy

BitNet-rs is a monorepo with multiple supported products, SDK surfaces,
backends, bindings, tests, docs, and historical evidence. The repository should
stay broad, but the public Cargo package surface must stay intentional.

The guiding rule is:

> Design seams like microcrates, implement most seams as module families, and
> publish only the seams we are willing to support as real contracts.

This policy exists to prevent accidental public surface area. A small file or a
single-responsibility seam is not, by itself, a reason to create or preserve a
public crate.

## Boundary classes

Every Rust boundary in this repository must fit one of these classes.

| Class | Meaning | Typical location | Default publish state |
| --- | --- | --- | --- |
| Public package | A crate users install, import, document against, or expect semver stability from. | `crates/<name>` or root package | Publishable when packaging checks pass. |
| Internal module family | A crate-grade design seam implemented inside an owner crate. | `crates/<owner>/src/<family>` | Not a Cargo package. |
| Dev-only crate | Test, fixture, benchmark, CI, or repository workflow support. | `crates/`, `tests/`, `tools/`, `xtask/` | `publish = false`. |
| Hardware lab crate | Experimental backend or hardware investigation code without a stable external story. | `crates/bitnet-*-lab`, backend submodules, or opt-in crates | `publish = false` until promoted. |
| Evidence/history | Reports, receipts, validation records, migration notes, and campaign tracking. | `docs/reports`, `docs/`, `ci/` | Not a package boundary. |
| Forced package boundary | A separate crate required by Rust, Cargo, FFI, proc macro, wasm, Python, or packaging constraints. | Case by case | Decided by boundary memo. |

## Non-negotiable rule for new crates

New `Cargo.toml` files are forbidden unless the boundary memo passes review.

A new package boundary is allowed only when it is stronger than an internal
module family. The memo must be committed with, or before, the new package.

## Boundary memo template

Use this template in the PR description or in a checked-in design note when a
crate is added, preserved as public, or promoted from internal to public.

```text
crate name:
owner:
boundary class:
audience:
public API:
semver promise:
standalone README/docs:
external consumers outside owner crate:
invariant owned by this boundary:
why a module family is not enough:
public dependency closure forced by this crate:
packaging command and result:
migration or collapse plan if temporary:
```

## Tests for a surviving public crate

A crate should survive as a public package only if it passes at least one strong
test:

- An outsider would knowingly choose it directly.
- Multiple surviving public packages need it as a stable contract.
- It owns a versioned schema, receipt, model contract, or other external
  compatibility promise.
- It is a hardware/backend surface with a documented target, feature surface,
  and standalone value.
- It is a binding or tool users install directly.
- A Cargo, Rust, FFI, wasm, Python, or build-system constraint forces a package
  boundary.

## Collapse signals

A crate should become an internal module family when most of these are true:

- It has one obvious owner crate.
- Its name reads as an implementation layer, such as `*-core`, `*-policy-core`,
  `*-state-core`, `*-snapshot-core`, `*-diagnostics-core`, or
  `*-contract-core`.
- It has no standalone README story.
- It has no independent semver meaning.
- It exists mainly to keep files small.
- It is consumed only by one public crate or one product adapter.
- Publishing it would force internal dependencies into the public graph.

## Module-family discipline

Collapsing a crate must not create a junk drawer. Internal module families keep
crate-grade design discipline:

- One owner crate.
- One responsibility.
- One public or `pub(crate)` facade.
- Private internals by default.
- Seam-focused tests.
- Clear dependency direction.
- No sibling deep imports.

A good shape is:

```text
crates/bitnet-inference/src/generation/
  mod.rs
  events.rs
  stop.rs
  policy.rs
  tests.rs
```

The facade controls what other modules can use:

```rust
pub(crate) mod stop;
pub mod events;

pub use events::{GenerationEvent, GenerationStats};
pub(crate) use stop::{StopDecision, StopPolicy};
```

Code outside the family should depend on `crate::generation` exports rather
than `crate::generation::stop::internal_detail`.

## Public package requirements

Every public package should have:

- A clear audience and install/import story.
- `description`, `license`, `readme`, and documentation metadata in
  `Cargo.toml`.
- A standalone README or equivalent docs section.
- A defined semver surface.
- Feature flags that describe user-facing capabilities instead of internal
  implementation accidents.
- A clean packaging dry run before publication.
- No dependency on unpublished runtime path crates unless the dependency is
  explicitly documented as temporary and guarded by migration notes.

Recommended checks for publishable crates:

```bash
cargo package --list -p <crate>
cargo publish --dry-run -p <crate>
cargo doc -p <crate> --no-deps
```

## Dev-only and internal package requirements

A workspace crate that remains separate for tests, tooling, or temporary
migration must make that status explicit:

```toml
publish = false
```

Dev-only crates should not be part of the public dependency closure of a
publishable crate. If a publishable crate temporarily depends on a dev-only or
collapse-candidate crate, document the migration note in the crate-boundary
inventory.

## Promotion path

A module family or dev-only crate can be promoted later. Promotion requires:

1. A boundary memo.
2. A standalone user story.
3. API documentation.
4. A packaging dry run.
5. Agreement on semver ownership.
6. An explicit decision about whether existing internal names remain private or
   become public API.

## Migration posture

The cleanup should be incremental and low drama:

1. Document the doctrine.
2. Inventory public surfaces, collapse candidates, and dev-only crates.
3. Add advisory `xtask crate-boundaries check` enforcement.
4. Collapse low-risk owner-local crates first.
5. Promote only the seams that earn a public contract.
6. Make boundary checks blocking after the inventory is trustworthy.

The goal is not fewer architectural seams. The goal is fewer accidental package
contracts.

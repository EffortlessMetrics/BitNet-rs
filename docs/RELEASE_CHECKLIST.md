# v0.2.0 Release Checklist

Pre-release gates for the v0.2.0 release. Every item must be checked before
tagging.

## Build & Test Gates

- [ ] All GPU microcrates compile: `cargo check --workspace`
- [ ] CPU tests pass: `cargo nextest run --workspace --no-default-features --features cpu`
- [ ] GPU feature compilation: `cargo check --no-default-features --features oneapi`
- [ ] Clippy clean: `cargo clippy --all-targets --no-default-features --features cpu -- -D warnings`
- [ ] Format check: `cargo fmt --all --check`
- [ ] Documentation builds: `cargo doc --workspace --no-deps`

## Release Artifacts

- [ ] CHANGELOG.md updated with all changes since v0.1.0
- [ ] Version bumped in workspace `Cargo.toml`
- [ ] README.md reflects new features (GPU support, multi-backend)
- [ ] COMPATIBILITY.md updated with new GPU backends and platform support
- [ ] Migration guide written (`docs/MIGRATION_GUIDE_v0.2.0.md`)
- [ ] Release notes finalized (`docs/v0.2.0_RELEASE_NOTES.md`)

## CI & Quality

- [ ] CI green on all platforms (Linux, macOS, Windows)
- [ ] Benchmark baselines recorded in `ci/inference.json`
- [ ] Fuzz targets run for 5+ minutes each (15 targets)
- [ ] License headers present on all new files
- [ ] THIRD_PARTY.md updated for new dependencies

## Publish

- [ ] Git tag `v0.2.0` created and pushed

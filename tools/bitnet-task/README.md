# bitnet-task

`bitnet-task` is the compatibility facade for migrated `scripts/*.sh` entrypoints.

Control-plane boundary:

- `xtask` owns internal developer workflows, CI orchestration, and new maintenance commands.
- `bitnet-task` owns only the command surface needed to preserve existing shell entrypoints while those scripts remain supported.
- `test-generation` intentionally narrows the old shell behavior by delegating to the existing `bitnet-models` integration smoke instead of compiling an ad hoc scratch binary.

Wrapper contract:

```bash
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec cargo run --quiet --locked --manifest-path "$ROOT/Cargo.toml" -p bitnet-task -- <subcommand> "$@"
```

Validation during the stabilization pass:

```bash
cargo check -p bitnet-task
cargo fmt --all --check
cargo test -p bitnet-task
scripts/tests/bitnet-task-wrapper-help-smoke.sh
```

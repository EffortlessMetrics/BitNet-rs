# CUDA-UX-005: Legacy Benchmark CUDA Fail-Closed

Status: implemented

## Scope

`bitnet benchmark` and its `bench` alias are legacy benchmarking surfaces. They
simulate benchmark work and do not emit governed CUDA receipts. Before this
item, `--device cuda` and `--device gpu` were accepted and silently mapped to
CPU, which could be mistaken for accelerator evidence.

## Change

The legacy benchmark command now accepts only `cpu` and `auto`. Accelerator
device labels fail closed with an error that points users to receipt-backed CUDA
ask/chat paths and governed benchmark receipts.

This preserves the current claim boundary:

- no `bitnet bench --device cuda` runtime benchmark claim;
- no dense CUDA speedup claim;
- no full CUDA residency claim;
- no BitNet packed I2S/QK256 proof from dense CUDA evidence;
- no tokenizer, loader, transformer, QK256, CUDA kernel, or server behavior
  changes.

## Validation

- `cargo test --locked -p bitnet-cli --test cli_arg_tests --no-default-features --features cpu,full-cli bench_ -- --nocapture`
- `cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli`
- `cargo fmt -p bitnet-cli -- --check`
- `cargo run --release --locked -p xtask --no-default-features -- campaign check nvidia-5070ti`
- `cargo run --release --locked -p xtask --no-default-features -- campaign generate --check`
- `git diff --check`

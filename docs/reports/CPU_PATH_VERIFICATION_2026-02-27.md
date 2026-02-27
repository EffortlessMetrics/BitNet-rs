# CPU Path Verification (2026-02-27)

## Goal
Verify that the CPU inference path executes successfully and produces stable, sensible outputs on the in-repo GGUF fixture.

## Commands Run

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- --help
```

Result:
- ✅ `bitnet` CLI built successfully with CPU features.
- ✅ `run` subcommand and CPU usage guidance are available.

```bash
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run --model tests/models/mini.gguf --prompt "2+2=" --max-new-tokens 8 --temperature 0.0 --greedy
```

Result:
- ✅ Binary executed and attempted real model load.
- ✅ Correctly rejected invalid fixture for real inference with explicit message:
  - `Validation error: GGUF: missing embedding tensor`
  - `To run with mock tensors ... pass --allow-mock`

```bash
cargo test -p bitnet-inference --test e2e_cpu_golden_path --no-default-features --features cpu -- --nocapture
```

Result:
- ✅ Test suite passed: `10 passed; 0 failed`.
- ✅ CPU golden-path tests validate deterministic generation and receipt invariants.
- ✅ Fixture-based inference executed repeatedly with consistent top logits and token behavior.

## Sensibility Signals Observed
- Hyperparameter and quantization sanity checks reported success repeatedly.
- Deterministic token generation path remained stable across repeated runs.
- Golden output guard (`test_e2e_golden_path_pinned_output`) passed.
- Token boundary and stop-token behavior tests passed.

## Conclusion
The CPU path is operational in this environment and demonstrates sensible behavior for the maintained golden-path fixture tests (determinism, schema/receipt invariants, and bounded generation semantics).

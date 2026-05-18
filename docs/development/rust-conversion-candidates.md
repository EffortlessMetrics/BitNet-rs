# Rust conversion candidates

This inventory captures non-Rust maintenance and validation logic that can be
converted into Rust and moved behind the repository's core control planes. It is
intended to keep the project biased toward Rust-native, typed, testable flows
while preserving compatibility shims only where they are the product surface.

## Placement rule

Move repository-internal automation into `xtask` by default. Move reusable model,
GGUF, tokenizer, quantization, or receipt logic into the owning workspace crate
first, then expose it through `xtask` or a CLI wrapper. Keep thin shell entrypoints
only when they preserve an existing user-facing command or CI contract.

## Current non-Rust surface

The audit used `rg --files` to avoid walking `target/` and found these relevant
non-Rust implementation surfaces:

| Surface | Count | Rust migration posture |
| --- | ---: | --- |
| Python files (`*.py`) | 61 | Migrate repository-internal generation, validation, parity, and report logic; keep Python binding examples/tests as product-surface coverage. |
| Shell files (`*.sh`) | 146 | Migrate policy, gates, validation, model setup, and release orchestration to `xtask`; leave tiny compatibility wrappers temporarily. |
| C/C++/CUDA/HIP/Metal/OpenCL/WGSL/GLSL shader sources | 57 | Do not blanket-migrate. Keep GPU shader languages and ABI examples; migrate host-side C/C++ shims only after Rust parity/ABI coverage exists. |
| PowerShell files (`*.ps1`) | 14 | Treat as Windows compatibility wrappers after equivalent `xtask` commands exist. |
| JavaScript/TypeScript files (`*.js`, `*.ts`) | 8 | Keep wasm/browser smoke helpers unless they start carrying core validation logic. |

## Highest-value conversion candidates

### 1. GGUF fixture generation and inspection

**Candidates:**

- `scripts/generate_gguf_fixtures.py`
- `scripts/inspect_gguf.py`
- `scripts/test_gguf_debug.py`
- `scripts/validate_gguf.sh`
- `scripts/validate-fixtures.sh`

**Move to:** `bitnet-gguf`, `bitnet-models`, and `xtask` subcommands.

**Why:** The Python fixture writer hand-encodes GGUF headers and metadata. That
logic belongs beside the Rust GGUF reader/writer contracts so tests, fixture
builders, and debugging tools share one schema implementation. `xtask` already
has `gen-fixtures` and `gen-mini-gguf` command slots, so the migration path is to
turn the Python scripts into Rust library fixtures plus `xtask` wrappers rather
than another standalone script.

**Acceptance target:** `cargo run --locked -p xtask --no-default-features -- gen-fixtures ...`
and `gen-mini-gguf` can fully replace the Python fixture/debug scripts, with
golden fixtures covered by Rust tests.

### 2. SafeTensors-to-GGUF validation pipeline

**Candidates:**

- `scripts/convert_safetensors_to_gguf.py`
- `scripts/convert_safetensors_to_gguf_validated.py`
- `scripts/fix_gguf_tokenizer.py`
- `scripts/export_clean_gguf.sh`
- `scripts/quantize_i2s_clean.sh`

**Move to:** `bitnet-st2gguf`, `bitnet-models`, `bitnet-tokenizers`, and `xtask`.

**Why:** Conversion correctness depends on tokenizer parity, metadata fidelity,
LayerNorm/F16 preservation, and quantization format detection. Those are core
model-loader contracts, not script-local concerns. The validated converter should
be decomposed into Rust library checks and a single `xtask`/CLI flow that emits a
machine-readable receipt.

**Acceptance target:** one Rust command performs conversion, tokenizer parity,
metadata validation, and receipt emission without shelling through Python for
normal CI/developer use.

### 3. Trace-diff and parity diagnostics

**Candidates:**

- `scripts/trace_diff.py`
- `scripts/compare_traces.py`
- `scripts/replay_parity.py`
- `scripts/check_greedy_argmax.py`
- `scripts/test-tokenizer-parity.py`
- `scripts/logit-parity.sh`
- `scripts/nll-parity.sh`
- `scripts/prop-greedy-parity.sh`
- `scripts/run_crossval_sweep.sh`

**Move to:** `crossval`, `bitnet-receipts`, and `xtask/src/crossval/`.

**Why:** First-divergence detection, token/logit/NLL parity, and deterministic
cross-validation are part of the inference quality contract. Keeping those
algorithms in Rust makes them share receipt schemas, error types, tokenizer
adapters, and CI feature gates. `xtask` already contains `trace-diff`,
`crossval`, `crossval-per-token`, and `parity-both` entrypoints, so the desired
shape is a Rust crossval library plus thin commands.

**Acceptance target:** shell/Python parity scripts become compatibility wrappers
or are deleted after Rust commands produce equivalent JSON receipts and concise
human diagnostics.

### 4. Property-based cross-validation metrics

**Candidates:**

- `crossval/props/metrics.py`
- `crossval/props/strategies.py`
- `crossval/props/run_model.py`
- `crossval/props/test_greedy_invariants.py`
- `crossval/props/test_greedy_parity.py`
- `crossval/props/test_logit_parity.py`
- `crossval/props/test_nll_parity.py`

**Move to:** Rust `proptest` tests in `crossval` and core metric helpers in a
small Rust module that can be reused by receipts.

**Why:** The Python suite includes real algorithms such as edit distance,
prefix/suffix matching, n-gram scoring, Kendall tau, prompt strategies, and
runner orchestration. These are valuable oracles and should participate in the
same Rust test runner, feature gates, deterministic environment handling, and
receipt model as the rest of the project.

**Acceptance target:** Rust `proptest` coverage exercises the same prompt spaces
and metrics, while any remaining Python is limited to external Python package
compatibility checks.

### 5. Policy, guard, and hygiene scripts

**Candidates:**

- `scripts/check-feature-gates.sh`
- `scripts/check-ignore-annotations.sh`
- `scripts/check-ignore-hygiene.sh`
- `scripts/check-serial-annotations.sh`
- `scripts/check-units.sh`
- `scripts/check-units-imports.sh`
- `scripts/check-patch-policy.sh`
- `scripts/check-codeowners-teams.sh`
- `scripts/json-schema-gate.sh`
- `scripts/security-audit.sh`
- `ci/scripts/grep-guards.sh`

**Move to:** `xtask/src/policy/`, `xtask/src/gates.rs`, and dedicated xtask
subcommands.

**Why:** These scripts encode repository policy. Shell/ripgrep implementations
are easy to drift from Cargo feature resolution, test-target metadata, and Rust
AST-level rules. Rust commands can provide structured diagnostics, test fixtures,
and stable exit codes while still using `rg`-style scanning internally where
appropriate.

**Acceptance target:** `make guards` and CI call Rust-native `xtask` checks for
policy decisions; any shell left under `scripts/` only dispatches to `xtask`.

### 6. Benchmark, performance, and receipt reporting

**Candidates:**

- `scripts/benchmark_comparison.py`
- `scripts/compare_performance.py`
- `scripts/detect-performance-regression.py`
- `scripts/generate_performance_report.py`
- `scripts/render_perf_md.py`
- `scripts/measure_perf_json.sh`
- `scripts/perf-gate.sh`
- `scripts/run-performance-benchmarks.sh`
- `scripts/update-baseline.sh`

**Move to:** `xtask`, `bitnet-receipts`, and benchmark crates.

**Why:** Performance claims and regression thresholds should be computed from the
same typed receipt schema that CI validates. Markdown rendering can stay a leaf
output, but threshold comparison, baseline loading, hardware labels, and schema
versions should be Rust-native.

**Acceptance target:** `xtask benchmark`, `xtask compare-metrics`, and receipt
verification own the JSON schema and regression decision; scripts no longer parse
or mutate benchmark JSON directly.

### 7. Model download/setup orchestration

**Candidates:**

- `scripts/prepare_test_model.sh`
- `scripts/fetch-pr-model.sh`
- `scripts/setup_model_storage.sh`
- `scripts/test_download.sh`
- `ci/fetch_bitnet_cpp.sh`
- `ci/use-bitnet-cpp-cache.sh`

**Move to:** `bitnet-download`, `xtask download-model`, `xtask fetch-models`, and
`xtask fetch-cpp`/`setup-cpp-auto`.

**Why:** Downloads require retry, offline mode, hashing, cache locking, and clear
exit codes. Those guarantees already align with Rust crates and `xtask`; shell
should not own correctness-critical download or cache behavior.

**Acceptance target:** CI and local docs use Rust commands for model and C++
reference setup; shell wrappers are compatibility-only.

## Surfaces to keep outside Rust for now

- **Python bindings and examples** under `crates/bitnet-py/` should remain Python
  because they verify the Python-facing API and migration story.
- **C ABI examples and headers** (`examples/ffi_simple.c`,
  `examples/c_compatibility_demo.c`, `crates/bitnet-ffi/include/`) should remain
  as compatibility proof for C consumers.
- **Vendored/reference C or C++ quantization code** should stay until Rust kernels
  have documented parity, because it is useful as a reference oracle.
- **GPU shader sources** (`*.cu`, `*.hip`, `*.metal`, `*.cl`, `*.wgsl`, `*.comp`)
  are backend-native kernel languages. The Rust migration target is host-side
  orchestration and shared kernel metadata, not replacing shader code blindly.

## Recommended migration order

1. Finish GGUF fixture/inspection migration because it is self-contained and
   shrinks Python from core test data generation.
2. Move policy/guard scripts to `xtask` because they reduce CI drift and have
   deterministic text/metadata inputs.
3. Consolidate trace-diff and parity diagnostics in Rust receipts.
4. Port crossval property metrics to Rust `proptest` after the diagnostic schema
   is stable.
5. Fold benchmark/report logic into typed receipt handling.
6. Replace model/setup shell wrappers last, once CI call sites are already using
   the Rust commands.

## Definition of done for each migration

A candidate is properly moved over when:

- the core algorithm lives in an owning Rust crate or `xtask` module, not in a
  standalone script;
- the command has Rust tests or golden fixtures;
- JSON output has an explicit schema/version when consumed by CI;
- docs and CI call the Rust command directly;
- the old script is removed or reduced to a documented compatibility wrapper;
- feature gates match the crate-level optional dependencies used by the command.

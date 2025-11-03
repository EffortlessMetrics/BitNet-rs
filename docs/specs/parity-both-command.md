# Parity-Both Command Specification

**Status**: Implementation Ready (Updated 2025-10-26)
**Version**: 1.1 (Enhanced with Infrastructure Analysis)
**Feature**: Dual-lane cross-validation with unified receipts
**Priority**: High (Developer Experience)
**Scope**: `xtask/src/main.rs`, `xtask/src/crossval/`, `crossval/src/receipt.rs`

---

## 1. Executive Summary

The `parity-both` command orchestrates dual-lane cross-validation by running **both** Rust and C++ backends (BitNet.cpp + llama.cpp) in a single invocation, automatically handling preflight checks, executing inference with shared Rust evaluation, and generating unified receipts with comparative metrics.

### 1.1 Problem Statement

**Current workflow** requires manual orchestration:
```bash
# Lane A: BitNet.cpp
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf --tokenizer tokenizer.json --cpp-backend bitnet \
  --receipt receipt_bitnet.json

# Lane B: llama.cpp
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf --tokenizer tokenizer.json --cpp-backend llama \
  --receipt receipt_llama.json

# Manual comparison
diff receipt_bitnet.json receipt_llama.json
```

**Gaps**:
- **Manual backend selection**: User must remember which backends to test
- **No auto-repair**: Missing backends require separate preflight + setup commands
- **Sequential execution**: Rust inference runs twice (once per backend)
- **Manual comparison**: User must interpret two separate receipts
- **Error-prone**: Easy to forget one backend or use different prompts

### 1.2 Desired Behavior

**New workflow** - "Two comparisons are one command away":
```bash
# Single command runs both backends, auto-repairs if needed
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --out-dir ci/parity

# Output:
# ✓ Preflight: BitNet.cpp available
# ✓ Preflight: llama.cpp available
# ⚙ Rust inference (shared): 4 tokens
# ⚙ C++ inference (BitNet): 4 tokens
# ⚙ C++ inference (llama): 4 tokens
# ✓ Lane A (BitNet.cpp): Parity OK (cos_sim=0.9999)
# ✓ Lane B (llama.cpp): Parity OK (cos_sim=0.9998)
# ✓ Receipts written: ci/parity/receipt_bitnet.json, ci/parity/receipt_llama.json
```

### 1.3 Key Goals

1. **Auto-repair by default**: Missing backends automatically installed and configured
2. **Dual-lane execution**: Both backends validated in single invocation
3. **Unified receipts**: Structured JSON output with comparative metrics
4. **Clear summary**: Human-readable status for each backend
5. **Exit code semantics**: 0=both pass, 1=either fails, 2=usage error

---

## 2. Command-Line Interface

### 2.1 Command Signature

```bash
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf <PATH> \
  --tokenizer <PATH> \
  [OPTIONS]
```

### 2.2 Required Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--model-gguf` | path | Path to GGUF model file | `models/model.gguf` |
| `--tokenizer` | path | Path to tokenizer.json file | `models/tokenizer.json` |

### 2.3 Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | string | "What is 2+2?" | Input prompt for inference |
| `--max-tokens` | usize | 4 | Maximum tokens to generate (excluding prompt) |
| `--cos-tol` | f32 | 0.999 | Cosine similarity threshold (0.0-1.0) |
| `--format` | enum | "text" | Output format: text or json |
| `--prompt-template` | enum | "auto" | Template: auto, raw, instruct, llama3-chat |
| `--system-prompt` | string | (none) | System prompt for chat templates |
| `--out-dir` | path | "ci" | Output directory for receipts |
| `--no-repair` | flag | false | Disable auto-repair of missing backends |
| `--verbose` | flag | false | Show detailed progress for each lane |
| `--dump-ids` | flag | false | Dump Rust token IDs to stderr |
| `--dump-cpp-ids` | flag | false | Dump C++ token IDs to stderr |
| `--metrics` | string | "mse" | Metrics to compute: mse,kl,topk |

### 2.4 Output Files

**Naming convention**:
```
{out_dir}/receipt_bitnet.json   # Lane A: BitNet.cpp backend
{out_dir}/receipt_llama.json    # Lane B: llama.cpp backend
```

### 2.5 Example Invocations

#### Minimal (uses defaults)
```bash
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json
```

#### Full options
```bash
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "Explain photosynthesis in one sentence" \
  --max-tokens 32 \
  --cos-tol 0.995 \
  --format json \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant" \
  --out-dir ci/parity \
  --verbose \
  --metrics mse,kl,topk
```

#### Disable auto-repair (manual setup)
```bash
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json \
  --no-repair
```

---

## 3. Workflow Architecture

### 3.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Preflight Both Backends (auto-repair by default)   │
├─────────────────────────────────────────────────────────────┤
│ • Check BitNet.cpp availability (HAS_BITNET constant)      │
│ • Check llama.cpp availability (HAS_LLAMA constant)        │
│ • If --no-repair: fail immediately if missing              │
│ • If auto-repair: setup-cpp-auto + rebuild xtask + retry   │
│ • Exit code 2 if either backend unavailable after repair   │
└─────────────────────────────────────────────────────────────┘
    ↓ (both backends available)
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Shared Setup (~40ms)                               │
├─────────────────────────────────────────────────────────────┤
│ • Template processing (auto-detect or explicit)            │
│ • Rust tokenization (once, reused for comparison)          │
│ • C++ tokenization for both backends                       │
│ • Token parity validation (fail-fast if mismatch)          │
└─────────────────────────────────────────────────────────────┘
    ↓ (token parity OK)
┌──────────────────────┬──────────────────────────────────────┐
│ STEP 3: Shared Rust Logits (once)                          │
│ • Load GGUF model                                           │
│ • Evaluate logits for all positions                        │
│ • Return logits matrix (reused for both lanes)             │
│ (~10-30s depending on model size and token count)          │
└─────────────────────────────────────────────────────────────┘
    ↓ (shared Rust logits ready)
┌──────────────────────┬──────────────────────────────────────┐
│ LANE A: BitNet.cpp   │ LANE B: llama.cpp                    │
├──────────────────────┼──────────────────────────────────────┤
│ STEP 4A: C++ Logits  │ STEP 4B: C++ Logits                  │
│ • BitnetSession ctx  │ • BitnetSession ctx (llama backend)  │
│ • C++ eval           │ • C++ eval                           │
│ (~10-30s)            │ (~10-30s)                            │
├──────────────────────┼──────────────────────────────────────┤
│ STEP 5A: Compare     │ STEP 5B: Compare                     │
│ • Per-position MSE   │ • Per-position MSE                   │
│ • Cosine similarity  │ • Cosine similarity                  │
│ • L2 distance        │ • L2 distance                        │
│ • Divergence detect  │ • Divergence detect                  │
│ (~100ms)             │ (~100ms)                             │
├──────────────────────┼──────────────────────────────────────┤
│ STEP 6A: Receipt     │ STEP 6B: Receipt                     │
│ • receipt_bitnet.json│ • receipt_llama.json                 │
│ (~50ms)              │ (~50ms)                              │
└──────────────────────┴──────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Unified Summary                                     │
├─────────────────────────────────────────────────────────────┤
│ Lane A (BitNet.cpp): ✓ Parity OK (cos_sim=0.9999)          │
│ Lane B (llama.cpp):  ✓ Parity OK (cos_sim=0.9998)          │
│                                                             │
│ Receipts written:                                           │
│   - ci/parity/receipt_bitnet.json                           │
│   - ci/parity/receipt_llama.json                            │
│                                                             │
│ Exit code: 0 (both lanes passed)                            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Optimization: Shared Rust Inference

**Current design** (sequential with shared Rust):
- Shared: Rust inference (~20s, once)
- Lane A: C++ BitNet (~15s)
- Lane B: C++ llama (~15s)
- **Total**: 50s (30% faster than sequential dual)

**Future parallel design** (with `--parallel`):
- Shared: Rust inference (~20s, once)
- Parallel: max(C++ BitNet ~15s, C++ llama ~15s) = ~15s
- **Total**: 35s (50% faster, ~2× speedup vs sequential dual)

### 3.3 Preflight Auto-Repair Flow

```
┌─────────────────────────────────────────┐
│ preflight_both_backends()               │
├─────────────────────────────────────────┤
│ 1. Check BitNet.cpp (HAS_BITNET)        │
│    ├─ Available → Continue              │
│    ├─ Missing + --no-repair → Fail      │
│    └─ Missing + auto-repair:            │
│        ├─ Run setup-cpp-auto            │
│        ├─ Rebuild xtask                 │
│        └─ Retry preflight               │
│ 2. Check llama.cpp (HAS_LLAMA)          │
│    └─ (Same logic)                      │
│ 3. Return Ok(()) if both available      │
└─────────────────────────────────────────┘
```

**Implementation Status**: Scaffolding in place (`xtask/src/crossval/parity_both.rs:340-467`), requires integration.

---

## 4. Receipt Schema

### 4.1 Schema v1.0.0 (Current)

**Location**: `crossval/src/receipt.rs`

```json
{
  "version": 1,
  "timestamp": "2025-10-26T14:30:00Z",
  "model": "models/model.gguf",
  "backend": "bitnet",
  "prompt": "What is 2+2?",
  "positions": 4,
  "thresholds": {
    "mse": 0.0001,
    "kl": 0.1,
    "topk": 0.8
  },
  "rows": [
    {
      "pos": 0,
      "mse": 1.23e-6,
      "max_abs": 0.0042,
      "kl": null,
      "topk_agree": null,
      "top5_rust": [128000, 1229, 374, 220, 17],
      "top5_cpp": [128000, 1229, 374, 220, 17]
    }
  ],
  "summary": {
    "all_passed": true,
    "first_divergence": null,
    "mean_mse": 2.15e-5,
    "mean_kl": null
  }
}
```

### 4.2 Dual-Lane Receipt Naming

**File naming convention**:
- `{out_dir}/receipt_bitnet.json` - Lane A (BitNet.cpp backend)
- `{out_dir}/receipt_llama.json` - Lane B (llama.cpp backend)

**Contract**: Both receipts use schema v1 with `backend` field indicating source:
- Lane A: `"backend": "bitnet.cpp"` or `"backend": "bitnet"`
- Lane B: `"backend": "llama.cpp"` or `"backend": "llama"`

**Backward compatibility**: Existing receipt verification tools work unchanged.

---

## 5. Comparison Logic

### 5.1 Per-Lane Metrics

Each lane computes identical metrics as `crossval-per-token`:

**Position-level metrics** (from `crossval/src/logits_compare.rs`):
- **Cosine similarity**: `dot(a,b) / (||a|| * ||b||)` — Range [0.0, 1.0]
- **L2 distance**: `sqrt(Σ(rust[i] - cpp[i])²)` — Euclidean distance
- **MSE**: `(L2 distance)²` — Mean squared error
- **Max absolute difference**: `max(|rust[i] - cpp[i]|)` — Peak divergence

**Optional metrics** (if `--metrics` includes):
- **KL divergence**: `Σ P(i) log(P(i)/Q(i))` on softmax distributions
- **Top-K agreement**: `|top_k(rust) ∩ top_k(cpp)| / k`

### 5.2 Pass/Fail Criteria

**Per-lane status**:
- **Pass**: `cosine_sim >= cos_tol` for all positions AND `mse <= threshold.mse`
- **Fail**: `cosine_sim < cos_tol` OR `mse > threshold.mse` at any position

**Divergence detection threshold** (from `crossval/src/logits_compare.rs:212`):
```rust
pub const COSINE_SIMILARITY_THRESHOLD: f32 = 1e-4; // (1.0 - cos_sim) > 1e-4
```

**Unified status** (exit code logic):
- **Exit 0**: Both Lane A and Lane B pass
- **Exit 1**: Either Lane A or Lane B fails
- **Exit 2**: Usage error (token mismatch, invalid args, backend unavailable)

### 5.3 Summary Output Format

#### Text Format (Default)

```
Parity-Both Cross-Validation Summary
═════════════════════════════════════════════════════════

Lane A: BitNet.cpp
──────────────────────────────────────────────────────────
Backend:          bitnet.cpp
Status:           ✓ Parity OK
First divergence: None
Mean MSE:         2.15e-5
Mean cosine sim:  0.99995
Receipt:          ci/parity/receipt_bitnet.json

Lane B: llama.cpp
──────────────────────────────────────────────────────────
Backend:          llama.cpp
Status:           ✓ Parity OK
First divergence: None
Mean MSE:         1.98e-5
Mean cosine sim:  0.99996
Receipt:          ci/parity/receipt_llama.json

Overall Status
──────────────────────────────────────────────────────────
Both lanes:       ✓ PASSED
Exit code:        0
```

#### JSON Format (`--format json`)

```json
{
  "status": "ok",
  "lanes": {
    "bitnet": {
      "backend": "bitnet",
      "status": "ok",
      "first_divergence": null,
      "mean_mse": 2.15e-5,
      "mean_cosine_sim": 0.99995,
      "receipt_path": "ci/parity/receipt_bitnet.json"
    },
    "llama": {
      "backend": "llama",
      "status": "ok",
      "first_divergence": null,
      "mean_mse": 1.98e-5,
      "mean_cosine_sim": 0.99996,
      "receipt_path": "ci/parity/receipt_llama.json"
    }
  },
  "overall": {
    "both_passed": true,
    "exit_code": 0
  }
}
```

---

## 6. Error Handling

### 6.1 Error Scenarios

| Scenario | Exit Code | Behavior |
|----------|-----------|----------|
| Backend missing + `--no-repair` | 2 | Fail with setup instructions |
| Backend missing + auto-repair failure | 2 | Fail with diagnostic output |
| Token parity mismatch | 2 | Fail-fast before logits comparison |
| Lane A parity fail, Lane B pass | 1 | Complete both lanes, report divergence |
| Lane A pass, Lane B parity fail | 1 | Complete both lanes, report divergence |
| Both lanes fail | 1 | Complete both lanes, report both divergences |
| Invalid arguments | 2 | Show usage error |

### 6.2 Partial Failure Handling

**Design principle**: Complete all lanes even if one fails.

**Example**: Lane A fails, Lane B succeeds
```
Lane A: BitNet.cpp
──────────────────────────────────────────────────────────
Status:           ✗ Parity FAILED
First divergence: Position 2 (cos_sim=0.9985 < threshold 0.999)
Receipt:          ci/parity/receipt_bitnet.json

Lane B: llama.cpp
──────────────────────────────────────────────────────────
Status:           ✓ Parity OK
Receipt:          ci/parity/receipt_llama.json

Overall Status
──────────────────────────────────────────────────────────
Both lanes:       ✗ FAILED (1 of 2 lanes failed)
Exit code:        1
```

**Rationale**: Users get complete diagnostic data for both backends, enabling root cause analysis.

---

## 7. Implementation Gaps & Status

### 7.1 Current State (MVP)

**✅ Implemented** (from `xtask/src/crossval/parity_both.rs:1-698`):
- Command scaffolding with `ParityBothArgs` struct
- `run_dual_lanes_and_summarize()` function with dual-lane orchestration
- `run_single_lane()` helper for per-backend evaluation
- `LaneResult` struct and summary functions
- `print_unified_summary()` with text and JSON formats
- Exit code logic (`determine_exit_code`, `both_passed`, `overall_status`)
- Unit tests for exit code logic

**❌ Gaps Requiring Implementation**:

#### Gap 1: Command Integration in `xtask/src/main.rs`
**Current**: `run()` function is scaffolding only (lines 75-120)
```rust
pub fn run(_args: &ParityBothArgs) -> Result<()> {
    // TODO: Uncomment run_dual_lanes_and_summarize call
    Ok(())  // Currently no-op
}
```

**What's Needed**:
1. Uncomment `run_dual_lanes_and_summarize()` call
2. Wire `ParityBothArgs` fields to function parameters
3. Add command variant to `Commands` enum in main.rs
4. Add handler in main command dispatch

**Effort**: 2-3 hours (Low complexity)

#### Gap 2: Auto-Repair Integration
**Current**: `preflight_backend_libs()` checks availability, but no auto-repair
**Location**: `xtask/src/crossval/preflight.rs`

**What's Needed**:
1. Extract auto-repair logic from manual workflow
2. Implement `auto_repair_backend(backend, verbose)` function
3. Implement `preflight_both_backends(auto_repair, verbose)` wrapper
4. Handle xtask rebuild and retry logic

**Effort**: 4-6 hours (Medium complexity - process management)

#### Gap 3: Parallel Lane Execution
**Current**: Sequential C++ evaluation in `run_single_lane()` loops
**Location**: `xtask/src/crossval/parity_both.rs:423-452`

**What's Needed**:
1. Add `--parallel` flag handling
2. Use `rayon::join()` or `std::thread` for parallel evaluation
3. Handle cross-thread error reporting
4. Test for FFI thread safety

**Effort**: 3-4 hours (Medium complexity - thread safety)

#### Gap 4: Help Text and CLI Registration
**Current**: Command not registered in main.rs `Commands` enum
**Location**: `xtask/src/main.rs`

**What's Needed**:
1. Add `ParityBoth` variant to `Commands` enum with full clap annotations
2. Add handler in main match statement
3. Update `--help` output validation tests

**Effort**: 1-2 hours (Low complexity)

### 7.2 Implementation Phases

| Phase | Tasks | Estimated Hours | Dependencies |
|-------|-------|----------------|--------------|
| **Phase 1** | Command integration (Gap 1, Gap 4) | 3-5 | None |
| **Phase 2** | Auto-repair (Gap 2) | 4-6 | Phase 1 |
| **Phase 3** | Parallel execution (Gap 3) | 3-4 | Phase 1, Phase 2 |
| **Testing** | Integration tests with real models | 3-4 | All phases |
| **Documentation** | Update CLAUDE.md, add examples | 1-2 | All phases |
| **Total** | | **14-21 hours** | (2-3 days) |

---

## 8. Data Flow & Architecture

### 8.1 Comparison Algorithm (from `crossval/src/logits_compare.rs`)

```
FOR each token position i in [0, min(rust_logits.len(), cpp_logits.len())):
    rust_vec = rust_logits[i]  (logit values for position i)
    cpp_vec = cpp_logits[i]    (logit values for position i)

    IF rust_vec.len() != cpp_vec.len():
        cosine_sim[i] = 0.0
        l2_dist[i] = INFINITY
        first_divergence = i
        CONTINUE

    // Cosine similarity: dot(a,b) / (||a|| * ||b||)
    cosine_sim[i] = cosine_similarity(rust_vec, cpp_vec)
    l2_dist[i] = sqrt(sum((rust_vec[j] - cpp_vec[j])^2))

    // Track maximum absolute difference
    max_abs[i] = max(|rust_vec[j] - cpp_vec[j]| for all j)

    // Divergence detection: (1.0 - cosine_sim) > threshold
    IF (1.0 - cosine_sim[i]) > COSINE_SIMILARITY_THRESHOLD AND first_divergence.is_none():
        first_divergence = i
```

### 8.2 Receipt Population Flow

```
New Receipt
    ↓ ParityReceipt::new(model, backend, prompt)
    ├─ version = 1
    ├─ timestamp = Utc::now()
    ├─ thresholds = Thresholds::default() (mse: 1e-4, kl: 0.1, topk: 0.8)
    ├─ rows = [] (empty, to be populated)
    └─ summary = { all_passed: true, first_divergence: None, mean_mse: 0.0 }
    ↓
Population Phase (add_position for each token)
    ↓ receipt.add_position(PositionMetrics { pos, mse, max_abs, ... })
    ↓ (repeat for each position)
    ↓
Finalization Phase
    ↓ receipt.finalize()
    ├─ positions = rows.len()
    ├─ summary.mean_mse = avg(all MSE values)
    ├─ summary.first_divergence = first position where mse > threshold.mse
    └─ summary.all_passed = first_divergence.is_none()
    ↓
Output
    ↓ receipt.write_to_file(path)?  // Writes JSON to disk
    └─ receipt ready for consumption by CI/diagnostics
```

### 8.3 Backend Selection (from `xtask/src/crossval/backend.rs`)

**CppBackend Enum**:
```rust
pub enum CppBackend {
    BitNet,  // microsoft/bitnet-b1.58-2B-4T-gguf, libbitnet*.so
    Llama,   // llama-3, llama-2, SmolLM3, libllama*.so + libggml*.so
}
```

**Auto-Detection Heuristics** (`from_model_path`):
1. **Tier 1 (BitNet)**: Path contains "bitnet" OR "microsoft/bitnet" → `CppBackend::BitNet`
2. **Tier 2 (Llama)**: Path contains "llama" → `CppBackend::Llama`
3. **Tier 3 (Default)**: No pattern found → `CppBackend::Llama` (conservative)

**Build-Time Detection** (from `crossval/build.rs`):
```rust
pub const HAS_BITNET: bool = const_str_eq(option_env!("CROSSVAL_HAS_BITNET"), "true");
pub const HAS_LLAMA: bool = const_str_eq(option_env!("CROSSVAL_HAS_LLAMA"), "true");
pub const BACKEND_STATE: &str = option_env!("CROSSVAL_BACKEND_STATE").unwrap_or("none");
```

**States**: `"full"` (both), `"llama"` (llama only), `"none"` (neither)

---

## 9. Acceptance Criteria

### AC1: Single Command Execution
**Requirement**: Single command runs both backends without user intervention

**Test**:
```bash
cargo run -p xtask --features crossval-all -- parity-both \
  --model-gguf models/model.gguf \
  --tokenizer models/tokenizer.json
```

**Pass criteria**:
- Command completes without requiring additional user input
- Both Lane A (BitNet.cpp) and Lane B (llama.cpp) evaluated
- Exit code 0 if both pass

**Implementation Status**: ✅ Scaffolding in place (lines 324-467), requires CLI integration

### AC2: Dual Receipt Naming
**Requirement**: Two receipt files created with clear naming (llama vs bitnet)

**Test**: Check output directory after execution

**Pass criteria**:
- `receipt_bitnet.json` exists with `"backend": "bitnet"` or `"backend": "bitnet.cpp"`
- `receipt_llama.json` exists with `"backend": "llama"` or `"backend": "llama.cpp"`
- Both use ParityReceipt schema v1.0.0

**Implementation Status**: ✅ Implemented (lines 416-418, 569-599)

### AC3: Token Comparison with Thresholds
**Requirement**: Token comparison with configurable thresholds (cosine similarity, L2 distance, MSE)

**Test**: Parse receipt JSON for metrics

**Pass criteria**:
- Each position has: `mse`, `max_abs`, `kl` (optional), `topk_agree` (optional)
- Summary has: `mean_mse`, `first_divergence`, `all_passed`
- Thresholds applied: `mse <= 0.0001`, `kl <= 0.1`, `topk >= 0.8`

**Implementation Status**: ✅ Implemented (lines 563-598)

### AC4: Exit Code Semantics
**Requirement**: Exit code 0=pass, 1=comparison fail, 2=error

**Test matrix**:

| Lane A | Lane B | Expected Exit Code |
|--------|--------|-------------------|
| Pass   | Pass   | 0 |
| Pass   | Fail   | 1 |
| Fail   | Pass   | 1 |
| Fail   | Fail   | 1 |

**Pass criteria**: All combinations return correct exit codes

**Implementation Status**: ✅ Implemented and tested (lines 264-277, tests 637-696)

### AC5: First Divergence Reporting
**Requirement**: Report first divergence with index and token triple (id/rust/cpp)

**Test**: Parse verbose output or receipt for divergence info

**Pass criteria**:
- Summary shows: `"first_divergence": <position>` or `null`
- Verbose mode shows: position, cosine_sim, mse, status symbol
- Receipt rows have: `pos`, `mse`, `max_abs` for all positions

**Implementation Status**: ✅ Implemented (lines 547-559, receipt rows 587-596)

### AC6: Auto-Repair Default
**Requirement**: Auto-repair enabled by default (can disable with `--no-repair`)

**Test**:
```bash
# Default: auto-repair ON
BITNET_CPP_DIR="" cargo run -p xtask -- parity-both \
  --model-gguf models/model.gguf --tokenizer models/tokenizer.json

# Explicit: auto-repair OFF
BITNET_CPP_DIR="" cargo run -p xtask -- parity-both \
  --model-gguf models/model.gguf --tokenizer models/tokenizer.json \
  --no-repair
```

**Pass criteria**:
- Default invocation attempts auto-repair (calls `setup-cpp-auto`, rebuilds xtask)
- `--no-repair` disables auto-repair and fails with setup instructions

**Implementation Status**: ❌ Requires implementation (Gap 2)

### AC7: CLI Integration with Help Text
**Requirement**: CLI integration with xtask main and help text

**Test**:
```bash
cargo run -p xtask --features crossval-all -- --help | grep parity-both
cargo run -p xtask --features crossval-all -- parity-both --help
```

**Pass criteria**:
- `parity-both` appears in main help
- `parity-both --help` shows all flags with descriptions
- Command dispatches correctly

**Implementation Status**: ❌ Requires registration (Gap 4)

---

## 10. Testing Strategy

### 10.1 Unit Tests

**File**: `xtask/src/crossval/parity_both.rs:620-697`

**Implemented**:
- ✅ `test_determine_exit_code_both_pass`
- ✅ `test_determine_exit_code_lane_a_fail`
- ✅ `test_determine_exit_code_lane_b_fail`
- ✅ `test_determine_exit_code_both_fail`
- ✅ `test_both_passed`
- ✅ `test_overall_status`

**Pending**:
- ❌ Token parity validation tests
- ❌ Receipt naming convention tests
- ❌ Metrics computation tests (MSE, cosine similarity)
- ❌ Auto-repair flow tests (mocked `setup-cpp-auto`)

### 10.2 Integration Tests

**Manual verification checklist**:

1. **Both backends available** (happy path):
   ```bash
   cargo run -p xtask --features crossval-all -- parity-both \
     --model-gguf models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
   # Expected: Exit 0, two receipts written, both pass
   ```

2. **Partial failure** (one lane fails):
   ```bash
   cargo run -p xtask --features crossval-all -- parity-both \
     --model-gguf models/model.gguf \
     --tokenizer models/tokenizer.json \
     --cos-tol 0.9999
   # Expected: Exit 1, both receipts written, clear divergence report
   ```

3. **JSON output**:
   ```bash
   cargo run -p xtask --features crossval-all -- parity-both \
     --model-gguf models/model.gguf \
     --tokenizer models/tokenizer.json \
     --format json | jq .
   # Expected: Valid JSON with lanes.bitnet and lanes.llama
   ```

4. **Verbose mode**:
   ```bash
   cargo run -p xtask --features crossval-all -- parity-both \
     --model-gguf models/model.gguf \
     --tokenizer models/tokenizer.json \
     --verbose
   # Expected: Detailed progress for preflight, shared setup, both lanes
   ```

---

## 11. Dependencies

### 11.1 Module Dependencies

```
xtask/src/crossval/parity_both.rs
├─ Depends on: xtask/src/crossval/backend.rs (CppBackend enum)
├─ Depends on: xtask/src/crossval/preflight.rs (preflight_backend_libs)
├─ Depends on: crossval/src/receipt.rs (ParityReceipt, Thresholds, PositionMetrics)
├─ Depends on: crossval/src/cpp_bindings.rs (BitnetSession API)
├─ Depends on: crossval/src/logits_compare.rs (compare_per_position_logits)
├─ Depends on: bitnet-tokenizers (load_tokenizer, encode)
└─ Depends on: bitnet-inference (eval_logits_all_positions)
```

### 11.2 Feature Gate Requirements

**Required features** for `parity-both`:
- `crossval-all` (unified: `crossval + ffi + inference`)
- OR combination: `crossval + ffi + inference`

**Build command**:
```bash
cargo build -p xtask --features crossval-all
```

---

## 12. Related Documentation

- `docs/howto/cpp-setup.md` - C++ reference setup guide
- `docs/explanation/dual-backend-crossval.md` - Dual-backend architecture
- `docs/specs/preflight-ux-parity.md` - Preflight enhancement spec
- `CLAUDE.md` - Project conventions and workflows (section: Cross-Validation CLI Reference)
- `/tmp/crossval_infrastructure_analysis.md` - Existing infrastructure reference
- `/tmp/backend_detection_analysis.md` - Backend detection logic

---

## 13. Summary

This specification defines the `parity-both` command, enabling dual-lane cross-validation with auto-repair by default, unified receipts, and clear summary output. The implementation follows BitNet.rs architectural patterns and integrates seamlessly with existing `crossval-per-token` infrastructure.

**Key Benefits**:
- **Developer experience**: "Two comparisons are one command away"
- **Auto-repair**: Missing backends automatically installed (planned)
- **Comprehensive coverage**: Both BitNet.cpp and llama.cpp validated
- **CI-ready**: Structured receipts with quality gates

**Current Status**:
- ✅ **70% complete**: Core orchestration and summary logic implemented
- ❌ **30% remaining**: CLI integration, auto-repair, parallel execution

**Estimated remaining effort**: 14-21 hours (2-3 days)

**Next steps**:
1. ✅ Specification reviewed and enhanced with infrastructure analysis
2. ❌ Implement Phase 1 (command CLI integration - Gap 1, Gap 4)
3. ❌ Implement Phase 2 (auto-repair - Gap 2)
4. ❌ Implement Phase 3 (parallel execution - Gap 3)
5. ❌ Add integration tests with real models
6. ❌ Update CLAUDE.md with usage examples

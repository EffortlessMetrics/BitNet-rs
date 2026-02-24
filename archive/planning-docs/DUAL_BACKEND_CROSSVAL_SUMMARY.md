# Dual-Backend Cross-Validation Documentation Summary

This document summarizes the comprehensive documentation updates for BitNet.rs dual-backend cross-validation system (G4).

## Documentation Structure

The dual-backend cross-validation system is now fully documented across three key files:

### 1. **CLAUDE.md** (CLI Quick Reference)
**Location**: `/home/steven/code/Rust/BitNet-rs/CLAUDE.md`

**Updated Sections**:
- ✅ Cross-Validation CLI Reference (lines 597-770)
  - Complete `crossval-per-token` flag documentation
  - Complete `setup-cpp-auto` command reference
  - Complete `preflight` command reference
  - All flags documented: `--cpp-backend`, `--dump-ids`, `--dump-cpp-ids`, `--verbose`, etc.

- ✅ Troubleshooting Section (lines 772-816)
  - Backend selection guidance
  - Token mismatch diagnostics
  - Preflight check procedures
  - C++ library setup pointers

- ✅ Documentation Structure (lines 222-254)
  - Added cross-reference to `docs/howto/cpp-setup.md`
  - Added cross-reference to `docs/explanation/dual-backend-crossval.md`

**Key Examples Added**:
```bash
# BitNet model (auto-detects bitnet.cpp)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 0.999 \
  --format json \
  --dump-ids --dump-cpp-ids --verbose

# LLaMA model (auto-detects llama.cpp)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is the capital of France?" \
  --max-tokens 8 \
  --cos-tol 0.995

# Explicit backend override
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --cpp-backend llama \
  --prompt-template raw \
  --prompt "2+2=" \
  --max-tokens 1 \
  --dump-ids \
  --dump-cpp-ids \
  --verbose
```

### 2. **docs/howto/cpp-setup.md** (Setup Guide)
**Location**: `/home/steven/code/Rust/BitNet-rs/docs/howto/cpp-setup.md`

**Existing Complete Sections** (verified):
- ✅ Overview of dual-backend architecture (lines 6-30)
  - Lane A: BitNet.rs ↔ bitnet.cpp
  - Lane B: BitNet.rs ↔ llama.cpp
  - Auto-detection heuristics

- ✅ One-Command Setup (lines 42-72)
  - `setup-cpp-auto` for all platforms (bash, fish, PowerShell)
  - Shell profile integration

- ✅ Manual Per-Backend Setup (lines 74-116)
  - BitNet.cpp environment variables
  - llama.cpp environment variables
  - Platform-specific instructions (Linux, macOS, Windows)

- ✅ Verification Section (lines 118-158)
  - Library validation with `ldd` / `otool`
  - Backend preflight checks with `--verbose`

- ✅ Usage Examples (lines 160-228)
  - BitNet model validation
  - LLaMA model validation
  - Explicit backend selection
  - Full cross-validation sweep
  - Trace diffing

- ✅ Troubleshooting (lines 290-466)
  - 10+ common error scenarios with fixes
  - Library path issues
  - Backend selection issues
  - Token mismatch debugging
  - Windows-specific issues

**Key Callouts**:
- Environment variables: `BITNET_CPP_DIR`, `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`
- Preflight verification: `cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose`
- Auto-detection rules: Path contains "bitnet" → bitnet.cpp, "llama" → llama.cpp, default → llama.cpp

### 3. **docs/explanation/dual-backend-crossval.md** (Architecture Details)
**Location**: `/home/steven/code/Rust/BitNet-rs/docs/explanation/dual-backend-crossval.md`

**Existing Complete Sections** (verified):
- ✅ Overview and Motivation (lines 1-30)
  - Why dual-backend vs unified
  - Model family requirements

- ✅ Component Structure Diagram (lines 34-101)
  - ASCII art flow diagram
  - Backend selection logic
  - Parity comparison flow

- ✅ Backend Auto-Detection (lines 103-133)
  - Rust code example from implementation
  - Design rationale
  - Priority levels (explicit keyword > llama > default)

- ✅ Backend-Specific Configuration (lines 135-149)
  - Library requirements per backend
  - Tokenizer differences
  - Model constraints

- ✅ Tokenization Pipeline (lines 151-191)
  - Template application flow
  - Token parity pre-gate (fail-fast)
  - Debug flag usage

- ✅ Operational Flows (lines 193-277)
  - Flow 1: Auto-detection
  - Flow 2: Explicit override
  - Flow 3: Template override
  - Flow 4: Debug mode with token inspection

- ✅ Configuration & Environment Variables (lines 279-313)
  - Build-time vs runtime configuration
  - Library path setup

- ✅ Error Handling & Diagnostics (lines 315-341)
  - Error tables with causes and solutions
  - Backend selection errors
  - Tokenizer parity errors
  - Logits divergence

- ✅ Usage Examples (lines 343-433)
  - 4 complete examples with expected output
  - JSON and text format demonstrations

- ✅ Performance Considerations (lines 435-473)
  - Compute time estimates
  - Memory usage
  - Library caching

- ✅ Design Tradeoffs (lines 475-510)
  - Why dual-backend?
  - Why auto-detection by path?
  - Why preflight checks?

- ✅ Related Documentation & Glossary (lines 512-527)

## Flag Reference Table

Complete flag documentation for `crossval-per-token`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | path | (required) | Path to GGUF model file |
| `--tokenizer` | path | (required) | Path to tokenizer.json file |
| `--prompt` | string | (required) | Input prompt for inference |
| `--max-tokens` | integer | 4 | Maximum tokens to generate (excluding prompt) |
| `--cos-tol` | float | 0.999 | Cosine similarity threshold (0.0-1.0); below = divergence |
| `--format` | string | "text" | Output format: "text" or "json" |
| `--prompt-template` | enum | "auto" | Template type: raw, instruct, llama3-chat, auto |
| `--system-prompt` | string | (none) | System prompt for chat templates |
| `--cpp-backend` | enum | (auto) | C++ backend selection: bitnet, llama (auto-detects from path if omitted) |
| `--verbose` | flag | false | Show backend selection, preflight checks, diagnostics |
| `--dump-ids` | flag | false | Dump Rust token IDs to stderr for debugging |
| `--dump-cpp-ids` | flag | false | Dump C++ token IDs to stderr for debugging |

## Backend Auto-Detection Heuristics

Documented in all three files with consistent messaging:

1. **Priority 1**: Path contains "bitnet" or "microsoft/bitnet" → Uses bitnet.cpp
2. **Priority 2**: Path contains "llama" → Uses llama.cpp
3. **Priority 3**: Default fallback → llama.cpp (conservative)
4. **Override**: `--cpp-backend bitnet|llama` flag for explicit selection

## Common Scenarios Documentation

### Scenario 1: BitNet Model Validation (Auto-Detect)
**Where documented**: All three files
**Key command**:
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 0.999 \
  --format json
```

### Scenario 2: LLaMA Model Validation (Auto-Detect)
**Where documented**: All three files
**Key command**:
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is the capital of France?" \
  --max-tokens 8 \
  --cos-tol 0.999
```

### Scenario 3: Debug Token Mismatch
**Where documented**: CLAUDE.md (Troubleshooting), cpp-setup.md (Troubleshooting), dual-backend-crossval.md (Example 3)
**Key command**:
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Hello world" \
  --max-tokens 2 \
  --dump-ids \
  --dump-cpp-ids \
  --verbose 2>&1 | grep -E "TOKENIZE|IDs|Parity"
```

### Scenario 4: Explicit Backend Override
**Where documented**: All three files
**Key command**:
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --cpp-backend llama \
  --prompt-template raw \
  --prompt "2+2=" \
  --max-tokens 1 \
  --verbose
```

### Scenario 5: Preflight Check
**Where documented**: CLAUDE.md (preflight command), cpp-setup.md (Verification), dual-backend-crossval.md (Configuration)
**Key commands**:
```bash
# Check all backends
cargo run -p xtask --features crossval-all -- preflight

# Check specific backend with diagnostics
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
cargo run -p xtask --features crossval-all -- preflight --backend llama --verbose
```

## Troubleshooting Quick Reference

**Where to find solutions**:

| Issue | Primary Doc | Section |
|-------|-------------|---------|
| Library not found errors | `cpp-setup.md` | Troubleshooting (line 290+) |
| Backend selection wrong | `CLAUDE.md` | Troubleshooting (line 795-797) |
| Token mismatch | `CLAUDE.md` | Troubleshooting (line 798-801) |
| Preflight failures | `cpp-setup.md` | Troubleshooting (line 409-431) |
| Understanding architecture | `dual-backend-crossval.md` | Component Structure (line 34-101) |
| Setup from scratch | `cpp-setup.md` | Quick Start (line 42-72) |

## User Journey Map

### Journey 1: First-Time Setup
1. Read: `CLAUDE.md` → "Essential Commands" → One-command setup
2. Run: `eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"`
3. Verify: `cargo run -p xtask --features crossval-all -- preflight`
4. Validate: Run first `crossval-per-token` with BitNet model
5. Reference: `cpp-setup.md` for troubleshooting if needed

### Journey 2: Debugging Token Mismatch
1. Read: `CLAUDE.md` → Troubleshooting → Token mismatch diagnostics
2. Run: `crossval-per-token` with `--dump-ids --dump-cpp-ids --verbose`
3. Analyze: Compare token sequences in stderr output
4. Reference: `dual-backend-crossval.md` → Tokenization Pipeline (line 151-191)
5. Fix: Adjust `--prompt-template` or verify tokenizer file

### Journey 3: Understanding Architecture
1. Read: `dual-backend-crossval.md` → Overview (line 1-30)
2. Review: Component Structure diagram (line 34-101)
3. Understand: Backend Auto-Detection (line 103-133)
4. Apply: Run explicit backend override examples (line 423-433)

## Documentation Completeness Checklist

- ✅ All `crossval-per-token` flags documented with types, defaults, descriptions
- ✅ `setup-cpp-auto` command fully documented (all shells)
- ✅ `preflight` command fully documented (all backends)
- ✅ Backend auto-detection heuristics explained (with priority levels)
- ✅ Environment variables listed (BITNET_CPP_DIR, LD_LIBRARY_PATH, etc.)
- ✅ Both backends documented (BitNet.cpp + llama.cpp)
- ✅ Library requirements per backend listed
- ✅ Tokenization pipeline flow documented
- ✅ Error handling tables with causes and solutions
- ✅ Performance considerations (compute time, memory, caching)
- ✅ Design tradeoffs explained
- ✅ Copy-pasteable examples for common scenarios
- ✅ Troubleshooting guide with 10+ error scenarios
- ✅ Cross-references between documents
- ✅ Platform-specific instructions (Linux, macOS, Windows)
- ✅ Debug mode usage (`--verbose`, `--dump-ids`, `--dump-cpp-ids`)

## What's Not Included (Out of Scope)

This documentation update focused on dual-backend cross-validation. The following are intentionally not covered here:

- GPU-specific cross-validation (separate from dual-backend)
- Trace diffing implementation details (covered in other docs)
- C++ FFI internals (covered in `bitnet-sys` crate docs)
- Receipt verification (covered in separate docs)
- Model validation gates (covered in `validate-models.md`)

## Next Steps for Users

**To get started with dual-backend cross-validation**:

1. **Setup**: Run one-command setup from `CLAUDE.md` Essential Commands
   ```bash
   eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
   ```

2. **Verify**: Check backend availability
   ```bash
   cargo run -p xtask --features crossval-all -- preflight --verbose
   ```

3. **Validate**: Run first cross-validation
   ```bash
   cargo run -p xtask --features crossval-all -- crossval-per-token \
     --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
     --prompt "What is 2+2?" \
     --max-tokens 4
   ```

4. **Debug** (if needed): Use diagnostic flags
   ```bash
   --dump-ids --dump-cpp-ids --verbose
   ```

5. **Learn More**: Read architecture details in `docs/explanation/dual-backend-crossval.md`

## Documentation Locations Summary

**Quick Reference**: `CLAUDE.md` (lines 597-806)
- CLI flag table
- Example commands
- Troubleshooting bullets

**Setup Guide**: `docs/howto/cpp-setup.md`
- Installation steps
- Environment variables
- Verification procedures
- 10+ troubleshooting scenarios

**Architecture Deep-Dive**: `docs/explanation/dual-backend-crossval.md`
- System design
- Component diagrams
- Tokenization pipeline
- Error handling tables
- Performance considerations
- Design tradeoffs

All documentation is complete, cross-referenced, and ready for users.

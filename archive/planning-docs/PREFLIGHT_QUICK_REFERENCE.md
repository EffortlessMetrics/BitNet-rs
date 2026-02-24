# Preflight & Diagnostics Quick Reference

## Key Files at a Glance

```
xtask/src/crossval/
├── backend.rs              ← CppBackend enum, auto-detection, required_libs()
├── preflight.rs           ← preflight_backend_libs(), print_backend_status()
└── mod.rs                 ← Module exports

xtask/src/main.rs
├── Lines 435-481          ← CrossvalPerToken command definition
├── Lines 960-971          ← Handler dispatch
└── Lines 2975-3239        ← crossval_per_token_cmd() implementation
    ├── Line 2991          ← Backend auto-detection
    ├── Line 3002          ← Preflight check (✅ WIRED)
    ├── Line 3031-3039     ← Rust tokenization
    ├── Line 3042-3046     ← C++ availability check (❌ DUPLICATE)
    ├── Line 3048-3094     ← C++ tokenization
    ├── Line 3101-3119     ← Token parity validation (FAIL-FAST)
    ├── Line 3122-3124     ← Rust logits eval
    ├── Line 3131-3166     ← C++ logits eval
    └── Line 3176-3239     ← Comparison & output

crossval/src/
├── backend.rs             ← crossval::CppBackend enum (different from xtask!)
├── token_parity.rs        ← Token mismatch error formatting (excellent!)
├── cpp_bindings.rs        ← C++ FFI wrappers
└── lib.rs

crossval/build.rs
└── Lines 86-150           ← Library detection, env var emission

crossval/tests/
├── dual_backend_integration.rs  ← 7 test categories
├── smoke.rs                     ← Environment validation
└── ...

xtask/tests/
└── preflight.rs           ← GPU preflight tests (not C++ backend tests)
```

## What's Wired vs Not Wired

| Item | Status | Location |
|------|--------|----------|
| Backend auto-detection | ✅ | xtask/src/crossval/backend.rs:50-61 |
| Preflight check in crossval-per-token | ✅ | xtask/src/main.rs:3002 |
| Token parity error messages | ✅ | crossval/src/token_parity.rs:158-258 |
| Build-time lib detection | ✅ | crossval/build.rs:86-150 |
| `--dump-ids` flag | ❌ | Documented in CLAUDE.md but not in code |
| `--dump-cpp-ids` flag | ❌ | Documented in CLAUDE.md but not in code |
| Verbose library diagnostics | ⚠️ | Minimal implementation |
| `xtask preflight` command | ❌ | Function exists but not wired to command |
| `BITNET_CPP_BACKEND` env override | ❌ | Test scaffolding only |
| `BITNET_CROSSVAL_VERBOSE` env var | ❌ | Test scaffolding only |

## Implementation Checklist

### To Wire `--dump-ids` and `--dump-cpp-ids`

1. [ ] Add to `CrossvalPerToken` struct (xtask/src/main.rs:~480)
   ```rust
   #[arg(long)]
   dump_ids: bool,
   
   #[arg(long)]
   dump_cpp_ids: bool,
   ```

2. [ ] Add to handler dispatch (xtask/src/main.rs:~970)
   ```rust
   dump_ids,
   dump_cpp_ids,
   ```

3. [ ] Add to function signature (xtask/src/main.rs:2975)
   ```rust
   dump_ids: bool,
   dump_cpp_ids: bool,
   ```

4. [ ] Implement token dumping (xtask/src/main.rs:~3034)
   ```rust
   if dump_ids {
       eprintln!("Rust token IDs: {:?}", token_ids);
   }
   ```

5. [ ] Implement C++ token dumping (xtask/src/main.rs:~3095)
   ```rust
   if dump_cpp_ids {
       eprintln!("C++ token IDs: {:?}", cpp_tokens);
   }
   ```

### To Create `xtask preflight` Command

1. [ ] Add `Preflight` variant to `Cmd` enum (xtask/src/main.rs)
   ```rust
   /// Show C++ backend availability status
   Preflight,
   ```

2. [ ] Add handler in match expression
   ```rust
   Cmd::Preflight => {
       crate::crossval::preflight::print_backend_status();
       Ok(())
   }
   ```

3. [ ] Done! Function already exists at `xtask/src/crossval/preflight.rs:85-121`

### To Consolidate Availability Checks

1. [ ] Check `preflight_backend_libs()` result (line 3002) instead of separate check
2. [ ] Remove/consolidate `bitnet_sys::is_available()` call (line 3042)
3. [ ] Test that preflight result determines whether to proceed

## Test Running

```bash
# Backend auto-detection tests (always run, no deps)
cargo test -p crossval test_backend_autodetect

# Preflight tests (always run, no deps)
cargo test -p crossval test_preflight_env_var_reporting

# Error handling tests (always run, no deps)
cargo test -p crossval test_backend_error_when_unavailable

# All non-ignored tests
cargo test -p crossval --no-default-features --features ffi

# Include ignored tests (requires C++ libs and model)
cargo test -p crossval -- --ignored
```

## Error Output When Libraries Missing

When preflight finds missing libraries, it shows:
- Backend name in error header
- Setup command for that backend
- Required libraries
- Actionable next steps

Example:
```
Backend 'bitnet.cpp' selected but required libraries not found.

Setup instructions:
1. Install C++ reference implementation:
   eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"
...
```

## Token Parity Errors

When token sequences don't match:
- Shows Rust and C++ token sequences
- Marks first difference with exact token values
- Backend-specific troubleshooting hints
- Example fix command (copy-pasteable)
- Exit code: 2

## Build-Time Detection

Environment variables emitted by `crossval/build.rs`:
- `CROSSVAL_HAS_BITNET=true|false` - BitNet libs found
- `CROSSVAL_HAS_LLAMA=true|false` - LLaMA libs found
- Checked at compile time, used at runtime via `option_env!()`

## Feature Flags

In `crossval/Cargo.toml`:
- `crossval` - Full C++ integration (implies `ffi`)
- `ffi` - FFI wrappers only
- No default features

In `xtask/Cargo.toml`:
- `inference` - Requires `bitnet-inference`, `bitnet-crossval`, etc.
- `ffi` - Requires `bitnet-sys`
- `crossval-all` - All three above

## Two Different CppBackend Types!

⚠️ **Important**: There are TWO `CppBackend` enums:

1. **xtask's**: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/backend.rs`
   - Used in CLI parsing and dispatch
   - Implements `ValueEnum` for clap
   - Methods: `from_model_path()`, `required_libs()`, `setup_command()`

2. **crossval's**: `/home/steven/code/Rust/BitNet-rs/crossval/src/backend.rs`
   - Used in token parity error messages
   - Simpler, just name and display
   - Methods: `from_name()`, `name()`, `full_name()`

They need to be converted between when calling crossval functions (see line 3104-3108 of xtask/src/main.rs).

## Verbose Mode Enhancement Ideas

Current verbose output (lines 2993-3018):
- Backend selection
- Template configuration

Could add:
- Library search paths checked (from build.rs output)
- Compile-time feature flags
- Environment variables in use
- Backend setup command (if missing)
- GGUF metadata inspection (chat_template, etc.)

## Next Steps to Implement

**Phase 1 (30 min)**: Wire `--dump-ids` and `--dump-cpp-ids` flags
- Highest ROI, already documented
- Minimal changes needed
- Users likely expecting it

**Phase 2 (20 min)**: Enhance verbose diagnostics  
- Better user experience
- Help with debugging
- Low risk changes

**Phase 3 (45 min)**: Create `xtask preflight` command
- Useful for CI/CD
- Function already exists
- Just need CLI wiring

**Phase 4 (30 min)**: Implement environment variable overrides
- `BITNET_CPP_BACKEND` - Force backend
- `BITNET_CROSSVAL_VERBOSE` - Extra logging
- Test scaffolding already in place

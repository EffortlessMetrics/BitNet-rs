# Validation Workflow Diagram

## Job Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       VALIDATION WORKFLOW                        │
│                     (validation.yml)                             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Security Guard       │
                    │                        │
                    │  ✓ Block correction    │
                    │    flags in CI         │
                    │  ✓ Verify strict mode  │
                    │  ✓ Check BITNET_*      │
                    │    environment vars    │
                    └────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   PASSED (0 minutes)    │
                    └────────────┬────────────┘
                                 │
                                 ▼
        ┌────────────────────────┴────────────────────────┐
        │                                                  │
        ▼                                                  ▼
┌──────────────────┐                              ┌──────────────────┐
│  Build Tools     │                              │  Build Tools     │
│  (Ubuntu)        │              ...             │  (macOS)         │
│                  │                              │                  │
│  ✓ bitnet-cli    │◄─────────────────────────────►  ✓ bitnet-cli    │
│  ✓ st2gguf       │    Parallel Execution        │  ✓ st2gguf       │
│  ✓ st-tools      │                              │  ✓ st-tools      │
└──────────────────┘                              └──────────────────┘
        │                                                  │
        │          ┌──────────────────┐                   │
        └─────────►│  Build Tools     │◄──────────────────┘
                   │  (Windows)       │
                   │                  │
                   │  ✓ bitnet-cli    │
                   │  ✓ st2gguf       │
                   │  ✓ st-tools      │
                   └──────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │        ALL BUILDS PASSED              │
        │        (5-10 minutes with cache)      │
        └───────────────────┬───────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────┐                    ┌──────────────────┐
│ Validation Tests │                    │ Validate Models  │
│                  │                    │                  │
│ Ubuntu           │                    │ Ubuntu Only      │
│ Windows          │                    │                  │
│ macOS            │                    │ ✓ BitNet I2S     │
│                  │                    │ ✓ Clean F16      │
│ ✓ workflow tests │                    │                  │
│ ✓ inspect tests  │                    │ Parse JSON       │
│ ✓ ln validation  │                    │ Check rulesets   │
│ ✓ architecture   │                    │ Detect suspicion │
│   detection      │                    │                  │
└──────────────────┘                    └──────────────────┘
        │                                       │
        │          Parallel Execution           │
        │          (2-5 minutes)                │
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Validation Summary   │
                │                       │
                │  ✓ Aggregate results  │
                │  ✓ Generate summary   │
                │  ✓ Check all gates    │
                │    passed             │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │    Quality Gate       │
                │                       │
                │  ✓ Final gate         │
                │  ✓ Blocks PR merge    │
                │  ✓ Provides guidance  │
                └───────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
  ┌──────────┐                          ┌──────────┐
  │ SUCCESS  │                          │  FAILURE │
  │          │                          │          │
  │ ✅ All    │                          │ ❌ Some   │
  │   gates  │                          │   gates  │
  │   passed │                          │   failed │
  │          │                          │          │
  │ Ready    │                          │ PR       │
  │ for      │                          │ blocked  │
  │ merge    │                          │          │
  └──────────┘                          └──────────┘
```

## Job Details

### 1. Security Guard (Critical)

```
┌───────────────────────────────────────────────────┐
│              SECURITY GUARD                       │
├───────────────────────────────────────────────────┤
│                                                   │
│  Checks:                                          │
│  ├─ BITNET_ALLOW_RUNTIME_CORRECTIONS not set     │
│  ├─ BITNET_CORRECTION_POLICY not set             │
│  ├─ BITNET_FIX_LN_SCALE not set (deprecated)     │
│  └─ BITNET_STRICT_MODE=1 is enabled              │
│                                                   │
│  Scans:                                           │
│  └─ .github/workflows/*.yml                       │
│                                                   │
│  Exit Codes:                                      │
│  ├─ 0: All checks passed                          │
│  └─ 1: Forbidden flag detected                    │
│                                                   │
│  Duration: < 1 minute                             │
│  Critical: YES (blocks all downstream jobs)       │
└───────────────────────────────────────────────────┘
```

### 2. Build Tools (Critical)

```
┌───────────────────────────────────────────────────┐
│              BUILD TOOLS                          │
├───────────────────────────────────────────────────┤
│                                                   │
│  Platforms:                                       │
│  ├─ Ubuntu (Linux x86_64)                         │
│  ├─ Windows (x86_64)                              │
│  └─ macOS (x86_64)                                │
│                                                   │
│  Tools Built:                                     │
│  ├─ bitnet-cli (--features cpu,full-cli)         │
│  ├─ bitnet-st2gguf                                │
│  ├─ st-ln-inspect                                 │
│  └─ st-merge-ln-f16                               │
│                                                   │
│  Verification:                                    │
│  ├─ Binary files exist                            │
│  └─ Binaries execute (--version/--help)           │
│                                                   │
│  Artifacts:                                       │
│  └─ {os}-validation-tools (7 days retention)     │
│                                                   │
│  Duration: 5-10 minutes (with cache)              │
│  Critical: YES (required for downstream jobs)     │
└───────────────────────────────────────────────────┘
```

### 3. Validation Tests (Critical)

```
┌───────────────────────────────────────────────────┐
│           VALIDATION TESTS                        │
├───────────────────────────────────────────────────┤
│                                                   │
│  Test Suites:                                     │
│  ├─ validation_workflow.rs                        │
│  │  ├─ Basic inspect invocation                   │
│  │  ├─ LayerNorm RMS validation                   │
│  │  ├─ Architecture detection                     │
│  │  ├─ Gate modes (auto, none, policy)            │
│  │  ├─ JSON output format                         │
│  │  └─ Exit code verification                     │
│  │                                                 │
│  └─ inspect_ln_stats.rs                           │
│     ├─ LayerNorm tensor identification            │
│     ├─ Projection weight validation               │
│     ├─ Quantized tensor handling                  │
│     └─ Output format tests                        │
│                                                   │
│  Platforms: Ubuntu, Windows, macOS                │
│                                                   │
│  Features: --no-default-features --features       │
│            cpu,full-cli                           │
│                                                   │
│  Artifacts:                                       │
│  └─ validation-test-report-{os} (30 days)        │
│                                                   │
│  Duration: 2-5 minutes                            │
│  Critical: YES (blocks merge)                     │
└───────────────────────────────────────────────────┘
```

### 4. Validate Models (Optional)

```
┌───────────────────────────────────────────────────┐
│            VALIDATE MODELS                        │
├───────────────────────────────────────────────────┤
│                                                   │
│  Models:                                          │
│  ├─ BitNet-I2S-2B                                 │
│  │  ├─ Path: models/.../ggml-model-i2_s.gguf     │
│  │  ├─ Expected: bitnet-b1.58:i2_s               │
│  │  └─ Validate: LN weights RMS                   │
│  │                                                 │
│  └─ Clean-F16                                     │
│     ├─ Path: models/clean/clean-f16.gguf         │
│     ├─ Expected: generic                          │
│     └─ Validate: LN weights RMS                   │
│                                                   │
│  Process:                                         │
│  ├─ Download tools from build-tools               │
│  ├─ Check if model exists (skip if not)           │
│  ├─ Run: inspect --ln-stats --json                │
│  ├─ Parse JSON output                             │
│  ├─ Verify ruleset matches expected               │
│  └─ Check for suspicious weights                  │
│                                                   │
│  Skip Condition:                                  │
│  └─ workflow_dispatch: skip_model_validation      │
│                                                   │
│  Artifacts:                                       │
│  └─ model-validation-{name} (30 days)            │
│                                                   │
│  Duration: 1-3 minutes per model                  │
│  Critical: NO (can be skipped)                    │
└───────────────────────────────────────────────────┘
```

### 5. Validation Summary (Required)

```
┌───────────────────────────────────────────────────┐
│          VALIDATION SUMMARY                       │
├───────────────────────────────────────────────────┤
│                                                   │
│  Aggregates:                                      │
│  ├─ security-guard result                         │
│  ├─ build-tools result                            │
│  ├─ validation-tests result                       │
│  └─ validate-models result                        │
│                                                   │
│  Outputs:                                         │
│  ├─ GitHub Actions step summary                   │
│  ├─ Configuration details                         │
│  ├─ Validation coverage                           │
│  └─ Platform coverage                             │
│                                                   │
│  Success Criteria:                                │
│  ├─ security-guard: must pass                     │
│  ├─ build-tools: must pass                        │
│  ├─ validation-tests: must pass                   │
│  └─ validate-models: pass or skipped              │
│                                                   │
│  Duration: < 1 minute                             │
│  Critical: YES (gates quality-gate)               │
└───────────────────────────────────────────────────┘
```

### 6. Quality Gate (Critical)

```
┌───────────────────────────────────────────────────┐
│             QUALITY GATE                          │
├───────────────────────────────────────────────────┤
│                                                   │
│  Purpose: Final gate to block PR merge           │
│                                                   │
│  Permissions:                                     │
│  ├─ checks: write                                 │
│  └─ pull-requests: write                          │
│                                                   │
│  Behavior:                                        │
│  ├─ Always runs (even if upstream fails)          │
│  ├─ Checks validation-summary result              │
│  ├─ Fails with detailed error if validation       │
│  │   did not pass                                 │
│  └─ Provides troubleshooting guidance             │
│                                                   │
│  Common Issues Reported:                          │
│  ├─ Correction flags set in CI                    │
│  ├─ Suspicious LayerNorm weights                  │
│  ├─ Build failures                                │
│  └─ Integration test failures                     │
│                                                   │
│  Duration: < 1 minute                             │
│  Critical: YES (blocks PR merge)                  │
└───────────────────────────────────────────────────┘
```

## Environment Configuration

```
┌─────────────────────────────────────────────┐
│        ENVIRONMENT VARIABLES                │
├─────────────────────────────────────────────┤
│                                             │
│  Cargo:                                     │
│  ├─ CARGO_TERM_COLOR=always                 │
│  ├─ RUST_BACKTRACE=1                        │
│  ├─ CARGO_INCREMENTAL=0                     │
│  └─ RUSTFLAGS="-D warnings"                 │
│                                             │
│  Validation:                                │
│  ├─ BITNET_STRICT_MODE=1                    │
│  ├─ BITNET_DETERMINISTIC=1                  │
│  ├─ BITNET_SEED=42                          │
│  └─ RAYON_NUM_THREADS=1                     │
│                                             │
│  Git Metadata (vergen-gix):                 │
│  ├─ VERGEN_GIT_SHA=${{ github.sha }}        │
│  ├─ VERGEN_GIT_BRANCH=${{ github.ref_name }}│
│  ├─ VERGEN_GIT_DESCRIBE=...                 │
│  └─ VERGEN_IDEMPOTENT=1                     │
│                                             │
│  Blocked (security-guard enforces):         │
│  ├─ BITNET_ALLOW_RUNTIME_CORRECTIONS ❌     │
│  ├─ BITNET_CORRECTION_POLICY ❌             │
│  └─ BITNET_FIX_LN_SCALE ❌ (deprecated)     │
└─────────────────────────────────────────────┘
```

## Trigger Conditions

```
┌─────────────────────────────────────────────┐
│           WORKFLOW TRIGGERS                 │
├─────────────────────────────────────────────┤
│                                             │
│  Push to main/develop:                      │
│  └─ Changes to validation-related paths     │
│                                             │
│  Pull Request to main/develop:              │
│  └─ Changes to validation-related paths     │
│                                             │
│  Workflow Dispatch (manual):                │
│  └─ Optional: skip_model_validation         │
│                                             │
│  Monitored Paths:                           │
│  ├─ crates/bitnet-cli/**                    │
│  ├─ crates/bitnet-st-tools/**               │
│  ├─ crates/bitnet-st2gguf/**                │
│  ├─ crates/bitnet-models/**                 │
│  ├─ scripts/validate_gguf.sh                │
│  ├─ scripts/export_clean_gguf.sh            │
│  └─ .github/workflows/validation.yml        │
└─────────────────────────────────────────────┘
```

## Timeline

```
Time  │ Job
──────┼────────────────────────────────────────
0:00  │ ┌─────────────────────────┐
      │ │ Security Guard          │
      │ └─────────────────────────┘
0:30  │          │
      │          ▼
1:00  │ ┌─────────┬─────────┬─────────┐
      │ │ Build   │ Build   │ Build   │
      │ │ Ubuntu  │ Windows │ macOS   │
      │ └─────────┴─────────┴─────────┘
      │       (Parallel Execution)
5:00  │
      │
10:00 │          │
      │          ▼
      │ ┌─────────────┬──────────────┐
      │ │ Validation  │ Validate     │
      │ │ Tests       │ Models       │
      │ └─────────────┴──────────────┘
      │    (Parallel Execution)
15:00 │
      │          │
      │          ▼
      │ ┌─────────────────────────┐
      │ │ Validation Summary      │
      │ └─────────────────────────┘
      │          │
      │          ▼
      │ ┌─────────────────────────┐
20:00 │ │ Quality Gate            │
      │ └─────────────────────────┘
      │          │
      │          ▼
      │   ✅ or ❌

Total: 15-25 minutes (with caching)
```

## Artifacts Flow

```
┌──────────────────┐
│   Build Tools    │
└────────┬─────────┘
         │
         ├─ Upload: {os}-validation-tools
         │          (binaries, 7 days)
         │
         ▼
┌──────────────────┐
│ Validate Models  │
└────────┬─────────┘
         │
         ├─ Download: {os}-validation-tools
         │            (from build-tools)
         │
         └─ Upload: model-validation-{name}
                    (JSON, reports, 30 days)

┌──────────────────┐
│Validation Tests  │
└────────┬─────────┘
         │
         └─ Upload: validation-test-report-{os}
                    (reports, 30 days)
```

## Success/Failure Paths

```
                    Start
                      │
                      ▼
             ┌────────────────┐
             │ Security Guard │
             └────────┬───────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
       Success                 Failure
          │                       │
          ▼                       ▼
    Build Tools            ❌ BLOCKED
          │
          ▼
    ┌──────────┐
    │  Tests   │
    └─────┬────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
 Success     Failure
    │           │
    ▼           │
 Summary ◄──────┘
    │
    ▼
┌──────────┐
│  Gate    │
└─────┬────┘
      │
  ┌───┴───┐
  │       │
  ▼       ▼
 ✅      ❌
Merge   Block
```

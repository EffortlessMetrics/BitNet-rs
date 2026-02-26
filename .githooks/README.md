# bitnet-rs Git Hooks

This directory contains Git hooks that enforce bitnet-rs quality standards locally before commits reach CI.

## Quick Setup

Enable hooks for this repository:

```bash
git config core.hooksPath .githooks
```

## Available Hooks

### pre-commit

Runs before every commit to check:

1. **#[ignore] Annotation Hygiene**
   - Ensures all `#[ignore]` attributes include a reason
   - Prevents bare `#[ignore]` markers from being committed
   - Enforces pattern: `#[ignore = "reason"]` or comment before attribute

2. **Environment Mutation Safety**
   - Warns about raw `std::env::set_var()` / `remove_var()` calls
   - Encourages EnvGuard pattern with `#[serial(bitnet_env)]`
   - Prevents test race conditions and environment pollution

## Hook Status

- **pre-commit**: Enabled (checks #[ignore] and env mutations)
- **commit-msg**: Not implemented
- **pre-push**: Not implemented

## Disabling Hooks (Not Recommended)

To temporarily bypass hooks:

```bash
git commit --no-verify
```

**Warning**: CI will still enforce these checks. Local hooks save time by catching issues early.

## Troubleshooting

### Hook Not Running

Verify hooks are enabled:
```bash
git config core.hooksPath
# Should output: .githooks
```

### False Positives

If the hook incorrectly flags valid code:
1. Check the pattern documentation in the hook file
2. Report issue to the bitnet-rs team
3. Use `--no-verify` as a temporary workaround

## Documentation

- Test patterns: `docs/development/test-suite.md`
- Environment testing: `docs/development/test-suite.md#environment-variable-testing`
- Ignored tests: `docs/development/test-suite.md#ignored-tests`

## Contributing

To add new hooks:
1. Create executable script in `.githooks/`
2. Document behavior in this README
3. Add corresponding CI check in `.github/workflows/ci.yml`
4. Ensure hook matches CI guard behavior

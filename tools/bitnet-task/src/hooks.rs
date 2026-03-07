use super::*;

pub(crate) fn cmd_install_hooks(root: &Path) -> Result<()> {
    println!("🔧 Installing Git hooks for BitNet-rs...");

    let hooks_dir = root.join(".git").join("hooks");
    fs::create_dir_all(&hooks_dir).context("creating .git/hooks")?;

    let pre_commit = r#"#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Running pre-commit checks..."

# 1. Check formatting
echo "📝 Checking formatting..."
if ! cargo fmt --all -- --check; then
  echo "❌ Code is not formatted. Run 'cargo fmt --all' to fix."
  exit 1
fi

# 2. Run clippy with strict checks
echo "🔍 Running clippy..."
if ! RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-features --all-targets -- -D warnings -D clippy::ptr_arg 2>/dev/null; then
  echo "❌ Clippy found issues. Please fix them before committing."
  exit 1
fi

# 3. Check banned patterns
echo "🚫 Checking for banned patterns..."
if ! bash scripts/hooks/banned-patterns.sh; then
  echo "❌ Found banned patterns. Please fix them before committing."
  exit 1
fi

# 4. Check that tests compile
echo "🧪 Checking tests compile..."
if ! cargo check --workspace --tests --no-default-features --features cpu 2>/dev/null; then
  echo "❌ Tests don't compile. Please fix them before committing."
  exit 1
fi

echo "✅ All pre-commit checks passed!"
"#;
    let pre_push = r#"#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Running pre-push checks..."

# 1. Run tests without execution to ensure they build
echo "🧪 Building tests..."
if ! cargo test --workspace --no-default-features --features cpu --no-run; then
  echo "❌ Tests failed to build. Please fix them before pushing."
  exit 1
fi

# 2. Run cargo-deny if available
if command -v cargo-deny &> /dev/null; then
  echo "🔒 Running cargo-deny security checks..."
  if ! cargo deny check --hide-inclusion-graph; then
    echo "⚠️  cargo-deny found issues. Consider fixing them."
    # Don't fail on cargo-deny, just warn
  fi
else
  echo "ℹ️  cargo-deny not installed. Install with: cargo install cargo-deny"
fi

echo "✅ All pre-push checks passed!"
"#;

    let pre_commit_path = hooks_dir.join("pre-commit");
    let pre_push_path = hooks_dir.join("pre-push");
    fs::write(&pre_commit_path, pre_commit)
        .with_context(|| format!("writing {}", pre_commit_path.display()))?;
    fs::write(&pre_push_path, pre_push)
        .with_context(|| format!("writing {}", pre_push_path.display()))?;

    #[cfg(unix)]
    {
        for path in [&pre_commit_path, &pre_push_path] {
            let mut perms = fs::metadata(path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms)?;
        }
    }
    #[cfg(not(unix))]
    {
        run_stream(
            root,
            "chmod",
            &[
                "+x",
                pre_commit_path.to_string_lossy().as_ref(),
                pre_push_path.to_string_lossy().as_ref(),
            ],
            &[],
        )?;
    }

    println!("✅ Git hooks installed successfully!");
    println!();
    println!("To use Python-based pre-commit instead (more features):");
    println!("  pip install pre-commit");
    println!("  pre-commit install");
    println!();
    println!("To install additional tools:");
    println!("  cargo install cargo-deny taplo-cli");
    Ok(())
}

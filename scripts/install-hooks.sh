#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Installing Git hooks for BitNet-rs..."

# Create .git/hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/usr/bin/env bash
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
EOF

# Create pre-push hook
cat > .git/hooks/pre-push << 'EOF'
#!/usr/bin/env bash
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
EOF

# Make hooks executable
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/pre-push

echo "✅ Git hooks installed successfully!"
echo ""
echo "To use Python-based pre-commit instead (more features):"
echo "  pip install pre-commit"
echo "  pre-commit install"
echo ""
echo "To install additional tools:"
echo "  cargo install cargo-deny taplo-cli"

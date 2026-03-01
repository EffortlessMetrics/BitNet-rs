#!/usr/bin/env bash
# Comprehensive crates.io readiness validation for BitNet-rs.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

if [[ ! -f Cargo.toml || ! -d crates ]]; then
  fail "Run this script from the BitNet-rs repository root"
fi

SKIP_TESTS=${SKIP_TESTS:-0}
SKIP_DOCS=${SKIP_DOCS:-0}

info "Checking Rust toolchain version..."
REQUIRED_VERSION=$(awk -F'"' '/^rust-version = / {print $2; exit}' Cargo.toml)
RUST_VERSION=$(rustc --version | awk '{print $2}')
if ! printf '%s\n%s\n' "$REQUIRED_VERSION" "$RUST_VERSION" | sort -V -C; then
  fail "Rust $RUST_VERSION is below required $REQUIRED_VERSION"
fi
pass "Rust version $RUST_VERSION meets workspace requirement"

info "Checking required cargo subcommands..."
command -v cargo >/dev/null || fail "cargo not found"
cargo fmt --version >/dev/null 2>&1 || fail "rustfmt component missing (install with: rustup component add rustfmt)"
cargo clippy --version >/dev/null 2>&1 || fail "clippy component missing (install with: rustup component add clippy)"
pass "Core toolchain commands are available"

info "Running format check..."
cargo fmt --all -- --check
pass "Formatting is correct"

info "Running clippy..."
cargo clippy --workspace --all-targets --all-features -- -D warnings
pass "Clippy passed"

info "Building workspace..."
cargo build --workspace --all-features
pass "Build passed"

if [[ "$SKIP_TESTS" != "1" ]]; then
  info "Running tests..."
  cargo test --workspace --all-features
  pass "Tests passed"
else
  warn "Skipping tests (SKIP_TESTS=1)"
fi

if [[ "$SKIP_DOCS" != "1" ]]; then
  info "Building documentation..."
  cargo doc --workspace --all-features --no-deps
  pass "Documentation build passed"
else
  warn "Skipping docs (SKIP_DOCS=1)"
fi

if command -v cargo-audit >/dev/null 2>&1; then
  info "Running cargo audit..."
  cargo audit
  pass "Security audit passed"
else
  warn "cargo-audit not installed; skipping security audit"
fi

if command -v cargo-deny >/dev/null 2>&1; then
  info "Running cargo deny..."
  cargo deny check
  pass "cargo deny passed"
else
  warn "cargo-deny not installed; skipping license/policy check"
fi

info "Validating metadata and packaging for publishable crates..."
PUBLISHABLE=$(python - <<'PY'
import json
import subprocess
import tomllib

meta = json.loads(subprocess.check_output(['cargo', 'metadata', '--no-deps', '--format-version', '1']))
workspace_members = set(meta['workspace_members'])
for pkg in meta['packages']:
  if pkg['id'] not in workspace_members:
    continue
  manifest = tomllib.load(open(pkg['manifest_path'], 'rb'))
  if manifest.get('package', {}).get('publish') is False:
    continue
  print(pkg['name'])
PY
)

PUBLISHABLE_LEAVES=$(python - <<'PY'
import json
import subprocess
import tomllib

meta = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version', '1']))
workspace_members = set(meta['workspace_members'])
publishable = set()
for pkg in meta['packages']:
  if pkg['id'] not in workspace_members:
    continue
  manifest = tomllib.load(open(pkg['manifest_path'], 'rb'))
  if manifest.get('package', {}).get('publish') is False:
    continue
  publishable.add(pkg['name'])

for pkg in meta['packages']:
  if pkg['id'] not in workspace_members:
    continue
  if pkg['name'] not in publishable:
    continue
  has_internal_publishable_dep = False
  for dep in pkg.get('dependencies', []):
    if dep.get('name') in publishable:
      has_internal_publishable_dep = True
      break
  if not has_internal_publishable_dep:
    print(pkg['name'])
PY
)

README_BY_PACKAGE=$(python - <<'PY'
import json,subprocess,tomllib
meta=json.loads(subprocess.check_output(['cargo','metadata','--no-deps','--format-version','1']))
for pkg in meta['packages']:
  name=pkg['name']
  m=tomllib.load(open(pkg['manifest_path'],'rb'))
  print(f"{name}\t{m.get('package',{}).get('readme','')}")
PY
)

for pkg in $PUBLISHABLE; do
  info "Packaging check for $pkg"
  cargo package -p "$pkg" --allow-dirty --no-verify --list >/tmp/bitnet-package-list.txt
  if ! grep -q '^Cargo.toml$' /tmp/bitnet-package-list.txt; then
    fail "$pkg package list does not include Cargo.toml"
  fi
  readme=$(printf '%s\n' "$README_BY_PACKAGE" | awk -F '\t' -v p="$pkg" '$1==p{print $2}')
  if [[ -n "$readme" ]] && ! grep -q "^${readme}$" /tmp/bitnet-package-list.txt; then
    fail "$pkg declares readme=$readme but it is not included in package"
  fi
  pass "$pkg package list looks good"
done

pass "All publishable crates pass packaging checks"

if [[ -n "$PUBLISHABLE_LEAVES" ]]; then
  info "Running crates.io dry-run publish checks for leaf crates..."
  for pkg in $PUBLISHABLE_LEAVES; do
    info "Dry-run publish for $pkg"
    cargo publish --dry-run -p "$pkg"
    pass "$pkg dry-run publish check passed"
  done
  pass "Leaf crate publish dry-runs passed"
else
  warn "No publishable leaf crates found; skipping dry-run publish checks"
fi

echo
echo "🚀 BitNet-rs crates.io readiness checks complete"

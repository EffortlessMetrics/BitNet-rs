#!/usr/bin/env bash
set -euo pipefail

out_dir="docs/generated/phase1"
mkdir -p "$out_dir"

# 1) Repo fingerprint (deterministic list of tracked files)
git ls-files | LC_ALL=C sort > "$out_dir/repo-fingerprint.txt"

# 2) Workspace dependency snapshot
cargo metadata --format-version 1 > "$out_dir/deps-workspace.json"

# 6) Public API manifest of existing snapshots
{
  echo "# Public API Snapshot Manifest"
  echo
  echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  for f in docs/api/rust/*.public-api.txt; do
    [ -e "$f" ] || continue
    sha=$(sha256sum "$f" | awk '{print $1}')
    printf -- '- `%s`: `%s`\n' "$f" "$sha"
  done
} > "$out_dir/public-api-manifest.md"

# 21) SBOM baseline via cargo metadata
cp "$out_dir/deps-workspace.json" "$out_dir/sbom-cargo-metadata.json"

# 22-23) Security / license scan report (non-fatal in generator)
if command -v cargo-deny >/dev/null 2>&1; then
  cargo deny check advisories bans licenses sources > "$out_dir/security-license-report.txt" 2>&1 || true
else
  echo "cargo-deny not installed in this environment." > "$out_dir/security-license-report.txt"
fi

# 26) Bench inventory baseline
{
  echo "# Bench Inventory"
  echo
  echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  rg --files benches benchmarks 2>/dev/null | LC_ALL=C sort | sed 's|^|- |'
} > "$out_dir/bench-inventory.md"

# 27) Size inventory baseline (dependency graph size proxy)
{
  echo "# Size Inventory (Dependency Proxy)"
  echo
  echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  echo "## Cargo.lock summary"
  echo
  echo "- lines: $(wc -l < Cargo.lock)"
  echo "- sha256: $(sha256sum Cargo.lock | awk '{print $1}')"
} > "$out_dir/size-inventory.md"

# 30) Churn report over last 30 days
{
  echo "# Churn Report (30 days)"
  echo
  echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  git log --since='30 days ago' --name-only --pretty=format: | sed '/^$/d' | \
    LC_ALL=C sort | uniq -c | sort -nr | head -n 100 | awk '{count=$1; $1=""; sub(/^ /, ""); print "- " $0 ": " count}' || true
} > "$out_dir/churn-30d.md"

# 36) Release notes preview baseline
{
  echo "# Release Notes Preview"
  echo
  echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  echo "## Recent commits"
  echo
  git log --pretty='- %h %s' -n 50
} > "$out_dir/release-notes-preview.md"

# Status file with stable hashes
{
  echo '{'
  echo "  \"generated_at_utc\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\","
  echo '  "artifacts": {'
  first=1
  for f in "$out_dir"/*; do
    [ -f "$f" ] || continue
    h=$(sha256sum "$f" | awk '{print $1}')
    base=$(basename "$f")
    if [ "$first" -eq 0 ]; then echo ','; fi
    first=0
    printf '    "%s": "%s"' "$base" "$h"
  done
  echo
  echo '  }'
  echo '}'
} > "$out_dir/status.json"

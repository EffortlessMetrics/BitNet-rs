#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

scripts/maintenance/generate_deps_graph.py
scripts/maintenance/generate_churn_report.py
scripts/maintenance/generate_status_report.py
scripts/maintenance/generate_coverage_summary.py || true
scripts/maintenance/generate_public_api_snapshot.sh
scripts/maintenance/generate_sbom.sh

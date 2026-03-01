#!/usr/bin/env python3
"""Generate repository maintenance artifacts.

Artifacts:
- docs/maintenance/deps.dot
- docs/maintenance/deps.svg (if graphviz `dot` is available)
- docs/maintenance/public_api_snapshot.md
- docs/maintenance/churn.md
- docs/maintenance/status.md
- docs/maintenance/sbom.json
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "maintenance"
SELECTED_API_CRATES = [
    "crates/bitnet-common",
    "crates/bitnet-inference",
    "crates/bitnet-quantization",
    "crates/bitnet-tokenizers",
]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=True)


def load_cargo_metadata() -> dict:
    cp = run(["cargo", "metadata", "--format-version", "1"])
    return json.loads(cp.stdout)


def generate_dependency_graph(meta: dict) -> None:
    workspace_ids = set(meta.get("workspace_members", []))
    id_to_name = {pkg["id"]: pkg["name"] for pkg in meta["packages"] if pkg["id"] in workspace_ids}
    name_to_pkg = {pkg["name"]: pkg for pkg in meta["packages"] if pkg["id"] in workspace_ids}

    edges: set[tuple[str, str]] = set()
    for pkg in name_to_pkg.values():
        src = pkg["name"]
        for dep in pkg.get("dependencies", []):
            tgt = dep["name"]
            if tgt in name_to_pkg:
                edges.add((src, tgt))

    lines = [
        "digraph workspace_deps {",
        "  rankdir=LR;",
        '  graph [fontname="Helvetica"];',
        '  node [shape=box, style=rounded, fontname="Helvetica"];',
        '  edge [color="#6b7280"];',
    ]
    for name in sorted(name_to_pkg):
        lines.append(f'  "{name}";')
    for src, tgt in sorted(edges):
        lines.append(f'  "{src}" -> "{tgt}";')
    lines.append("}")

    dot_path = DOCS / "deps.dot"
    dot_path.write_text("\n".join(lines) + "\n")

    try:
        run(["dot", "-Tsvg", str(dot_path), "-o", str(DOCS / "deps.svg")])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def iter_public_items(crate_path: Path) -> Iterable[str]:
    src = crate_path / "src"
    if not src.exists():
        return []

    patterns = [
        re.compile(r"^\s*pub\s+(?:async\s+)?fn\s+([A-Za-z0-9_]+)"),
        re.compile(r"^\s*pub\s+struct\s+([A-Za-z0-9_]+)"),
        re.compile(r"^\s*pub\s+enum\s+([A-Za-z0-9_]+)"),
        re.compile(r"^\s*pub\s+trait\s+([A-Za-z0-9_]+)"),
        re.compile(r"^\s*pub\s+type\s+([A-Za-z0-9_]+)"),
        re.compile(r"^\s*pub\s+const\s+([A-Za-z0-9_]+)"),
        re.compile(r"^\s*pub\s+mod\s+([A-Za-z0-9_]+)"),
    ]

    items: set[str] = set()
    for rs_file in sorted(src.rglob("*.rs")):
        rel = rs_file.relative_to(crate_path)
        for line in rs_file.read_text(errors="ignore").splitlines():
            for pat in patterns:
                m = pat.match(line)
                if m:
                    items.add(f"{rel}:{m.group(1)}")
                    break
    return sorted(items)


def generate_public_api_snapshot() -> None:
    out = ["# Public API Snapshot", "", f"Generated: {dt.datetime.now(dt.UTC).isoformat()}Z", ""]
    for crate_rel in SELECTED_API_CRATES:
        crate = ROOT / crate_rel
        out.append(f"## `{crate_rel}`")
        items = list(iter_public_items(crate))
        out.append(f"Public item count: **{len(items)}**")
        out.append("")
        out.extend([f"- `{item}`" for item in items])
        out.append("")

    (DOCS / "public_api_snapshot.md").write_text("\n".join(out) + "\n")


def _git_lines(*args: str) -> list[str]:
    cp = run(["git", *args], check=False)
    if cp.returncode != 0:
        return []
    return [ln for ln in cp.stdout.splitlines() if ln.strip()]


def generate_churn_report() -> None:
    lines_90 = _git_lines("log", "--since=90.days", "--name-only", "--pretty=format:")
    lines_30 = _git_lines("log", "--since=30.days", "--name-only", "--pretty=format:")

    def summarize(lines: list[str]) -> Counter[str]:
        c: Counter[str] = Counter()
        for path in lines:
            if path.startswith("docs/"):
                key = "docs"
            elif "/" in path:
                key = path.split("/", 1)[0]
            else:
                key = "."
            c[key] += 1
        return c

    top_dirs_90 = summarize(lines_90).most_common(10)
    top_files_90 = Counter(lines_90).most_common(15)
    top_dirs_30 = summarize(lines_30).most_common(10)

    out = [
        "# Churn Hotspot Report",
        "",
        f"Generated: {dt.datetime.now(dt.UTC).isoformat()}Z",
        "",
        "## Top directories (90 days)",
        "",
    ]
    out.extend([f"- `{name}`: {count} edits" for name, count in top_dirs_90])
    out += ["", "## Top directories (30 days)", ""]
    out.extend([f"- `{name}`: {count} edits" for name, count in top_dirs_30])
    out += ["", "## Most touched files (90 days)", ""]
    out.extend([f"- `{name}`: {count} edits" for name, count in top_files_90])

    (DOCS / "churn.md").write_text("\n".join(out) + "\n")


def generate_status_report(meta: dict) -> None:
    cargo_toml = (ROOT / "Cargo.toml").read_text()
    msrv = "unknown"
    for pat in [r"rust-version\s*=\s*\"([^\"]+)\"", r"rust-version\.workspace\s*=\s*true"]:
        m = re.search(pat, cargo_toml)
        if m:
            msrv = m.group(1) if m.groups() else "workspace"
            break

    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    workspace_count = len(meta.get("workspace_members", []))

    out = [
        "# Repository Invariants Status",
        "",
        f"Generated: {dt.datetime.now(dt.UTC).isoformat()}Z",
        "",
        f"- **MSRV**: `{msrv}`",
        f"- **Workspace crate count**: `{workspace_count}`",
        f"- **Workflow count**: `{len(workflows)}`",
        "",
        "## CI workflows",
        "",
    ]
    out.extend([f"- `{wf.name}`" for wf in workflows])
    out += ["", "## Contract / schema hints", ""]
    for path in sorted((ROOT / "scripts").glob("*contract*")):
        out.append(f"- `scripts/{path.name}`")

    (DOCS / "status.md").write_text("\n".join(out) + "\n")


def generate_sbom(meta: dict) -> None:
    packages = []
    for pkg in sorted(meta["packages"], key=lambda p: p["name"]):
        packages.append(
            {
                "name": pkg["name"],
                "version": pkg["version"],
                "license": pkg.get("license"),
                "repository": pkg.get("repository"),
                "manifest_path": str(Path(pkg["manifest_path"]).relative_to(ROOT) if str(pkg["manifest_path"]).startswith(str(ROOT)) else Path(pkg["manifest_path"])),
            }
        )

    sbom = {
        "bomFormat": "BitNet-rs-internal",
        "generated": dt.datetime.now(dt.UTC).isoformat() + "Z",
        "packageCount": len(packages),
        "packages": packages,
    }
    (DOCS / "sbom.json").write_text(json.dumps(sbom, indent=2) + "\n")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    meta = load_cargo_metadata()
    generate_dependency_graph(meta)
    generate_public_api_snapshot()
    generate_churn_report()
    generate_status_report(meta)
    generate_sbom(meta)


if __name__ == "__main__":
    main()

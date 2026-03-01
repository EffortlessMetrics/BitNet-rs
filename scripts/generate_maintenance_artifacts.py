#!/usr/bin/env python3
"""Generate deterministic maintenance artifacts for architecture and governance tracking."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STATUS_DIR = REPO_ROOT / "docs" / "status"
API_DIR = REPO_ROOT / "docs" / "api" / "rust"


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    return result.stdout


def workspace_members() -> list[str]:
    cargo_toml = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    members: list[str] = []
    in_members = False
    for raw in cargo_toml.splitlines():
        line = raw.strip()
        if line.startswith("members = ["):
            in_members = True
            continue
        if in_members and line.startswith("]"):
            break
        if not in_members:
            continue
        if '"' not in line:
            continue
        value = line.split('"')[1]
        if value in {".", "tests", "tests-new", "xtask", "crossval", "fuzz", "xtask-build-helper"}:
            continue
        if value.startswith("crates/"):
            members.append(value.split("/", 1)[1])
    return sorted(set(members))


def crate_edges() -> dict[str, list[str]]:
    metadata = json.loads(run(["cargo", "metadata", "--format-version", "1", "--no-deps"]))
    name_for_manifest: dict[str, str] = {}
    for pkg in metadata["packages"]:
        manifest = pathlib.Path(pkg["manifest_path"])
        if "crates" in manifest.parts:
            name_for_manifest[str(manifest)] = pkg["name"]

    workspace = set(name_for_manifest.values())
    edges: dict[str, list[str]] = collections.defaultdict(list)
    for pkg in metadata["packages"]:
        src = pkg["name"]
        if src not in workspace:
            continue
        for dep in pkg.get("dependencies", []):
            target = dep["name"]
            if target in workspace and target != src:
                edges[src].append(target)
    for src, deps in edges.items():
        edges[src] = sorted(set(deps))
    return dict(sorted(edges.items()))


def write_dependency_graph() -> None:
    edges = crate_edges()
    lines = ["graph TD"]
    for src, deps in edges.items():
        if not deps:
            lines.append(f"  {src}")
        for dep in deps:
            lines.append(f"  {src} --> {dep}")

    mmd = "\n".join(lines) + "\n"
    (STATUS_DIR / "dependency-graph.mmd").write_text(mmd, encoding="utf-8")

    summary = [
        "# Crate Dependency Graph",
        "",
        "Generated from `cargo metadata --no-deps` and encoded as Mermaid for reviewable diffs.",
        "",
        f"- Workspace crates with internal edges: `{len(edges)}`",
        f"- Internal dependency edges: `{sum(len(v) for v in edges.values())}`",
        "",
        "```mermaid",
        *lines,
        "```",
        "",
    ]
    (STATUS_DIR / "dependency-graph.md").write_text("\n".join(summary), encoding="utf-8")


def write_api_surface_summary() -> None:
    rows = []
    for path in sorted(API_DIR.glob("*.public-api.txt")):
        text = path.read_text(encoding="utf-8")
        normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip() + "\n"
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        exported_items = sum(1 for line in text.splitlines() if re.search(r"\bpub\b", line))
        rows.append((path.name, exported_items, digest))

    lines = [
        "# Public API Snapshot Summary",
        "",
        "Stable summary of checked-in API snapshots under `docs/api/rust`.",
        "",
        "| Snapshot | `pub` lines | SHA-256 (12 chars) |",
        "|---|---:|---|",
    ]
    for name, count, digest in rows:
        lines.append(f"| `{name}` | {count} | `{digest}` |")

    lines.append("")
    (STATUS_DIR / "public-api-summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_churn_report() -> None:
    stdout = run([
        "git",
        "log",
        "--since=90 days ago",
        "--name-only",
        "--pretty=format:",
        "--",
        "crates",
        "docs",
        ".github/workflows",
    ])
    files = [line.strip() for line in stdout.splitlines() if line.strip()]
    by_dir = collections.Counter(pathlib.Path(p).parts[0] if pathlib.Path(p).parts else p for p in files)
    by_file = collections.Counter(files)

    lines = [
        "# Churn Hotspot Report (90 days)",
        "",
        "Based on `git log --since=90 days ago --name-only`.",
        "",
        "## Top directories",
        "",
        "| Directory | Touches |",
        "|---|---:|",
    ]
    for directory, count in by_dir.most_common(10):
        lines.append(f"| `{directory}` | {count} |")

    lines.extend(["", "## Top files", "", "| File | Touches |", "|---|---:|"])
    for file_path, count in by_file.most_common(15):
        lines.append(f"| `{file_path}` | {count} |")
    lines.append("")

    (STATUS_DIR / "churn.md").write_text("\n".join(lines), encoding="utf-8")


def write_repo_invariants() -> None:
    cargo_toml = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    msrv_match = re.search(r"rust-version\s*=\s*\"([^\"]+)\"", cargo_toml)
    msrv = msrv_match.group(1) if msrv_match else "unknown"

    workflows = sorted(p.name for p in (REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    specs = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "docs" / "api").glob("**/*.json"))

    lines = [
        "# Repository Invariants",
        "",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}`",
        f"- MSRV (`workspace.package.rust-version`): `{msrv}`",
        f"- Workspace crates: `{len(workspace_members())}`",
        f"- CI workflows: `{len(workflows)}`",
        f"- API/contract JSON files: `{len(specs)}`",
        "",
        "## Contract files",
        "",
    ]
    lines.extend(f"- `{spec}`" for spec in specs)
    lines.extend(["", "## CI workflow files", ""])
    lines.extend(f"- `.github/workflows/{wf}`" for wf in workflows)
    lines.append("")

    (STATUS_DIR / "invariants.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    write_dependency_graph()
    write_api_surface_summary()
    write_churn_report()
    write_repo_invariants()


if __name__ == "__main__":
    main()

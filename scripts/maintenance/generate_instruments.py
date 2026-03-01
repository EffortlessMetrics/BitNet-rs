#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "instruments"
PUBLIC_API_CRATES = [
    "bitnet",
    "bitnet-cli",
    "bitnet-inference",
    "bitnet-server",
    "bitnet-tokenizers",
]


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True)


def cargo_metadata() -> dict[str, Any]:
    return json.loads(run(["cargo", "metadata", "--format-version", "1", "--all-features"]))


def build_workspace_maps(meta: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    packages = {p["id"]: p for p in meta["packages"]}
    members = set(meta["workspace_members"])
    return packages, members


def write_dependency_graph(meta: dict[str, Any], packages: dict[str, dict[str, Any]], members: set[str]) -> None:
    lines = ["graph TD"]
    resolve_nodes = {node["id"]: node for node in meta["resolve"]["nodes"]}

    member_ids = sorted(members, key=lambda pid: packages[pid]["name"])
    for pid in member_ids:
        pkg = packages[pid]
        crate = pkg["name"]
        lines.append(f'    {safe(crate)}["{crate}"]')

    for pid in member_ids:
        node = resolve_nodes.get(pid)
        if not node:
            continue
        source = packages[pid]["name"]
        for dep in sorted(node["deps"], key=lambda d: d["name"]):
            dep_id = dep["pkg"]
            if dep_id not in members:
                continue
            target = packages[dep_id]["name"]
            if source == target:
                continue
            lines.append(f"    {safe(source)} --> {safe(target)}")

    content = "\n".join(lines) + "\n"
    (OUT_DIR / "deps.mmd").write_text(content)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def public_symbols(crate_dir: Path) -> list[str]:
    lib_rs = crate_dir / "src" / "lib.rs"
    if not lib_rs.exists():
        return []
    out: list[str] = []
    for line in lib_rs.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("pub ") or stripped.startswith("pub("):
            if stripped.startswith("pub(crate)"):
                continue
            out.append(stripped)
    return sorted(set(out))


def write_public_api_snapshot(meta: dict[str, Any], packages: dict[str, dict[str, Any]], members: set[str]) -> None:
    by_name = {packages[mid]["name"]: packages[mid] for mid in members}
    lines = [
        "# Public API Snapshot",
        "",
        "Lightweight snapshot of top-level `pub` declarations in selected library crates.",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    for crate in PUBLIC_API_CRATES:
        pkg = by_name.get(crate)
        if not pkg:
            continue
        crate_dir = Path(pkg["manifest_path"]).parent
        symbols = public_symbols(crate_dir)
        lines.append(f"## {crate}")
        lines.append("")
        if not symbols:
            lines.append("_No `src/lib.rs` public items found._")
            lines.append("")
            continue
        lines.append(f"Public items: {len(symbols)}")
        lines.append("")
        for symbol in symbols:
            lines.append(f"- `{symbol}`")
        lines.append("")

    (OUT_DIR / "public_api.md").write_text("\n".join(lines))


def write_sbom(meta: dict[str, Any], packages: dict[str, dict[str, Any]], members: set[str]) -> None:
    resolve_nodes = {node["id"]: node for node in meta["resolve"]["nodes"]}
    components = []
    for pid in sorted(members, key=lambda p: packages[p]["name"]):
        pkg = packages[pid]
        node = resolve_nodes.get(pid, {"deps": []})
        components.append(
            {
                "name": pkg["name"],
                "version": pkg["version"],
                "license": pkg.get("license"),
                "manifest_path": pkg["manifest_path"],
                "dependencies": sorted(
                    [
                        packages[d["pkg"]]["name"]
                        for d in node.get("deps", [])
                        if d["pkg"] in members
                    ]
                ),
            }
        )

    sbom = {
        "schema": "bitnet-maintenance-sbom-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "component_count": len(components),
        "components": components,
    }
    (OUT_DIR / "sbom.json").write_text(json.dumps(sbom, indent=2) + "\n")


def git_output(args: list[str], fallback: str = "n/a") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return fallback


def write_churn_report() -> None:
    lines = [
        "# Churn Report (last 90 days)",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Most touched files",
        "",
    ]
    try:
        raw = run(["git", "log", "--since=90.days", "--name-only", "--pretty=format:"])
        counts: dict[str, int] = {}
        for path in raw.splitlines():
            path = path.strip()
            if not path:
                continue
            counts[path] = counts.get(path, 0) + 1
        top_files = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:25]
        if top_files:
            lines.append("| File | Touches |")
            lines.append("|---|---:|")
            for path, count in top_files:
                lines.append(f"| `{path}` | {count} |")
        else:
            lines.append("No churn data available.")
    except subprocess.CalledProcessError:
        lines.append("Unable to compute churn data in this environment.")

    lines.append("")
    lines.append("## Most touched directories")
    lines.append("")
    try:
        raw = run(["git", "log", "--since=90.days", "--name-only", "--pretty=format:"])
        counts: dict[str, int] = {}
        for path in raw.splitlines():
            path = path.strip()
            if not path:
                continue
            p = Path(path)
            parent = str(p.parent) if str(p.parent) != "." else "(root)"
            counts[parent] = counts.get(parent, 0) + 1
        top_dirs = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        if top_dirs:
            lines.append("| Directory | Touches |")
            lines.append("|---|---:|")
            for path, count in top_dirs:
                lines.append(f"| `{path}` | {count} |")
    except subprocess.CalledProcessError:
        pass

    (OUT_DIR / "churn.md").write_text("\n".join(lines) + "\n")


def extract_msrv() -> str:
    cargo_toml = (ROOT / "Cargo.toml").read_text()
    match = re.search(r"rust-version\s*=\s*\"([^\"]+)\"", cargo_toml)
    return match.group(1) if match else "unknown"


def write_status_report(meta: dict[str, Any], members: set[str]) -> None:
    lines = [
        "# Repository Invariants Status",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- **MSRV**: `{extract_msrv()}`",
        f"- **Workspace members**: `{len(members)}`",
        f"- **Default members**: `{len(meta.get('workspace_default_members', []))}`",
        f"- **HEAD commit**: `{git_output(['rev-parse', '--short', 'HEAD'])}`",
        f"- **Current branch**: `{git_output(['rev-parse', '--abbrev-ref', 'HEAD'])}`",
        "",
        "## Gate Workflows",
        "",
    ]

    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    for wf in workflows:
        lines.append(f"- `{wf.name}`")

    lines.append("")
    lines.append("## Key generated instruments")
    lines.append("")
    lines.append("- `docs/instruments/deps.mmd`")
    lines.append("- `docs/instruments/public_api.md`")
    lines.append("- `docs/instruments/sbom.json`")
    lines.append("- `docs/instruments/churn.md`")

    (OUT_DIR / "status.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = cargo_metadata()
    packages, members = build_workspace_maps(meta)
    write_dependency_graph(meta, packages, members)
    write_public_api_snapshot(meta, packages, members)
    write_sbom(meta, packages, members)
    write_churn_report()
    write_status_report(meta, members)


if __name__ == "__main__":
    main()

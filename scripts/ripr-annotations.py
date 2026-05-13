#!/usr/bin/env python3
import json
import signal
from pathlib import Path

signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def escape_message(value):
    return (
        str(value)
        .replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def escape_property(value):
    return escape_message(value).replace(",", "%2C")


def repo_relative(path):
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def print_annotation(level, file, line, title, body):
    print(
        f"::{level} file={escape_property(file)},line={line},"
        f"title={escape_property(title)}::{escape_message(body)}"
    )


def emit_comment_annotations(data):
    for item in data.get("comments", []):
        file = item.get("path") or item.get("file")
        line = item.get("line")
        title = item.get("title") or "RIPR"
        body = item.get("body") or item.get("message") or ""

        if not file or not line:
            continue

        print_annotation("warning", repo_relative(file), line, title, body)


def emit_finding_annotations(data):
    for finding in data.get("findings", []):
        probe = finding.get("probe") or {}
        file = probe.get("file")
        line = probe.get("line")

        if not file or not line:
            continue

        classification = finding.get("classification") or "ripr"
        severity = finding.get("severity") or "note"
        confidence = finding.get("confidence")
        expression = probe.get("expression")
        next_step = (
            finding.get("suggested_next_action")
            or finding.get("recommended_next_step")
            or "Review RIPR evidence for this changed line."
        )

        body_parts = [str(next_step)]
        if expression:
            body_parts.append(f"Expression: {expression}")
        if confidence is not None:
            body_parts.append(f"Confidence: {confidence}")

        level = "warning" if severity == "warning" else "notice"
        print_annotation(
            level,
            repo_relative(file),
            line,
            f"RIPR {classification}",
            " | ".join(body_parts),
        )


path = Path("target/ripr/review/comments.json")
if not path.exists():
    raise SystemExit(0)

data = json.loads(path.read_text(encoding="utf-8"))

emit_comment_annotations(data)
emit_finding_annotations(data)

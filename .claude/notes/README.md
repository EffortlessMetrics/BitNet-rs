# Claude Notes — Zettelkasten-lite

Durable, atomic, linkable knowledge base for Claude agents working on bitnet-rs.

## Directory Layout

```
.claude/notes/
├── README.md          # This file — rules, templates, constraints
├── moc/               # Maps of Content (index files)
│   ├── 00-index.md    # Front door — links to all MOCs
│   ├── 10-ci.md       # CI/xtask policies
│   ├── 20-determinism.md  # Seed invariants, reproducibility
│   ├── 30-interop.md  # FFI, cross-platform, adapters
│   └── 40-security.md # Repo hygiene, no-blob policy
└── z/                 # Zettels (atomic notes)
    └── YYYYMMDDTHHMI-slug.md
```

## Zettel Template

```markdown
# Title

- **id**: YYYYMMDDTHHMI-slug
- **tags**: #tag1 #tag2
- **links**: [[00-index]]

## Context

Why does this note exist?

## Decision

What was decided or observed?

## Evidence

Supporting data, logs, links.

## Consequences

What follows from this?

## Follow-ups

- [ ] Action item
```

## Hard Rules

1. **One idea per zettel.** Never append to a growing log file.
2. **Naming**: `YYYYMMDDTHHMI-slug.md` (e.g., `20260301T1420-apple-silicon-team-process.md`).
3. **Max ~150 lines per file.** Split if longer.
4. **Link, don't duplicate.** Use `[[id]]` or relative paths (`../moc/10-ci.md`).
5. **MOCs are indexes, not content.** Keep them as link collections with brief scope descriptions.
6. **No binary files.** Text only.
7. **Prefer structured sections** (Context/Decision/Evidence/Consequences/Follow-ups).

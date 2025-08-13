# xtask Developer Guide

`xtask` collects reliable, reproducible workflows for this repo. It's designed
to be **idempotent**, **cache-aware**, and **CI-friendly**.

## `download-model`

Fetches a GGUF model from Hugging Face (or a mirror) safely.

### Flags

- `--id <org/repo>`: HF repo id (default: BitNet model)
- `--file <path>`: artifact path within the repo (default: ggml-model-i2_s.gguf)
- `--out <dir>`: output folder (default: `models/`)
- `--rev <ref>` (alias: `--ref`): pin to branch/tag/commit (`resolve/<rev>`)
- `--sha256 <HEX>`: verify integrity after download
- `--force`: redownload (also clears `.part` if present)
- `--no-progress` / `--quiet`: hide progress (TTY is auto-detected)
- `--verbose`: print decisions, statuses, and retry reasons
- `--base-url <url>`: override base, e.g. a corporate mirror

### How It Works (Safety Guarantees)

- **Resume**: uses `Range` and validates `Content-Range` alignment; if the
  server ignores/misaligns ranges, automatically restarts from 0.
- **Cache**: honors `ETag`/`Last-Modified` (HEAD or 1-byte probe), and
  performs a conditional **full GET** when `start==0` to avoid redownloading.
- **429/Retry-After**: respects `Retry-After` before `error_for_status()`.
- **Disk space**: checks **remaining** bytes (not total) with 50 MB headroom,
  re-checks if it needs to restart from 0 mid-flight.
- **Durability**: streams into `file.part` via buffered writes; `fsync`s file,
  renames atomically to the final path, and fsyncs the **parent directory**.
  `.etag` / `.lastmod` are persisted via atomic temp-file + `fsync` + rename.
- **Locks**: creates `file.lock` next to `file.part` (single writer).
- **Ctrl-C**: leaves `.part` for resume and removes `.lock`.
- **Security**: path traversal is blocked (we only use the leaf filename).
- **Proxies**: respects `HTTP[S]_PROXY` automatically via `reqwest`.

### File Layout

```
models/
  <id-with-dashes>/
    <file>             # final artifact
    <file>.part        # partial (if interrupted)
    <file>.lock        # exclusive download lock
    <file>.etag        # cache metadata (atomic write)
    <file>.lastmod     # cache metadata (atomic write)
```

### Troubleshooting

- **416** on ranged: we restart from 0 automatically.
- **206 but missing/invalid `Content-Range`**: treated as unsafe; restart from 0.
- **304** via HEAD/range/full GET: early return; optional SHA check still runs.
- **SHA mismatch**: file + metadata removed; exits with code `13`.
- **Rate limited (429)**: we back off per `Retry-After`; exits `12` if exhausted.

---

## `fetch-cpp`

Fetches & builds the C++ reference. On success, validates that
`~/.cache/bitnet_cpp/bitnet-llama-cli` exists.

```bash
cargo xtask fetch-cpp --tag b1-65-ggml [--force] [--clean]
```

---

## `crossval` / `full-crossval`

- `crossval` auto-discovers models in `models/` unless `--model` is provided.
- `full-crossval` runs: download → fetch-cpp → crossval, with helpful errors.

---

## `gen-fixtures`

Creates deterministic, GGUF-like metadata JSON + dummy weight blobs:

```bash
cargo xtask gen-fixtures --size tiny|small|medium --output test-fixtures/
```

---

## `clean-cache`

Interactive cleanup with size reporting; removes `target/`, `~/.cache/bitnet_cpp/`,
`crossval/fixtures/`, and `models/`.

```bash
cargo xtask clean-cache
```

---

## Exit Codes

| Code | Meaning                 |
|-----:|-------------------------|
| 0    | success                 |
| 10   | no space                |
| 11   | auth error              |
| 12   | rate limited            |
| 13   | hash mismatch           |
| 14   | network error           |
| 130  | interrupted (Ctrl-C)    |
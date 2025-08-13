# xtask Developer Guide

`xtask` collects reliable, reproducible workflows for this repo. It's designed
to be **idempotent**, **cache-aware**, and **CI-friendly**.

## Exit Codes

| Code | Meaning              | Common Causes |
|-----:|---------------------|---------------|
| 0    | Success             | All operations completed |
| 10   | No space            | Disk full, insufficient space |
| 11   | Auth error          | 401/403, invalid HF_TOKEN |
| 12   | Rate limited        | 429, too many requests |
| 13   | Hash mismatch       | SHA256 verification failed |
| 14   | Network error       | Connection issues, 404, timeouts |
| 130  | Interrupted         | Ctrl-C during operation |

## `download-model`

Fetches a GGUF model from Hugging Face (or a mirror) safely.

### Flags Summary

| Flag | Description | Default |
|------|-------------|---------|
| `--id <org/repo>` | HF repository identifier | `microsoft/bitnet-b1.58-2B-4T-gguf` |
| `--file <path>` | File path within repository | `ggml-model-i2_s.gguf` |
| `--out <dir>` | Output directory | `models/` |
| `--rev <ref>` | Pin to branch/tag/commit | `main` |
| `--sha256 <HEX>` | Verify SHA256 after download | None |
| `--force` | Force redownload | `false` |
| `--no-progress` | Hide progress bar | `false` (auto-detects TTY) |
| `--verbose` | Debug output | `false` |
| `--base-url <url>` | Alternative repository URL | `https://huggingface.co` |

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
- **Proxies**: automatically respects `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` environment variables.

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
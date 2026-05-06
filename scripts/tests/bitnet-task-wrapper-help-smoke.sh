#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_PATH="$ROOT/Cargo.toml"

mapfile -t WRAPPERS < <(
  while IFS= read -r wrapper; do
    # We only treat top-level scripts/*.sh wrappers as in-scope for this compatibility smoke.
    if rg -q '^[[:space:]]*(exec[[:space:]]+)?(cargo[[:space:]]+run[[:space:]].*bitnet-task([[:space:]]|$)|bitnet-task[[:space:]]+--)' "$wrapper"; then
      printf '%s\n' "$wrapper"
    fi
  done < <(find "$ROOT/scripts" -maxdepth 1 -type f -name '*.sh' | sort)
)

if [ "${#WRAPPERS[@]}" -eq 0 ]; then
  echo "no bitnet-task wrappers found"
  exit 1
fi

assert_normalized_wrapper() {
  local wrapper="$1"

  if ! rg -q '^exec cargo run --quiet --locked --manifest-path "\$ROOT/Cargo.toml" -p bitnet-task --' \
    "$wrapper"; then
    echo "wrapper is not normalized: ${wrapper#$ROOT/}" >&2
    echo "expected an exec line using the bitnet-task facade contract" >&2
    sed -n '1,40p' "$wrapper" >&2
    exit 1
  fi
}

assert_invocation() {
  local label="$1"
  local wrapper="$2"
  shift 2
  local wrapper_args=()
  local expected=()
  local parsing=wrapper_args
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--expected--" ]; then
      parsing=expected
      shift
      continue
    fi
    if [ "$parsing" = "wrapper_args" ]; then
      wrapper_args+=("$1")
    else
      expected+=("$1")
    fi
    shift
  done

  local tmpdir
  tmpdir="$(mktemp -d)"
  mkdir -p "$tmpdir/bin"

  cat >"$tmpdir/bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
{
  for arg in "$@"; do
    printf '%s\n' "$arg"
  done
} >"$BITNET_TASK_FAKE_CARGO_LOG"
EOF
  chmod +x "$tmpdir/bin/cargo"

  (
    cd "$tmpdir"
    PATH="$tmpdir/bin:$PATH" \
      BITNET_TASK_FAKE_CARGO_LOG="$tmpdir/cargo.log" \
      bash "$wrapper" "${wrapper_args[@]}"
  ) >/dev/null

  mapfile -t actual <"$tmpdir/cargo.log"
  if [ "${#actual[@]}" -ne "${#expected[@]}" ]; then
    echo "compatibility smoke failed for $label: expected ${#expected[@]} args, got ${#actual[@]}" >&2
    printf 'expected:\n' >&2
    printf '  %s\n' "${expected[@]}" >&2
    printf 'actual:\n' >&2
    printf '  %s\n' "${actual[@]}" >&2
    exit 1
  fi

  for i in "${!expected[@]}"; do
    if [ "${actual[$i]}" != "${expected[$i]}" ]; then
      echo "compatibility smoke failed for $label at arg[$i]" >&2
      echo "  expected: ${expected[$i]}" >&2
      echo "  actual:   ${actual[$i]}" >&2
      exit 1
    fi
  done

  rm -rf "$tmpdir"
}

echo "==> wrapper help smoke"
for wrapper in "${WRAPPERS[@]}"; do
  echo "==> ${wrapper#$ROOT/}"
  assert_normalized_wrapper "$wrapper"
  (
    cd /tmp
    bash "$wrapper" --help >/dev/null
  )
done

echo "==> wrapper compatibility smoke"
assert_invocation \
  "perf positional rewrite" \
  "$ROOT/scripts/perf_phase1_quant_probe.sh" \
  fixtures/model.gguf \
  fixtures/tokenizer.json \
  --sentinel \
  --expected-- \
  run \
  --quiet \
  --locked \
  --manifest-path \
  "$MANIFEST_PATH" \
  -p \
  bitnet-task \
  -- \
  perf-phase1-quant-probe \
  --model \
  fixtures/model.gguf \
  --tokenizer \
  fixtures/tokenizer.json \
  --sentinel

assert_invocation \
  "perf flag passthrough" \
  "$ROOT/scripts/perf_phase1_quant_probe.sh" \
  --model \
  flagged.gguf \
  --tokenizer \
  flagged.json \
  --sentinel \
  --expected-- \
  run \
  --quiet \
  --locked \
  --manifest-path \
  "$MANIFEST_PATH" \
  -p \
  bitnet-task \
  -- \
  perf-phase1-quant-probe \
  --model \
  flagged.gguf \
  --tokenizer \
  flagged.json \
  --sentinel

assert_invocation \
  "vendor default injection" \
  "$ROOT/scripts/vendor_ggml_quants.sh" \
  --expected-- \
  run \
  --quiet \
  --locked \
  --manifest-path \
  "$MANIFEST_PATH" \
  -p \
  bitnet-task \
  -- \
  vendor-ggml-quants \
  master

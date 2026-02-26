#!/usr/bin/env bash
# Receipt sanity validation for BitNet-rs inference receipts
set -euo pipefail

RECEIPT_FILE="${1:-ci/inference.json}"

if [[ ! -f "$RECEIPT_FILE" ]]; then
    echo "ERROR: Receipt file not found: $RECEIPT_FILE"
    exit 1
fi

echo "Validating receipt: $RECEIPT_FILE"

# Validate receipt structure and required fields
jq -e '
  .receipt.compute_path == "real" and
  (.receipt.backend == "cpu" or .receipt.backend == "gpu" or .receipt.backend == "cuda") and
  (.receipt.backend != "gpu" or (.receipt.kernels | length) > 0)
' "$RECEIPT_FILE" > /dev/null

if [[ $? -eq 0 ]]; then
    echo "✅ Receipt validation passed"

    # Display summary
    BACKEND=$(jq -r '.receipt.backend' "$RECEIPT_FILE")
    COMPUTE_PATH=$(jq -r '.receipt.compute_path' "$RECEIPT_FILE")
    echo "   Backend: $BACKEND"
    echo "   Compute Path: $COMPUTE_PATH"

    if [[ "$BACKEND" == "gpu" || "$BACKEND" == "cuda" ]]; then
        KERNEL_COUNT=$(jq '.receipt.kernels | length' "$RECEIPT_FILE")
        echo "   GPU Kernels: $KERNEL_COUNT"
    fi
else
    echo "❌ Receipt validation failed"
    exit 1
fi

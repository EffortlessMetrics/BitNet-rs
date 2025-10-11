#!/bin/bash
# Flake detection script for test_cross_crate_strict_mode_consistency

echo "=== Flake Detection Run - 10 iterations ==="
pass_count=0
fail_count=0

for i in {1..10}; do
    echo "Run $i:"
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS=1

    cargo test --workspace --no-default-features --features cpu test_cross_crate_strict_mode_consistency 2>&1 > /tmp/flake_test_$i.log

    if grep -q "test result: ok" /tmp/flake_test_$i.log && grep -q "1 passed" /tmp/flake_test_$i.log; then
        echo "  PASS"
        pass_count=$((pass_count + 1))
    else
        echo "  FAIL or FILTERED"
        fail_count=$((fail_count + 1))
        # Show relevant error
        grep -A 2 "test_cross_crate" /tmp/flake_test_$i.log | head -5
    fi
done

echo ""
echo "=== Summary ==="
echo "Passed: $pass_count/10"
echo "Failed: $fail_count/10"
echo "Reproduction rate: $((fail_count * 10))%"

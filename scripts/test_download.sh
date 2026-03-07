#!/bin/bash
set -ex

# Setup mock environment
mkdir -p tests/tmp/mock/model/resolve/main
echo "dummy model data" > tests/tmp/mock/model/resolve/main/model.gguf
echo "{}" > tests/tmp/mock/model/resolve/main/tokenizer.json

# Start python http server in background
cd tests/tmp
python3 -m http.server 8080 &
SERVER_PID=$!
cd ../..

# Wait for server to start
sleep 2

# Build xtask
cargo build -p xtask --no-default-features

# Test normal download from mock server
if ! ./target/debug/xtask download-model --base-url http://localhost:8080 --id mock/model --file model.gguf --out tests/tmp/downloaded --force; then
    echo "Download failed!"
    kill $SERVER_PID
    return 1
fi

# Verify download
if [ ! -f tests/tmp/downloaded/mock-model/model.gguf ]; then
    echo "Download failed!"
    kill $SERVER_PID
    return 1
fi

# Test offline mode (should succeed since file is in cache)
if ! ./target/debug/xtask download-model --offline --id mock/model --file model.gguf --out tests/tmp/downloaded; then
    echo "Offline failed!"
    kill $SERVER_PID
    return 1
fi

# Test offline mode with missing file (should fail)
if ./target/debug/xtask download-model --offline --id mock/model --file missing.gguf --out tests/tmp/downloaded; then
    echo "Offline mode should have failed for missing file!"
    kill $SERVER_PID
    return 1
fi

kill $SERVER_PID
echo "Tests passed!"

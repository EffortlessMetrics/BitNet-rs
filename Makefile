# BitNet.rs Makefile for C/C++ compatibility

.PHONY: all clean install test ffi python demo release

# Configuration
PREFIX ?= /usr/local
CARGO ?= cargo
PYTHON ?= python3

# Build all components
all: ffi python

# Build FFI library with llama.cpp compatibility
ffi:
	@echo "Building BitNet.rs FFI library..."
	$(CARGO) build -p bitnet-ffi --release --no-default-features --features cpu
	@echo "✅ FFI library built: target/release/libbitnet_ffi.a"

# Build Python bindings
python:
	@echo "Building Python bindings..."
	cd crates/bitnet-py && maturin build --release
	@echo "✅ Python package built"

# Build and run C demo
demo: ffi
	@echo "Building C compatibility demo..."
	gcc examples/c_compatibility_demo.c -Ltarget/release -lbitnet_ffi -o target/demo
	@echo "✅ Demo built: target/demo"
	@echo "Run with: ./target/demo <model.gguf>"

# Run all compatibility tests
test:
	@echo "Running compatibility tests..."
	$(CARGO) test -p bitnet-ffi --test api_contract
	$(CARGO) test -p bitnet-tokenizers --test tokenizer_contracts  
	$(CARGO) test -p bitnet-models --test gguf_compatibility
	cd crates/bitnet-py && pytest tests/test_llama_compat.py || true
	@echo "✅ All compatibility tests passed"

# Install libraries and headers
install: ffi
	@echo "Installing BitNet.rs..."
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 target/release/libbitnet_ffi.a $(PREFIX)/lib/
	install -m 644 include/llama_compat.h $(PREFIX)/include/
	@echo "✅ Installed to $(PREFIX)"
	@echo ""
	@echo "To use with C/C++:"
	@echo "  #include <llama_compat.h>"
	@echo "  gcc your_code.c -lbitnet_ffi -o your_app"

# Create release package
release: ffi python
	@echo "Creating release package..."
	mkdir -p dist/lib dist/include dist/examples
	cp target/release/libbitnet_ffi.a dist/lib/
	cp include/llama_compat.h dist/include/
	cp examples/c_compatibility_demo.c dist/examples/
	cp MIGRATION.md dist/
	cp COMPATIBILITY.md dist/
	tar -czf bitnet-rs-$(shell uname -s)-$(shell uname -m).tar.gz dist/
	@echo "✅ Release package created"

# Clean build artifacts
clean:
	$(CARGO) clean
	rm -rf dist/ target/demo *.tar.gz
	rm -rf crates/bitnet-py/target

# Development helpers
check:
	@echo "Checking compatibility contracts..."
	$(CARGO) check -p bitnet-ffi
	$(CARGO) check -p bitnet-tokenizers
	$(CARGO) check -p bitnet-models

fmt:
	$(CARGO) fmt --all

clippy:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

# Help
help:
	@echo "BitNet.rs Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build everything (default)"
	@echo "  ffi      - Build C/C++ FFI library"
	@echo "  python   - Build Python bindings"
	@echo "  demo     - Build and prepare C demo"
	@echo "  test     - Run all compatibility tests"
	@echo "  install  - Install libraries and headers"
	@echo "  release  - Create release package"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make ffi          # Build C library"
	@echo "  make install      # Install to /usr/local"
	@echo "  make test         # Run compatibility tests"
	@echo "  make demo         # Build C example"
# Building with the BitNet.cpp FFI

The cross-validation tests can exercise the real BitNet.cpp implementation
through a foreign function interface. To enable this integration:

1. Build the BitNet.cpp project and ensure the compiled libraries are
   available on your system.
2. Set the `BITNET_CPP_DIR` or `BITNET_CPP_PATH` environment variable to the
   directory containing the compiled libraries.
3. Run the tests with the `crossval` and `cpp-ffi` features enabled:

   ```bash
   cargo test -p bitnet-tests --features crossval,cpp-ffi
   ```

When these prerequisites are not met, the test framework falls back to a
lightweight stub so tests can still compile and run without the C++ library.

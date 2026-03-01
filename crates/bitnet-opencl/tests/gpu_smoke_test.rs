//! GPU smoke test for optional real hardware validation.

#[test]
#[ignore = "requires OpenCL runtime - run with --ignored on GPU machine"]
fn smoke_test_real_gpu() {
    // This test is a placeholder for real hardware validation.
    // When run on a machine with OpenCL runtime:
    // 1. Initialize OpenCL platform/device
    // 2. Create context and command queue
    // 3. Compile a trivial vector-add kernel
    // 4. Execute kernel and verify results
    //
    // Since we don't link to OpenCL by default, this test
    // documents the intended real-hardware validation path.
    eprintln!("smoke_test_real_gpu: would init OpenCL and run vector add");
    eprintln!("Skipped: no OpenCL runtime linked.");
}

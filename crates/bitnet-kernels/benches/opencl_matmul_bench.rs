//! Placeholder benchmark for OpenCL matmul operations.
//!
//! Requires a real OpenCL device; skips gracefully when unavailable.

use criterion::{Criterion, criterion_group, criterion_main};

fn opencl_matmul_benchmark(_c: &mut Criterion) {
    // TODO: Implement OpenCL matmul benchmarks when hardware is available
    eprintln!("OpenCL matmul benchmarks require an OpenCL-capable device — skipping");
}

criterion_group!(benches, opencl_matmul_benchmark);
criterion_main!(benches);

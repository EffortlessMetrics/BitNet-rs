use crate::error::RunnerError;
use crate::runner::{KernelRunner, matmul_workgroups};

/// WGSL shader for naive matrix multiplication.
///
/// Layout: A is M×K row-major, B is K×N row-major, C is M×N row-major.
const MATMUL_SHADER: &str = r#"
struct Dims {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn matmul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if row >= dims.M || col >= dims.N {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.K; i = i + 1u) {
        sum = sum + a[row * dims.K + i] * b[i * dims.N + col];
    }
    c[row * dims.N + col] = sum;
}
"#;

/// The workgroup size used in the matmul shader (must match WGSL).
const WORKGROUP_SIZE: u32 = 8;

/// Dimensions struct matching the WGSL `Dims` layout (16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

/// High-level GPU matrix multiplication interface.
pub struct MatmulRunner {
    runner: KernelRunner,
}

impl MatmulRunner {
    /// Create a new `MatmulRunner`, initializing the GPU device.
    pub async fn new() -> Result<Self, RunnerError> {
        Ok(Self { runner: KernelRunner::new().await? })
    }

    /// Perform GPU matrix multiplication: C = A × B.
    ///
    /// - `a`: M×K matrix in row-major order
    /// - `b`: K×N matrix in row-major order
    /// - Returns: M×N result matrix in row-major order
    pub async fn multiply(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, RunnerError> {
        let expected_a = (m * k) as usize;
        if a.len() != expected_a {
            return Err(RunnerError::InvalidDimensions {
                expected: expected_a,
                actual: a.len(),
                name: "matrix A",
            });
        }
        let expected_b = (k * n) as usize;
        if b.len() != expected_b {
            return Err(RunnerError::InvalidDimensions {
                expected: expected_b,
                actual: b.len(),
                name: "matrix B",
            });
        }

        let kernel = self.runner.compile_shader(MATMUL_SHADER, "matmul_main")?;

        let buf_a = self.runner.create_buffer_f32(a);
        let buf_b = self.runner.create_buffer_f32(b);
        let output_size = (m * n) as u64 * std::mem::size_of::<f32>() as u64;
        let buf_c = self.runner.create_output_buffer(output_size);

        let dims = Dims { m, n, k, _pad: 0 };
        let buf_dims = self.runner.create_uniform_buffer(bytemuck::bytes_of(&dims));

        let bind_group = self.runner.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &kernel.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_dims.as_entire_binding() },
            ],
        });

        let workgroups = matmul_workgroups(m, n, WORKGROUP_SIZE);
        self.runner.dispatch(&kernel, &bind_group, workgroups);

        self.runner.read_buffer_f32(&buf_c, (m * n) as usize).await
    }
}

/// CPU reference implementation of matrix multiplication for validation.
///
/// - `a`: M×K matrix in row-major order
/// - `b`: K×N matrix in row-major order
/// - Returns: M×N result matrix in row-major order
pub fn cpu_matmul(a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Vec<f32> {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    assert_eq!(a.len(), m * k, "matrix A size mismatch");
    assert_eq!(b.len(), k * n, "matrix B size mismatch");

    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- CPU reference tests (no GPU needed) ----

    #[test]
    fn cpu_matmul_2x2() {
        #[rustfmt::skip]
        let a = vec![1.0, 2.0,
                     3.0, 4.0];
        #[rustfmt::skip]
        let b = vec![5.0, 6.0,
                     7.0, 8.0];
        let c = cpu_matmul(&a, &b, 2, 2, 2);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn cpu_matmul_identity_2x2() {
        #[rustfmt::skip]
        let a = vec![3.0, 7.0,
                     5.0, 11.0];
        #[rustfmt::skip]
        let identity = vec![1.0, 0.0,
                            0.0, 1.0];
        let c = cpu_matmul(&a, &identity, 2, 2, 2);
        assert_eq!(c, vec![3.0, 7.0, 5.0, 11.0]);
    }

    #[test]
    fn cpu_matmul_identity_4x4() {
        #[rustfmt::skip]
        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        #[rustfmt::skip]
        let identity: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let c = cpu_matmul(&a, &identity, 4, 4, 4);
        assert_eq!(c, a);
    }

    #[test]
    fn cpu_matmul_non_square() {
        // A: 2×3, B: 3×2 → C: 2×2
        #[rustfmt::skip]
        let a = vec![1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0];
        #[rustfmt::skip]
        let b = vec![7.0,  8.0,
                     9.0,  10.0,
                     11.0, 12.0];
        let c = cpu_matmul(&a, &b, 2, 2, 3);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_matmul_1x1() {
        let c = cpu_matmul(&[3.0], &[7.0], 1, 1, 1);
        assert_eq!(c, vec![21.0]);
    }

    #[test]
    fn cpu_matmul_row_times_col() {
        // A: 1×3 row vector, B: 3×1 column vector → 1×1
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = cpu_matmul(&a, &b, 1, 1, 3);
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(c, vec![32.0]);
    }

    #[test]
    fn cpu_matmul_col_times_row() {
        // A: 3×1 column vector, B: 1×3 row vector → 3×3 outer product
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = cpu_matmul(&a, &b, 3, 3, 1);
        #[rustfmt::skip]
        let expected = vec![
            4.0, 5.0, 6.0,
            8.0, 10.0, 12.0,
            12.0, 15.0, 18.0,
        ];
        assert_eq!(c, expected);
    }

    #[test]
    fn cpu_matmul_zeros() {
        let a = vec![0.0; 4];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = cpu_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "matrix A size mismatch")]
    fn cpu_matmul_wrong_a_size() {
        cpu_matmul(&[1.0, 2.0], &[1.0; 4], 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "matrix B size mismatch")]
    fn cpu_matmul_wrong_b_size() {
        cpu_matmul(&[1.0; 4], &[1.0, 2.0], 2, 2, 2);
    }

    // ---- GPU tests (require runtime) ----

    #[test]
    #[ignore = "requires GPU runtime — run manually with --ignored"]
    fn gpu_matmul_2x2() {
        pollster::block_on(async {
            let runner = MatmulRunner::new().await.unwrap();
            #[rustfmt::skip]
            let a = vec![1.0, 2.0,
                         3.0, 4.0];
            #[rustfmt::skip]
            let b = vec![5.0, 6.0,
                         7.0, 8.0];
            let c = runner.multiply(&a, &b, 2, 2, 2).await.unwrap();
            let expected = cpu_matmul(&a, &b, 2, 2, 2);
            for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "mismatch at index {i}: got {got}, expected {want}"
                );
            }
        });
    }

    #[test]
    #[ignore = "requires GPU runtime — run manually with --ignored"]
    fn gpu_matmul_4x4() {
        pollster::block_on(async {
            let runner = MatmulRunner::new().await.unwrap();
            #[rustfmt::skip]
            let a: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ];
            #[rustfmt::skip]
            let b: Vec<f32> = vec![
                16.0, 15.0, 14.0, 13.0,
                12.0, 11.0, 10.0, 9.0,
                8.0, 7.0, 6.0, 5.0,
                4.0, 3.0, 2.0, 1.0,
            ];
            let c = runner.multiply(&a, &b, 4, 4, 4).await.unwrap();
            let expected = cpu_matmul(&a, &b, 4, 4, 4);
            for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "mismatch at index {i}: got {got}, expected {want}"
                );
            }
        });
    }

    #[test]
    #[ignore = "requires GPU runtime — run manually with --ignored"]
    fn gpu_matmul_non_square() {
        pollster::block_on(async {
            let runner = MatmulRunner::new().await.unwrap();
            // A: 2×3, B: 3×4 → C: 2×4
            #[rustfmt::skip]
            let a = vec![1.0, 2.0, 3.0,
                         4.0, 5.0, 6.0];
            #[rustfmt::skip]
            let b = vec![1.0, 2.0, 3.0, 4.0,
                         5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0];
            let c = runner.multiply(&a, &b, 2, 4, 3).await.unwrap();
            let expected = cpu_matmul(&a, &b, 2, 4, 3);
            for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "mismatch at index {i}: got {got}, expected {want}"
                );
            }
        });
    }

    #[test]
    #[ignore = "requires GPU runtime — run manually with --ignored"]
    fn gpu_matmul_identity() {
        pollster::block_on(async {
            let runner = MatmulRunner::new().await.unwrap();
            #[rustfmt::skip]
            let a = vec![3.0, 7.0,
                         5.0, 11.0];
            #[rustfmt::skip]
            let identity = vec![1.0, 0.0,
                                0.0, 1.0];
            let c = runner.multiply(&a, &identity, 2, 2, 2).await.unwrap();
            for (i, (got, want)) in c.iter().zip(a.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "mismatch at index {i}: got {got}, expected {want}"
                );
            }
        });
    }
}

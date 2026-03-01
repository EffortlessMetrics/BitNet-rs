#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug, Clone, Copy)]
enum FuzzKernelType {
    Matmul,
    RmsNorm,
    Attention,
    Qk256Gemv,
    Softmax,
    Elementwise,
}

#[derive(Arbitrary, Debug)]
struct KernelParamInput {
    kernel_type: FuzzKernelType,
    workgroup_x: u32,
    workgroup_y: u32,
    workgroup_z: u32,
    global_x: u64,
    global_y: u64,
    global_z: u64,
    shared_mem_bytes: u32,
    problem_m: u32,
    problem_n: u32,
    problem_k: u32,
    threads_per_block: u32,
    tile_m: u16,
    tile_n: u16,
    n_heads: u16,
    head_dim: u16,
    eps: f32,
    causal: bool,
    extra_params: Vec<u32>,
}

const MAX_THREADS_PER_BLOCK: u32 = 1024;
const MAX_SHARED_MEM: u32 = 48 * 1024; // 48 KB typical
const MAX_DISPATCH_DIM: u64 = 65535;

fn validate_launch_config(
    threads_per_block: u32,
    shared_mem: u32,
    grid: (u64, u64, u64),
) -> Result<(), &'static str> {
    if threads_per_block == 0 {
        return Err("zero threads per block");
    }
    if threads_per_block > MAX_THREADS_PER_BLOCK {
        return Err("exceeds max threads per block");
    }
    if shared_mem > MAX_SHARED_MEM {
        return Err("exceeds shared memory limit");
    }
    if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
        return Err("zero grid dimension");
    }
    if grid.0 > MAX_DISPATCH_DIM || grid.1 > MAX_DISPATCH_DIM || grid.2 > MAX_DISPATCH_DIM {
        return Err("grid dimension exceeds limit");
    }
    Ok(())
}

fn validate_tile_config(tile_m: u16, tile_n: u16, threads: u32) -> Result<(), &'static str> {
    if tile_m == 0 || tile_n == 0 {
        return Err("zero tile dimension");
    }
    let tile_area = (tile_m as u32).checked_mul(tile_n as u32).ok_or("tile overflow")?;
    // Tiles shouldn't exceed thread count (each thread handles >= 1 element)
    if threads > 0 && tile_area > threads.saturating_mul(64) {
        return Err("tile too large for thread count");
    }
    Ok(())
}

fn select_kernel_config(
    kernel: FuzzKernelType,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(u32, (u64, u64, u64), u32), &'static str> {
    if m == 0 || n == 0 {
        return Err("zero problem dimension");
    }
    match kernel {
        FuzzKernelType::Matmul => {
            if k == 0 {
                return Err("zero inner dimension for matmul");
            }
            let threads = 256u32;
            let grid_x = ((m + 31) / 32) as u64;
            let grid_y = ((n + 31) / 32) as u64;
            if grid_x > MAX_DISPATCH_DIM || grid_y > MAX_DISPATCH_DIM {
                return Err("matmul grid too large");
            }
            Ok((threads, (grid_x, grid_y, 1), 0))
        }
        FuzzKernelType::RmsNorm => {
            let threads = (n as u32).min(MAX_THREADS_PER_BLOCK).max(1);
            let grid_x = m as u64;
            if grid_x > MAX_DISPATCH_DIM {
                return Err("rmsnorm grid too large");
            }
            Ok((threads, (grid_x, 1, 1), 0))
        }
        FuzzKernelType::Attention => {
            if k == 0 {
                return Err("zero head_dim");
            }
            let threads = 256u32;
            let grid_x = m as u64; // n_heads * seq_len_q
            let grid_y = 1u64;
            if grid_x > MAX_DISPATCH_DIM {
                return Err("attention grid too large");
            }
            Ok((threads, (grid_x, grid_y, 1), (k as u32).saturating_mul(4)))
        }
        FuzzKernelType::Qk256Gemv => {
            if k == 0 || k % 256 != 0 {
                return Err("QK256 k must be positive multiple of 256");
            }
            let threads = 128u32;
            let grid_x = ((n + 3) / 4) as u64;
            let grid_y = m as u64;
            if grid_x > MAX_DISPATCH_DIM || grid_y > MAX_DISPATCH_DIM {
                return Err("qk256 grid too large");
            }
            Ok((threads, (grid_x, grid_y, 1), 256 * 4))
        }
        FuzzKernelType::Softmax => {
            let threads = (n as u32).min(MAX_THREADS_PER_BLOCK).max(1);
            let grid_x = m as u64;
            if grid_x > MAX_DISPATCH_DIM {
                return Err("softmax grid too large");
            }
            Ok((threads, (grid_x, 1, 1), (n as u32).saturating_mul(4)))
        }
        FuzzKernelType::Elementwise => {
            let total = (m as u64).saturating_mul(n as u64);
            let threads = 256u32;
            let blocks = total.saturating_add(255) / 256;
            if blocks > MAX_DISPATCH_DIM {
                return Err("elementwise grid too large");
            }
            Ok((threads, (blocks.max(1), 1, 1), 0))
        }
    }
}

fn validate_eps(eps: f32) -> Result<f32, &'static str> {
    if eps.is_nan() || eps.is_infinite() {
        return Err("non-finite epsilon");
    }
    if eps <= 0.0 {
        return Err("epsilon must be positive");
    }
    if eps > 1.0 {
        return Err("epsilon too large");
    }
    Ok(eps)
}

fuzz_target!(|input: KernelParamInput| {
    let m = input.problem_m as usize;
    let n = input.problem_n as usize;
    let k = input.problem_k as usize;

    // Validate workgroup dimensions — must not panic
    let wg = (input.workgroup_x, input.workgroup_y, input.workgroup_z);
    let _ = validate_launch_config(
        wg.0.saturating_mul(wg.1).saturating_mul(wg.2).max(1),
        input.shared_mem_bytes,
        (input.global_x, input.global_y, input.global_z),
    );

    // Validate tile config — must not panic
    let _ = validate_tile_config(input.tile_m, input.tile_n, input.threads_per_block);

    // Kernel selection with arbitrary problem sizes — must not panic
    let _ = select_kernel_config(input.kernel_type, m, n, k);

    // Attention-style head/dim validation — must not panic
    let n_heads = input.n_heads as usize;
    let head_dim = input.head_dim as usize;
    if n_heads > 0 && head_dim > 0 && m > 0 && n > 0 {
        let _ = select_kernel_config(
            FuzzKernelType::Attention,
            n_heads.saturating_mul(m),
            n,
            head_dim,
        );
    }

    // Epsilon validation — must not panic
    let _ = validate_eps(input.eps);

    // Causal mask bounds check — must not panic
    if input.causal && m > 0 && n > 0 {
        let mask_elems = (m as u64).saturating_mul(n as u64);
        let _ = mask_elems;
    }

    // Iterate extra params with capped iterations — must not panic
    let mut checksum: u64 = 0;
    for &p in input.extra_params.iter().take(256) {
        checksum = checksum.wrapping_add(p as u64);
    }
    let _ = checksum;
});

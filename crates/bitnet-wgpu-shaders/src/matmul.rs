//! Matrix multiplication WGSL compute shaders.
//!
//! Provides naive, tiled (shared-memory), and matrix-vector variants.

/// Naive matrix multiplication: C = A × B (M×K · K×N → M×N).
///
/// Workgroup size: [16, 16, 1]. Each thread computes one output element.
pub const MATMUL_NAIVE_SRC: &str = r"
struct MatmulParams {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

@compute @workgroup_size(16, 16, 1)
fn matmul_naive(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;

    if row >= params.M || col >= params.N {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.K; i = i + 1u) {
        sum = sum + a[row * params.K + i] * b[i * params.N + col];
    }
    result[row * params.N + col] = sum;
}
";

/// Tiled matrix multiplication using workgroup shared memory.
///
/// `TILE_SIZE` = 16. Workgroup size: [16, 16, 1].
/// Uses `workgroupBarrier()` to synchronise shared-memory loads.
pub const MATMUL_TILED_SRC: &str = r"
struct MatmulParams {
    M: u32,
    N: u32,
    K: u32,
}

const TILE_SIZE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.y * TILE_SIZE + lid.y;
    let col = wgid.x * TILE_SIZE + lid.x;
    let local_idx = lid.y * TILE_SIZE + lid.x;

    var sum: f32 = 0.0;
    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile from A
        let a_col = t * TILE_SIZE + lid.x;
        if row < params.M && a_col < params.K {
            tile_a[local_idx] = a[row * params.K + a_col];
        } else {
            tile_a[local_idx] = 0.0;
        }

        // Load tile from B
        let b_row = t * TILE_SIZE + lid.y;
        if b_row < params.K && col < params.N {
            tile_b[local_idx] = b[b_row * params.N + col];
        } else {
            tile_b[local_idx] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[lid.y * TILE_SIZE + i] * tile_b[i * TILE_SIZE + lid.x];
        }

        workgroupBarrier();
    }

    if row < params.M && col < params.N {
        result[row * params.N + col] = sum;
    }
}
";

/// Matrix-vector multiplication: y = A × x (M×K · K → M).
///
/// Workgroup size: [256, 1, 1]. Each thread computes one output row.
pub const MATMUL_VEC_SRC: &str = r"
struct MatmulParams {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

@compute @workgroup_size(256, 1, 1)
fn matmul_vec(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let row = gid.x;
    if row >= params.M {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.K; i = i + 1u) {
        sum = sum + a[row * params.K + i] * b[i];
    }
    result[row] = sum;
}
";

// Matmul kernel (kernels/ copy matching shaders/matmul.wgsl)
// C = A × B — A is M×K, B is K×N, C is M×N

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params { m: u32, n: u32, k: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) { return; }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        sum = sum + a[row * params.k + i] * b[i * params.n + col];
    }
    result[row * params.n + col] = sum;
// Matrix multiplication compute shader for WebGPU inference.
// Computes C = A * B where A is (M×K), B is (K×N), C is (M×N).
// Uses tiled approach with 16×16 workgroup size for coalesced memory access.

struct Params {
    m: u32,
    n: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>; // TILE_SIZE * TILE_SIZE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum: f32 = 0.0;
    let num_tiles = (params.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile from A
        let a_col = t * TILE_SIZE + local_col;
        if (row < params.m && a_col < params.k) {
            tile_a[local_row * TILE_SIZE + local_col] = a[row * params.k + a_col];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile from B
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < params.k && col < params.n) {
            tile_b[local_row * TILE_SIZE + local_col] = b[b_row * params.n + col];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        // Accumulate partial dot product from this tile
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + i]
                      * tile_b[i * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < params.m && col < params.n) {
        c[row * params.n + col] = sum;
    }
}

//! Element-wise WGSL compute shaders.
//!
//! All kernels use workgroup size [256, 1, 1] and operate on flat `array<f32>`
//! storage buffers with a uniform `length` parameter for bounds checking.

/// Element-wise addition: result[i] = a[i] + b[i].
pub const ADD_SRC: &str = r"
struct Params { length: u32, }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.length { return; }
    result[idx] = a[idx] + b[idx];
}
";

/// Element-wise multiplication: result[i] = a[i] * b[i].
pub const MUL_SRC: &str = r"
struct Params { length: u32, }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.length { return; }
    result[idx] = a[idx] * b[idx];
}
";

/// Scalar multiplication: result[i] = a[i] * scalar.
pub const SCALE_SRC: &str = r"
struct Params {
    length: u32,
    scalar: f32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.length { return; }
    result[idx] = a[idx] * params.scalar;
}
";

/// `ReLU` activation: result[i] = max(0, a[i]).
pub const RELU_SRC: &str = r"
struct Params { length: u32, }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.length { return; }
    result[idx] = max(a[idx], 0.0);
}
";

/// GELU activation (tanh approximation):
/// result[i] = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub const GELU_SRC: &str = r"
struct Params { length: u32, }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

@compute @workgroup_size(256, 1, 1)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.length { return; }
    let x = a[idx];
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    result[idx] = 0.5 * x * (1.0 + tanh(inner));
}
";

/// `SiLU` (Swish) activation: result[i] = x / (1 + exp(-x)).
pub const SILU_SRC: &str = r"
struct Params { length: u32, }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.length { return; }
    let x = a[idx];
    result[idx] = x / (1.0 + exp(-x));
}
";

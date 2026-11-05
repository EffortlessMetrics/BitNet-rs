//! Gradient accumulation buffer for fine-tuning support (OpenCL / CPU reference).
//!
//! Provides gradient accumulation over multiple micro-batches, global-norm and
//! value clipping, mixed-precision FP16↔FP32 conversion, dynamic loss scaling,
//! checkpoint save/load for resumable training, and statistics computation.
//! All operations have CPU reference implementations so the module compiles and
//! tests without an actual OpenCL runtime.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// OpenCL kernel source (embedded, not compiled without a runtime)
// ---------------------------------------------------------------------------

/// OpenCL kernel source for gradient accumulation operations.
pub const GRAD_ACCUM_CL: &str = r#"
// ---- gradient accumulation kernels ----

__kernel void grad_accumulate(
    __global float* accum,
    __global const float* grads,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        accum[gid] += grads[gid];
    }
}

__kernel void grad_average(
    __global float* accum,
    const float inv_steps,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        accum[gid] *= inv_steps;
    }
}

__kernel void grad_zero(
    __global float* accum,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        accum[gid] = 0.0f;
    }
}

// ---- clipping kernels ----

__kernel void grad_compute_sq_sum(
    __global const float* data,
    __global float* partial,
    __local float* scratch,
    const uint n)
{
    uint lid = get_local_id(0);
    uint gid = get_global_id(0);
    float val = (gid < n) ? data[gid] : 0.0f;
    scratch[lid] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = get_local_size(0) >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[get_group_id(0)] = scratch[0];
}

__kernel void grad_clip_by_norm(
    __global float* data,
    const float scale,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        data[gid] *= scale;
    }
}

__kernel void grad_clip_by_value(
    __global float* data,
    const float min_val,
    const float max_val,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        data[gid] = clamp(data[gid], min_val, max_val);
    }
}

// ---- mixed-precision kernels ----

__kernel void fp32_to_fp16(
    __global const float* src,
    __global half* dst,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        dst[gid] = convert_half(src[gid]);
    }
}

__kernel void fp16_to_fp32(
    __global const half* src,
    __global float* dst,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        dst[gid] = convert_float(src[gid]);
    }
}

// ---- loss-scaling kernels ----

__kernel void grad_scale(
    __global float* data,
    const float factor,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        data[gid] *= factor;
    }
}

__kernel void grad_check_overflow(
    __global const float* data,
    __global uint* overflow_flag,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        float v = data[gid];
        if (isinf(v) || isnan(v)) {
            atomic_or(overflow_flag, 1u);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// GradBuffer
// ---------------------------------------------------------------------------

/// Accumulation buffer for gradients backed by `f32` storage.
#[derive(Debug, Clone)]
pub struct GradBuffer {
    data: Vec<f32>,
    name: String,
}

impl GradBuffer {
    /// Create a new zero-initialized gradient buffer of the given size.
    pub fn new(size: usize, name: impl Into<String>) -> Self {
        Self { data: vec![0.0; size], name: name.into() }
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Name / label of the buffer.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Read-only access to the underlying data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable access to the underlying data.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Set all elements to zero.
    pub fn zero(&mut self) {
        self.data.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Element-wise accumulate `grads` into this buffer (CPU reference).
    pub fn accumulate(&mut self, grads: &[f32]) {
        assert_eq!(self.data.len(), grads.len(), "gradient size mismatch");
        for (a, g) in self.data.iter_mut().zip(grads.iter()) {
            *a += g;
        }
    }

    /// Scale every element by `factor`.
    pub fn scale(&mut self, factor: f32) {
        self.data.iter_mut().for_each(|v| *v *= factor);
    }

    /// Compute the L2 norm of the buffer.
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Check whether any element is NaN or Inf.
    pub fn has_overflow(&self) -> bool {
        self.data.iter().any(|v| v.is_nan() || v.is_infinite())
    }
}

impl fmt::Display for GradBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GradBuffer({}, len={})", self.name, self.data.len())
    }
}

// ---------------------------------------------------------------------------
// GradAccumulator
// ---------------------------------------------------------------------------

/// Accumulates gradients over N micro-batches, then averages.
#[derive(Debug)]
pub struct GradAccumulator {
    buffers: HashMap<String, GradBuffer>,
    accum_steps: usize,
    current_step: usize,
}

impl GradAccumulator {
    /// Create a new accumulator that averages over `accum_steps` micro-batches.
    pub fn new(accum_steps: usize) -> Self {
        assert!(accum_steps > 0, "accum_steps must be > 0");
        Self { buffers: HashMap::new(), accum_steps, current_step: 0 }
    }

    /// Register a named gradient buffer of the given size.
    pub fn register_buffer(&mut self, name: impl Into<String>, size: usize) {
        let name = name.into();
        self.buffers.insert(name.clone(), GradBuffer::new(size, name));
    }

    /// Number of registered buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Configured accumulation steps.
    pub fn accum_steps(&self) -> usize {
        self.accum_steps
    }

    /// Current micro-batch step within the accumulation window.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Accumulate a set of named gradients for one micro-batch.
    ///
    /// Returns `true` when the accumulation window is complete (i.e. after
    /// `accum_steps` calls) and the caller should apply the averaged
    /// gradients.
    pub fn accumulate(&mut self, gradients: &HashMap<String, Vec<f32>>) -> bool {
        for (name, grads) in gradients {
            if let Some(buf) = self.buffers.get_mut(name) {
                buf.accumulate(grads);
            }
        }
        self.current_step += 1;
        self.current_step >= self.accum_steps
    }

    /// Average all buffers by `1/accum_steps` and reset the step counter.
    /// Call this when [`accumulate`](Self::accumulate) returns `true`.
    pub fn average_and_reset(&mut self) {
        let inv = 1.0 / self.accum_steps as f32;
        for buf in self.buffers.values_mut() {
            buf.scale(inv);
        }
        self.current_step = 0;
    }

    /// Zero all buffers without changing the step counter.
    pub fn zero_grads(&mut self) {
        for buf in self.buffers.values_mut() {
            buf.zero();
        }
    }

    /// Read-only access to a named buffer.
    pub fn buffer(&self, name: &str) -> Option<&GradBuffer> {
        self.buffers.get(name)
    }

    /// Mutable access to a named buffer.
    pub fn buffer_mut(&mut self, name: &str) -> Option<&mut GradBuffer> {
        self.buffers.get_mut(name)
    }

    /// Iterate over all buffers.
    pub fn buffers(&self) -> impl Iterator<Item = (&str, &GradBuffer)> {
        self.buffers.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Whether an accumulation window is in progress.
    pub fn is_accumulating(&self) -> bool {
        self.current_step > 0 && self.current_step < self.accum_steps
    }
}

// ---------------------------------------------------------------------------
// GradClipping
// ---------------------------------------------------------------------------

/// Method used to clip gradients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipMethod {
    /// Clip by global L2 norm: if ‖g‖ > max_norm, scale g by max_norm/‖g‖.
    GlobalNorm(f32),
    /// Clip each element to [−value, value].
    Value(f32),
    /// No clipping.
    None,
}

impl fmt::Display for ClipMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GlobalNorm(n) => write!(f, "global_norm({n})"),
            Self::Value(v) => write!(f, "value({v})"),
            Self::None => write!(f, "none"),
        }
    }
}

/// Gradient clipping utilities (CPU reference).
pub struct GradClipping;

impl GradClipping {
    /// Clip gradients by global L2 norm across all provided buffers.
    ///
    /// Returns the original global norm before clipping.
    pub fn clip_by_global_norm(buffers: &mut [&mut GradBuffer], max_norm: f32) -> f32 {
        let total_sq: f32 = buffers.iter().flat_map(|b| b.data().iter()).map(|v| v * v).sum();
        let global_norm = total_sq.sqrt();
        if global_norm > max_norm && global_norm > 0.0 {
            let scale = max_norm / global_norm;
            for buf in buffers.iter_mut() {
                buf.scale(scale);
            }
        }
        global_norm
    }

    /// Clip every element in a single buffer by global norm.
    pub fn clip_buffer_by_norm(buf: &mut GradBuffer, max_norm: f32) -> f32 {
        let norm = buf.l2_norm();
        if norm > max_norm && norm > 0.0 {
            buf.scale(max_norm / norm);
        }
        norm
    }

    /// Clamp every element to [−value, value].
    pub fn clip_by_value(buf: &mut GradBuffer, value: f32) {
        for v in buf.data_mut().iter_mut() {
            *v = v.clamp(-value, value);
        }
    }

    /// Apply a [`ClipMethod`] to a single buffer. Returns the pre-clip norm
    /// (meaningful for norm-based clipping; 0.0 for value / none).
    pub fn apply(buf: &mut GradBuffer, method: ClipMethod) -> f32 {
        match method {
            ClipMethod::GlobalNorm(max) => Self::clip_buffer_by_norm(buf, max),
            ClipMethod::Value(v) => {
                Self::clip_by_value(buf, v);
                0.0
            }
            ClipMethod::None => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// MixedPrecisionGrad — FP16 ↔ FP32 helpers (software, no `half` crate dep)
// ---------------------------------------------------------------------------

/// Convert f32 to f16 bits (IEEE 754 half-precision).
#[inline]
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    if exponent == 0xFF {
        // Inf / NaN
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }
    let exp = exponent - 127 + 15;
    if exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow → Inf
    }
    if exp <= 0 {
        return sign as u16; // underflow → zero (flush to zero)
    }
    (sign | ((exp as u32) << 10) | (mantissa >> 13)) as u16
}

/// Convert f16 bits to f32.
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x03FF) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // ±0
        }
        // subnormal
        let mut m = mantissa;
        let mut e: i32 = 1;
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        let exp = ((127 - 15 + e) as u32) << 23;
        let mant = (m & 0x03FF) << 13;
        return f32::from_bits(sign | exp | mant);
    }
    if exponent == 31 {
        let mant = if mantissa != 0 { 0x007F_FFFF } else { 0 };
        return f32::from_bits(sign | 0x7F80_0000 | mant);
    }
    let exp = ((exponent as i32 - 15 + 127) as u32) << 23;
    let mant = mantissa << 13;
    f32::from_bits(sign | exp | mant)
}

/// Simulates FP16 by rounding an f32 through half-precision.
#[inline]
fn f32_to_f16_to_f32(v: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(v))
}

/// Mixed-precision gradient utilities.
///
/// The workflow is: compute in FP16, accumulate in FP32, apply in FP16.
pub struct MixedPrecisionGrad;

impl MixedPrecisionGrad {
    /// Convert an FP32 slice to FP16 (returned as u16 words).
    pub fn fp32_to_fp16(src: &[f32]) -> Vec<u16> {
        src.iter().map(|&v| f32_to_f16_bits(v)).collect()
    }

    /// Convert FP16 words back to FP32.
    pub fn fp16_to_fp32(src: &[u16]) -> Vec<f32> {
        src.iter().map(|&bits| f16_bits_to_f32(bits)).collect()
    }

    /// Roundtrip an FP32 slice through FP16 and back. Useful for simulating
    /// FP16 quantisation error.
    pub fn roundtrip(data: &[f32]) -> Vec<f32> {
        data.iter().map(|&v| f32_to_f16_to_f32(v)).collect()
    }

    /// Accumulate FP16 gradients into an FP32 accumulation buffer.
    pub fn accumulate_fp16_into_fp32(accum: &mut [f32], fp16_grads: &[u16]) {
        assert_eq!(accum.len(), fp16_grads.len(), "size mismatch");
        for (a, &bits) in accum.iter_mut().zip(fp16_grads.iter()) {
            *a += f16_bits_to_f32(bits);
        }
    }

    /// Convert an FP32 buffer to FP16, apply element-wise to a model weight
    /// buffer (simulated as f32 after roundtrip).
    pub fn apply_as_fp16(weights: &mut [f32], grads_fp32: &[f32], lr: f32) {
        assert_eq!(weights.len(), grads_fp32.len(), "size mismatch");
        for (w, &g) in weights.iter_mut().zip(grads_fp32.iter()) {
            let g16 = f32_to_f16_to_f32(g);
            *w -= lr * g16;
        }
    }

    /// Check if any FP16 value overflows (> 65504 or < −65504 in f32).
    pub fn would_overflow_fp16(data: &[f32]) -> bool {
        const FP16_MAX: f32 = 65504.0;
        data.iter().any(|v| v.abs() > FP16_MAX || v.is_nan() || v.is_infinite())
    }
}

// ---------------------------------------------------------------------------
// GradScaler — dynamic loss scaling for mixed precision
// ---------------------------------------------------------------------------

/// Dynamic loss scaler for mixed-precision training.
///
/// Scales the loss up before the backward pass so that small FP16 gradients
/// don't underflow, then unscales the gradients before the optimiser step.
/// If an overflow is detected during unscale, the step is skipped and the
/// scale factor is reduced.
#[derive(Debug, Clone)]
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_last_overflow: usize,
    overflow_count: u64,
    enabled: bool,
}

impl GradScaler {
    /// Create a new scaler with default hyper-parameters.
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_last_overflow: 0,
            overflow_count: 0,
            enabled: true,
        }
    }

    /// Create a scaler with custom growth/backoff parameters.
    pub fn with_params(
        initial_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
    ) -> Self {
        Self {
            scale: initial_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            steps_since_last_overflow: 0,
            overflow_count: 0,
            enabled: true,
        }
    }

    /// Current scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Whether the scaler is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Total number of overflow events observed so far.
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Scale a loss value before the backward pass.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        if self.enabled { loss * self.scale } else { loss }
    }

    /// Un-scale gradients after the backward pass.
    /// Returns `true` if the gradients are valid (no overflow).
    pub fn unscale(&self, buf: &mut GradBuffer) -> bool {
        if !self.enabled {
            return true;
        }
        let inv_scale = 1.0 / self.scale;
        buf.scale(inv_scale);
        !buf.has_overflow()
    }

    /// Update the scale factor after a step.
    ///
    /// `found_overflow` should be `true` if any buffer had overflow during
    /// [`unscale`](Self::unscale).
    pub fn update(&mut self, found_overflow: bool) {
        if !self.enabled {
            return;
        }
        if found_overflow {
            self.scale *= self.backoff_factor;
            self.steps_since_last_overflow = 0;
            self.overflow_count += 1;
        } else {
            self.steps_since_last_overflow += 1;
            if self.steps_since_last_overflow >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_last_overflow = 0;
            }
        }
    }

    /// Disable the scaler (passthrough mode).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable the scaler.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

impl fmt::Display for GradScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradScaler(scale={}, overflows={}, enabled={})",
            self.scale, self.overflow_count, self.enabled,
        )
    }
}

// ---------------------------------------------------------------------------
// GradCheckpointer — save / load gradient state
// ---------------------------------------------------------------------------

/// Serialisable snapshot of gradient state for resumable training.
#[derive(Debug, Clone, PartialEq)]
pub struct GradCheckpoint {
    /// Buffer name → gradient data.
    pub buffers: HashMap<String, Vec<f32>>,
    /// Current micro-batch step in the accumulation window.
    pub current_step: usize,
    /// Accumulation steps configured.
    pub accum_steps: usize,
    /// Loss-scaler state (scale, overflow_count).
    pub scaler_state: Option<(f32, u64)>,
}

/// Save / restore gradient state to a simple in-memory representation.
pub struct GradCheckpointer;

impl GradCheckpointer {
    /// Snapshot the current accumulator (and optionally scaler) state.
    pub fn save(accumulator: &GradAccumulator, scaler: Option<&GradScaler>) -> GradCheckpoint {
        let buffers = accumulator
            .buffers
            .iter()
            .map(|(name, buf)| (name.clone(), buf.data().to_vec()))
            .collect();
        GradCheckpoint {
            buffers,
            current_step: accumulator.current_step,
            accum_steps: accumulator.accum_steps,
            scaler_state: scaler.map(|s| (s.scale, s.overflow_count)),
        }
    }

    /// Restore accumulator (and optionally scaler) from a checkpoint.
    pub fn load(
        checkpoint: &GradCheckpoint,
        accumulator: &mut GradAccumulator,
        scaler: Option<&mut GradScaler>,
    ) {
        accumulator.accum_steps = checkpoint.accum_steps;
        accumulator.current_step = checkpoint.current_step;
        for (name, data) in &checkpoint.buffers {
            if let Some(buf) = accumulator.buffers.get_mut(name) {
                assert_eq!(buf.len(), data.len(), "checkpoint buffer size mismatch for '{name}'");
                buf.data_mut().copy_from_slice(data);
            } else {
                let mut buf = GradBuffer::new(data.len(), name.clone());
                buf.data_mut().copy_from_slice(data);
                accumulator.buffers.insert(name.clone(), buf);
            }
        }
        if let Some((saved_scale, saved_overflows)) = checkpoint.scaler_state
            && let Some(s) = scaler
        {
            s.scale = saved_scale;
            s.overflow_count = saved_overflows;
        }
    }

    /// Serialise a checkpoint to bytes (simple format: header + f32 data).
    pub fn to_bytes(checkpoint: &GradCheckpoint) -> Vec<u8> {
        let mut out = Vec::new();
        // Magic + version
        out.extend_from_slice(b"GCKP");
        out.extend_from_slice(&1u32.to_le_bytes());
        // accum_steps, current_step
        out.extend_from_slice(&(checkpoint.accum_steps as u64).to_le_bytes());
        out.extend_from_slice(&(checkpoint.current_step as u64).to_le_bytes());
        // scaler state
        let has_scaler = checkpoint.scaler_state.is_some();
        out.push(u8::from(has_scaler));
        if let Some((scale, overflows)) = checkpoint.scaler_state {
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&overflows.to_le_bytes());
        }
        // buffer count
        out.extend_from_slice(&(checkpoint.buffers.len() as u64).to_le_bytes());
        // Deterministic order
        let mut names: Vec<&String> = checkpoint.buffers.keys().collect();
        names.sort();
        for name in names {
            let data = &checkpoint.buffers[name];
            let name_bytes = name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&(data.len() as u64).to_le_bytes());
            for &v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Deserialise a checkpoint from bytes produced by [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Option<GradCheckpoint> {
        if bytes.len() < 4 + 4 + 8 + 8 + 1 {
            return None;
        }
        let mut pos = 0usize;

        let read_u32 = |p: &mut usize, b: &[u8]| -> Option<u32> {
            if *p + 4 > b.len() {
                return None;
            }
            let v = u32::from_le_bytes(b[*p..*p + 4].try_into().ok()?);
            *p += 4;
            Some(v)
        };
        let read_u64 = |p: &mut usize, b: &[u8]| -> Option<u64> {
            if *p + 8 > b.len() {
                return None;
            }
            let v = u64::from_le_bytes(b[*p..*p + 8].try_into().ok()?);
            *p += 8;
            Some(v)
        };
        let read_f32 = |p: &mut usize, b: &[u8]| -> Option<f32> {
            if *p + 4 > b.len() {
                return None;
            }
            let v = f32::from_le_bytes(b[*p..*p + 4].try_into().ok()?);
            *p += 4;
            Some(v)
        };

        // Magic
        if &bytes[pos..pos + 4] != b"GCKP" {
            return None;
        }
        pos += 4;
        // Version
        let version = read_u32(&mut pos, bytes)?;
        if version != 1 {
            return None;
        }
        let accum_steps = read_u64(&mut pos, bytes)? as usize;
        let current_step = read_u64(&mut pos, bytes)? as usize;

        if pos >= bytes.len() {
            return None;
        }
        let has_scaler = bytes[pos] != 0;
        pos += 1;
        let scaler_state = if has_scaler {
            let scale = read_f32(&mut pos, bytes)?;
            let overflows = read_u64(&mut pos, bytes)?;
            Some((scale, overflows))
        } else {
            None
        };

        let buf_count = read_u64(&mut pos, bytes)? as usize;
        let mut buffers = HashMap::with_capacity(buf_count);
        for _ in 0..buf_count {
            let name_len = read_u32(&mut pos, bytes)? as usize;
            if pos + name_len > bytes.len() {
                return None;
            }
            let name = String::from_utf8(bytes[pos..pos + name_len].to_vec()).ok()?;
            pos += name_len;
            let data_len = read_u64(&mut pos, bytes)? as usize;
            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                data.push(read_f32(&mut pos, bytes)?);
            }
            buffers.insert(name, data);
        }

        Some(GradCheckpoint { buffers, current_step, accum_steps, scaler_state })
    }
}

// ---------------------------------------------------------------------------
// GradStats
// ---------------------------------------------------------------------------

/// Statistics computed over a gradient buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct GradStats {
    /// L2 (Euclidean) norm.
    pub l2_norm: f32,
    /// L1 norm (sum of absolute values).
    pub l1_norm: f32,
    /// Maximum absolute value.
    pub max_abs: f32,
    /// Minimum absolute value.
    pub min_abs: f32,
    /// Mean value.
    pub mean: f32,
    /// Fraction of elements that are exactly zero.
    pub sparsity: f32,
    /// Number of elements that are NaN or Inf.
    pub overflow_count: usize,
    /// Total number of elements.
    pub count: usize,
}

impl GradStats {
    /// Compute statistics over a [`GradBuffer`].
    pub fn from_buffer(buf: &GradBuffer) -> Self {
        Self::from_slice(buf.data())
    }

    /// Compute statistics over a raw f32 slice.
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                l2_norm: 0.0,
                l1_norm: 0.0,
                max_abs: 0.0,
                min_abs: 0.0,
                mean: 0.0,
                sparsity: 1.0,
                overflow_count: 0,
                count: 0,
            };
        }
        let count = data.len();
        let mut sum = 0.0f64;
        let mut sq_sum = 0.0f64;
        let mut abs_sum = 0.0f64;
        let mut max_abs: f32 = 0.0;
        let mut min_abs: f32 = f32::MAX;
        let mut zeros = 0usize;
        let mut overflows = 0usize;
        for &v in data {
            if v.is_nan() || v.is_infinite() {
                overflows += 1;
                continue;
            }
            let a = v.abs();
            sum += v as f64;
            sq_sum += (v as f64) * (v as f64);
            abs_sum += a as f64;
            if a > max_abs {
                max_abs = a;
            }
            if a < min_abs {
                min_abs = a;
            }
            if v == 0.0 {
                zeros += 1;
            }
        }
        let finite_count = count - overflows;
        if finite_count == 0 {
            return Self {
                l2_norm: 0.0,
                l1_norm: 0.0,
                max_abs: 0.0,
                min_abs: 0.0,
                mean: 0.0,
                sparsity: 0.0,
                overflow_count: overflows,
                count,
            };
        }
        Self {
            l2_norm: (sq_sum as f32).sqrt(),
            l1_norm: abs_sum as f32,
            max_abs,
            min_abs,
            mean: (sum / count as f64) as f32,
            sparsity: zeros as f32 / count as f32,
            overflow_count: overflows,
            count,
        }
    }
}

impl fmt::Display for GradStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradStats(l2={:.4}, max={:.4}, sparsity={:.2}%, overflows={})",
            self.l2_norm,
            self.max_abs,
            self.sparsity * 100.0,
            self.overflow_count,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- GradBuffer ----------------------------------------------------------

    #[test]
    fn test_grad_buffer_new_zeroed() {
        let buf = GradBuffer::new(128, "test");
        assert_eq!(buf.len(), 128);
        assert!(!buf.is_empty());
        assert!(buf.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_grad_buffer_empty() {
        let buf = GradBuffer::new(0, "empty");
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_grad_buffer_accumulate() {
        let mut buf = GradBuffer::new(4, "acc");
        buf.accumulate(&[1.0, 2.0, 3.0, 4.0]);
        buf.accumulate(&[0.5, 0.5, 0.5, 0.5]);
        assert_eq!(buf.data(), &[1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_grad_buffer_scale() {
        let mut buf = GradBuffer::new(3, "s");
        buf.accumulate(&[2.0, 4.0, 6.0]);
        buf.scale(0.5);
        assert_eq!(buf.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_grad_buffer_zero() {
        let mut buf = GradBuffer::new(4, "z");
        buf.accumulate(&[1.0, 2.0, 3.0, 4.0]);
        buf.zero();
        assert!(buf.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_grad_buffer_l2_norm() {
        let mut buf = GradBuffer::new(2, "n");
        buf.accumulate(&[3.0, 4.0]);
        let norm = buf.l2_norm();
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_grad_buffer_has_overflow_nan() {
        let mut buf = GradBuffer::new(3, "o");
        buf.data_mut()[1] = f32::NAN;
        assert!(buf.has_overflow());
    }

    #[test]
    fn test_grad_buffer_has_overflow_inf() {
        let mut buf = GradBuffer::new(3, "o");
        buf.data_mut()[2] = f32::INFINITY;
        assert!(buf.has_overflow());
    }

    #[test]
    fn test_grad_buffer_no_overflow() {
        let mut buf = GradBuffer::new(3, "ok");
        buf.accumulate(&[1.0, -2.0, 3.0]);
        assert!(!buf.has_overflow());
    }

    #[test]
    fn test_grad_buffer_display() {
        let buf = GradBuffer::new(10, "weights");
        assert!(format!("{buf}").contains("weights"));
        assert!(format!("{buf}").contains("10"));
    }

    #[test]
    fn test_grad_buffer_name() {
        let buf = GradBuffer::new(1, "layer0.weight");
        assert_eq!(buf.name(), "layer0.weight");
    }

    #[test]
    fn test_grad_buffer_single_element() {
        let mut buf = GradBuffer::new(1, "single");
        buf.accumulate(&[42.0]);
        assert_eq!(buf.data(), &[42.0]);
        assert!((buf.l2_norm() - 42.0).abs() < 1e-6);
    }

    // -- GradAccumulator -----------------------------------------------------

    #[test]
    fn test_accumulator_basic_flow() {
        let mut acc = GradAccumulator::new(3);
        acc.register_buffer("w", 4);

        let g1: HashMap<String, Vec<f32>> = [("w".into(), vec![1.0, 2.0, 3.0, 4.0])].into();
        let g2: HashMap<String, Vec<f32>> = [("w".into(), vec![2.0, 3.0, 4.0, 5.0])].into();
        let g3: HashMap<String, Vec<f32>> = [("w".into(), vec![3.0, 4.0, 5.0, 6.0])].into();

        assert!(!acc.accumulate(&g1));
        assert!(!acc.accumulate(&g2));
        assert!(acc.accumulate(&g3)); // 3rd step → ready

        acc.average_and_reset();
        let buf = acc.buffer("w").unwrap();
        // Mean of [1,2,3],[2,3,4],[3,4,5],[4,5,6] per element:
        // (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4, (4+5+6)/3=5
        for (i, &expected) in [2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            assert!(
                (buf.data()[i] - expected).abs() < 1e-5,
                "element {i}: {} != {expected}",
                buf.data()[i],
            );
        }
    }

    #[test]
    fn test_accumulator_step_counter() {
        let mut acc = GradAccumulator::new(2);
        acc.register_buffer("b", 1);
        assert_eq!(acc.current_step(), 0);
        let g: HashMap<String, Vec<f32>> = [("b".into(), vec![1.0])].into();
        acc.accumulate(&g);
        assert_eq!(acc.current_step(), 1);
        assert!(acc.is_accumulating());
    }

    #[test]
    fn test_accumulator_zero_grads() {
        let mut acc = GradAccumulator::new(1);
        acc.register_buffer("x", 2);
        let g: HashMap<String, Vec<f32>> = [("x".into(), vec![5.0, 6.0])].into();
        acc.accumulate(&g);
        acc.zero_grads();
        let buf = acc.buffer("x").unwrap();
        assert!(buf.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_accumulator_multiple_buffers() {
        let mut acc = GradAccumulator::new(2);
        acc.register_buffer("a", 2);
        acc.register_buffer("b", 3);
        assert_eq!(acc.buffer_count(), 2);

        let g1: HashMap<String, Vec<f32>> =
            [("a".into(), vec![1.0, 1.0]), ("b".into(), vec![2.0, 2.0, 2.0])].into();
        let g2: HashMap<String, Vec<f32>> =
            [("a".into(), vec![3.0, 3.0]), ("b".into(), vec![4.0, 4.0, 4.0])].into();

        acc.accumulate(&g1);
        acc.accumulate(&g2);
        acc.average_and_reset();

        let a = acc.buffer("a").unwrap();
        assert!((a.data()[0] - 2.0).abs() < 1e-5);
        let b = acc.buffer("b").unwrap();
        assert!((b.data()[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_accumulator_single_step() {
        let mut acc = GradAccumulator::new(1);
        acc.register_buffer("w", 2);
        let g: HashMap<String, Vec<f32>> = [("w".into(), vec![10.0, 20.0])].into();
        assert!(acc.accumulate(&g));
        acc.average_and_reset();
        let buf = acc.buffer("w").unwrap();
        assert!((buf.data()[0] - 10.0).abs() < 1e-5);
        assert!((buf.data()[1] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_accumulator_ignores_unknown_buffer() {
        let mut acc = GradAccumulator::new(1);
        acc.register_buffer("known", 1);
        let g: HashMap<String, Vec<f32>> = [("unknown".into(), vec![99.0])].into();
        acc.accumulate(&g); // should not panic
        let buf = acc.buffer("known").unwrap();
        assert_eq!(buf.data(), &[0.0]);
    }

    #[test]
    fn test_accumulator_is_not_accumulating_at_start() {
        let acc = GradAccumulator::new(4);
        assert!(!acc.is_accumulating());
    }

    #[test]
    fn test_accumulator_accum_steps() {
        let acc = GradAccumulator::new(8);
        assert_eq!(acc.accum_steps(), 8);
    }

    #[test]
    fn test_accumulator_buffer_mut() {
        let mut acc = GradAccumulator::new(1);
        acc.register_buffer("m", 2);
        {
            let buf = acc.buffer_mut("m").unwrap();
            buf.data_mut()[0] = 42.0;
        }
        assert_eq!(acc.buffer("m").unwrap().data()[0], 42.0);
    }

    #[test]
    fn test_accumulator_buffers_iter() {
        let mut acc = GradAccumulator::new(1);
        acc.register_buffer("x", 1);
        acc.register_buffer("y", 1);
        let names: Vec<&str> = acc.buffers().map(|(n, _)| n).collect();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"y"));
    }

    // -- GradClipping --------------------------------------------------------

    #[test]
    fn test_clip_global_norm_above_threshold() {
        let mut buf = GradBuffer::new(4, "c");
        buf.data_mut().copy_from_slice(&[3.0, 4.0, 0.0, 0.0]);
        // Norm = 5.0; clip to 1.0
        let orig = GradClipping::clip_buffer_by_norm(&mut buf, 1.0);
        assert!((orig - 5.0).abs() < 1e-5);
        let after = buf.l2_norm();
        assert!(after <= 1.0 + 1e-5, "norm after clip: {after}");
    }

    #[test]
    fn test_clip_global_norm_below_threshold() {
        let mut buf = GradBuffer::new(2, "c");
        buf.data_mut().copy_from_slice(&[0.3, 0.4]);
        let orig = GradClipping::clip_buffer_by_norm(&mut buf, 10.0);
        assert!((orig - 0.5).abs() < 1e-5);
        // Data should be unchanged
        assert!((buf.data()[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_clip_by_value() {
        let mut buf = GradBuffer::new(5, "v");
        buf.data_mut().copy_from_slice(&[-10.0, -0.5, 0.0, 0.5, 10.0]);
        GradClipping::clip_by_value(&mut buf, 1.0);
        assert_eq!(buf.data(), &[-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_clip_multi_buffer_global_norm() {
        let mut b1 = GradBuffer::new(2, "a");
        let mut b2 = GradBuffer::new(2, "b");
        b1.data_mut().copy_from_slice(&[3.0, 0.0]);
        b2.data_mut().copy_from_slice(&[0.0, 4.0]);
        // Global norm = sqrt(9+16) = 5
        let orig = GradClipping::clip_by_global_norm(&mut [&mut b1, &mut b2], 1.0);
        assert!((orig - 5.0).abs() < 1e-5);
        let total_sq: f32 = b1.data().iter().chain(b2.data().iter()).map(|v| v * v).sum();
        assert!(total_sq.sqrt() <= 1.0 + 1e-5);
    }

    #[test]
    fn test_clip_method_apply_none() {
        let mut buf = GradBuffer::new(2, "n");
        buf.data_mut().copy_from_slice(&[100.0, -100.0]);
        GradClipping::apply(&mut buf, ClipMethod::None);
        assert_eq!(buf.data(), &[100.0, -100.0]);
    }

    #[test]
    fn test_clip_method_apply_value() {
        let mut buf = GradBuffer::new(3, "v");
        buf.data_mut().copy_from_slice(&[-5.0, 0.0, 5.0]);
        GradClipping::apply(&mut buf, ClipMethod::Value(2.0));
        assert_eq!(buf.data(), &[-2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_clip_method_apply_global_norm() {
        let mut buf = GradBuffer::new(2, "gn");
        buf.data_mut().copy_from_slice(&[3.0, 4.0]);
        let orig = GradClipping::apply(&mut buf, ClipMethod::GlobalNorm(1.0));
        assert!((orig - 5.0).abs() < 1e-5);
        assert!(buf.l2_norm() <= 1.0 + 1e-5);
    }

    #[test]
    fn test_clip_zero_gradients() {
        let mut buf = GradBuffer::new(4, "z");
        // All zeros — norm is 0, clipping should be a no-op
        let orig = GradClipping::clip_buffer_by_norm(&mut buf, 1.0);
        assert!(orig.abs() < 1e-9);
        assert!(buf.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_clip_method_display() {
        assert_eq!(format!("{}", ClipMethod::None), "none");
        assert!(format!("{}", ClipMethod::GlobalNorm(1.5)).contains("1.5"));
        assert!(format!("{}", ClipMethod::Value(0.5)).contains("0.5"));
    }

    // -- MixedPrecisionGrad --------------------------------------------------

    #[test]
    fn test_fp32_to_fp16_roundtrip() {
        let data = vec![1.0f32, 0.5, -0.25, 0.0, 100.0];
        let fp16 = MixedPrecisionGrad::fp32_to_fp16(&data);
        let back = MixedPrecisionGrad::fp16_to_fp32(&fp16);
        for (i, (&orig, &rt)) in data.iter().zip(back.iter()).enumerate() {
            assert!((orig - rt).abs() < 0.01, "element {i}: {orig} != {rt}");
        }
    }

    #[test]
    fn test_roundtrip_helper() {
        let data = vec![1.0, 2.0, 3.0];
        let rt = MixedPrecisionGrad::roundtrip(&data);
        for (i, (&orig, &r)) in data.iter().zip(rt.iter()).enumerate() {
            assert!((orig - r).abs() < 0.01, "element {i}");
        }
    }

    #[test]
    fn test_accumulate_fp16_into_fp32() {
        let grads_f32 = vec![1.0f32, 2.0, 3.0];
        let fp16 = MixedPrecisionGrad::fp32_to_fp16(&grads_f32);
        let mut accum = vec![0.0f32; 3];
        MixedPrecisionGrad::accumulate_fp16_into_fp32(&mut accum, &fp16);
        for (i, (&a, &e)) in accum.iter().zip(grads_f32.iter()).enumerate() {
            assert!((a - e).abs() < 0.01, "element {i}: {a} != {e}");
        }
    }

    #[test]
    fn test_apply_as_fp16() {
        let mut weights = vec![10.0f32, 20.0, 30.0];
        let grads = vec![1.0, 2.0, 3.0];
        MixedPrecisionGrad::apply_as_fp16(&mut weights, &grads, 0.1);
        // w -= lr * round(g)
        assert!((weights[0] - 9.9).abs() < 0.01);
        assert!((weights[1] - 19.8).abs() < 0.01);
        assert!((weights[2] - 29.7).abs() < 0.02);
    }

    #[test]
    fn test_would_overflow_fp16() {
        assert!(!MixedPrecisionGrad::would_overflow_fp16(&[1.0, 100.0, -50.0]));
        assert!(MixedPrecisionGrad::would_overflow_fp16(&[70000.0]));
        assert!(MixedPrecisionGrad::would_overflow_fp16(&[f32::NAN]));
        assert!(MixedPrecisionGrad::would_overflow_fp16(&[f32::INFINITY]));
    }

    #[test]
    fn test_fp16_small_values() {
        // Very small values that survive FP16
        let data = vec![0.001, 0.0001, -0.001];
        let rt = MixedPrecisionGrad::roundtrip(&data);
        for (i, (&orig, &r)) in data.iter().zip(rt.iter()).enumerate() {
            assert!((orig - r).abs() < 0.001, "element {i}: {orig} != {r}");
        }
    }

    #[test]
    fn test_fp16_zero() {
        let rt = MixedPrecisionGrad::roundtrip(&[0.0]);
        assert_eq!(rt, vec![0.0]);
    }

    #[test]
    fn test_fp16_negative() {
        let data = vec![-1.0, -2.0, -3.0];
        let rt = MixedPrecisionGrad::roundtrip(&data);
        for (i, (&orig, &r)) in data.iter().zip(rt.iter()).enumerate() {
            assert!((orig - r).abs() < 0.01, "element {i}");
        }
    }

    // -- GradScaler ----------------------------------------------------------

    #[test]
    fn test_scaler_scale_loss() {
        let scaler = GradScaler::new(1024.0);
        assert!((scaler.scale_loss(1.0) - 1024.0).abs() < 1e-3);
    }

    #[test]
    fn test_scaler_unscale_roundtrip() {
        let scaler = GradScaler::new(256.0);
        let mut buf = GradBuffer::new(3, "u");
        // Simulate scaled grads
        buf.data_mut().copy_from_slice(&[256.0, 512.0, -256.0]);
        let valid = scaler.unscale(&mut buf);
        assert!(valid);
        assert!((buf.data()[0] - 1.0).abs() < 1e-5);
        assert!((buf.data()[1] - 2.0).abs() < 1e-5);
        assert!((buf.data()[2] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_scaler_overflow_detection() {
        let scaler = GradScaler::new(256.0);
        let mut buf = GradBuffer::new(2, "of");
        buf.data_mut().copy_from_slice(&[f32::INFINITY, 1.0]);
        let valid = scaler.unscale(&mut buf);
        assert!(!valid);
    }

    #[test]
    fn test_scaler_update_on_overflow() {
        let mut scaler = GradScaler::new(1024.0);
        scaler.update(true);
        assert!((scaler.scale() - 512.0).abs() < 1e-3);
        assert_eq!(scaler.overflow_count(), 1);
    }

    #[test]
    fn test_scaler_growth_after_interval() {
        let mut scaler = GradScaler::with_params(100.0, 2.0, 0.5, 3);
        scaler.update(false);
        scaler.update(false);
        // After 2 steps scale is still 100
        assert!((scaler.scale() - 100.0).abs() < 1e-3);
        scaler.update(false); // 3rd step → grow
        assert!((scaler.scale() - 200.0).abs() < 1e-3);
    }

    #[test]
    fn test_scaler_disable() {
        let mut scaler = GradScaler::new(1024.0);
        scaler.disable();
        assert!(!scaler.is_enabled());
        assert!((scaler.scale_loss(1.0) - 1.0).abs() < 1e-6);

        let mut buf = GradBuffer::new(2, "d");
        buf.data_mut().copy_from_slice(&[5.0, 10.0]);
        scaler.unscale(&mut buf);
        // Passthrough — no change
        assert!((buf.data()[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaler_enable() {
        let mut scaler = GradScaler::new(100.0);
        scaler.disable();
        scaler.enable();
        assert!(scaler.is_enabled());
    }

    #[test]
    fn test_scaler_display() {
        let scaler = GradScaler::new(512.0);
        let s = format!("{scaler}");
        assert!(s.contains("512"));
        assert!(s.contains("enabled=true"));
    }

    #[test]
    fn test_scaler_multiple_overflows_reduce_scale() {
        let mut scaler = GradScaler::new(1024.0);
        for _ in 0..4 {
            scaler.update(true);
        }
        // 1024 * 0.5^4 = 64
        assert!((scaler.scale() - 64.0).abs() < 1e-3);
        assert_eq!(scaler.overflow_count(), 4);
    }

    #[test]
    fn test_scaler_no_growth_after_overflow_resets() {
        let mut scaler = GradScaler::with_params(100.0, 2.0, 0.5, 2);
        scaler.update(false); // step 1
        scaler.update(true); // overflow → resets counter
        scaler.update(false); // step 1 again
        // Scale should be 50 (one backoff), not grown
        assert!((scaler.scale() - 50.0).abs() < 1e-3);
    }

    // -- GradCheckpointer ----------------------------------------------------

    #[test]
    fn test_checkpoint_save_load_roundtrip() {
        let mut acc = GradAccumulator::new(4);
        acc.register_buffer("w1", 3);
        acc.register_buffer("w2", 2);
        let g: HashMap<String, Vec<f32>> =
            [("w1".into(), vec![1.0, 2.0, 3.0]), ("w2".into(), vec![4.0, 5.0])].into();
        acc.accumulate(&g);
        acc.accumulate(&g);

        let scaler = GradScaler::new(512.0);
        let ckpt = GradCheckpointer::save(&acc, Some(&scaler));

        // Restore into a fresh accumulator
        let mut acc2 = GradAccumulator::new(1);
        acc2.register_buffer("w1", 3);
        acc2.register_buffer("w2", 2);
        let mut scaler2 = GradScaler::new(1.0);
        GradCheckpointer::load(&ckpt, &mut acc2, Some(&mut scaler2));

        assert_eq!(acc2.accum_steps(), 4);
        assert_eq!(acc2.current_step(), 2);
        assert_eq!(acc2.buffer("w1").unwrap().data(), &[2.0, 4.0, 6.0]);
        assert!((scaler2.scale() - 512.0).abs() < 1e-3);
    }

    #[test]
    fn test_checkpoint_bytes_roundtrip() {
        let mut acc = GradAccumulator::new(3);
        acc.register_buffer("a", 2);
        let g: HashMap<String, Vec<f32>> = [("a".into(), vec![7.0, 8.0])].into();
        acc.accumulate(&g);
        let ckpt = GradCheckpointer::save(&acc, None);
        let bytes = GradCheckpointer::to_bytes(&ckpt);
        let ckpt2 = GradCheckpointer::from_bytes(&bytes).expect("deserialize");
        assert_eq!(ckpt, ckpt2);
    }

    #[test]
    fn test_checkpoint_bytes_with_scaler() {
        let mut acc = GradAccumulator::new(2);
        acc.register_buffer("b", 1);
        let mut scaler = GradScaler::new(256.0);
        scaler.update(true); // overflow_count=1, scale=128
        let ckpt = GradCheckpointer::save(&acc, Some(&scaler));
        let bytes = GradCheckpointer::to_bytes(&ckpt);
        let ckpt2 = GradCheckpointer::from_bytes(&bytes).expect("deserialize");
        assert_eq!(ckpt2.scaler_state, Some((128.0, 1)));
    }

    #[test]
    fn test_checkpoint_invalid_bytes() {
        assert!(GradCheckpointer::from_bytes(&[]).is_none());
        assert!(GradCheckpointer::from_bytes(b"GCKP").is_none());
        assert!(GradCheckpointer::from_bytes(b"XXXX0000000000000000000").is_none());
    }

    #[test]
    fn test_checkpoint_load_creates_missing_buffer() {
        let mut acc = GradAccumulator::new(2);
        acc.register_buffer("existing", 2);
        let g: HashMap<String, Vec<f32>> = [("existing".into(), vec![1.0, 2.0])].into();
        acc.accumulate(&g);
        let ckpt = GradCheckpointer::save(&acc, None);

        // Load into an accumulator that doesn't have the buffer yet
        let mut acc2 = GradAccumulator::new(1);
        GradCheckpointer::load(&ckpt, &mut acc2, None);
        assert!(acc2.buffer("existing").is_some());
        assert_eq!(acc2.buffer("existing").unwrap().data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_checkpoint_no_scaler() {
        let acc = GradAccumulator::new(1);
        let ckpt = GradCheckpointer::save(&acc, None);
        assert!(ckpt.scaler_state.is_none());
        let bytes = GradCheckpointer::to_bytes(&ckpt);
        let ckpt2 = GradCheckpointer::from_bytes(&bytes).unwrap();
        assert!(ckpt2.scaler_state.is_none());
    }

    // -- GradStats -----------------------------------------------------------

    #[test]
    fn test_stats_basic() {
        let mut buf = GradBuffer::new(4, "s");
        buf.data_mut().copy_from_slice(&[1.0, -2.0, 3.0, 0.0]);
        let stats = GradStats::from_buffer(&buf);
        assert_eq!(stats.count, 4);
        // l2 = sqrt(1+4+9) = sqrt(14)
        assert!((stats.l2_norm - 14.0f32.sqrt()).abs() < 1e-4);
        assert!((stats.l1_norm - 6.0).abs() < 1e-4);
        assert!((stats.max_abs - 3.0).abs() < 1e-6);
        assert!((stats.min_abs - 0.0).abs() < 1e-6);
        assert!((stats.mean - 0.5).abs() < 1e-4);
        assert!((stats.sparsity - 0.25).abs() < 1e-6);
        assert_eq!(stats.overflow_count, 0);
    }

    #[test]
    fn test_stats_empty() {
        let stats = GradStats::from_slice(&[]);
        assert_eq!(stats.count, 0);
        assert!((stats.sparsity - 1.0).abs() < 1e-6);
        assert_eq!(stats.overflow_count, 0);
    }

    #[test]
    fn test_stats_all_zeros() {
        let stats = GradStats::from_slice(&[0.0, 0.0, 0.0]);
        assert!((stats.sparsity - 1.0).abs() < 1e-6);
        assert!((stats.l2_norm).abs() < 1e-6);
    }

    #[test]
    fn test_stats_with_overflow() {
        let stats = GradStats::from_slice(&[1.0, f32::NAN, f32::INFINITY]);
        assert_eq!(stats.overflow_count, 2);
        assert_eq!(stats.count, 3);
    }

    #[test]
    fn test_stats_single_element() {
        let stats = GradStats::from_slice(&[42.0]);
        assert!((stats.l2_norm - 42.0).abs() < 1e-4);
        assert!((stats.mean - 42.0).abs() < 1e-4);
        assert!((stats.max_abs - 42.0).abs() < 1e-6);
        assert!((stats.min_abs - 42.0).abs() < 1e-6);
        assert!((stats.sparsity).abs() < 1e-6);
    }

    #[test]
    fn test_stats_display() {
        let stats = GradStats::from_slice(&[1.0, 2.0, 3.0]);
        let s = format!("{stats}");
        assert!(s.contains("l2="));
        assert!(s.contains("sparsity="));
    }

    #[test]
    fn test_stats_negative_only() {
        let stats = GradStats::from_slice(&[-1.0, -2.0, -3.0]);
        assert!((stats.mean - (-2.0)).abs() < 1e-4);
        assert!((stats.max_abs - 3.0).abs() < 1e-6);
        assert!((stats.min_abs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_all_overflow() {
        let stats = GradStats::from_slice(&[f32::NAN, f32::INFINITY]);
        assert_eq!(stats.overflow_count, 2);
        assert!((stats.l2_norm).abs() < 1e-6);
    }

    // -- Edge cases & property-style tests -----------------------------------

    #[test]
    fn test_accumulate_then_average_is_mean() {
        // Property: accumulate N identical vectors then average = that vector.
        let n = 5;
        let vec = vec![3.0, 7.0, 11.0];
        let mut acc = GradAccumulator::new(n);
        acc.register_buffer("v", vec.len());

        let g: HashMap<String, Vec<f32>> = [("v".into(), vec.clone())].into();
        for _ in 0..n {
            acc.accumulate(&g);
        }
        acc.average_and_reset();
        let buf = acc.buffer("v").unwrap();
        for (i, &expected) in vec.iter().enumerate() {
            assert!(
                (buf.data()[i] - expected).abs() < 1e-4,
                "element {i}: {} != {expected}",
                buf.data()[i],
            );
        }
    }

    #[test]
    fn test_accumulate_diverse_then_average() {
        // Property: mean of [1,2,3,4,5] = 3 per element
        let mut acc = GradAccumulator::new(5);
        acc.register_buffer("x", 1);
        for i in 1..=5 {
            let g: HashMap<String, Vec<f32>> = [("x".into(), vec![i as f32])].into();
            acc.accumulate(&g);
        }
        acc.average_and_reset();
        let buf = acc.buffer("x").unwrap();
        assert!((buf.data()[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_large_tensor() {
        let size = 100_000;
        let mut buf = GradBuffer::new(size, "big");
        let grads: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        buf.accumulate(&grads);
        assert_eq!(buf.len(), size);
        assert!((buf.data()[0]).abs() < 1e-6);
        assert!((buf.data()[999] - 0.999).abs() < 1e-4);
    }

    #[test]
    fn test_clip_preserves_direction() {
        let mut buf = GradBuffer::new(4, "dir");
        buf.data_mut().copy_from_slice(&[6.0, -8.0, 0.0, 0.0]);
        // Norm = 10; clip to 5 → scale by 0.5
        GradClipping::clip_buffer_by_norm(&mut buf, 5.0);
        assert!((buf.data()[0] - 3.0).abs() < 1e-4);
        assert!((buf.data()[1] - (-4.0)).abs() < 1e-4);
    }

    #[test]
    fn test_scaler_scale_unscale_cancel() {
        // scale then unscale should recover original values (if no overflow)
        let scaler = GradScaler::new(128.0);
        let original = vec![1.0, -0.5, 0.25];
        let mut buf = GradBuffer::new(3, "su");
        // Simulate: loss * scale → backward → grads * scale
        for (i, &v) in original.iter().enumerate() {
            buf.data_mut()[i] = v * scaler.scale();
        }
        let valid = scaler.unscale(&mut buf);
        assert!(valid);
        for (i, &v) in original.iter().enumerate() {
            assert!((buf.data()[i] - v).abs() < 1e-5, "element {i}: {} != {v}", buf.data()[i]);
        }
    }

    #[test]
    fn test_mixed_precision_accumulate_multiple() {
        let mut accum = vec![0.0f32; 2];
        let g1 = MixedPrecisionGrad::fp32_to_fp16(&[1.0, 2.0]);
        let g2 = MixedPrecisionGrad::fp32_to_fp16(&[3.0, 4.0]);
        MixedPrecisionGrad::accumulate_fp16_into_fp32(&mut accum, &g1);
        MixedPrecisionGrad::accumulate_fp16_into_fp32(&mut accum, &g2);
        assert!((accum[0] - 4.0).abs() < 0.01);
        assert!((accum[1] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_full_training_loop_simulation() {
        // Simulate 2 accumulation windows of size 2
        let mut acc = GradAccumulator::new(2);
        acc.register_buffer("w", 2);
        let scaler = GradScaler::new(256.0);

        for window in 0..2 {
            for _micro in 0..2 {
                let g: HashMap<String, Vec<f32>> =
                    [("w".into(), vec![1.0 + window as f32, 2.0])].into();
                acc.accumulate(&g);
            }
            acc.average_and_reset();
            let buf = acc.buffer_mut("w").unwrap();
            let valid = scaler.unscale(buf);
            assert!(valid);
            GradClipping::clip_buffer_by_norm(buf, 10.0);
            acc.zero_grads();
        }
        assert_eq!(scaler.overflow_count(), 0);
    }

    // -- OpenCL kernel source ------------------------------------------------

    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!GRAD_ACCUM_CL.is_empty());
    }

    #[test]
    fn test_kernel_source_contains_accumulate() {
        assert!(GRAD_ACCUM_CL.contains("grad_accumulate"));
    }

    #[test]
    fn test_kernel_source_contains_average() {
        assert!(GRAD_ACCUM_CL.contains("grad_average"));
    }

    #[test]
    fn test_kernel_source_contains_clip() {
        assert!(GRAD_ACCUM_CL.contains("grad_clip_by_norm"));
        assert!(GRAD_ACCUM_CL.contains("grad_clip_by_value"));
    }

    #[test]
    fn test_kernel_source_contains_overflow_check() {
        assert!(GRAD_ACCUM_CL.contains("grad_check_overflow"));
    }

    #[test]
    fn test_kernel_source_contains_fp16_conversion() {
        assert!(GRAD_ACCUM_CL.contains("fp32_to_fp16"));
        assert!(GRAD_ACCUM_CL.contains("fp16_to_fp32"));
    }

    #[test]
    fn test_kernel_source_contains_scale() {
        assert!(GRAD_ACCUM_CL.contains("grad_scale"));
    }

    #[test]
    fn test_kernel_source_contains_zero() {
        assert!(GRAD_ACCUM_CL.contains("grad_zero"));
    }
}

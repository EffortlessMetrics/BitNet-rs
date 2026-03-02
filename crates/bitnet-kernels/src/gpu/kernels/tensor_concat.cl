/// Tensor concatenation along axis 0 (batch dimension).
///
/// Copies src into dst at the given byte offset. Each work-item handles
/// one element.
__kernel void concat_axis0(
    __global const float* src,
    __global float* dst,
    const uint dst_offset,
    const uint num_elements
) {
    const uint i = get_global_id(0);
    if (i < num_elements) {
        dst[dst_offset + i] = src[i];
    }
}

/// Tensor concatenation along an arbitrary axis.
///
/// For each element in the source tensor, computes its destination index
/// in the concatenated output. Parameters encode the tensor geometry:
///   outer_size  — product of dims before the concat axis
///   src_axis    — size of the source along the concat axis
///   inner_size  — product of dims after the concat axis
///   dst_axis    — size of the destination along the concat axis
///   axis_offset — where this source starts within the concat axis
__kernel void concat_general(
    __global const float* src,
    __global float* dst,
    const uint outer_size,
    const uint src_axis,
    const uint inner_size,
    const uint dst_axis,
    const uint axis_offset
) {
    const uint idx = get_global_id(0);
    const uint total = outer_size * src_axis * inner_size;
    if (idx >= total) return;

    const uint inner_idx = idx % inner_size;
    const uint axis_idx  = (idx / inner_size) % src_axis;
    const uint outer_idx = idx / (src_axis * inner_size);

    const uint dst_idx = outer_idx * (dst_axis * inner_size)
                       + (axis_offset + axis_idx) * inner_size
                       + inner_idx;
    dst[dst_idx] = src[idx];
}

/// Split: copy a contiguous slice from src into dst.
/// Inverse of concat_axis0 — extracts `num_elements` starting at `src_offset`.
__kernel void split_axis0(
    __global const float* src,
    __global float* dst,
    const uint src_offset,
    const uint num_elements
) {
    const uint i = get_global_id(0);
    if (i < num_elements) {
        dst[i] = src[src_offset + i];
    }
}

/// General split along arbitrary axis.
__kernel void split_general(
    __global const float* src,
    __global float* dst,
    const uint outer_size,
    const uint src_axis,
    const uint inner_size,
    const uint dst_axis,
    const uint axis_offset
) {
    const uint idx = get_global_id(0);
    const uint total = outer_size * dst_axis * inner_size;
    if (idx >= total) return;

    const uint inner_idx = idx % inner_size;
    const uint axis_idx  = (idx / inner_size) % dst_axis;
    const uint outer_idx = idx / (dst_axis * inner_size);

    const uint src_idx = outer_idx * (src_axis * inner_size)
                       + (axis_offset + axis_idx) * inner_size
                       + inner_idx;
    dst[idx] = src[src_idx];
}

/// Padded copy: copy src into dst, filling remainder with pad_value.
__kernel void padded_copy(
    __global const float* src,
    __global float* dst,
    const uint src_len,
    const uint dst_len,
    const float pad_value
) {
    const uint i = get_global_id(0);
    if (i < dst_len) {
        dst[i] = (i < src_len) ? src[i] : pad_value;
    }
}

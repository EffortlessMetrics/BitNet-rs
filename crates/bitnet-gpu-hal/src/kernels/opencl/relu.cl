__kernel void relu(
    __global const float* input,
    __global float* output,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        output[i] = fmax(input[i], 0.0f);
    }
}

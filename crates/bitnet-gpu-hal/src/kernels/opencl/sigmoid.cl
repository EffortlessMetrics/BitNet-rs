__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}

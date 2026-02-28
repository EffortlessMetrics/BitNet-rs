__kernel void mish(
    __global const float* input,
    __global float* output,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
        output[i] = x * tanh(sp);
    }
}

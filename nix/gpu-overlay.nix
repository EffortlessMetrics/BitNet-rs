# GPU-specific Nix overlay for BitNet-rs
# Adds OpenCL, Vulkan, and GPU development tools
final: prev: {
  bitnet-gpu-env = final.buildEnv {
    name = "bitnet-gpu-env";
    paths = with final; [
      # OpenCL
      ocl-icd
      opencl-headers
      clinfo

      # Vulkan
      vulkan-headers
      vulkan-loader
      vulkan-tools
      vulkan-validation-layers

      # GPU tools
      intel-gpu-tools # intel_gpu_top

      # Build tools
      cmake
      pkg-config
    ];
  };
}

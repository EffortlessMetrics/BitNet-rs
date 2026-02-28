# GPU development shell for BitNet-rs
{ pkgs, rustStable, nativeDeps, commonEnv, ... }:
pkgs.mkShell {
  name = "bitnet-gpu-dev";

  buildInputs = nativeDeps ++ [
    rustStable
  ] ++ (with pkgs; [
    # OpenCL runtime
    ocl-icd
    opencl-headers

    # Vulkan
    vulkan-headers
    vulkan-loader

    # System
    pkg-config
    clinfo
  ]);

  shellHook = ''
    export RUSTUP_TOOLCHAIN=${commonEnv.RUSTUP_TOOLCHAIN}
    export RUSTC_WRAPPER=${commonEnv.RUSTC_WRAPPER}
    export CARGO_NET_GIT_FETCH_WITH_CLI=${commonEnv.CARGO_NET_GIT_FETCH_WITH_CLI}
    export CARGO_INCREMENTAL=${commonEnv.CARGO_INCREMENTAL}
    export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"

    echo "╭─────────────────────────────────────────────╮"
    echo "│ 🎮 BitNet-rs GPU development shell          │"
    echo "├─────────────────────────────────────────────┤"
    echo "│ rustc:  $(rustc --version | cut -d' ' -f2) │"
    echo "│ cargo:  $(cargo --version | cut -d' ' -f2) │"
    echo "├─────────────────────────────────────────────┤"
    echo "│ OpenCL: $(clinfo --list 2>/dev/null | head -1 || echo 'not available')"
    echo "│ Vulkan: $(vulkaninfo --summary 2>/dev/null | grep 'apiVersion' | head -1 || echo 'not available')"
    echo "├─────────────────────────────────────────────┤"
    echo "│ Build with GPU features:                    │"
    echo "│  • cargo build --no-default-features \\      │"
    echo "│        --features gpu                       │"
    echo "│  • cargo nextest run --workspace \\          │"
    echo "│        --no-default-features --features gpu │"
    echo "╰─────────────────────────────────────────────╯"
  '';

  # Tell OpenCL where to find ICDs
  OCL_ICD_VENDORS = "${pkgs.ocl-icd}/etc/OpenCL/vendors";

  # Vulkan ICD
  VK_ICD_FILENAMES =
    "${pkgs.vulkan-loader}/share/vulkan/icd.d/intel_icd.x86_64.json";
}

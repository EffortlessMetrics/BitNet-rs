fn main() {
    // Only build the C shim when the feature is on.
    if std::env::var("CARGO_FEATURE_IQ2S_FFI").is_ok() {
        let mut build = cc::Build::new();
        build
            .file("csrc/ggml_quants_shim.c")
            .include("csrc")
            .include("csrc/ggml/include")
            .include("csrc/ggml/src")
            // Define for IQ quantization family
            .define("GGML_USE_K_QUANTS", None)
            .define("QK_IQ2_S", "256")
            // Optimization and warnings
            .flag_if_supported("-O3")
            .flag_if_supported("-fPIC")
            .flag_if_supported("-Wno-sign-compare")
            .flag_if_supported("-Wno-unused-parameter")
            .flag_if_supported("-Wno-unused-function")
            .flag_if_supported("-Wno-unused-variable")
            .flag_if_supported("-Wno-unused-but-set-variable")
            .compile("bitnet_ggml_quants_shim");
            
        println!("cargo:rerun-if-changed=csrc/ggml_quants_shim.c");
        println!("cargo:rerun-if-changed=csrc/ggml/src/ggml-quants.c");
        println!("cargo:rerun-if-changed=csrc/ggml/src/ggml-quants.h");
        println!("cargo:rerun-if-changed=csrc/ggml/include/ggml/ggml.h");
    }
}
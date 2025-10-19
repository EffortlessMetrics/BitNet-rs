fn main() {
    // Only build the C shim when the feature is on.
    if std::env::var("CARGO_FEATURE_IQ2S_FFI").is_ok() {
        use std::{fs, path::Path};

        // Inject vendored commit into the crate's env
        let marker = Path::new("csrc/VENDORED_GGML_COMMIT");
        let commit = fs::read_to_string(marker).unwrap_or_else(|_| "unknown".into());
        let commit = commit.trim().to_string();
        println!("cargo:rustc-env=BITNET_GGML_COMMIT={}", commit);

        // In CI, fail fast if the marker is missing or unknown
        if std::env::var("CI").is_ok() && commit == "unknown" {
            panic!(
                "VENDORED_GGML_COMMIT is 'unknown' in CI.\n\
                 Run: cargo xtask vendor-ggml --commit <sha>\n\
                 Or set crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT"
            );
        }
        let ggml_quants_src =
            fs::read_to_string("csrc/ggml/src/ggml-quants.c").expect("read ggml-quants.c");

        let mut build = cc::Build::new();
        if ggml_quants_src.contains("assert(k % QK_IQ2_S == 0)") {
            build.define("BITNET_IQ2S_DEQUANT_NEEDS_QK_MULTIPLE", None);
        }

        build
            .file("csrc/ggml_quants_shim.c")
            .file("csrc/ggml_consts.c") // Constants extraction
            .include("csrc") // Local includes (use -I, warnings visible)
            // AC6: Use -isystem for vendored GGML headers (third-party code)
            // This suppresses warnings from the vendored GGML implementation
            // while preserving warnings from our shim code.
            .flag("-isystemcsrc/ggml/include")
            .flag("-isystemcsrc/ggml/src")
            // Define for IQ quantization family
            .define("GGML_USE_K_QUANTS", None)
            .define("QK_IQ2_S", "256")
            // Optimization and warnings
            .flag_if_supported("-O3")
            .flag_if_supported("-fPIC")
            // AC6: Selective warning suppression for vendored code patterns
            // These affect the vendored GGML code but not our local shim code
            // (which uses -I, not -isystem)
            .flag_if_supported("-Wno-sign-compare")
            .flag_if_supported("-Wno-unused-parameter")
            .flag_if_supported("-Wno-unused-function")
            .flag_if_supported("-Wno-unused-variable")
            .flag_if_supported("-Wno-unused-but-set-variable")
            .compile("bitnet_ggml_quants_shim");

        println!("cargo:rerun-if-changed=csrc/ggml_quants_shim.c");
        println!("cargo:rerun-if-changed=csrc/ggml_consts.c");
        println!("cargo:rerun-if-changed=csrc/ggml/src/ggml-quants.c");
        println!("cargo:rerun-if-changed=csrc/ggml/src/ggml-quants.h");
        println!("cargo:rerun-if-changed=csrc/ggml/include/ggml/ggml.h");
        println!("cargo:rerun-if-changed=csrc/VENDORED_GGML_COMMIT");
    }
}

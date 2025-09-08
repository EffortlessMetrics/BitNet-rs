use pyo3_build_config::get;

fn main() {
    // Use PyO3's build configuration to get proper Python linking info
    let config = get();

    // Let PyO3 handle the linking directives automatically based on the ABI
    // This ensures compatibility with abi3-py312 and proper symbol resolution
    if let Some(lib_dir) = &config.lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }

    // Only add explicit library links if needed for the specific ABI
    if config.abi3
        && let Some(lib_name) = &config.lib_name
    {
        println!("cargo:rustc-link-lib={}", lib_name);
    }

    println!("cargo:rerun-if-changed=build.rs");
}

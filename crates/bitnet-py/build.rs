fn main() {
    // Use PyO3's build configuration to set up linker args for Python extension modules.
    // In pyo3-build-config 0.28+, this replaces the removed `get()` API.
    pyo3_build_config::add_extension_module_link_args();

    println!("cargo:rerun-if-changed=build.rs");
}

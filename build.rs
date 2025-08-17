use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // Minimal, dependency-free metadata so the build never blocks on 'vergen'
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("cargo:rustc-env=BITNET_BUILD_TS={ts}");
}

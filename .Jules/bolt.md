## 2024-05-22 - WASM Build Configuration
**Learning:** Rust crates relying on C-bindings (like `onig`) must be made optional features to support `wasm32-unknown-unknown` targets, which often lack a full libc or C compiler environment.
**Action:** When porting to WASM, check dependency trees for native libraries and use `default-features = false` or feature flags to exclude them.

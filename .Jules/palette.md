## 2026-01-30 - Vanilla JS Tab Accessibility
**Learning:** The `bitnet-wasm` browser example uses vanilla JS/HTML with `div`s for tabs. While visually functional, this is inaccessible. Adding ARIA roles (`tablist`, `tab`, `tabpanel`) and keyboard listeners (`keydown` for Enter/Space) allows for significant accessibility gains without altering the visual design or needing complex framework dependencies.
**Action:** For legacy or vanilla JS components, prefer augmenting existing DOM elements with ARIA and event listeners over rewriting them into semantic elements if retaining exact styling is critical.

## 2026-01-30 - Frontend Verification dependencies
**Learning:** The `crates/bitnet-wasm/examples/browser` frontend depends on a generated WASM package (`./pkg/bitnet_wasm.js`) that may not be present in the source tree. This causes script load errors that block verification of unrelated UI logic.
**Action:** When verifying frontend logic in environments where the backend/WASM cannot be built, create a mock JS module for the missing dependency to allow the rest of the application to initialize enough for UI testing.

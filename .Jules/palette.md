# Palette's Journal

## 2025-10-21 - Raw Static Site Constraints in WASM Examples
**Learning:** The `bitnet-wasm` frontend examples are raw HTML/JS/CSS without a build system or test runner. Accessibility improvements must rely on vanilla JS DOM manipulation and cannot leverage component libraries or standard testing frameworks.
**Action:** When modifying these examples, verify changes with custom scripts (e.g. Playwright) and ensure vanilla JS implementation is robust (e.g. `aria` attribute management).

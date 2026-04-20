# Palette's Journal - Critical Learnings

## 2024-05-22 - Converting Interactive Divs to Buttons
**Learning:** When converting interactive `div` elements (like tabs) to `button` elements for accessibility, user agent styles (background, border, padding, font) often override custom styles.
**Action:** Always include a CSS reset for the new button class (e.g., `background: transparent; border: none; font: inherit;`) to maintain the original visual design while gaining semantic benefits.

## 2024-05-22 - Mocking WASM for Frontend Verification
**Learning:** Frontend code that imports WASM modules (like `pkg/bitnet_wasm.js`) fails to run in isolation if the WASM build artifacts are missing.
**Action:** Create a mock JS file that exports the necessary functions and classes (even if empty) to allow the frontend logic to execute and be verified without a full WASM build.

## 2024-05-22 - Keyboard Navigation in Custom Tabs
**Learning:** Custom tab implementations using ARIA roles (`tablist`, `tab`) often miss the expected keyboard interaction pattern (arrow keys to navigate), making them inaccessible to keyboard users despite having semantic roles.
**Action:** Always implement a `keydown` handler for custom tab components to support ArrowRight/ArrowLeft/Home/End navigation and automatic activation.

## 2024-05-22 - Semantic Accessibility of Scrollable Output Containers
**Learning:** Using `<label>` tags to identify non-interactive scrollable containers (like `div`s with `overflow-y: auto` containing terminal output) violates HTML semantics, as `<label>` is strictly for form controls. This can confuse screen readers.
**Action:** Replace `<label>` with a visually identical styled `<div>` given a unique ID. Link it to the output container using `aria-labelledby`. Ensure the output container has `tabindex="0"` for keyboard scrollability and `role="region"` so screen readers announce its semantic purpose as a landmark.

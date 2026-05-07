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

## 2024-05-23 - Accessible Scrollable Dynamic Regions
**Learning:** Using `<label>` tags for non-interactive `<div>` elements with `overflow-y: auto` is semantically incorrect because labels are meant for form controls. Additionally, asynchronously updating containers lack announcements for screen readers.
**Action:** Replace the `<label>` with a styled `<div>` or `<span>`, assign an `id` to it, and use `aria-labelledby`, `role="region"`, and `tabindex="0"` on the scrollable container. Add `aria-live="polite"` to dynamically updating text regions to ensure proper screen reader support.

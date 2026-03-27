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

## 2024-05-23 - Accessibility of Scrollable Output Regions
**Learning:** Using `<label>` elements for non-interactive content blocks (like `<div>` containers showing output) is semantically incorrect and ignored by screen readers. Furthermore, scrollable content blocks that don't contain focusable elements are inaccessible to keyboard-only users.
**Action:** To label non-interactive containers, use styled text elements (like `<div class="label-text">`) and link them to the container via `aria-labelledby`. To make scrollable content blocks accessible to keyboard users, add `tabindex="0"` and `role="region"`.

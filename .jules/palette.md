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

## 2026-03-22 - Replacing Labels for Non-Interactive Output Divs
**Learning:** In HTML, standard `<div>` containers (such as scrollable output areas) cannot be labeled directly with `<label>` tags. Screen readers will ignore the label unless it points to an actionable form element. Additionally, `overflow-y: auto` boxes cannot be navigated to by keyboard users unless given a specific focus treatment.
**Action:** When a scrollable output container needs to be labeled for accessibility, replace `<label>` tags with explicitly styled `<div>` elements having IDs, and link them to the output container using `aria-labelledby="..."`. Provide the output container with `tabindex="0"` and `role="region"` so keyboard users can scroll it and screen readers can read it correctly.

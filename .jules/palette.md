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

## 2024-05-22 - Focus Visibility and Box Sizing
**Learning:** Adding `box-sizing: border-box` to inputs prevents them from overflowing containers when padding is applied and `width: 100%` is used. Furthermore, relying on default browser focus rings is insufficient for accessibility; explicit `:focus-visible` styles with custom outlines ensure clear visual feedback for keyboard users across different platforms.
**Action:** Always include `box-sizing: border-box` for standard form inputs and define clear `:focus-visible` styles for better keyboard navigation.

## 2024-05-22 - Improved Range Slider Layout
**Learning:** Range sliders with adjacent value readouts can suffer from poor layout and spacing when placed in standard `div` containers. The label and value span can become disjointed or visually misaligned with the slider.
**Action:** Group the label and value span in a flex container (`display: flex; justify-content: space-between;`) above the range input to create a clear visual header for the slider control. Style the value readout distinctively (e.g., using a pill-shaped badge) to emphasize it as an interactive state value rather than static text.
## 2024-05-24 - Confirm Destructive Actions
**Learning:** Destructive actions that result in data loss or immediate page reloads (like resetting settings) must have a confirmation prompt to prevent accidental activation and poor UX.
**Action:** Always add a confirmation step (e.g., using `confirm()`) or a custom confirmation modal before executing destructive actions or operations that force a full page reload.
## 2024-10-18 - Keyboard Shortcuts for Text Areas
**Learning:** For chat-like or generative UI, requiring users to switch from keyboard to mouse to click "Generate" breaks the interaction flow. While screen readers have native ways to navigate, sighted power users greatly benefit from explicit keyboard shortcuts.
**Action:** Always provide explicit keyboard shortcuts (e.g., Ctrl+Enter) for primary actions associated with large text inputs, visually indicate them using `<kbd>` tags (with `aria-hidden="true"` so they don't clutter screen reader output), and expose the shortcut semantically using the `aria-keyshortcuts` attribute on the input element itself.

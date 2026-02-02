## 2024-11-25 - Legacy Tab Accessibility
**Learning:** The browser examples use `div` elements for tabs without any semantic roles or keyboard support, making them completely inaccessible to screen readers and keyboard users.
**Action:** When working on legacy/simple frontend examples in this repo, always check for non-semantic interactive elements and retrofit them with WAI-ARIA roles (`tablist`, `tab`, `tabpanel`) and keyboard handlers (`Enter`/`Space`) to ensure basic accessibility without changing the visual design.

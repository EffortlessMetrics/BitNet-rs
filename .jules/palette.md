## 2026-02-24 - Semantic Tabs
**Learning:** In browser examples, tabs implemented as `div`s with `onclick` are completely inaccessible to keyboard users and screen readers.
**Action:** Use `<button role="tab">` elements for tabs, along with `role="tablist"` and `role="tabpanel"`. Ensure `aria-selected` is updated dynamically.

## 2026-02-13 - Tab Navigation Accessibility
**Learning:** Interactive elements implemented as `<div>`s are invisible to keyboard users and screen readers, creating a major accessibility barrier.
**Action:** Always use semantic `<button>` elements for tabs, or strictly implement `role="tab"`, `tabindex="0"`, and `keydown` handlers (Enter/Space) if `<div>`s are absolutely necessary.

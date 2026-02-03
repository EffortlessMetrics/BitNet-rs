## 2026-02-03 - Accessible Tabs Implementation
**Learning:** Tab interfaces using non-semantic elements (like `div`) require explicit ARIA roles (`tablist`, `tab`, `tabpanel`) and manual keyboard navigation (Arrow keys) to be accessible. Simply adding `onclick` is insufficient for keyboard and screen reader users.
**Action:** When creating custom tabs, always implement the full WAI-ARIA Tab Design Pattern, including `aria-selected`, `tabindex` management, and keyboard event handlers for focus movement.

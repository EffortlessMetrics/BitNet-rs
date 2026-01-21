# Palette's Journal

## 2025-10-21 - Accessible Tabs without Semantic Buttons
**Learning:** Using `<div>` with `role="tab"` and manual keyboard handling was necessary here because global `button` styles were too invasive to override cleanly. This avoided "fighting the framework" (or lack thereof) while still achieving full A11y compliance.
**Action:** When working in legacy or simple codebases with aggressive global styles, prefer semantic ARIA roles on neutral elements over struggling with semantic HTML elements that carry heavy style baggage.

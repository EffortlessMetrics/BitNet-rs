## 2025-10-24 - Accessibility in Custom Tabs
**Learning:** The browser example used `div` elements for tabs, completely excluding keyboard users. This is a common pattern in "lightweight" demos that unintentionally excludes assistive tech.
**Action:** When seeing custom tab implementations, immediately check for `role="tablist"`, `role="tab"`, and keyboard event handlers (Arrows/Home/End).

## 2025-10-24 - Work Superseded
**Learning:** Sometimes work gets superseded by larger refactors. Always check for recent major PRs before starting small fixes.
**Action:** Before starting, check recent PRs or ask about ongoing refactors.

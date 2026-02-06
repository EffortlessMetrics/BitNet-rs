## 2025-10-21 - Vanilla JS Tab Accessibility
**Learning:** Legacy or example code often uses "div soup" (divs with onclick) for interactive elements like tabs, breaking accessibility.
**Action:** Replace interactive divs with semantic `<button>` elements, add `role="tablist/tab/tabpanel"`, and use CSS resets (border/background: none) to maintain the original visual design while gaining native keyboard and screen reader support.

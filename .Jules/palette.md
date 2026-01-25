# Palette's Journal - Critical UX & Accessibility Learnings

## 2025-10-21 - Accessible Tabs Pattern
**Learning:** The application was using `<div>` elements for tab navigation, which makes them inaccessible to keyboard users and screen readers.
**Action:** Always use `<button>` elements for tab controls to ensure native keyboard accessibility. Use `role="tab"`, `aria-selected`, and `aria-controls` to properly link tabs to their content panels and communicate state to assistive technology.

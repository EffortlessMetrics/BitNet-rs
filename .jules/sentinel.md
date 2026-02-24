## 2025-10-21 - Overly Permissive CORS Configuration (CLOSED)
**Vulnerability:** The default CORS configuration was hardcoded to allow all origins (`*`).
**Status:** This issue was addressed in recent refactoring work (PRs #608–#638). This entry is kept for historical context.
**Learning:** Always check for existing or in-progress refactoring efforts before starting new security fixes.

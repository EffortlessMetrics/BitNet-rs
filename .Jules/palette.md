## 2024-10-22 - Vanilla JS Verification Strategy
**Learning:** For frontend examples lacking a build system (no package.json), verifying interaction requires mocking missing WASM/module dependencies and serving static files via a temporary Python HTTP server.
**Action:** Use `python3 -m http.server` and create dummy ES modules for verification scripts to avoid build dependency blockers.

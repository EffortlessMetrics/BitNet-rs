## 2025-10-21 - Hardcoded Grafana Password in Docker Compose
**Vulnerability:** Found `GF_SECURITY_ADMIN_PASSWORD=admin` hardcoded in `docker-compose.yml`.
**Learning:** Development configurations often leak into production-like artifacts. Docker Compose files are frequently used for both local dev and production, creating a risk when defaults are insecure.
**Prevention:** Use shell parameter expansion with error enforcement (`${VAR:?error}`) in configuration files to force explicit secret injection, rather than providing weak defaults.

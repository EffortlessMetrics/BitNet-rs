## 2025-11-03 - Hardcoded Secrets in Docker Compose
**Vulnerability:** Found `GF_SECURITY_ADMIN_PASSWORD=admin` and SMTP credentials hardcoded in `docker-compose.yml` and `alertmanager.yml`.
**Learning:** Default configurations in infrastructure-as-code files are often deployed as-is, creating persistent security risks.
**Prevention:** Use environment variable substitution (e.g., `${VAR:-default}`) for all credentials, and document required secrets clearly.

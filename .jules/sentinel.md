## 2025-10-21 - Hardcoded Secrets in Infrastructure Configuration
**Vulnerability:** A hardcoded administrative password ('admin') was found in the `docker-compose.yml` file for the Grafana service.
**Learning:** Development configurations often prioritize convenience over security, leading to hardcoded secrets that can accidentally be deployed to production or exposed in public repositories. Even if intended for local development, these defaults condition users to insecure practices.
**Prevention:** Always use environment variables for sensitive data, even in development configurations. Provide a template (e.g., `.env.example`) with placeholder values or instructions, but never commit actual secrets to the repository.

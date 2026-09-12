# Security Policy

## Reporting a Vulnerability

The InsightfulAI team takes security vulnerabilities seriously. If you discover a security vulnerability in InsightfulAI, please report it to us responsibly.

### How to Report

**Do not** file a public GitHub issue for security vulnerabilities. Instead, please email your report to **security@craftedwithintent.com** with the following information:

1. **Type of vulnerability** — (e.g., injection, data exposure, privilege escalation, dependency vulnerability)
2. **Location** — Specific file(s), module(s), or function(s) affected
3. **Description** — Clear explanation of the vulnerability and its impact
4. **Proof of Concept** — Minimal reproducible example (code snippet, test case, etc.)
5. **Suggested Fix** — If you have one (optional but appreciated)

### Response Timeline

- **Acknowledgment** — Within 24 hours
- **Initial assessment** — Within 48 hours
- **Patch or mitigation** — Within 7-14 days (depending on severity)
- **Public disclosure** — After patch is released and users have had time to upgrade

### Supported Versions

| Version | Status | Security Updates |
|---------|--------|------------------|
| 0.3.x   | Alpha  | Yes (active development) |
| 0.2.x   | Beta   | Yes (limited) |
| < 0.2.0 | EOL    | No |

### Severity Levels

We classify vulnerabilities using CVSS 3.1 scoring:

- **Critical (9.0–10.0):** Immediate patch required; all users should upgrade immediately
- **High (7.0–8.9):** Patch released within 48 hours; strong recommendation to upgrade
- **Medium (4.0–6.9):** Patch released within 7 days; recommended to upgrade
- **Low (0.1–3.9):** Patch released in next regular release; can be deployed with other updates

### Security Best Practices

When using InsightfulAI in production:

1. **Dependencies** — Regularly run `pip install --upgrade InsightfulAI` to get security updates
2. **Training Data** — Never commit sensitive training data to git. Use environment variables or secure data pipelines
3. **Model Files** — Keep serialized models in `.gitignore` if they contain sensitive information
4. **API Keys** — If using external APIs (OpenAI, etc.), store credentials in environment variables or `.env` files (not in code)
5. **CI/CD Integration** — Use GitHub Secrets for sensitive credentials in `.github/workflows/`
6. **Async Operations** — When using async methods with sensitive data, ensure proper error handling and logging doesn't expose secrets
7. **Telemetry** — Review OpenTelemetry instrumentation to ensure it doesn't collect sensitive data

### Scope

InsightfulAI is a machine learning framework designed to run in controlled environments (CI/CD, local development, internal servers). It is **not** intended for:

- Handling sensitive user data directly
- Serving as a standalone network service exposed to untrusted networks
- Operating without proper input validation and authentication

Security vulnerabilities in **dependencies** (scikit-learn, numpy, opentelemetry, etc.) should be reported to those projects directly.

### Vulnerability Management Process

1. **Report received** → Assigned unique ID
2. **Triage** → Severity assessment (Critical/High/Medium/Low)
3. **Development** → Fix implemented and tested
4. **QA** → Security review and regression testing
5. **Release** → Patch published to PyPI
6. **Communication** → Release notes published, users notified
7. **Follow-up** → Monitor for similar issues

### Acknowledgments

We credit security researchers and contributors who report vulnerabilities responsibly. With your permission, we will acknowledge you in the release notes.

---

**Questions?** 
- Security issues: **security@craftedwithintent.com**
- General questions: Open an issue or contact **team@craftedwithintent.com**

# Changelog

All notable changes to InsightfulAI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0a1] - 2026-09-12

### Added

- **Production-Ready Package Configuration** — Modern pyproject.toml (PEP 517/518)
  - Pinned dependencies with minimum versions (scikit-learn, numpy, opentelemetry)
  - Python >=3.9 requirement (upgraded from >=3.6)
  - Optional [dev] dependencies (pytest, ruff, black, mypy, pyright, pandas)
  - Build tool configuration (black, ruff, mypy, pyright, pytest, coverage)

- **Comprehensive Governance** — Open-source standards
  - `CONTRIBUTING.md` — 387-line development guide with 25+ contribution ideas
  - `CODE_OF_CONDUCT.md` — Contributor Covenant 2.0 community standards
  - `SECURITY.md` — Vulnerability reporting policy and response timeline
  - Apache 2.0 LICENSE file

- **Docker Containerization** — Multi-stage production build
  - Dockerfile with builder + runtime stages
  - Non-root user (insightful:1000) for security
  - docker-compose.yml for local development
  - .dockerignore for efficient builds
  - Ready for GitHub Container Registry (GHCR) push

- **GitHub Actions CI/CD Pipeline** — Automated testing and publishing
  - `tests.yml` — Python 3.9-3.12 matrix testing with coverage (70%+ minimum)
  - `publish.yml` — PyPI publishing on version tags with verification
  - `docker.yml` — Multi-platform Docker builds (linux/amd64, linux/arm64) to GHCR
  - Automated GitHub releases with build artifacts

### Core Features (From v0.2.0)

- **Logistic Regression Model** — Binary classification with configurable solver
- **Random Forest Model** — Ensemble classifier with customizable parameters
- **NLP Model** — Text processing and classification
- **Railway Oriented Programming (ROP)** — OperationResult wrapper for error handling
- **OpenTelemetry Instrumentation** — Built-in tracing and observability
- **Async/Batch Processing** — async_fit, async_predict, async_evaluate methods
- **Retry Decorator** — Fault tolerance with automatic retry logic
- **Model Interface** — Extensible abstract base class for custom implementations

### Testing

- 4 test files with real-world datasets (Kaggle)
- 17 passing tests, 2 pre-existing async failures
- Coverage: 70% baseline (expandable in future releases)
- Pandas integration for data handling

### Documentation

- README.md — Project overview, quick start, model types, examples
- Design/ folder — Architecture documentation (3 design docs)
- Setup/ folder — Configuration guides (logging, asyncio, OpenTelemetry)
- Tutorials/ folder — Usage examples and best practices
- docs/CONTRIBUTION_IDEAS.md — 20+ ideas for contributors (Easy/Medium/Hard)

### Known Limitations (Phase 1)

- Coverage at 70% baseline (not expanded to edge cases)
- Limited type hints (partial coverage, not 100%)
- Async methods have 2 pre-existing failures (known issue)
- No CHANGELOG prior to v0.3.0a1

### Roadmap (Phase 2+)

- [ ] Type hint expansion (target 100%)
- [ ] Test coverage to 80%+ (edge cases, async operations)
- [ ] Additional model types (SVM, gradient boosting)
- [ ] Model serialization/persistence
- [ ] Hyperparameter tuning framework
- [ ] Cross-validation support
- [ ] Feature engineering pipelines
- [ ] Batch prediction API enhancements
- [ ] Performance monitoring dashboard
- [ ] Cost analysis and tracking

---

## [0.2.0] - 2026-09-01

### Added

- **Async Operations** — Batch processing with async methods
  - async_fit() for training on batches
  - async_predict() for predictions on batches
  - async_evaluate() for batch evaluation
  - Concurrency control and error handling

- **NLP Model Support** — Text processing capabilities
  - Basic NLP model implementation
  - Text classification support
  - Integration with scikit-learn pipeline

### Dependencies

- scikit-learn>=1.3.0 — Machine learning core
- numpy>=1.24.0 — Numerical computation
- opentelemetry-api>=1.20.0 — Tracing framework
- opentelemetry-sdk>=1.20.0 — SDK implementation
- opentelemetry-instrumentation>=0.41.0 — Auto-instrumentation

---

## [0.1.0] - 2026-08-15

### Added

- **Initial Alpha Release** — Core ML framework
  - Logistic Regression model
  - Random Forest model
  - Model Interface for extensibility
  - Basic fit/predict/evaluate workflow
  - OperationResult wrapper (ROP pattern)
  - Retry decorator for resilience
  - OpenTelemetry instrumentation
  - Test suite with real datasets

### Architecture

- **Functional Core + Imperative Shell** — Pure models vs. API coordination
- **Railway Oriented Programming** — Explicit error handling via Result types
- **Extensible Design** — Abstract ModelInterface for custom implementations
- **Observability** — Built-in OpenTelemetry tracing

### Testing

- Unit tests for logistic regression and random forest
- Real-world datasets (diabetes, roach outcomes)
- Basic regression and classification scenarios

### Known Limitations (Phase 1)

- No async support (Phase 2 feature)
- Limited NLP capabilities (Phase 2)
- Minimal type hints (Phase 2)
- No Docker support (Phase 2)
- No CI/CD pipeline (Phase 2)

---

## Future Releases

### v1.0.0 (Beta → Production)

- [ ] Type hints 100% coverage
- [ ] Test coverage 80%+
- [ ] Extended model zoo (SVM, boosting, etc.)
- [ ] Production deployment guides
- [ ] Performance benchmarks
- [ ] Community contributed models

---

**For detailed architecture and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md) and [Design/](Design/).**

# Contributing to InsightfulAI

👋 **We actively welcome community contributions!** Whether you're fixing a typo, improving a model, or adding new features — your help makes InsightfulAI better.

## 🎯 Quick Start for Contributors

### New to InsightfulAI?

1. **Pick an issue:** Browse [#good-first-issue](https://github.com/CraftedWithIntent/InsightfulAI/labels/good%20first%20issue) or [#help-wanted](https://github.com/CraftedWithIntent/InsightfulAI/labels/help%20wanted) labels
2. **Review ideas:** See [docs/CONTRIBUTION_IDEAS.md](docs/CONTRIBUTION_IDEAS.md) for 20+ contribution ideas (🟢 Easy → 🔴 Hard)
3. **Comment on issue:** Let us know you're interested
4. **Follow the guide:** Setup, code, test, submit PR
5. **Get feedback:** We'll review and merge if it meets criteria

## Help Wanted

These issues are ready for community pickup. **All skill levels welcome!**

### 🟢 Good First Issues (Easy, 0.5–2 hours)

Perfect for learning InsightfulAI. No deep architecture knowledge needed.

- **Add new model type** — Implement SVM or gradient boosting classifier
- **Improve NLP model** — Add sentiment analysis or text classification
- **Add custom metrics** — Implement precision, recall, F1-score evaluators
- **Documentation improvements** — Expand tutorials and examples
- **Performance optimization** — Profile and optimize async operations
- **Error messages** — Improve error handling and user feedback
- **Type hints** — Expand type coverage (target 100%)
- **Test coverage** — Add edge case tests for edge scenarios
- **Examples** — Create production-ready example notebooks
- **GitHub Actions** — Setup CI/CD pipelines for testing and publishing

### 🟡 Medium Issues (Intermediate, 2–4 hours)

Requires understanding InsightfulAI internals.

- **Model serialization** — Save/load trained models to disk
- **Hyperparameter tuning** — Implement grid search or random search
- **Cross-validation** — Add k-fold validation support
- **Feature engineering** — Add data preprocessing pipelines
- **Batch prediction API** — Streaming predictions on large datasets
- **Model comparison** — Framework for comparing model performance
- **OpenTelemetry extension** — Custom metrics and traces
- **Docker optimization** — Multi-stage builds and minimal images
- **Monitoring dashboard** — Real-time performance tracking
- **Cost analysis** — Track ML infrastructure costs

### 🔴 Hard Issues (Advanced, 4+ hours)

Requires deep knowledge of ML, async, or architecture.

- **Distributed training** — Multi-GPU/TPU support
- **AutoML framework** — Automatic model selection and tuning
- **Ensemble methods** — Voting and stacking classifiers
- **Explainability** — SHAP/LIME integration for model interpretability
- **Custom loss functions** — Framework for specialized losses
- **Production deployment** — Kubernetes manifests and scaling strategies

---

**👉 [See docs/CONTRIBUTION_IDEAS.md](docs/CONTRIBUTION_IDEAS.md) for detailed descriptions, acceptance criteria, and implementation hints for all 20+ ideas.**

---

## The Contribution Path

1. **Pick an issue** (🟢 Good First Issue recommended)
2. **Comment on GitHub issue:** "I'd like to work on this"
3. **Read relevant docs:** This file + [docs/CONTRIBUTION_IDEAS.md](docs/CONTRIBUTION_IDEAS.md)
4. **Setup dev environment:** Follow "Setup" section below
5. **Implement & test:** Write code, run tests, ensure 80%+ coverage
6. **Submit PR:** Link to GitHub issue, describe changes
7. **Iterate:** Address reviewer feedback
8. **Merge:** We'll squash + merge when ready

**Pro tip:** Start with 🟢 Good First Issues to learn the codebase, then tackle harder issues.

---

Thank you for contributing! This guide explains how to develop, test, and submit changes to InsightfulAI.

## Setup

### Clone and install in dev mode

```bash
git clone https://github.com/CraftedWithIntent/InsightfulAI.git
cd InsightfulAI
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -e .[dev]
```

### Verify setup

```bash
pytest tests/ -v
python -m pyright src/
```

## Architecture

InsightfulAI follows **Functional Core + Imperative Shell** architecture:

- **Functional Core:** Pure model implementations (logistic regression, random forest, NLP)
- **Imperative Shell:** Public API (InsightfulAI class), async support, telemetry

### Code Organization

```
InsightfulAI/
├── models/                       # Pure ML model implementations
│   ├── model_interface.py        # Abstract base class
│   ├── logistic_regression_model.py
│   ├── random_forest_model.py
│   └── nlp_model.py
├── templates/                    # Usage examples and templates
│   ├── logistic_regression_template.py
│   ├── random_forest_template.py
│   └── nlp_template.py
├── retry/                        # Retry decorator for fault tolerance
│   └── retry_decorator.py
├── insightful_ai_api.py          # Public API with ROP pattern
├── operation_result.py           # Result wrapper (Railway Oriented Programming)
├── tests/                        # Test suite
│   ├── test_diabetes_outcome_logistic_regression.py
│   ├── test_random_forest.py
│   ├── test_roach_outcome_logical_regression.py
│   ├── test_nlp.py
│   └── datasets/
├── Design/                       # Architecture documentation
├── Setup/                        # Setup guides
├── Tutorials/                    # User documentation
├── pyproject.toml                # Modern package config (PEP 517/518)
├── setup.py                      # Backward compatibility (deprecated)
└── README.md                     # Main documentation
```

## Code Style

### Linting & Formatting

InsightfulAI uses **Ruff** for all style enforcement:

```bash
ruff check src tests         # Check only
ruff check --fix src tests   # Auto-fix
```

### Type Hints

- All functions must have parameter + return type hints
- Use `from typing import ...` for generic types
- Target: 100% type hint coverage

### Imports

- Group: stdlib, third-party, local (in that order)
- Alphabetical within each group
- Ruff auto-sorts on `--fix`

### Docstrings

- Use triple-quoted docstrings for all public functions, classes, modules
- Format: Google-style (Args, Returns, Raises, Example)
- Required for: Public APIs, models, utilities

## Testing

### Run tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_diabetes_outcome_logistic_regression.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing

# With markers
pytest tests/ -v -k "logistic"
```

### Coverage Requirements

- Minimum: **80%**
- Target: **90%+**
- Enforced by CI/CD

### Writing Tests

**Test file naming:** `test_<module>.py`

**Testing async operations:**
```python
import pytest
from insightful_ai_api import InsightfulAI

@pytest.mark.asyncio
async def test_async_predict():
    model = InsightfulAI(model_type="logistic_regression")
    # Set up training data
    model.fit(X_train, y_train)
    # Test async predict
    result = await model.async_predict([X_test_batch_1, X_test_batch_2])
    assert result.is_success
    assert isinstance(result.value, list)
```

**Mocking external dependencies:**
```python
from unittest.mock import patch, MagicMock

@patch("models.logistic_regression_model.LogisticRegression")
def test_model_initialization(mock_sklearn):
    from insightful_ai_api import InsightfulAI
    model = InsightfulAI(model_type="logistic_regression", C=0.5)
    # Verify sklearn was called with correct parameters
    assert model is not None
```

### Adding New Models

1. **Create model class** in `models/<name>_model.py`:
   ```python
   from models.model_interface import ModelInterface
   
   class MyModel(ModelInterface):
       """Your model implementation."""
       
       def __init__(self, param1: float = 1.0):
           self.param1 = param1
           self.model = None
       
       def fit(self, X, y) -> None:
           """Train the model."""
           # Implementation
           pass
       
       def predict(self, X):
           """Make predictions."""
           # Implementation
           pass
       
       def evaluate(self, X, y) -> float:
           """Evaluate model accuracy."""
           # Implementation
           pass
       
       async def async_fit(self, X_batches, y_batches) -> None:
           """Async training on batches."""
           # Implementation
           pass
       
       async def async_predict(self, X_batches):
           """Async predictions on batches."""
           # Implementation
           pass
       
       async def async_evaluate(self, X_batches, y_batches):
           """Async evaluation on batches."""
           # Implementation
           pass
   ```

2. **Register in API** (`insightful_ai_api.py`):
   ```python
   from models.my_model import MyModel
   
   # In __init__ method:
   elif model_type == "my_model":
       self.model = MyModel(**kwargs)
   ```

3. **Add test** in `tests/test_my_model.py`:
   ```python
   def test_my_model_training():
       from insightful_ai_api import InsightfulAI
       model = InsightfulAI(model_type="my_model", param1=2.0)
       # Set up training data
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       assert predictions is not None
   ```

4. **Update docs:**
   - Add to README.md model types table
   - Create usage example in templates/
   - Add tutorial in Tutorials/

## PR Workflow

### Before You Start

1. **Check for open PRs:** `gh pr list --state open`
2. **Verify main clean:** `git log main --oneline | head -1`
3. **Search codebase** for existing implementations (zero duplication policy)
4. **Create feature branch:** `git checkout -b feat/issue-NUMBER-description`

### During Development

- Keep scope small: **Max 5 files per PR** (excludes config files, generated files)
- Build frequently: Run tests locally
- Run tests: `pytest tests/ --cov=.`
- Update CHANGELOG.md with your changes

### Submitting PR

1. **Commit message format:**
   ```
   feat: Brief description (#ISSUE_NUMBER)
   
   Longer explanation of what changed and why.
   Include test coverage summary.
   ```

2. **Create PR via CLI:**
   ```bash
   git push origin feat/issue-NUMBER-description
   gh pr create --title "feat: Brief description (#ISSUE_NUMBER)" \
     --body "Detailed description, testing notes, design decisions"
   ```

3. **Wait for CI:** All checks must pass (ruff, pytest, type checking)

4. **Address feedback:** Push fixes to same branch (auto-updates PR)

5. **Merge:** Reviewer squashes + merges

## Release Process

### Version Bumping

InsightfulAI uses semantic versioning: **MAJOR.MINOR.PATCH**

- **MAJOR:** Breaking API changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Tag commit:**
   ```bash
   git tag v0.4.0
   git push origin v0.4.0
   ```
4. **Build and publish to PyPI:**
   ```bash
   pip install build twine
   python -m build
   twine upload dist/insightful*.whl
   ```

## Troubleshooting

### "ImportError: No module named 'insightful_ai_api'"

**Cause:** Package not installed in dev mode

**Fix:** Run `pip install -e .` from repository root

### "pytest: error: unrecognized arguments: --cov"

**Cause:** Coverage plugin not installed

**Fix:** Run `pip install -e .[dev]`

### "ModuleNotFoundError: No module named 'opentelemetry'"

**Cause:** Dependencies not installed

**Fix:** Run `pip install -e .[dev]`

## Questions?

- Open a GitHub Issue: [Issues](https://github.com/CraftedWithIntent/InsightfulAI/issues)
- Start a Discussion: [Discussions](https://github.com/CraftedWithIntent/InsightfulAI/discussions)
- Review architecture: [Design/](Design/)

---

**Thank you for making InsightfulAI better!**

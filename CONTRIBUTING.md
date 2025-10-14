# Contributing to ELF

Thank you for your interest in contributing to ELF (Electrical Load Forecasting)! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Error messages or screenshots

### Suggesting Features

We welcome feature suggestions! Please create an issue describing:
- The feature you'd like to see
- Why it would be useful
- How it might work
- Any examples from other tools

### Code Contributions

1. **Fork the repository**
2. **Create a branch** for your feature
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
6. **Push to your fork**
7. **Create a Pull Request**

## Development Setup

### 1. Clone Your Fork

```bash
git clone https://github.com/YOUR-USERNAME/ELF.git
cd ELF
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Development Tools

```bash
pip install pytest black flake8 mypy
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### Formatting

Use `black` for code formatting:
```bash
black *.py
```

### Linting

Check code quality with `flake8`:
```bash
flake8 *.py --max-line-length=100
```

## Project Structure

```
ELF/
├── app.py              # Streamlit application
├── models.py           # ML model implementations
├── data_utils.py       # Data processing utilities
├── config.py           # Configuration settings
├── example.py          # Command-line example
├── test_imports.py     # Import tests
├── requirements.txt    # Dependencies
├── Dockerfile         # Docker configuration
└── *.md               # Documentation
```

## Adding New Features

### Adding a New ML Model

1. **Add training method to `models.py`**:
```python
def train_your_model(self, X_train, y_train):
    """Train Your Model."""
    model = YourModel(parameters)
    model.fit(X_train, y_train)
    self.models['Your Model'] = model
    return model
```

2. **Update `train_all_models()`**:
```python
def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
    # ... existing models ...
    print("Training Your Model...")
    self.train_your_model(X_train, y_train)
```

3. **Update `config.py`**:
```python
MODEL_NAMES = [
    'Linear Regression',
    # ... existing models ...
    'Your Model'
]
```

4. **Test the model**

### Adding New Features

1. **Modify `data_utils.py`** to create new features:
```python
def preprocess_data(df, lookback=24):
    # ... existing features ...
    df['your_feature'] = calculate_your_feature(df)
    feature_cols.append('your_feature')
```

2. **Test with different datasets**

### Adding Data Sources

1. **Create new loader in `data_utils.py`**:
```python
def load_custom_data(filepath):
    """Load data from custom source."""
    df = pd.read_csv(filepath)
    # Process to match expected format
    return df
```

2. **Update `app.py`** to include option

## Testing

### Running Tests

Currently minimal tests exist. To add tests:

1. **Create test file**: `test_models.py`, `test_data_utils.py`
2. **Use pytest**:
```python
import pytest
from models import LoadForecastingModels

def test_linear_regression():
    # Your test code
    pass
```
3. **Run tests**:
```bash
pytest
```

### Manual Testing

1. **Test data generation**:
```bash
python -c "from data_utils import generate_sample_data; print(generate_sample_data(days=10))"
```

2. **Test models**:
```bash
python example.py
```

3. **Test UI**:
```bash
streamlit run app.py
```

## Documentation

### Updating Documentation

When adding features:
1. Update README.md with overview
2. Update USER_GUIDE.md with usage instructions
3. Update ARCHITECTURE.md if structure changes
4. Add docstrings to new functions/classes

### Documentation Style

```python
def function_name(param1, param2):
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> function_name(1, 2)
        3
    """
    pass
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tested locally
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No unnecessary files included

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How you tested the changes

## Screenshots
If applicable

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process

1. Maintainer reviews code
2. Feedback provided if needed
3. You make requested changes
4. Approved and merged

## Areas for Contribution

### High Priority

- [ ] Unit tests for all modules
- [ ] Real data loader (CSV, database)
- [ ] Model persistence (save/load)
- [ ] API endpoint for predictions
- [ ] More evaluation metrics
- [ ] Feature importance visualization

### Medium Priority

- [ ] Additional ML models
- [ ] Hyperparameter tuning
- [ ] Multi-step ahead forecasting
- [ ] Anomaly detection
- [ ] Confidence intervals
- [ ] Model explainability (SHAP)

### Nice to Have

- [ ] Web authentication
- [ ] Database integration
- [ ] Automated reporting
- [ ] Email notifications
- [ ] Mobile responsive design
- [ ] Dark mode UI

## Code Review Criteria

### What We Look For

✅ **Good**:
- Clear, readable code
- Appropriate comments
- Efficient algorithms
- Error handling
- Consistent style

❌ **Avoid**:
- Unclear variable names
- Excessive complexity
- No error handling
- Breaking existing features
- Large, unfocused changes

## Questions?

- Create an issue for questions
- Tag issues appropriately
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- README.md
- Release notes
- Project documentation

Thank you for contributing to ELF! 🎉

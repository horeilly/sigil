# Contributing to Sigil

Thank you for your interest in contributing to the Sigil project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/circle-packing.git
cd circle-packing
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

## Code Style

This project uses **Black** for code formatting with a line length of 100 characters.

### Before Committing

Always format your code with Black:
```bash
black sigil/ main.py test_basic.py examples/
```

Check that your code is formatted correctly:
```bash
black --check sigil/ main.py test_basic.py examples/
```

### Additional Quality Checks

**Type checking with mypy:**
```bash
mypy sigil/
```

**Linting with flake8:**
```bash
flake8 sigil/ main.py test_basic.py examples/
```

**Run tests:**
```bash
pytest
```

## Code Standards

- **Type Hints**: All functions and methods must have complete type annotations
- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Testing**: Add tests for new functionality (pytest framework)
- **Line Length**: Maximum 100 characters (enforced by Black)
- **Imports**: Use absolute imports within the package

## Pull Request Process

1. Create a new branch for your feature or bugfix
2. Make your changes following the code style guidelines
3. Add or update tests as needed
4. Run all quality checks (Black, mypy, flake8, pytest)
5. Update documentation if needed
6. Submit a pull request with a clear description of changes

## Questions?

Feel free to open an issue for any questions or concerns.
# Contributing to SIARE

Thank you for your interest in contributing to SIARE! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/siare.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install dev dependencies: `pip install -e ".[dev,full]"`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Run linting: `ruff check siare/`
5. Run type checking: `pyright siare/`
6. Commit your changes with a descriptive message
7. Push and create a pull request

## Code Style

- Follow PEP 8
- Use type hints for all public functions
- Write docstrings for public classes and functions
- Keep functions focused and small
- Add tests for new functionality

## Testing

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- Test file naming: `test_<module_name>.py`
- Test function naming: `test_<function>_<scenario>_<expected>`

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation as needed
- Ensure CI passes before requesting review
- Write a clear PR description

## Reporting Issues

When reporting bugs, please include:
- Python version
- SIARE version
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Questions?

Open a GitHub Discussion or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

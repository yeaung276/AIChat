# Dev command
uv run uvicorn main:app --reload --port 8080

# Test command
uv run pytest --cov=aichat --cov-report=html tests/

## Viewing test
open htmlcov/index.html


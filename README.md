# Dev command
uv run uvicorn aichat.app:app --reload --port 8080

# Test command
uv run pytest --cov=aichat --cov-report=html tests/

## Viewing test
open htmlcov/index.html

# Development in docker
`docker run`

# Training
### Dataset
link: `https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai/data`

### Generate data
 uv run python -m training.data

### Run evaluation
uv run python -m experiments.evaluate_llm
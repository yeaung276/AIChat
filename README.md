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

uv add vllm --extra-index-url https://wheels.vllm.ai/nightly --index-strategy unsafe-best-match --prerelease=allow

1. feedback form and its questions, ask gpt for the questions
2. implement the first situation updater
3. test app fully on colab
4. update design section to include justification. 
5. rework on implementation's auth and other inimportant parts and replace it with this contextual situation updater talking about single turn limitation and how this fix it.
6. Discuss about model choices and why that particular model for the situation context updater. link providfed
https://claude.ai/chat/9aa3d3c7-c213-4827-8f89-087c91e46d92
7. safety guardrails and its implementations, "keywords" detection.

## Forms
Form Title: Companion AI Video Chat — Session Feedback
Description: Thank you for taking part in this study. Please answer the following questions based on your conversation experience just now. There are no right or wrong answers.
Q1. The conversation felt natural and easy to follow.
Linear scale: 1 (Strongly Disagree) → 5 (Strongly Agree)
Q2. The AI's responses felt relevant to what I said.
Linear scale: 1 (Strongly Disagree) → 5 (Strongly Agree)
Q6. Overall, how satisfied were you with this conversation experience?
Linear scale: 1 (Very Unsatisfied) → 10 (Very Satisfied)
Q7. I would be willing to have another conversation with this AI.
Linear scale: 1 (Strongly Disagree) → 5 (Strongly Agree)
Q11. Is there anything else you'd like to share about your experience?
Paragraph (short answer, optional)

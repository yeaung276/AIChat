def build_prompt(situation: str, emotion: str, user: str, agent: str | None = None):
    prompt = (
        "Below is a situation you are in, paired with an input that provides current emotion "
        "and further context. Write an emotional and appropriate response.\n"
        "### Situation:\n"
        f"{situation}\n"
        "### Emotion:\n"
        f"{emotion}\n"
        "### User:\n"
        f"{user}\n"
        "### You:\n"
        f"{agent or ''}"
    )
    return prompt
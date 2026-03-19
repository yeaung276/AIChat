def build_prompt(situation: str, user: str, emotion: str | None = None, answer_type = "long", agent: str | None = None, extra_turns = []):
    extra_turns_str = "".join(
        f"### {turn['actor']}:\n{turn['message']}\n"
        for turn in (extra_turns or [])
    )
    prompt = (
        "Below is a situation you are in, paired with an input that provides current emotion "
        "and further context. Write an emotional and appropriate response.\n"
        "### Situation:\n"
        f"{situation}\n"
        + (f"### Emotion:\n{emotion}\n" if emotion else "")
        + extra_turns_str
        + "### User:\n"
        f"{user}\n"
        "### Answer Type:\n"
        f"{answer_type}\n"
        "### You:\n"
        f"{agent or ''}"
    )
    return prompt
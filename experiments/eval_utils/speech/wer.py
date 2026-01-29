from jiwer import wer

def evaluate_wer(refs: list[str], hyps: list[str]):
    return wer(refs, hyps)
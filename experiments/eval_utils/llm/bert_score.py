from bert_score import BERTScorer


scorer = BERTScorer(
    model_type="roberta-base",
    lang="en",
)


def evaluate_bertscore(preds, refs):
    P, R, F1 = scorer.score(preds, refs)

    return {
        "bertscore_p": P.mean().item(), # type: ignore
        "bertscore_r": R.mean().item(), # type: ignore
        "bertscore_f1": F1.mean().item(), # type: ignore
    }

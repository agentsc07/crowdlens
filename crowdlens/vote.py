from . import config


def vote(probs, threshold=config.THRESHOLD, spread=config.UNCLEAR_SPREAD) -> str:
    if not probs:
        return "unclear"
    if max(probs) - min(probs) > spread:
        return "unclear"
    return "yes" if sum(probs) / len(probs) >= threshold else "no"


def count(verdicts):
    yes = sum(v.label == "yes" for v in verdicts)
    unclear = sum(v.label == "unclear" for v in verdicts)
    return yes, yes + unclear

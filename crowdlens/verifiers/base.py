import math
from typing import Protocol


class Verifier(Protocol):
    def p_yes(self, crop_path: str, attribute: str) -> float: ...


def normalise(logit_yes: float, logit_no: float, logit_unclear: float):
    m = max(logit_yes, logit_no, logit_unclear)
    e = [math.exp(x - m) for x in (logit_yes, logit_no, logit_unclear)]
    s = sum(e)
    return tuple(x / s for x in e)

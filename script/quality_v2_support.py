"""Shared evidence contracts for the independent, broader offline audit."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    family: str
    title: str
    severity: str
    run: Callable[[Any], dict]
    inputs: Any = None
    level: str = "function"


class CheckFailure(AssertionError):
    def __init__(self, message, *, expected=None, observed=None):
        super().__init__(message)
        self.expected = expected
        self.observed = observed


def expect(observed, expected, message="Behavior differs from the customer contract"):
    if observed != expected:
        raise CheckFailure(message, expected=expected, observed=observed)


def require(condition, message, *, expected=None, observed=None):
    if not condition:
        raise CheckFailure(message, expected=expected, observed=observed)

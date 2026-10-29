"""Goal cards for continued customer conversations (no application changes)."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Journey:
    id: str
    area: str
    persona: str
    goal: str
    run: Callable[[Any], dict]
    planned_inputs: list

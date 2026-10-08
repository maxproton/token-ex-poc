"""Fixed tasks with machine-checkable answers. The task never changes, only the padding does."""
import re
from dataclasses import dataclass

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    expected: str  # lower case


TASKS = [
    Task("arith", "What is 17 multiplied by 23? Reply with only the number.", "391"),
    Task("capital", "What is the capital city of Australia? Reply with only the city name.", "canberra"),
    Task("symbol", "What is the chemical symbol for sodium? Reply with only the symbol.", "na"),
]


def check_answer(task: Task, text: str) -> bool:
    return re.search(rf"\b{re.escape(task.expected)}\b", (text or "").lower()) is not None

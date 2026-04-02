COUNTING_KEYWORDS = [
    "몇 개",
    "몇개",
    "개수",
    "몇 가지",
    "몇가지",
    "몇 종류",
    "몇종류",
    "몇 번",
    "몇번",
    "몇 마리",
    "몇마리",
    "몇",
]


def is_counting_question(question: str) -> bool:
    """Return True if the question is asking about a count/quantity."""
    return any(kw in question for kw in COUNTING_KEYWORDS)

"""Korean recycling keyword extraction & translation for GroundingDINO prompts.

질문에서 탐지 대상 키워드를 추출 → 영어로 변환.
GroundingDINO는 영어 프롬프트에서 성능이 월등히 좋으므로 한→영 변환 필수.

Usage:
    from utils.ko2en import extract_grounding_prompt

    prompt = extract_grounding_prompt(
        "사진에 보이는 재활용 가능한 플라스틱 병은 몇 개입니까?"
    )
    # → "plastic bottle"

    prompt = extract_grounding_prompt(
        "사진 속 흰색 텀블러의 재질은 무엇인가요?"
    )
    # → "white tumbler"
"""
from __future__ import annotations

import re

# ── Korean → English vocabulary ──────────────────────────────────────────────
# 긴 것이 먼저 매칭되도록 사용 시 길이 역순 정렬 필수

_OBJECT_KO_EN: dict[str, str] = {
    # ── 복합 명사 (긴 것 우선) ──
    "발포 스티로폼 상자": "styrofoam box",
    "발포 스티로폼": "styrofoam",
    "스티로폼 상자": "styrofoam box",
    "투명 플라스틱 컵": "clear plastic cup",
    "플라스틱 음료 용기": "plastic beverage container",
    "플라스틱 빨대": "plastic straw",
    "플라스틱 뚜껑": "plastic cap",
    "플라스틱 용기": "plastic container",
    "플라스틱 컵": "plastic cup",
    "플라스틱 병": "plastic bottle",
    "플라스틱 봉투": "plastic bag",
    "튜브형 용기": "tube container",
    "음료 페트": "beverage PET bottle",
    "음료 용기": "beverage container",
    "음료 컵": "beverage cup",
    "알루미늄 캔": "aluminum can",
    "음료캔": "beverage can",
    "철캔": "steel can",
    "금속 캔": "metal can",
    "골판지 상자": "cardboard box",
    "종이 상자": "paper box",
    "종이상자": "paper box",
    "비닐 포장재": "plastic wrapping",
    "비닐봉지": "plastic bag",
    "종이컵 홀더": "paper cup holder",
    "과자 상자": "snack box",
    "포장 상자": "packaging box",
    "유리 병": "glass bottle",
    "종이 포장지": "paper wrapper",
    "종이컵": "paper cup",
    "종이팩": "paper carton",
    "종이류": "paper products",

    # ── 단일 명사 ──
    "스티로폼": "styrofoam",
    "페트병": "PET bottle",
    "페트": "PET bottle",
    "플라스틱": "plastic",
    "유리병": "glass bottle",
    "유리": "glass",
    "골판지": "cardboard",
    "캔": "can",
    "종이": "paper",
    "비닐": "plastic bag",
    "병": "bottle",
    "컵": "cup",
    "상자": "box",
    "용기": "container",
    "봉지": "bag",
    "봉투": "bag",
    "뚜껑": "cap",
    "빨대": "straw",
    "포장지": "wrapper",
    "포장재": "packaging",
    "팩": "carton",
    "텀블러": "tumbler",
    "형광등": "fluorescent lamp",
    "전구": "light bulb",
    "풍선": "balloon",
    "빈병": "empty bottle",
    "재활용품": "recyclable item",
    "쓰레기": "trash",
    "망": "mesh bag",
}

_COLOR_KO_EN: dict[str, str] = {
    "흰색": "white",
    "하얀색": "white",
    "하얀": "white",
    "검은색": "black",
    "검정색": "black",
    "검정": "black",
    "빨간색": "red",
    "빨간": "red",
    "파란색": "blue",
    "파란": "blue",
    "초록색": "green",
    "초록": "green",
    "노란색": "yellow",
    "노란": "yellow",
    "주황색": "orange",
    "분홍색": "pink",
    "보라색": "purple",
    "은색": "silver",
    "갈색": "brown",
    "투명": "transparent",
    "투명한": "transparent",
}

# 사전은 길이 역순으로 캐시
_OBJECT_SORTED = sorted(_OBJECT_KO_EN.items(), key=lambda x: -len(x[0]))
_COLOR_SORTED = sorted(_COLOR_KO_EN.items(), key=lambda x: -len(x[0]))

# ── 질문 전처리 패턴 ────────────────────────────────────────────────────────

_LOCATION_PREFIXES = [
    "이 사진에 보이는",
    "사진에 보이는 재활용품 중에서",
    "사진에 보이는 재활용 가능한",
    "사진에 보이는 재활용품 중",
    "사진에 보이는",
    "사진 속 재활용품 중",
    "사진 속 재활용 가능한",
    "사진 속",
    "사진에서 재활용품으로 분류할 수 있는",
    "사진에서 재활용 가능한",
    "사진에서",
    "이 상자 안에 들어있는 재활용품 중",
    "책상 위에 있는 재활용 가능한",
    "책상 위에 놓인 재활용 가능한",
    "책상 위에 놓인",
    "책상 위에",
]

_QUESTION_SUFFIXES = re.compile(
    r"(?:"
    r"의\s*(?:재질|재활용\s*분류|종류|색상|색깔|개수|뚜껑|브랜드)"
    r"|은\s*(?:몇|어떤|무엇)"
    r"|는\s*(?:몇|어떤|무엇|총)"
    r"|이\s*(?:몇|어떤|무엇)"
    r"|[은는이가을를에서의]?\s*(?:몇\s*개|개수|몇\s*가지|몇\s*종류)"
    r").*$"
)

_PARTICLES = re.compile(r"[은는이가을를의에서으로로도만]+$")


# ── Core extraction ──────────────────────────────────────────────────────────

def _find_object(text: str) -> str | None:
    """텍스트에서 한국어 물품 키워드를 찾아 영어로 반환 (longest match)."""
    for ko, en in _OBJECT_SORTED:
        if ko in text:
            return en
    return None


def _find_color(text: str) -> str | None:
    """텍스트에서 색상 키워드를 찾아 영어로 반환."""
    for ko, en in _COLOR_SORTED:
        if ko in text:
            return en
    return None


def _find_object_position(text: str) -> tuple[str | None, int]:
    """물품 키워드와 그 시작 위치 반환."""
    for ko, en in _OBJECT_SORTED:
        idx = text.find(ko)
        if idx >= 0:
            return en, idx
    return None, -1


def extract_grounding_prompt(
    question: str,
    choices: dict[str, str] | None = None,
    fallback: str = "recyclable item",
) -> str:
    """한국어 질문에서 GroundingDINO용 영어 프롬프트 생성.

    Args:
        question: 한국어 질문 텍스트
        choices: {"a": "...", "b": "...", "c": "...", "d": "..."} 선택지 (선택)
        fallback: 키워드 미발견 시 기본 프롬프트

    Returns:
        영어 프롬프트 (예: "plastic bottle", "white tumbler")

    Examples:
        >>> extract_grounding_prompt("사진에 보이는 재활용 가능한 플라스틱 병은 몇 개입니까?")
        'plastic bottle'
        >>> extract_grounding_prompt("사진 속 흰색 텀블러의 재질은 무엇인가요?")
        'white tumbler'
        >>> extract_grounding_prompt("사진에 보이는 재활용품 중 알루미늄 캔은 몇 개 있나요?")
        'aluminum can'
        >>> extract_grounding_prompt("사진에 보이는 재활용 가능한 유리병은 몇 개입니까?")
        'glass bottle'
    """
    text = question.strip()

    # Step 1: 질문 원문에서 직접 물품 + 색상 매칭 (가장 정확)
    obj_en, obj_pos = _find_object_position(text)
    color_en = None

    if obj_en and obj_pos > 0:
        prefix_chunk = text[max(0, obj_pos - 5):obj_pos]
        color_en = _find_color(prefix_chunk)

    if obj_en:
        if color_en and color_en not in obj_en:
            return f"{color_en} {obj_en}"
        return obj_en

    # Step 2: prefix/suffix 제거 후 재시도
    cleaned = text
    for prefix in sorted(_LOCATION_PREFIXES, key=len, reverse=True):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    cleaned = _QUESTION_SUFFIXES.sub("", cleaned).strip()
    cleaned = _PARTICLES.sub("", cleaned).strip()

    obj_en = _find_object(cleaned)
    if obj_en:
        color_en = _find_color(cleaned)
        if color_en and color_en not in obj_en:
            return f"{color_en} {obj_en}"
        return obj_en

    # Step 3: 선택지에서 힌트 추출
    if choices:
        choice_text = " ".join(str(v) for v in choices.values())
        obj_en = _find_object(choice_text)
        if obj_en:
            return obj_en

    return fallback


def extract_grounding_prompt_from_row(
    row: dict,
    fallback: str = "recyclable item",
) -> str:
    """DataFrame row(dict)에서 바로 추출. question + a/b/c/d 컬럼 활용."""
    question = str(row.get("question", ""))
    choices = None
    if all(k in row for k in ("a", "b", "c", "d")):
        choices = {k: str(row[k]) for k in ("a", "b", "c", "d")}
    return extract_grounding_prompt(question, choices, fallback)


# ── CLI: 변환 결과 확인 ──────────────────────────────────────────────────────

def _demo():
    """데모: 몇 가지 예시 질문에 대한 변환 결과 출력."""
    samples = [
        "사진에 보이는 재활용 가능한 플라스틱 병은 몇 개입니까?",
        "사진 속 흰색 텀블러의 재질은 무엇인가요?",
        "사진에 보이는 재활용품 중 알루미늄 캔은 몇 개 있나요?",
        "사진에 보이는 재활용 가능한 유리병은 몇 개입니까?",
        "사진 속 재활용품 중 플라스틱 재질인 것은 무엇인가요?",
        "사진에 보이는 '빅파이' 과자 상자의 재활용 분류는 무엇인가요?",
        "사진에 보이는 재활용품 중에서 형광등의 개수는 몇 개인가요?",
        "사진에 보이는 초록색 병은 몇 개입니까?",
        "사진 속 투명 플라스틱 컵에 붙어 있는 노란색 라벨에는 어떤 그림이 그려져 있나요?",
        "사진에 보이는 재활용품 중 골판지 재질의 물건은 몇 개입니까?",
        "사진 속 재활용 쓰레기봉투 안에 들어있는 캔의 개수는 몇 개인가요?",
        "사진에 보이는 재활용 가능한 비닐봉지는 몇 개입니까?",
        "사진에 보이는 에그 클럽 샌드위치 포장지는 어떤 재질로 보이나요?",
        "사진 속 음료 컵의 재활용 분류는 무엇인가요?",
    ]
    print("=== Korean → English GroundingDINO Prompt ===\n")
    for q in samples:
        en = extract_grounding_prompt(q)
        print(f"  Q: {q}")
        print(f"  → {en}\n")


def _batch_csv(csv_path: str, limit: int = 0):
    """CSV 전체에 대한 변환 결과 통계."""
    import csv
    from collections import Counter

    prompts = Counter()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            en = extract_grounding_prompt_from_row(row)
            prompts[en] += 1

    print(f"=== Prompt distribution ({sum(prompts.values())} questions) ===\n")
    for prompt, cnt in prompts.most_common():
        print(f"  {cnt:5d}  {prompt}")
    print(f"\n  Unique prompts: {len(prompts)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Korean → English prompt converter")
    parser.add_argument("--csv", help="CSV 파일 경로 (배치 분석)")
    parser.add_argument("--limit", type=int, default=0,
                        help="CSV 최대 행 수 (0=전체)")
    parser.add_argument("--question", "-q", help="단일 질문 변환")
    args = parser.parse_args()

    if args.question:
        print(extract_grounding_prompt(args.question))
    elif args.csv:
        _batch_csv(args.csv, args.limit)
    else:
        _demo()

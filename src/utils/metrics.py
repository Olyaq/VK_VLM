import re

def normalize_one_word(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # take first token as "one word"
    return s.split(" ")[0] if s else ""

def exact_match_one_word(pred: str, gold: str) -> int:
    return int(normalize_one_word(pred) == normalize_one_word(gold))

def parse_choice_letter(text: str) -> str:
    # Returns first letter A-E found
    text = (text or "").strip().upper()
    for ch in text:
        if ch in ["A","B","C","D","E"]:
            return ch
    return ""

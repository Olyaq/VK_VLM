def ensure_image_tag(text: str) -> str:
    text = text or ""
    if "<image>" not in text:
        return "<image>\n" + text
    return text

def gqa_one_word_prompt(question_ru: str) -> str:
    # In deepvk training they used post-prompt "Ответь одним словом."
    # (see model card of deepvk/llava-saiga-8b)
    return ensure_image_tag(question_ru.strip()) + "\nОтветь одним словом."

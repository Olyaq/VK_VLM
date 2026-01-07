from typing import Any, Dict, Tuple
from .prompts import ensure_image_tag

def extract_llava_conversation(example: Dict[str, Any]) -> Tuple[str, str]:
    """Extract (user_text, assistant_text) from a LLaVA-style 'conversations' list.

    deepvk/LLaVA-Instruct-ru viewer shows 'conversations' entries with keys:
    - from: 'human' / 'gpt'
    - value: string (often starts with '<image>\n...')
    """
    conv = example.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2:
        raise ValueError("Example has no valid 'conversations' field")

    def norm_role(r: str) -> str:
        r = (r or "").lower()
        if r in {"human", "user"}:
            return "user"
        if r in {"gpt", "assistant"}:
            return "assistant"
        return r

    user_text = None
    assistant_text = None
    for msg in conv:
        role = norm_role(msg.get("from") or msg.get("role"))
        val = msg.get("value") or msg.get("content")
        if role == "user" and user_text is None:
            user_text = ensure_image_tag(val)
        elif role == "assistant" and user_text is not None:
            assistant_text = val
            break

    if user_text is None or assistant_text is None:
        raise ValueError("Could not extract user/assistant messages from 'conversations'")

    return user_text, assistant_text

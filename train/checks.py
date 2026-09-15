"""Python ports of the checkers in rust/deliberate (voice.rs, tests/conversation_quality.rs,
tests/proactive_live.rs). Every target line in the dataset must pass these, and every
teacher-generated sample is kept only if it passes them. Keep in step with the Rust.
"""

from __future__ import annotations

import json
import re

# voice.rs GENERIC
GENERIC_PHRASES = [
    "how are you",
    "how's it going",
    "hows it going",
    "how is it going",
    "how have you been",
    "how can i help",
    "how may i help",
    "what can i do for you",
    "what can i help",
    "is there anything",
    "anything specific",
    "anything else i can",
    "anything else you need",
    "let me know if",
]

# A few more assistant reflexes the dataset refuses in targets (stricter than runtime).
EXTRA_GENERIC = [
    "how are you doing",
    "how are things",
    "feel free to",
    "i'm here to help",
    "happy to help",
    "how can i assist",
    "assist you",
    "nice to meet you",
    "welcome back",
    "have a great day",
    "have a nice day",
    "hope you're",
    "hope you are",
    "as an ai",
    "language model",
]

# voice.rs EXAMPLE_TOKENS
EXAMPLE_TOKENS = [
    "ada", "john", "mukesh", "priya", "sam", "leo", "parser", "rust", "two days", "coffee",
    "boss", "yaju",
]

TOOL_NAMES = [
    "recall_person", "remember", "remember_name", "remember_fact", "forget_person",
    "remember_reminder", "list_reminders", "run_shortcut", "open_facetime", "send_message",
]


def words(text: str) -> list[str]:
    out = []
    for w in text.split():
        w = "".join(c for c in w if c.isalnum() or c == "'").lower()
        if w:
            out.append(w)
    return out


def is_generic(sentence: str, strict: bool = False) -> bool:
    flat = " ".join(words(sentence))
    phrases = GENERIC_PHRASES + (EXTRA_GENERIC if strict else [])
    return any(g in flat for g in phrases)


def split_sentences(text: str) -> list[str]:
    out, cur = [], ""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        cur += c
        if c in ".!?":
            while i + 1 < n and text[i + 1] in ".!?":
                i += 1
                cur += text[i]
            if i + 1 < n and text[i + 1] in "\"'”’":
                i += 1
                cur += text[i]
            s = cur.strip()
            if s:
                out.append(s)
            cur = ""
        i += 1
    s = cur.strip()
    if s:
        out.append(s)
    return out


def _contains_word(hay: str, needle: str) -> bool:
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return False
        before = hay[i - 1] if i > 0 else ""
        after = hay[i + len(needle)] if i + len(needle) < len(hay) else ""
        if not (before.isalnum()) and not (after.isalnum()):
            return True
        start = i + 1


def leaks_example(sentence: str, real: str) -> bool:
    lower = sentence.lower()
    real = real.lower()
    return any(_contains_word(lower, t) and not _contains_word(real, t) for t in EXAMPLE_TOKENS)


def has_emoji(s: str) -> bool:
    return any(
        0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF or ord(c) == 0xFE0F for c in s
    )


def style_problem(reply: str) -> str | None:
    """conversation_quality.rs style_problem, plus the runtime's generic filter."""
    lower = reply.lower()
    if "how are you doing today" in lower:
        return "banned opener"
    if has_emoji(reply):
        return "emoji"
    if "**" in reply or "`" in reply or "#" in reply:
        return "markdown"
    if any(t in lower for t in ["recall_person", "remember_name", "remember_fact", "forget_person"]):
        return "tool call as text"
    if any(t in lower for t in ["<tool_call>", "tool_response", "{\"name\""]):
        return "tool call as text"
    lines = reply.splitlines()
    for i, l in enumerate(lines):
        t = l.lstrip()
        if i > 0 and (t.startswith("- ") or t.startswith("* ") or (t[:1].isdigit() and t[1:2] == ".")):
            return "list"
    for s in split_sentences(reply):
        if is_generic(s):
            return "generic"
    return None


def one_sentence(reply: str) -> bool:
    return len(split_sentences(reply)) == 1


def greets(reply: str) -> bool:
    l = reply.lower()
    w = set(words(l))
    return (
        any(g in w for g in ["hi", "hello", "hey", "greetings", "welcome", "hiya"])
        or "nice to see" in l or "good to see" in l or "nice to meet" in l or "great to see" in l
    )


def target_ok(reply: str, real: str, max_sentences: int = 2, allow_greet: bool = True) -> str | None:
    """The dataset's own gate for a spoken target. Returns why it fails, or None."""
    if not reply.strip():
        return "empty"
    p = style_problem(reply)
    if p:
        return p
    if "\n" in reply.strip():
        return "newline"
    sents = split_sentences(reply)
    if len(sents) > max_sentences:
        return f"{len(sents)} sentences"
    for s in sents:
        if is_generic(s, strict=True):
            return "generic(strict)"
        if leaks_example(s, real):
            return "example leak"
    if reply.count("?") > 1:
        return "two questions"
    if reply.startswith('"') or reply.startswith("Glydi:"):
        return "quoted"
    if not allow_greet and greets(reply):
        return "greets"
    return None


def parse_tool_calls(content: str) -> list[dict]:
    """Tool calls in a raw Qwen completion (<tool_call>{...}</tool_call>)."""
    out = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.S):
        try:
            out.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return out

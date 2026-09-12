"""Remembering things about people without being asked to.

The bot had a `remember_fact` tool from the start and never once used it: two
people enrolled, twelve face embeddings, zero facts. Models are reluctant to
interrupt a friendly exchange to file paperwork, and asking one to decide
mid-sentence whether something is worth keeping competes directly with the job
of replying quickly.

So this runs *after* the bot has finished speaking, on a background thread,
reading the exchange that just happened and extracting anything durable. It is
entirely off the critical path -- if it is slow, or fails, the conversation is
unaffected. The cost is one small extra call per turn.

What counts as durable is deliberately narrow. "I'm a teacher" is worth keeping.
"I'm tired today" is not: it will be false tomorrow, and a bot that greets you
by recalling your bad mood from last week is unsettling rather than clever.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request

from loguru import logger

EXTRACT_PROMPT = """\
You extract durable facts about a person from a snippet of conversation.

Keep only things that will still be true in a month and that the person would \
expect a friendly acquaintance to remember: their job or year group, where they \
live or study, family, hobbies, preferences, projects they are working on, \
things they explicitly ask you to remember.

Discard: anything about the present moment (mood, weather, what they are doing \
right now), anything you inferred rather than heard, pleasantries, and anything \
sensitive they did not clearly volunteer -- health, beliefs, money.

Write each fact as one short sentence in the third person, starting with their \
name.

People they mention by name go in "relations", not in facts: {"relation": \
"friend", "other": "Sony"} means "their friend is Sony". Use one plain word for \
the relation (friend, brother, sister, mother, father, wife, husband, son, \
daughter, colleague, boss, teacher, classmate, neighbour, partner). Only when a \
name and a relation were both actually said.

Return strict JSON: {"facts": ["..."], "relations": [{"relation": "...", \
"other": "..."}]}. Return empty lists if there is nothing worth keeping -- that \
is the common case and is fine."""

# Extraction goes to the same local server as the conversation by default, so
# a fully local install stays fully local. It is a separate, smaller call than
# the conversation, so the model can differ: GLYDI_MEMORY_MODEL overrides it.
# When the conversation runs on Gemini, extraction rides the same key.
LLM = os.environ.get("GLYDI_LLM", "local").strip().lower()
LOCAL_URL = os.environ.get("GLYDI_LOCAL_LLM_URL", "http://localhost:11434/v1").rstrip("/")
LOCAL_MODEL = os.environ.get("GLYDI_LOCAL_MODEL", "qwen2.5:3b")
# Only Gemini has its own extractor; every other conversation provider
# extracts on the local server, so the model name must be a local one.
MODEL = os.environ.get(
    "GLYDI_MEMORY_MODEL",
    "gemini-3.1-flash-lite" if LLM == "gemini" else LOCAL_MODEL,
)
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/"


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def _ask_local(user: str) -> str:
    data = _post_json(
        f"{LOCAL_URL}/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": user},
            ],
            "max_tokens": 200,
            "temperature": 0.0,
            # Every OpenAI-compatible local server honours this; it is the
            # difference between JSON and JSON wrapped in a helpful sentence.
            "response_format": {"type": "json_object"},
            **({"reasoning_effort": "none"} if MODEL.startswith("qwen3") else {}),
        },
        {},
    )
    return data["choices"][0]["message"].get("content") or ""


def _ask_gemini(user: str) -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        return ""
    data = _post_json(
        f"{GEMINI_ENDPOINT}{MODEL}:generateContent",
        {
            "system_instruction": {"parts": [{"text": EXTRACT_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": 200, "temperature": 0.0},
        },
        {"X-goog-api-key": key},
    )
    parts = data["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _parse_facts(text: str) -> list[str]:
    return _parse(text)[0]


def _parse(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """(facts, relations) from the extractor's JSON; tolerant of fences."""
    text = text.strip()
    if not text:
        return [], []
    # Models wrap JSON in fences more often than not.
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    if text.lstrip().startswith("json"):
        text = text.lstrip()[4:]
    data = json.loads(text)
    facts = [f.strip() for f in data.get("facts", []) if isinstance(f, str) and f.strip()]
    relations = []
    for r in data.get("relations", []) or []:
        if isinstance(r, dict) and str(r.get("relation", "")).strip() and str(r.get("other", "")).strip():
            relations.append((str(r["relation"]).strip().lower(), str(r["other"]).strip()))
    return facts, relations


def _extract(name: str, said: str, replied: str) -> list[str]:
    return _extract_all(name, said, replied)[0]


def _extract_all(name: str, said: str, replied: str) -> tuple[list[str], list[tuple[str, str]]]:
    user = (
        f"The person is called {name}.\n\n{name} said: {said}\n"
        f"You replied: {replied}"
    )
    if LLM == "local":
        text = _ask_local(user)
    elif LLM == "gemini":
        text = _ask_gemini(user)
    else:
        # Claude / OpenAI conversations: there is no matching cheap extractor
        # wired up, and we will not quietly send transcripts to a third service
        # the operator did not choose. Fall back to the local server if it is
        # there; otherwise extraction is simply off.
        try:
            text = _ask_local(user)
        except Exception:  # noqa: BLE001
            return [], []
    return _parse(text)


def remember_in_background(store, person_id: str, name: str, said: str,
                           replied: str) -> None:
    """Fire and forget. Never raises into the conversation."""

    def work() -> None:
        try:
            facts, relations = _extract_all(name, said, replied)
            for relation, other in relations:
                if other.strip().lower() == name.strip().lower():
                    continue
                if store.relate(person_id, relation, other):
                    logger.info(f"remembered: {name}'s {relation} is {other}")
            if not facts:
                return
            existing = {f.lower() for f in (store.get(person_id).facts or ())}
            for fact in facts:
                # Cheap dedupe. Without it the same fact accumulates every time
                # the subject comes up, and the recall answer turns into a list
                # of near-identical sentences.
                if fact.lower() in existing:
                    continue
                store.remember(person_id, fact)
                existing.add(fact.lower())
                logger.info(f"remembered about {name}: {fact}")
        except Exception as exc:  # noqa: BLE001 -- memory must never break talking
            logger.debug(f"fact extraction skipped: {exc}")

    threading.Thread(target=work, name="glydi-memory", daemon=True).start()


CONDENSE_PROMPT = """\
You keep a running summary of a spoken conversation between Glydi (a robot) and \
the people in the room, so Glydi can remember what was said earlier once the \
transcript no longer fits. Merge the previous summary with the new turns into \
one plain paragraph of at most 120 words: who said what that matters, anything \
asked of Glydi, anything unresolved. Keep names. Drop greetings and filler. \
Return strict JSON: {"summary": "..."}."""


def _turns_text(messages: list) -> str:
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if content.startswith("[room]"):
            content = content.split("\n\n", 1)[-1]
        lines.append(f"{'Glydi' if role == 'assistant' else 'Person'}: {content.strip()}")
    return "\n".join(lines)


def condense(previous: str, dropped: list) -> str:
    """Fold turns that no longer fit the context into the running summary."""
    turns = _turns_text(dropped)
    if not turns.strip():
        return previous
    user = (f"Previous summary: {previous or '(none)'}\n\nNew turns:\n{turns}")
    data = _post_json(
        f"{LOCAL_URL}/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": CONDENSE_PROMPT},
                {"role": "user", "content": user},
            ],
            "max_tokens": 220,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        {},
    )
    text = data["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    summary = str(json.loads(text).get("summary", "")).strip()
    return summary or previous

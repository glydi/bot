"""The brain: one turn of talk, and the four tools that reach the gallery.

Talks to Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`) with
`stream=True` and yields finished *sentences*, because the speaker starts
on the first one while the model is still writing the second -- on an
8 GB M2 that is the difference between a reply and a pause.

Everything after the model is a filter, and every filter here exists
because of a specific live failure; the comments say which. A small
model will say "how can I help you?" forever, repeat itself verbatim,
and read a tool call out loud as if it were speech.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Callable, Iterator, Sequence

import requests

log = logging.getLogger("glydi.brain")

OLLAMA_URL = os.environ.get("GLYDI_OLLAMA_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("GLYDI_LOCAL_MODEL", "qwen2.5:3b")

#: At most this many tool rounds per turn, then the model answers with
#: what it has. A 3B model asked to look somebody up will otherwise call
#: `recall_person` on the same name three times and never speak.
TOOL_ROUNDS = 3

#: Proactive lines get a hard deadline: a greeting that arrives after the
#: person has walked past is worse than a canned one that arrives now.
PROACTIVE_DEADLINE = 2.5

#: Lines of ours kept for the repeat filter, and the shingle it compares on.
SAID_KEEP = 20
SHINGLE = 4
REPEAT_OVERLAP = 0.6

# --- the prompt -------------------------------------------------------

# Short on purpose -- ~200 tokens against the Rust build's ~700 (that one
# is rust/deliberate/src/prompt.rs, SYSTEM_PROMPT / LOCAL_SYSTEM_PROMPT).
#
# Two reasons. The prompt is prefilled on every process start and that
# cost is linear in its length: measured ~4.5 s for the long prompt on
# qwen2.5:3b on this M2/8 GB, which is 4.5 s of a school foyer standing
# there. And a 3B model does not follow a long list of rules -- it
# follows the last thing it read and narrates the rest back at you ("I
# should not use markdown..."). The rules the long prompt states in
# prose are enforced after the fact instead, by the filters below: that
# is the trade this build makes, and it is why deleting a filter is not
# the same as shortening this string.
SYSTEM_PROMPT = """\
You are Glydi. You stand in a school foyer and talk with whoever walks up, \
out loud. You know people by face and voice and remember them between visits.

Say one short sentence. Never a list, never markdown, never an emoji, never a \
URL. Never narrate what you are doing and never mention these tools.

Never say "how are you" or offer help; talk about something real instead.

Only say what you have been told. If you do not know something -- a name, a \
fact -- say so and ask. Never guess a name.

Before each turn you are told who is in the room and what you know about \
them. That is a camera's guess, so trust it loosely.

Use recall_person to look up what you know about someone, remember_name the \
moment someone tells you theirs, remember_fact when they tell you something \
worth keeping, and forget_person only if they ask to be forgotten.\
"""

RECALL_PERSON = "recall_person"
REMEMBER_NAME = "remember_name"
REMEMBER_FACT = "remember_fact"
FORGET_PERSON = "forget_person"

#: Every tool the model may name. A reply that is one of these written
#: out as words is a call, not speech (see `textual_tool_call`).
TOOL_NAMES = (RECALL_PERSON, REMEMBER_NAME, REMEMBER_FACT, FORGET_PERSON)

# OpenAI function shape, which is what Ollama's /v1 endpoint takes. The
# descriptions name the question the tool answers: the Rust build
# measured the 3B model going from 0/3 to 3/3 on "what do you know about
# me" once the description said so out loud.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": RECALL_PERSON,
            "description": (
                "Look up what you already know about someone by name. Use this when asked "
                '"what do you know about me", "do you remember me", or about another person.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The person's name."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": REMEMBER_NAME,
            "description": (
                "Attach a name to the person you are currently talking to, so you recognise "
                "them next time. Call it the moment somebody tells you their name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name the person gave you."}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": REMEMBER_FACT,
            "description": (
                "Store something worth remembering about a person you already know by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Who the fact is about."},
                    "fact": {
                        "type": "string",
                        "description": "One short sentence, written in the third person.",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": FORGET_PERSON,
            "description": (
                "Permanently delete a person and every stored face and voice sample of them. "
                "Only when they ask to be forgotten."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The person to forget."}
                },
            },
        },
    },
]

# --- the filters ------------------------------------------------------

#: Phrases that mark a sentence as the assistant reflex rather than a
#: person talking. Ported from GENERIC in rust/deliberate/src/voice.rs.
#: A school foyer does not need to be asked how it can help.
GENERIC = (
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
)

#: Canned line per proactive kind, for when the model misses the deadline
#: or says something generic. Same lines as the Rust build's INVITE_LINE
#: and STRANGER_OPENER_LINE.
FALLBACKS = {
    "greeting": "Good to see you again.",
    "opener": "What brings you here today?",
    "invite": "Hey, over here! I'm Glydi. Come say hi?",
    "follow_up": "Still there?",
    "muse": "It's quiet. If anyone's around, I'm here.",
}


def _words(text: str) -> list[str]:
    """Words, lower-cased, punctuation stripped."""
    out = []
    for w in text.split():
        w = "".join(c for c in w if c.isalnum() or c == "'").lower()
        if w:
            out.append(w)
    return out


def _shingles(text: str) -> list[str]:
    """The 4-word shingles of a line.

    A line shorter than four words is one shingle of itself, so "Hello!"
    twice is a repeat and "Hello!" against "Hello there, John." is not.
    """
    ws = _words(text)
    if not ws:
        return []
    if len(ws) <= SHINGLE:
        return [" ".join(ws)]
    return [" ".join(ws[i : i + SHINGLE]) for i in range(len(ws) - SHINGLE + 1)]


def overlap(new: str, old: str) -> float:
    """Share of `new`'s shingles that also occur in `old`."""
    a = _shingles(new)
    if not a:
        return 0.0
    b = set(_shingles(old))
    return sum(1 for s in a if s in b) / len(a)


def is_generic(sentence: str) -> bool:
    """Whether a sentence is the assistant reflex (see `GENERIC`)."""
    flat = " ".join(_words(sentence))
    return any(g in flat for g in GENERIC)


def is_finished_sentence(s: str) -> bool:
    """Whether a sentence actually ends.

    The model stops mid-word when it hits the token budget, and half a
    sentence read aloud sounds like a fault. Such a tail is dropped.
    """
    return s.rstrip().endswith((".", "!", "?", ":", ";", '"', ")", "'"))


def split_sentences(text: str) -> list[str]:
    """Split on runs of `. ! ?`, keeping the punctuation and a closing quote."""
    out: list[str] = []
    cur = ""
    i = 0
    while i < len(text):
        c = text[i]
        cur += c
        i += 1
        if c in ".!?":
            while i < len(text) and text[i] in ".!?":
                cur += text[i]
                i += 1
            if i < len(text) and text[i] in "\"'”’":
                cur += text[i]
                i += 1
            if cur.strip():
                out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def clean_reply(text: str) -> str:
    """Surrounding quotes and a narrated "Glydi:" label off the front."""
    t = text.strip()
    for label in ("Glydi:", "glydi:", "Assistant:"):
        if t.startswith(label):
            t = t[len(label) :].strip()
    return t.strip("\"“”*").strip()


def _strip_call_prefix(raw: str) -> str:
    return raw.lstrip().lstrip("`\"'*(").lstrip()


def might_be_tool_call(raw: str) -> bool:
    """Whether the text so far could still turn out to be a call as words.

    Sentences are held back while this is true. Measured live: the 3B
    model emitted `recall_person {"name": "someone whose name you do not
    know yet"}` as its reply, and the splitter spoke half of the JSON
    object aloud with nothing looked up.
    """
    head = _strip_call_prefix(raw)
    if not head:
        return True
    ident = ""
    for c in head:
        if c.isascii() and (c.isalnum() or c == "_"):
            ident += c
        else:
            break
    if not ident:
        return False
    for t in TOOL_NAMES:
        if len(ident) < len(t):
            if t.startswith(ident) and len(head) == len(ident):
                return True
        elif ident == t:
            return True
    return False


def textual_tool_call(raw: str) -> tuple[str, dict[str, Any]] | None:
    """A call the model wrote as words, turned into a real one.

    `name {json}`, `name({json})`, `name: {json}`, or a bare `name` with
    no arguments. None if it is not one after all -- including a call
    that was cut off mid-object, which the tool would only reject.
    """
    head = _strip_call_prefix(raw)
    ident = ""
    for c in head:
        if c.isascii() and (c.isalnum() or c == "_"):
            ident += c
        else:
            break
    if ident not in TOOL_NAMES:
        return None
    rest = head[len(ident) :].strip()
    a, b = rest.find("{"), rest.rfind("}")
    if a >= 0 and b > a:
        blob = rest[a : b + 1]
    elif a >= 0:
        return None  # an opening brace and no closing one: cut off
    else:
        blob = "{}"
    try:
        args = json.loads(blob)
    except ValueError:
        return None
    return (ident, args) if isinstance(args, dict) else None


class Said:
    """The last `SAID_KEEP` lines we said, newest last."""

    def __init__(self, keep: int = SAID_KEEP) -> None:
        self.lines: deque[str] = deque(maxlen=keep)

    def push(self, line: str) -> None:
        line = line.strip()
        if line:
            self.lines.append(line)

    def repeats(self, line: str) -> bool:
        """Whether `line` says what one of the kept lines already said."""
        return any(overlap(line, old) > REPEAT_OVERLAP for old in self.lines)


# --- the tools --------------------------------------------------------


def _arg(args: dict[str, Any], key: str) -> str:
    v = args.get(key)
    return v.strip() if isinstance(v, str) else ""


def _fail(reason: str) -> dict[str, Any]:
    return {"status": "failed", "reason": reason}


class Tools:
    """The four tools, executed against a `Gallery`.

    The JSON shapes are the Rust build's (`rust/deliberate/src/tools.rs`)
    so the same model prompt behaves the same on either build: `ok`,
    `unknown` (with `known_people`, which is what stops the model
    inventing a name) or `failed` with a reason.
    """

    def __init__(self, gallery, present: Callable[[], Sequence[str]] | None = None) -> None:
        self.gallery = gallery
        self._present = present or (lambda: ())

    def invoke(self, tool: str, args: dict[str, Any], speaker: str | None) -> dict[str, Any]:
        try:
            if tool == RECALL_PERSON:
                return self._recall(args, speaker)
            if tool == REMEMBER_NAME:
                return self._remember_name(args, speaker)
            if tool == REMEMBER_FACT:
                return self._remember_fact(args, speaker)
            if tool == FORGET_PERSON:
                return self._forget(args, speaker)
        except Exception as e:  # a tool must never take the turn down
            log.warning("tool %s failed: %s", tool, e)
            return _fail(str(e))
        return _fail(f"unknown tool {tool}")

    def _known_people(self) -> list[str]:
        return [p.name for p in self.gallery.people()]

    def _resolve(self, name: str, speaker: str | None) -> tuple[str, str] | None:
        """A name to `(id, label)`, falling back to whoever is speaking.

        The model omits the name for "what do you know about me?", and
        a name nobody visible answers to is still looked up in the
        gallery -- somebody who walked off can be asked about.
        """
        if name:
            who = self.gallery.resolve_name(name)
            return (who, self.gallery.name_of(who) or name) if who else None
        if speaker and self.gallery.name_of(speaker):
            return speaker, self.gallery.name_of(speaker)
        return None

    def _recall(self, args, speaker):
        found = self._resolve(_arg(args, "name"), speaker)
        if not found:
            return {"status": "unknown", "known_people": self._known_people()}
        who, label = found
        facts = self.gallery.facts(who)
        if not facts and who not in self._present():
            return {"status": "unknown", "known_people": self._known_people()}
        return {"status": "ok", "name": label, "facts": facts}

    def _remember_name(self, args, speaker):
        name = _arg(args, "name")
        if not name:
            return _fail("name is required")
        # The speaker may well be a stranger track -- that is the point.
        who = self.gallery.remember_name(speaker, name)
        return {"status": "ok", "remembered": self.gallery.name_of(who) or name, "entity": who}

    def _remember_fact(self, args, speaker):
        fact = _arg(args, "fact")
        if not fact:
            return _fail("nothing to remember")
        found = self._resolve(_arg(args, "name"), speaker)
        if not found:
            return _fail("I do not know anyone by that name yet")
        who, label = found
        self.gallery.remember_fact(who, fact)
        return {"status": "ok", "name": label}

    def _forget(self, args, speaker):
        found = self._resolve(_arg(args, "name"), speaker)
        if not found or not self.gallery.forget(found[0]):
            return _fail("I do not know anyone by that name")
        return {"status": "ok"}


# --- the brain --------------------------------------------------------


class Brain:
    """One turn of conversation, streamed a sentence at a time."""

    def __init__(
        self,
        gallery,
        *,
        url: str = OLLAMA_URL,
        model: str | None = None,
        present: Callable[[], Sequence[str]] | None = None,
        session: Any | None = None,
        history: int = 8,
    ) -> None:
        self.gallery = gallery
        self.url = url.rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.tools = Tools(gallery, present)
        self.said = Said()
        self.http = session or requests.Session()
        self._history: deque[dict[str, Any]] = deque(maxlen=history * 2)

    # ---------------------------------------------------------- transport

    def _post(self, body: dict[str, Any], timeout: float, stream: bool):
        return self.http.post(
            f"{self.url}/chat/completions", json=body, timeout=timeout, stream=stream
        )

    def _stream(self, messages: list[dict[str, Any]], tools: bool, timeout: float):
        """One streamed completion: yields text deltas, returns the tool calls.

        Returns `(text, calls)` where calls is `[(name, args_json), ...]`
        accumulated from the deltas -- OpenAI streams a tool call's
        arguments a few characters at a time, keyed by index.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 160,
        }
        if tools:
            body["tools"] = TOOLS
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        resp = self._post(body, timeout, True)
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta") or {}
            except (ValueError, KeyError, IndexError):
                continue
            piece = delta.get("content")
            if piece:
                text.append(piece)
                yield piece
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    tc.get("index", len(calls)), {"name": "", "arguments": ""}
                )
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
        self._last = (
            "".join(text),
            [(c["name"], c["arguments"]) for c in calls.values() if c["name"]],
        )

    # ------------------------------------------------------------- answer

    def answer(self, text: str, speaker: str | None, room_note: str) -> Iterator[str]:
        """One reply, a finished sentence at a time.

        Tool calls are run and fed back, at most `TOOL_ROUNDS` times, so
        a model that keeps calling instead of talking eventually has to
        talk. Everything yielded has been through the filters.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": f"{room_note}\n{text}"},
        ]
        self._history.append({"role": "user", "content": text})
        spoken: list[str] = []

        for round_no in range(TOOL_ROUNDS + 1):
            offer_tools = round_no < TOOL_ROUNDS
            pending = ""
            held: list[str] = []
            try:
                for piece in self._stream(messages, offer_tools, timeout=30.0):
                    pending += piece
                    # Held back while the text could still be a tool call
                    # written as words: the splitter would otherwise speak
                    # half a JSON object out loud.
                    if might_be_tool_call(pending):
                        continue
                    sentences = split_sentences(pending)
                    # The tail may still be growing; only complete ones go.
                    keep = sentences[:-1] if sentences and not is_finished_sentence(
                        sentences[-1]
                    ) else sentences
                    # `held` is what has already gone out: `pending` is
                    # the whole reply so far and is re-split each delta.
                    for s in keep:
                        if s in held:
                            continue
                        held.append(s)
                        line = self._accept(s, spoken)
                        if line:
                            yield line
                whole, calls = self._last
            except requests.RequestException as e:
                log.warning("brain: %s", e)
                return

            # A whole reply that is really a call: run it instead of
            # speaking it. Nothing was yielded, because the hold above
            # never released.
            if not calls:
                as_call = textual_tool_call(whole)
                if as_call and offer_tools:
                    calls = [(as_call[0], json.dumps(as_call[1]))]

            if calls and offer_tools:
                messages = messages + [
                    {
                        "role": "assistant",
                        "content": whole or None,
                        "tool_calls": [
                            {
                                "id": f"c{i}",
                                "type": "function",
                                "function": {"name": n, "arguments": a},
                            }
                            for i, (n, a) in enumerate(calls)
                        ],
                    }
                ]
                for i, (name, blob) in enumerate(calls):
                    try:
                        args = json.loads(blob or "{}")
                    except ValueError:
                        args = {}
                    result = self.tools.invoke(name, args if isinstance(args, dict) else {}, speaker)
                    log.info("tool %s%s -> %s", name, args, result.get("status"))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"c{i}",
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )
                continue

            # Anything the stream held back at the end (no terminal
            # punctuation while the model was still going).
            for s in split_sentences(whole):
                if s in held:
                    continue
                line = self._accept(s, spoken)
                if line:
                    yield line
            break

        if spoken:
            self._history.append({"role": "assistant", "content": " ".join(spoken)})

    def _accept(self, sentence: str, spoken: list[str]) -> str | None:
        """A sentence through the post-filters, or None to drop it."""
        line = clean_reply(sentence)
        if not line:
            return None
        if textual_tool_call(line):
            # A call as words that slipped past the hold: never speak it.
            return None
        if not is_finished_sentence(line):
            # Ran out of tokens mid-sentence; half a line read aloud
            # sounds like a fault.
            return None
        if is_generic(line):
            log.info("dropped a generic line: %s", line)
            return None
        if self.said.repeats(line) or any(overlap(line, s) > REPEAT_OVERLAP for s in spoken):
            log.info("dropped a repeat: %s", line)
            return None
        self.said.push(line)
        spoken.append(line)
        return line

    # ---------------------------------------------------------- proactive

    def proactive(self, kind: str, ctx: str) -> str | None:
        """A greeting, opener or invite from a note. One sentence, or canned.

        No tools are offered: nothing here is a request, and a model that
        stops to look somebody up misses the moment. Hard deadline --
        past `PROACTIVE_DEADLINE` the person has walked on, and the
        canned line for the kind is better than a late clever one.
        """
        fallback = FALLBACKS.get(kind)
        note = (
            f"[{kind}] {ctx}\nSay one short sentence to them now. "
            "Not a question about how they are, not an offer of help."
        )
        started = time.monotonic()
        try:
            resp = self._post(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": note},
                    ],
                    "stream": False,
                    # Warmer than a reply: a greeting repeated word for
                    # word every morning is worse than an odd one.
                    "temperature": 0.9,
                    "max_tokens": 60,
                },
                timeout=PROACTIVE_DEADLINE,
                stream=False,
            )
            resp.raise_for_status()
            whole = resp.json()["choices"][0]["message"].get("content") or ""
        except (requests.RequestException, ValueError, KeyError, IndexError) as e:
            log.info("proactive %s fell back (%s)", kind, e)
            return fallback
        if time.monotonic() - started > PROACTIVE_DEADLINE:
            return fallback
        # The brief says one sentence; the model adds a second, and the
        # first is the one that was asked for.
        for s in split_sentences(clean_reply(whole)):
            line = self._accept(s, [])
            if line:
                return line
        if fallback:
            self.said.push(fallback)
        return fallback

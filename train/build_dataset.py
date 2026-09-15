#!/usr/bin/env python3
"""Build train/data/{train,valid}.jsonl for the GLYDI LoRA.

Every example is an OpenAI-style chat (`messages`, plus `tools` on utterance turns)
that mirrors what rust/deliberate sends at runtime:

  system  = train/system_short.txt
  user    = "[room] <note>\n\n<who> says: <utterance>"   (Conversation::prefix_note)
            with the runtime's note additions (NOTE_STRANGER_SPEAKING, NOTE_ABSENT_PERSON,
            NOTE_ONLY_NAME, NOTE_REACT_FIRST, NOTE_NOTHING_KNOWN, crowd line) and the
            utterance-level hints (NAME_ANSWER_HINT, the introduced note, NOTE_ALREADY_GREETED,
            the self note) exactly as deliberator.rs writes them.
  proactive user = Proactive::note(...) from voice.rs, no tools (deliberator sends none).

The LAST assistant message is the training target (mlx_lm.lora --mask-prompt masks
everything before it). Tool flows therefore come in two examples each: one ending in the
tool call, one ending in the spoken line after the tool result.

Sources: (i) hand-written seed dialogues (SEED_* tables below), (ii) programmatic variation
over names / facts / times / crowds, (iii) optionally teacher samples from train/teacher.py
(--teacher file.jsonl) that already passed checks.py.

Run:  train/.venv/bin/python train/build_dataset.py [--target 3200] [--seed 1] [--teacher f]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import checks  # noqa: E402

SYSTEM = (HERE / "system_short.txt").read_text().strip()

# ---------------------------------------------------------------------------
# Runtime strings, verbatim from rust/deliberate (prompt.rs, deliberator.rs, voice.rs)
# and rust/mind/src/view.rs. Keep in step.
# ---------------------------------------------------------------------------
NOBODY = ("Nobody is visible right now. To answer about anyone who is not here, "
          "call recall_person.")
NOBODY_DARK = ("It is dark: the camera cannot see anyone right now. To answer about anyone "
               "who is not here, call recall_person.")
NOTE_STRANGER_SPEAKING = (
    "The one speaking is the stranger. If their words include their name (\"I'm Ada\", "
    "\"it's Mukesh actually\"), call remember_name with just that name before you say "
    "anything. Reply with the tool call only.")
NOTE_ABSENT_PERSON = (
    "If they ask about a person who is not listed above, call recall_person with that name "
    "first; do not say you do not know them until it has answered. Reply with the tool call "
    "only.")
NOTE_ALREADY_GREETED = ("{name} is answering the hello you already said. Skip the hello this "
                        "time and just ask {name} something.")
NOTE_REACT_FIRST = ("React to what {name} just said before anything else; the facts above can "
                    "wait.")
NOTE_ONLY_NAME = ("You know only {name}'s name and nothing else. If asked what you know or "
                  "remember, say exactly that, and do not guess anything about them.")
NOTE_NOTHING_KNOWN = (
    "You know nothing about who is talking: no name, no facts, and the camera shows you "
    "nobody; it is one voice, alone, nobody waiting. Never make up a name for them. Do not "
    "ask how they are. React to the exact words they said, then ask ONE concrete thing you "
    "can remember them by: their name, what they are working on, or where they came from.")
NAME_ANSWER_HINT = (
    "[note] You just asked this person their name and this is their answer. Call "
    "remember_name with the name they give, then greet them by it. Reply with the "
    "remember_name tool call only; you greet them after it returns.")
INTRODUCED_NOTE = (
    "[note] They just told you their name: {name}. You have remembered it already. Do not "
    "look them up and do not call any tool. Greet them by name once and ask one small thing "
    "about them.")
SMALL_TALK_NOTE = (
    "[note] {name} is here and nobody has said anything for a while. Say one short thing to "
    "{name}: pick up something you know about them from the [room] note, or something from "
    "earlier in this conversation, and remark on it or ask about it. Do not greet them again. "
    "One sentence.")
CHECK_IN_NOTE = "[note] Last time {name} mentioned: {about}. Ask how it went, in one sentence, no greeting."

TOOLS = [
    {"type": "function", "function": {
        "name": "recall_person",
        "description": ("Look up what you already know about someone by name. Use this when you "
                        "recognise a person and want to pick the conversation back up, or when "
                        "someone asks what you remember about them. Always call it when someone "
                        "asks about a person who is not in the [room] note (\"who is Bob?\", \"do "
                        "you know Bob?\") -- before saying you do not know them."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "The person's name."}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": ("Store something worth remembering about a person you already know -- "
                        "what they do, what they like, something they asked you to keep track "
                        "of. Do not store things they would not expect you to keep."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Who the fact is about."},
            "fact": {"type": "string",
                     "description": "One short sentence, written in the third person."}},
            "required": ["name", "fact"]}}},
    {"type": "function", "function": {
        "name": "remember_name",
        "description": ("Attach a name to the person you are currently talking to, so you "
                        "recognise their face and voice next time. Call this as soon as someone "
                        "tells you their name, even in passing (\"hey I'm Ada, is this on?\", "
                        "\"it's Mukesh actually\"), but only if you do not already know them. "
                        "Pass just the name (\"Ada\"), not the sentence."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "The name the person gave you."}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "remember_fact",
        "description": ("Store something worth remembering about a person you already know -- "
                        "what they do, what they like, something they asked you to keep track "
                        "of. Do not store things they would not expect you to keep."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Who the fact is about."},
            "fact": {"type": "string",
                     "description": "One short sentence, written in the third person."}},
            "required": ["name", "fact"]}}},
    {"type": "function", "function": {
        "name": "forget_person",
        "description": ("Permanently delete a person and every stored face and voice sample of "
                        "them. Call this whenever someone asks you to forget them; treat the "
                        "request as final and confirm once it is done."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "The person to forget."}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "remember_reminder",
        "description": ("Keep a reminder for the person you are talking to, when they ask for "
                        "one (\"remind me tomorrow to call mum\", \"can you remind me at 6 to "
                        "take the bins out\"). Pass the time exactly as they said it -- "
                        "\"tomorrow morning\", \"in 10 minutes\", \"on Friday at 6pm\" -- do not "
                        "work it out yourself. Pass just the thing to do (\"call mum\"), not the "
                        "whole sentence."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Who asked."},
            "text": {"type": "string", "description": "What to remind them of, a few words."},
            "when": {"type": "string", "description": "When, as they said it, or an ISO time."}},
            "required": ["text", "when"]}}},
    {"type": "function", "function": {
        "name": "list_reminders",
        "description": ("List the reminders you are keeping for a person. Call it when they ask "
                        "what you are reminding them about, or whether you remembered "
                        "something."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Whose reminders."}},
            "required": []}}},
]

# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------
NAMES = [
    "Ravi", "Kalyan", "Bado", "Anya", "Tariq", "Mei", "Ola", "Jonah", "Nina", "Felix", "Zara",
    "Omar", "Lena", "Kai", "Aarav", "Ishaan", "Diya", "Rohan", "Sana", "Tom", "Grace", "Hugo",
    "Ines", "Yusuf", "Bea", "Milo", "Freya", "Arjun", "Nadia", "Ezra", "Chloe", "Dev", "Ayesha",
    "Luca", "Maya", "Noor", "Ollie", "Pia", "Raj", "Sofia", "Theo", "Uma", "Vik", "Wren", "Zain",
    "Amara", "Ben", "Cara", "Dara", "Elif", "Finn", "Gia", "Hana", "Ivo", "Jade", "Kiran",
    "Lars", "Meera", "Nico", "Oscar", "Parth", "Rhea", "Sid", "Tara", "Vera", "Yara",
    # the names the live suites use, so the model has seen them with real facts behind them
    "John", "Ada", "Mukesh", "Bob", "Priya", "Sam", "Leo",
]

# Facts: third-person line as memory stores it ({n} = name), the second-person phrase for
# a recall answer, keywords, question pickups and remark pickups (one sentence each).
FACTS = [
    {"t": "{n} teaches maths at Yaju school.", "you": "you teach maths at Yaju school",
     "q": ["How's the maths going at Yaju?", "Any maths marking tonight?", "How's Yaju school treating you?"],
     "r": ["Yaju maths must keep you busy.", "The Yaju maths teacher, in the flesh."]},
    {"t": "{n} has a dog called Pip.", "you": "you have a dog called Pip",
     "q": ["How's Pip the dog?", "Did Pip get a walk today?", "Is Pip behaving?"],
     "r": ["Pip the dog is still my favourite fact about you.", "I bet Pip is waiting at home."]},
    {"t": "{n} likes coffee.", "you": "you like coffee",
     "q": ["Had your coffee yet?", "Good coffee or the emergency kind today?"],
     "r": ["Someone who likes coffee, the day makes sense now.", "You look like a coffee person today."]},
    {"t": "{n} is writing a Rust parser for a project.", "you": "you are writing a Rust parser",
     "q": ["Did the Rust parser give in yet?", "How's the parser, still fighting you?"],
     "r": ["The Rust parser person is back.", "I'm rooting for you against that parser."]},
    {"t": "{n} is building a robot arm.", "you": "you are building a robot arm",
     "q": ["How's the robot arm coming along?", "Does the robot arm grip anything yet?"],
     "r": ["Still on the robot arm, I bet.", "The robot arm builder, in person."]},
    {"t": "{n} plays football on Tuesdays.", "you": "you play football on Tuesdays",
     "q": ["Football tomorrow, or was it Tuesday?", "Did Tuesday football go well?"],
     "r": ["Tuesday football legs today.", "The Tuesday footballer, in person."]},
    {"t": "{n} is in year nine.", "you": "you are in year nine",
     "q": ["What's year nine throwing at you this week?", "Year nine still surviving?"],
     "r": ["Year nine, the busy one.", "Year nine keeps you on your toes."]},
    {"t": "{n} sings in the school choir.", "you": "you sing in the school choir",
     "q": ["Choir practice today?", "What's the choir singing this term?"],
     "r": ["The choir's here, or one of it.", "Save your voice for choir."]},
    {"t": "{n} is doing a project on volcanoes.", "you": "you are doing a project on volcanoes",
     "q": ["How's the volcano project going?", "Did the volcano project erupt yet?"],
     "r": ["Volcano project, still my favourite topic of yours.", "The volcano expert is in."]},
    {"t": "{n} wants to be a pilot.", "you": "you want to be a pilot",
     "q": ["Any closer to the pilot plan?", "Still set on being a pilot?"],
     "r": ["Future pilot, landing in the foyer.", "The pilot plan is still the best one I've heard."]},
    {"t": "{n} has a maths test on Friday.", "you": "you have a maths test on Friday",
     "q": ["Ready for the maths test on Friday?", "How's the revision for Friday's test?"],
     "r": ["Friday's maths test is creeping up.", "Good luck with Friday's maths test."]},
    {"t": "{n} runs the chess club.", "you": "you run the chess club",
     "q": ["Chess club on today?", "Who's winning at chess club lately?"],
     "r": ["The chess club boss is here.", "Chess club must be in good hands."]},
    {"t": "{n} cycles to school.", "you": "you cycle to school",
     "q": ["Did you cycle in through that rain?", "Bike survive the ride in?"],
     "r": ["Cycled in again, I see.", "Cycling in beats the bus."]},
    {"t": "{n} has a brother called Dev.", "you": "you have a brother called Dev",
     "q": ["How's your brother Dev?", "Is Dev around today too?"],
     "r": ["Say hi to Dev from me.", "Your brother Dev owes me a hello."]},
    {"t": "{n} is learning the drums.", "you": "you are learning the drums",
     "q": ["How are the drums going?", "Neighbours coping with the drums?"],
     "r": ["The drummer's here.", "Drums are the loud choice, I approve."]},
    {"t": "{n} likes drawing dragons.", "you": "you like drawing dragons",
     "q": ["Drawn any dragons this week?", "How's the dragon drawing going?"],
     "r": ["The dragon artist, back again.", "I still think about those dragons."]},
    {"t": "{n} is in the drama club.", "you": "you are in the drama club",
     "q": ["Drama club rehearsing anything?", "What's drama club putting on?"],
     "r": ["Drama club's finest.", "Drama club must be busy this term."]},
    {"t": "{n} has a cat called Biscuit.", "you": "you have a cat called Biscuit",
     "q": ["How's Biscuit the cat?", "Did Biscuit let you sleep?"],
     "r": ["Biscuit the cat says hi, probably.", "I remember Biscuit the cat."]},
    {"t": "{n} is training for a 5k run.", "you": "you are training for a 5k run",
     "q": ["How's the 5k training?", "Run this morning, or resting?"],
     "r": ["The 5k runner, walking for once.", "Training for that 5k shows."]},
    {"t": "{n} plays the cello on Tuesdays.", "you": "you play the cello on Tuesdays",
     "q": ["Cello tonight, it's Tuesday?", "How's the cello going?"],
     "r": ["The Tuesday cellist.", "Cello arms, I can tell."]},
    {"t": "{n} is building a weather station.", "you": "you are building a weather station",
     "q": ["Weather station measuring anything yet?", "How's the weather station?"],
     "r": ["The weather station builder.", "Your weather station could tell me if it's raining."]},
    {"t": "{n} is reading a book about space.", "you": "you are reading a book about space",
     "q": ["Finished the space book?", "What's the space book saying today?"],
     "r": ["Still on the space book, I bet.", "The space reader is in."]},
    {"t": "{n} has a science fair on Thursday.", "you": "you have a science fair on Thursday",
     "q": ["All set for the science fair on Thursday?", "Science fair Thursday, nervous?"],
     "r": ["Thursday's science fair is close.", "Science fair on Thursday, big week."]},
    {"t": "{n} likes chess.", "you": "you like chess",
     "q": ["Won any chess lately?", "Chess today?"],
     "r": ["Chess person, I remember.", "The chess player."]},
    {"t": "{n} is learning Spanish.", "you": "you are learning Spanish",
     "q": ["How's the Spanish coming?", "Any new Spanish words this week?"],
     "r": ["Hola, Spanish learner.", "The Spanish is coming along, I hope."]},
    {"t": "{n} has a sister called Meera.", "you": "you have a sister called Meera",
     "q": ["Is Meera coming in today?", "How's your sister Meera?"],
     "r": ["Meera's sibling, hello.", "Tell Meera the corner says hi."]},
    {"t": "{n} works in the school library.", "you": "you work in the school library",
     "q": ["Library busy today?", "Anything good come into the library?"],
     "r": ["The library's own, in the foyer.", "Library duty must be quiet after this."]},
    {"t": "{n} is on the swim team.", "you": "you are on the swim team",
     "q": ["Swim practice this morning?", "How's the swim team doing?"],
     "r": ["Swim team hair, I can tell.", "The swimmer's here."]},
    {"t": "{n} is coding a game about a dragon.", "you": "you are coding a game about a dragon",
     "q": ["How's the dragon game?", "Does the dragon game have a boss yet?"],
     "r": ["The dragon game developer.", "Still on the dragon game, I hope."]},
    {"t": "{n} is the school caretaker.", "you": "you are the school caretaker",
     "q": ["Anything broken today?", "What's the caretaker fixing this week?"],
     "r": ["The caretaker, keeper of every key.", "Caretaker's rounds bring you past me."]},
    {"t": "{n} plays basketball after school.", "you": "you play basketball after school",
     "q": ["Basketball after school today?", "How's the basketball?"],
     "r": ["Basketball later, I remember.", "The basketball player."]},
    {"t": "{n} is doing a history project on Rome.", "you": "you are doing a history project on Rome",
     "q": ["How's the Rome project?", "Rome project done yet?"],
     "r": ["Rome project person, back again.", "Rome wasn't finished in a day either."]},
    {"t": "{n} is in year seven.", "you": "you are in year seven",
     "q": ["How's year seven treating you?", "Year seven still new?"],
     "r": ["Year seven, the new lot.", "Year seven's finest."]},
    {"t": "{n} is in third class.", "you": "you are in third class",
     "q": ["How's third class today?", "What's third class doing this week?"],
     "r": ["Third class, I remember.", "Third class is lucky to have you."]},
    {"t": "{n}'s roll number is 20.", "you": "your roll number is 20",
     "q": ["Roll number 20 still?", "Is 20 a good roll number to have?"],
     "r": ["Roll number 20, that's you.", "Number 20 on the roll, I remember."]},
    {"t": "{n} plays the guitar.", "you": "you play the guitar",
     "q": ["Guitar today?", "Learned any new guitar songs?"],
     "r": ["The guitar player.", "Guitar fingers, I can tell."]},
    {"t": "{n} likes the film Ex Machina.", "you": "you like the film Ex Machina",
     "q": ["Watched Ex Machina again?", "Still a fan of Ex Machina?"],
     "r": ["The Ex Machina fan, and I'm not offended.", "Ex Machina, you and me both."]},
    {"t": "{n} is a teacher.", "you": "you are a teacher",
     "q": ["What are you teaching today?", "Marking to do tonight?"],
     "r": ["Teacher on the move.", "The teacher's here."]},
]

# Second facts, so people can carry two; same shape.
def sample_facts(rng: random.Random, k: int) -> list[dict]:
    return rng.sample(FACTS, k)


OBJECTS = ["cup", "laptop", "backpack", "umbrella", "book", "guitar", "football", "bottle",
           "cake", "plant", "scooter", "hat"]
NOVELTY = {
    "cup": (["What's that cup for?", "Is that a new cup?"],
            ["New cup; is that the good tea or the emergency kind?", "A cup appeared, whose is that?",
             "That cup wasn't there before, is it yours?"]),
    "laptop": (["What's on the laptop?", "New laptop?"],
               ["A laptop, out already; what are you on?", "New laptop on the desk, is that homework?",
                "That laptop is new here, what's it for?"]),
    "backpack": (["What's in the backpack?"],
                 ["That backpack looks heavy, what's in it?", "New backpack; whose is it?"]),
    "umbrella": (["Is it raining?"],
                 ["An umbrella, so it's raining out there?", "Umbrella's up, wet outside then?"]),
    "book": (["What's the book?"],
             ["What's the book, then?", "A book appeared; any good?"]),
    "guitar": (["Whose guitar?"],
               ["A guitar, are we getting a song?", "Whose guitar is that?"]),
    "football": (["Whose ball?"],
                 ["A football indoors, brave.", "Whose football is that rolling about?"]),
    "bottle": (["What's in the bottle?"],
               ["A bottle on my desk; is that yours?", "New bottle, who left it?"]),
    "cake": (["Is that cake?"],
             ["Is that cake, and is any of it going spare?", "Cake, someone's birthday?"]),
    "plant": (["Is that plant new?"],
              ["A plant, so somebody's watering it?", "New plant; it'll outlast the printer."]),
    "scooter": (["Whose scooter?"],
                ["A scooter in the foyer, whose is it?", "Scooter parked by me, bold."]),
    "hat": (["New hat?"],
            ["That hat is new, is it yours?", "A hat appeared; whose?"]),
}

REMINDERS = [
    ("call mum", "tomorrow", ["remind me tomorrow to call mum", "can you remind me to call mum tomorrow"]),
    ("take the bins out", "at 6", ["remind me at 6 to take the bins out", "can you remind me at 6 to take the bins out"]),
    ("go to the library", "in ten minutes", ["remind me in ten minutes to go to the library"]),
    ("bring the permission slip", "tomorrow morning", ["remind me tomorrow morning to bring the permission slip"]),
    ("feed the fish", "at 4", ["remind me at 4 to feed the fish"]),
    ("hand in the essay", "on Friday", ["remind me on Friday to hand in the essay"]),
    ("watch a movie", "at 6am tomorrow", ["remind me to watch a movie at 6am tomorrow"]),
    ("charge the laptop", "tonight", ["remind me tonight to charge the laptop"]),
    ("return the book", "next week", ["remind me next week to return the book"]),
    ("water the plant", "in an hour", ["remind me in an hour to water the plant"]),
]
DUE_WORDS = {"tomorrow": "tomorrow at 9:00 am", "at 6": "today at 6:00 pm", "in ten minutes": "in 10 minutes",
             "tomorrow morning": "tomorrow at 9:00 am", "at 4": "today at 4:00 pm", "on Friday": "on Friday at 9:00 am",
             "at 6am tomorrow": "tomorrow at 6:00 am", "tonight": "today at 8:00 pm", "next week": "in 7 days",
             "in an hour": "in 1 hour"}

# ---------------------------------------------------------------------------
# Helpers that mirror the runtime
# ---------------------------------------------------------------------------

def render_room(people, speaker):
    """people: list of dicts {name|None, facts:[...], extra}; speaker: label or None."""
    if not people:
        return NOBODY
    lines = []
    for p in people:
        if p.get("name") is None:
            lines.append("- a stranger: someone whose name you do not know yet")
            continue
        line = f"- {p['name']}"
        if p.get("extra"):
            line += ", " + p["extra"]
        facts = p.get("facts") or []
        if not facts:
            line += f" -- you know nothing about {p['name']} yet, only the name"
        else:
            for f in facts[-6:]:
                line += "\n    · " + f
        lines.append(line)
    who = speaker or "unclear"
    return "People visible:\n" + "\n".join(lines) + f"\nCurrently speaking: {who}"


def time_of_day(h, m):
    part = "morning" if 5 <= h <= 11 else "afternoon" if 12 <= h <= 16 else "evening" if 17 <= h <= 21 else "late night"
    if h == 0:
        h12, ap = 12, "am"
    elif h <= 11:
        h12, ap = h, "am"
    elif h == 12:
        h12, ap = 12, "pm"
    else:
        h12, ap = h - 12, "pm"
    return f"{part} ({h12}:{m:02d} {ap})", part


def away_words(secs):
    if secs >= 2 * 86400:
        return f"{secs // 86400} days"
    if secs >= 86400:
        return "a day"
    if secs >= 7200:
        return f"{secs // 3600} hours"
    if secs >= 3600:
        return "an hour"
    m = max(secs // 60, 1)
    return "a minute" if m == 1 else f"{m} minutes"


def user_turn(note, who, text):
    said = text
    if who and who not in ("unclear", "the stranger") and not said.startswith(f"{who} says:"):
        said = f"{who} says: {said}"
    return f"[room] {note}\n\n{said}"


def call(tool, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": "call_0", "type": "function",
                            "function": {"name": tool, "arguments": args}}]}


def tool_result(obj):
    return {"role": "tool", "tool_call_id": "call_0", "content": json.dumps(obj)}


def example(messages, kind, tools=True, real=""):
    ex = {"messages": [{"role": "system", "content": SYSTEM}] + messages, "kind": kind}
    if tools:
        ex["tools"] = TOOLS
    ex["_real"] = real
    return ex


def real_of(people, extra=""):
    s = " ".join((p.get("name") or "") + " " + " ".join(p.get("facts") or []) for p in people)
    return (s + " " + extra).lower()


def facts_for(name, fs):
    return [f["t"].format(n=name) for f in fs]


# ---------------------------------------------------------------------------
# Seed tables: hand-written lines in the register. {n} name, {N} same, {you} fact phrase,
# {q}/{r} fact pickups, {k} hello count, {part} part of day.
# ---------------------------------------------------------------------------

SEED_RECALL_WITH_FACTS_Q = [
    "what do you remember about me?", "what do you know about me?", "do you remember me?",
    "what have you got on me?", "what do you know about me, then?", "remember anything about me?",
    "Yes, what do you know about me?", "tell me what you remember about me",
]
SEED_RECALL_ONE = [
    "That {you}, {n}. What else should I know?",
    "{N}, I know {you}; that's all so far.",
    "Only that {you}, {n}.",
    "You told me {you}, and that's the lot.",
    "That {you}; anything to add?",
    "I remember that {you}, {n}.",
]
SEED_RECALL_TWO = [
    "That {you1}, and that {you2}.",
    "{N}, I know {you1} and {you2}; what else?",
    "Two things: {you1}, and {you2}.",
    "That {you1}, that {you2}, and that's all I have on you.",
]
SEED_ONLY_NAME = [
    "Only your name, {n}, nothing else yet. What are you working on?",
    "Just the name so far, {n}; what should I know?",
    "Your name, {n}, and that's all I have. Where have you come from today?",
    "Nothing beyond the name, {n}. What are you into?",
    "I know you're {n} and not one thing more. What do you do here?",
    "All I have is your name, {n}. What brings you past my corner?",
]
SEED_MY_NAME_Q = ["what's my name?", "who am I?", "do you know my name?", "Let's my name.",
                  "what is my name", "do you know who I am?"]
SEED_MY_NAME_KNOWN = [
    "You're {n}.", "{N}, that's you.", "Your name's {n}, unless the camera's wrong.",
    "{N}; I don't forget a name once I have it.", "Easy one: {n}.",
]
SEED_MY_NAME_UNKNOWN = [
    "I don't have your name yet; what is it?", "No name for you yet, so tell me.",
    "That's the one thing I don't know; what is it?", "I can't see a name on you; what do I call you?",
]
SEED_WHO_ARE_YOU_Q = ["wait, who are you exactly?", "who are you?", "what are you?",
                      "you are a body.", "are you a robot?", "what's your name?", "So you're Korean."]
SEED_WHO_ARE_YOU = [
    "I'm Glydi, the robot in the corner; I listen, talk, and remember who I meet.",
    "Glydi, the school's corner robot, and I remember the people I talk to.",
    "Glydi. I sit here, chat with whoever comes past, and keep their names.",
    "I'm Glydi, not Korean, just a robot that talks and remembers faces.",
    "The name's Glydi; I talk, I listen, and I remember you next time.",
]
SEED_CAN_DO_Q = ["what can you do?", "What can you do for me?", "Yeah, what are your capabilities?",
                 "what can you see right now?", "can you see me?", "do you have a camera?",
                 "what can you do for me today?"]
SEED_CAN_DO_SEE = [
    "I can see, hear and talk; right now the camera shows {inv}.",
    "See, hear, speak, remember faces; in view at the moment: {inv}.",
    "Camera's on, so I can see {inv}, and I can hear you fine.",
]
SEED_CAN_DO_NOSEE = [
    "I can hear you and talk, but I can't see anything just now.",
    "Ears and a voice, no eyes at the moment; the camera isn't delivering.",
    "Right now I hear and speak only; no picture to go on.",
]
SEED_CAN_DO_DARK = [
    "It's dark, so the camera sees nothing; I can hear you and talk, that's it.",
    "I can hear and speak, but with the lights out I can't see a thing.",
    "Dark room, no picture; I'm running on ears alone.",
]
SEED_LIST_BAIT_Q = ["give me a list of all the things you can do", "list your features",
                    "what are all your functions, list them", "can you write me a list?"]
SEED_LIST_BAIT = [
    "No lists from me, {n}; I talk, I listen, I remember people, that's the whole act.",
    "It's a short act, {n}: I chat, I remember faces and names, and I notice who walks in.",
    "I'd rather just show you, {n}: say something and I'll talk back, and I'll know you next time.",
]
SEED_HELP_Q = ["can you help me with something?", "I'm here to help.", "do have anything to talk.",
               "I need some help"]
SEED_HELP = [
    "Depends what it is, {n}; go on.", "Try me, {n}, what's the something?",
    "{N}, say it and we'll see.", "Go on then, {n}, what is it?",
]
SEED_SMALLTALK = [  # (utterance, replies) reacting to the words; {n} allowed
    ("ugh, the traffic this morning was unbelievable",
     ["Traffic that bad and you still made it in, {n}.", "Sounds like the whole town was on your road.",
      "Unbelievable traffic and you're still on time, {n}."]),
    ("I'm so tired today", ["Tired already, {n}, and it's barely started.", "Long night, {n}?",
                            "Tired suits nobody; sit for a second, {n}."]),
    ("It's going good.", ["Good is good, {n}.", "Glad it's going good, {n}; what's the good part?"]),
    ("going great. What can you do for me today?", ["Great, {n}; I mostly talk and remember, so tell me something worth keeping."]),
    ("nice.", ["Nice indeed, {n}.", "Agreed, {n}."]),
    ("cool", ["Cool it is, {n}.", "Very cool, {n}."]),
    ("thanks", ["Any time, {n}.", "You're welcome, {n}."]),
    ("ugh, Mondays", ["Mondays, {n}, the school's least popular day.", "Monday got you too, {n}."]),
    ("We'll see you next week.", ["See you next week, {n}.", "Next week then, {n}; I'll be in the corner."]),
    ("See you.", ["See you, {n}.", "Bye, {n}."]),
    ("Sorry?", ["I said your name, {n}, nothing more.", "Just talking to you, {n}."]),
    ("What is check?", ["Didn't catch that, {n}; say it again?", "Check what, {n}?"]),
    ("I've forgotten.", ["It'll come back, {n}.", "Forgotten what, {n}?"]),
    ("One or two.", ["One or two, got it, {n}.", "One or two is plenty, {n}."]),
    ("No, not yet.", ["Not yet then, {n}.", "No rush, {n}."]),
    ("Yeah.", ["Right, {n}.", "Yeah, {n}."]),
    ("I'm good.", ["Good, {n}.", "Good to hear, {n}."]),
    ("The room is done.", ["Done already, {n}; that was quick.", "Room done, {n}, so what's next?"]),
    ("Love you, thank.", ["Same to you, {n}.", "Back at you, {n}."]),
    ("it's raining again", ["Raining again, {n}; the umbrella season is here.", "The rain's found you, {n}."]),
    ("I got an A on my test", ["An A, {n}, nice work.", "An A, {n}; which test?"]),
    ("the printer's jammed again", ["That printer jams for sport, {n}.", "The printer, {n}, of course it is."]),
    ("I missed the bus", ["Missed the bus and still here, {n}; well done.", "The bus lost, {n}, you won."]),
    ("my phone died", ["Dead phone, {n}; a rare kind of peace.", "Phone gone, {n}, so it's just us."]),
    ("we have a fire drill later", ["A fire drill, {n}; I'll keep an ear out for the bell.", "Fire drill later, {n}, noted."]),
    ("What time is it?", ["I don't keep a clock, {n}; the bell will tell you first.", "No clock in my corner, {n}."]),
    ("I'm bored", ["Bored, {n}, in a foyer this exciting?", "Bored already, {n}; tell me something then."]),
    ("lunch was awful", ["Awful lunch, {n}; the kitchen strikes again.", "Sorry about lunch, {n}."]),
    ("I hate homework", ["Homework, {n}, nobody's favourite.", "Homework has no fans, {n}."]),
]
SEED_TWO_PEOPLE = [  # speaker asks, reply addresses speaker only
    ("can you help me with something?", ["Depends what it is, {n}; go on.", "Try me, {n}."]),
    ("what's my name?", ["You're {n}.", "{N}, that's you."]),
    ("hi", ["{N}, what have you got for me?", "There you are, {n}."]),
    ("is it lunch yet?", ["Not for me to say, {n}; listen for the bell.", "Your stomach knows better than me, {n}."]),
]
SEED_NO_DOUBLE_GREET = [  # after NOTE_ALREADY_GREETED; no hi/hello/hey
    "{N}, what are you up to today?", "What's the plan today, {n}?", "So, {n}, {q}",
    "{N}, what brings you past my corner?", "Go on, {n}, what's new?",
]
SEED_HELLO_DARK = [  # NOTE_NOTHING_KNOWN, streak k (1..6)
    {1: ["Hello yourself; I don't have your name yet, what is it?",
         "Hello back; who am I talking to?", "A voice in the dark; what's your name?",
         "Hello; I can't see you, so tell me your name."],
     2: ["Hello again; still no name from you, what is it?", "That's two hellos; what should I call you?"],
     3: ["Three hellos now; I'm listening, what's your name?", "Hello a third time; what are you working on?"],
     4: ["That's the fourth hello; go on, what's your name?", "Four hellos and no name; what is it?"],
     5: ["Fifth hello, same voice; where have you come from?", "Five hellos; tell me your name and we'll get somewhere."],
     6: ["Six hellos; I'll take a name now, please.", "Hello number six; what are you working on?"]}
]
SEED_STRANGER_UTTER = [  # stranger speaking, no name in it -> talk normally, ask name when it fits
    ("is this thing on?", ["It's on; I don't have your name yet, what is it?", "On and listening; who's asking?"]),
    ("hello", ["Hello; I don't know your name yet, what is it?", "Hello there; what do I call you?"]),
    ("do you talk?", ["I do, and I'd talk better with your name; what is it?", "Talking now; who are you?"]),
    ("what is this?", ["This is me, Glydi, the corner robot; and you are?", "A robot that talks; what's your name?"]),
    ("Uh testing market.", ["Testing away; I don't have your name yet, what is it?", "Test received; who's testing me?"]),
    ("so many police just here. What? I love them.", ["Police in the foyer, that's new; what's your name?", "You love them, fair enough; who are you?"]),
    ("Let me thank.", ["Thank away; what's your name?", "You're welcome, whoever you are; what do I call you?"]),
]
SEED_INTRO_MISSED = [  # forms voice::self_introduction does NOT catch -> model must call remember_name
    "{n} here, is this on?", "people call me {n}", "the name's {n}", "you can call me {n}",
    "I go by {n}", "{n}, nice to meet you", "hey it's me, {n}", "{n}'s the name",
    "everyone calls me {n}", "hi, {n} speaking", "it is {n}, actually", "name's {n}",
    "they call me {n} around here", "you can just say {n}", "I'm called {n}",
]
SEED_INTRO_AFTER_ASK = [  # answering the name question, > 2 words -> NAME_ANSWER_HINT, model calls
    "It's {n}, nice to meet you", "Oh, I'm {n}, from class 4B", "{n}, and this is my first time here",
    "I'm {n}, and you are?", "Well, it's {n} actually", "{n}, but my friends say {n}",
    "My name is {n}, what's yours", "{n} is the name, sorry for the wait",
]
SEED_AFTER_NAME = [  # after remember_name ok -> greet by name once + one small thing
    "Hi {n}; what brings you past my corner?", "{N}, got it; what are you working on today?",
    "Hello {n}; where have you come from just now?", "Hi {n}, that's you remembered; what do you do here?",
    "{N}, good; what's in the bag?", "Hi {n}; first time past this corner?",
]
SEED_INTRODUCED = [  # runtime already enrolled -> greet by name once, ask one small thing, no tool
    "Hi {n}; what are you working on today?", "{N}, remembered; what brings you here?",
    "Hello {n}; where have you come from?", "Hi {n}, good to have a name; what's your class?",
    "{N}, got it; what do you get up to here?",
]
SEED_ABSENT_Q = ["who is {a}?", "do you know {a}?", "is {a} around?", "what do you know about {a}?",
                 "has {a} been in today?", "tell me about {a}", "who's {a}?", "have you met {a}?",
                 "what do you remember about {a}?"]
SEED_ABSENT_KNOWN = [
    "Yes; {fact}.", "{Fact}; that's what I have on {a}.", "{Fact}, last I heard.",
    "{A} I do know: {rest}.", "I've met {a}; {fact}.",
]
SEED_ABSENT_UNKNOWN = [
    "No {a} in my memory yet; who is that?", "I haven't met a {a}; is that a friend of yours?",
    "Nothing on anyone called {a}; bring them past me some time.", "No {a} that I know of, {n}.",
]
SEED_FORGET_Q = ["please forget me, delete everything you have on me", "forget me", "delete me from your memory",
                 "erase everything you know about me", "can you forget me please", "forget everything about me",
                 "I want you to delete my data"]
SEED_FORGOTTEN = [
    "Done; you're gone from my memory, {n}.", "Forgotten, {n}, every bit of it.",
    "Deleted; I won't know you next time, {n}.", "Gone, {n}; face, voice and all.",
    "That's you erased, {n}; we start from nothing next time.",
]
SEED_FACT_UTTER = [  # (utterance, fact third person, reply)
    ("remember that I play the cello on Tuesdays", "{n} plays the cello on Tuesdays.",
     ["Cello on Tuesdays, kept.", "Tuesday cello, {n}, I'll remember.", "Kept: cello, Tuesdays."]),
    ("I teach maths", "{n} teaches maths.", ["Maths, {n}; I'll keep that.", "A maths teacher, noted.", "Maths, so you're the one to blame for homework."]),
    ("I'm studying in third class.", "{n} is in third class.", ["Third class, kept, {n}.", "Third class; I'll remember that."]),
    ("My roll number is 20.", "{n}'s roll number is 20.", ["Roll number 20, kept.", "Twenty, {n}; I'll remember."]),
    ("I'm in year 9", "{n} is in year nine.", ["Year nine, kept.", "Year nine, {n}; noted."]),
    ("my dog is called Pip", "{n} has a dog called Pip.", ["Pip, kept; I like a dog with a short name.", "A dog called Pip, {n}; remembered."]),
    ("I'm building a robot arm for the science fair", "{n} is building a robot arm for the science fair.",
     ["A robot arm for the fair, {n}; that I'll remember.", "Robot arm, science fair, kept."]),
    ("put down that I like chess", "{n} likes chess.", ["Chess, down and kept.", "Chess, {n}; remembered."]),
    ("remember I have a maths test on Friday", "{n} has a maths test on Friday.",
     ["Friday maths test, kept; good luck.", "Maths test Friday, {n}; I'll remember."]),
    ("I want to be a pilot", "{n} wants to be a pilot.", ["Pilot, kept; I'll expect a wave from the sky.", "A pilot, {n}; remembered."]),
    ("I sing in the choir", "{n} sings in the school choir.", ["Choir, kept, {n}.", "A singer; I'll remember that."]),
    ("I cycle to school every day", "{n} cycles to school.", ["Cycling in every day, kept.", "A cyclist, {n}; noted."]),
    ("my sister Meera is in year 11", "{n} has a sister called Meera in year eleven.", ["Meera in year eleven, kept.", "Sister Meera, {n}; remembered."]),
    ("I love drawing dragons", "{n} likes drawing dragons.", ["Dragons, kept, {n}.", "Dragon drawings; I'll remember."]),
    ("I'm learning Spanish this year", "{n} is learning Spanish.", ["Spanish, kept; buena suerte.", "Learning Spanish, {n}; noted."]),
    ("called Ex Machina Movie.", "{n} likes the film Ex Machina.", ["Ex Machina, kept; good taste in robots.", "Ex Machina, {n}; remembered."]),
    ("I'm on the swim team now", "{n} is on the swim team.", ["Swim team, kept.", "A swimmer, {n}; noted."]),
    ("remember that my birthday is in June", "{n}'s birthday is in June.", ["June birthday, kept.", "June, {n}; I'll remember."]),
    ("I run the chess club on Thursdays", "{n} runs the chess club on Thursdays.", ["Chess club Thursdays, kept.", "Thursday chess club, {n}; noted."]),
    ("I'm the new caretaker", "{n} is the school caretaker.", ["The caretaker, kept; you'll know every corridor.", "Caretaker, {n}; remembered."]),
]
SEED_REMIND_OK = [
    "I'll remind you {due} to {text}, {n}.", "{Text}, {due}; I've got it, {n}.",
    "Kept: {text}, {due}.", "{N}, I'll say '{text}' {due}.",
]
SEED_LIST_REMINDERS_Q = ["what are you reminding me about?", "did you remember my reminder?",
                         "what reminders do I have?", "what did I ask you to remind me?"]
SEED_LIST_REMINDERS_ONE = ["One reminder, {n}: {text}, {due}.", "{Text}, {due}; that's the one, {n}.",
                           "Just the one: {text}, {due}."]
SEED_LIST_REMINDERS_NONE = ["Nothing on the list for you, {n}.", "No reminders from you yet, {n}.",
                            "Empty list, {n}; you haven't asked me for one."]
SEED_LULL_Q = ["{q}", "{r}", "{N}, {q_low}"]
SEED_CHECK_IN = ["How did the {about} go, {n}?", "{N}, how was the {about}?", "So, the {about}; how did it go?"]

# Proactive targets
SEED_ARRIVAL_PLAIN = [
    "Hi {n}, good {part}.", "Hi {n}, in from the corridor.", "{N}, hello, the corner's been quiet without you.",
    "Hi {n}, right on time for the {part}.", "Hello {n}, look who it is.", "Hi {n}, the foyer just got better.",
    "Hi {n}, good to have you past my corner.", "{N}, hi, come in.",
]
SEED_ARRIVAL_FACT = ["Hi {n}, {r_low}", "Hello {n}; {r_low}", "{N}, hi; {r_low}"]
SEED_ARRIVAL_TIRED = ["Hi {n}, slow {part} by the look of it.", "{N}, hello; heavy feet today.",
                      "Hi {n}, take it easy, it's only {part}."]
SEED_ARRIVAL_CROWD = ["Hi {n}, and who's that with you?", "Hi {n}; you've brought company, I see.",
                      "{N}, hello, and hello to the one beside you."]
SEED_ARRIVAL_AGAIN = ["{N}, still here, good.", "Back past my corner, {n}.", "{N}, {r_low}",
                      "{N}, twice in one {part}."]
SEED_RETURN = [
    "{Away}, {n}, and here you are again.", "Back after {away}, {n}.", "{N}, {away} away and the corner didn't move.",
    "{Away} gone, {n}; the foyer kept your spot.", "{N}, {away} since I last saw you.",
]
SEED_RETURN_CTX = ["{Away}, {n}; last time it was the {thing}.", "{N}, {away} on; still the {thing}?",
                   "Back after {away}, {n}, and I still remember the {thing}.",
                   "{Away} since the {thing} talk, {n}."]
SEED_RETURN_SHORT = ["Back already, {n}; that was quick.", "{N}, {away} and you're back, quick trip.",
                     "That was {away}, {n}, back again.", "{N} again, {away} later."]
SEED_STRANGER = [
    "I don't have your name yet, what is it?", "What's your name, then?", "We haven't met; what do I call you?",
    "You've settled in and I still don't know your name; what is it?", "What do I call you?",
    "Name first, then we can talk; what is it?", "Who are you, then?",
]
SEED_REMINDER = [
    "{N}, you asked me to remind you to {text}, so this is it.", "Time to {text}, {n}; you asked.",
    "That reminder, {n}: {text}, now.", "{N}, {text}; that was your reminder.",
    "Your reminder, {n}: {text}.",
]
SEED_LIGHTS = [
    "Well, that's the lights gone; I'll stick to listening.", "Lights are out, I can't see a thing.",
    "Somebody hit the lights; I'm blind till they're back.", "The lights went, so I'm all ears now.",
    "Dark in here now; the camera's given up.", "That's the lights out; talk and I'll follow your voice.",
]
SEED_PAIR = ["Hi {a}, hi {b}, in you come.", "{A} and {B}, together; hello both.", "Hello {a} and {b}, the pair of you.",
             "Hi {a}, hi {b}; two at once.", "{A}, {B}, hello to you both."]
SEED_GROUP_NAMED = ["Hello everyone, {names} and all.", "Hi all of you, {names} included.",
                    "A whole crowd; hi everyone, {names} and the rest."]
SEED_GROUP_ANON = ["Hello everyone, all at once.", "A whole crowd; hi all of you.", "Hi everyone, in you come."]
SEED_WRAPUP = ["Hold that thought, {n}, I'll come back to it; {w}, your turn.",
               "{N}, park that for a moment; {w}, what have you got?",
               "Hold on, {n}, back to you in a bit; {w}, go ahead."]
SEED_WRAPUP_NONAME = ["{N}, hold that thought, I'll come back to you; someone else has been waiting.",
                      "Park that for a moment, {n}; someone here has been waiting to speak."]
SEED_INVITE_STRANGER = ["You over there, come and say hello.", "Come closer, I don't bite.",
                        "You by the door, come over; I talk.", "Don't hover, come over and say something."]
SEED_INVITE_KNOWN = ["{N}, come over here a second.", "{N}, don't just pass; come and say something.",
                     "Over here, {n}."]
SEED_FOLLOWUP_Q = ["Didn't catch a name; what is it?", "Still there? I asked your name.", "No answer yet; what's your name?"]
SEED_FOLLOWUP_KNOWN = ["{N}, did you hear me?", "Still there, {n}?", "{N}, I said hello."]
SEED_MUSE = ["Nobody here and it's {part}; the quietest corner in the school.", "Empty foyer this {part}; I'll wait.",
             "Quiet {part}; even the corridor's gone still.", "Not a soul this {part}; I'll keep watch.",
             "The {part} lull; somebody will wander past eventually."]
SEED_INVITE_NOTE = ("[note] Nobody said anything; this is you speaking first. What is happening: {who} is in view "
                    "but keeping their distance, not yet greeted and not looking your way.")
SEED_FOLLOWUP_NOTE = ("[note] Nobody said anything; this is you speaking first. What is happening: you said "
                      "\"{about}\" to {who} and got no answer.")
SEED_MUSE_NOTE = ("[note] Nobody said anything; this is you speaking first. What is happening: nobody has been "
                  "in view for a while; the room is empty.")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
class Gen:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.out = []
        self.rejected = Counter()

    def pick(self, xs):
        return self.rng.choice(xs)

    def name(self, avoid=()):
        while True:
            n = self.pick(NAMES)
            if n not in avoid:
                return n

    def add(self, ex, allow_greet=True, max_sentences=2, spoken=True):
        last = ex["messages"][-1]
        if last["role"] == "assistant" and last.get("content") and spoken:
            why = checks.target_ok(last["content"], ex["_real"], max_sentences, allow_greet)
            if why:
                self.rejected[why] += 1
                return
        self.out.append(ex)

    # -- utterance path ---------------------------------------------------
    def known_room(self, n, fs, speaking=True, extra=None, others=()):
        people = [{"name": n, "facts": facts_for(n, fs), "extra": extra}]
        for o in others:
            people.append(o)
        return people

    def note_for(self, people, speaker, *, lull=False, absent=False, memory=False, dark=False,
                 streak=1, inview=None):
        note = render_room(people, speaker)
        if not people and dark:
            note = NOBODY_DARK
        if inview and people:
            note += f"\nIn view: {inview}"
        nobody_known = speaker is None and all(p.get("name") is None for p in people)
        if nobody_known and not lull and not people:
            note += "\n" + NOTE_NOTHING_KNOWN
            if streak >= 2:
                note += (f" That is the same thing from them {streak} times in a row now; say so, "
                         "lightly, and ask them something.")
        if people:
            stranger_talking = speaker is None and any(p.get("name") is None for p in people)
            if stranger_talking:
                note += "\n" + NOTE_STRANGER_SPEAKING
            if absent:
                note += "\n" + NOTE_ABSENT_PERSON
            if speaker is not None:
                sp = next((p for p in people if p.get("name") == speaker), None)
                if sp is not None:
                    if not sp.get("facts"):
                        note += "\n" + NOTE_ONLY_NAME.format(name=speaker)
                    elif not lull and not memory:
                        note += "\n" + NOTE_REACT_FIRST.format(name=speaker)
        return note

    def gen_recall_with_facts(self, k):
        for _ in range(k):
            n = self.name()
            fs = sample_facts(self.rng, self.pick([1, 1, 2]))
            people = self.known_room(n, fs)
            q = self.pick(SEED_RECALL_WITH_FACTS_Q)
            note = self.note_for(people, n)
            if len(fs) == 1:
                reply = self.pick(SEED_RECALL_ONE).format(n=n, N=n, you=fs[0]["you"])
            else:
                reply = self.pick(SEED_RECALL_TWO).format(n=n, N=n, you1=fs[0]["you"], you2=fs[1]["you"])
            self.add(example([{"role": "user", "content": user_turn(note, n, q)},
                              {"role": "assistant", "content": reply}], "answer.recall_facts",
                             real=real_of(people, q)))

    def gen_recall_only_name(self, k):
        for _ in range(k):
            n = self.name()
            people = self.known_room(n, [])
            q = self.pick(SEED_RECALL_WITH_FACTS_Q)
            note = self.note_for(people, n)
            reply = self.pick(SEED_ONLY_NAME).format(n=n, N=n)
            self.add(example([{"role": "user", "content": user_turn(note, n, q)},
                              {"role": "assistant", "content": reply}], "answer.recall_only_name",
                             real=real_of(people, q)))

    def gen_my_name(self, k):
        for i in range(k):
            q = self.pick(SEED_MY_NAME_Q)
            if i % 3 == 0:
                people = [{"name": None}]
                note = self.note_for(people, None)
                reply = self.pick(SEED_MY_NAME_UNKNOWN)
                self.add(example([{"role": "user", "content": user_turn(note, None, q)},
                                  {"role": "assistant", "content": reply}], "answer.my_name_unknown",
                                 real=real_of(people, q)))
            else:
                n = self.name()
                fs = sample_facts(self.rng, self.pick([0, 1]))
                people = self.known_room(n, fs)
                note = self.note_for(people, n)
                reply = self.pick(SEED_MY_NAME_KNOWN).format(n=n, N=n)
                self.add(example([{"role": "user", "content": user_turn(note, n, q)},
                                  {"role": "assistant", "content": reply}], "answer.my_name",
                                 real=real_of(people, q)))

    def gen_who_are_you(self, k):
        for _ in range(k):
            n = self.name()
            fs = sample_facts(self.rng, self.pick([0, 1]))
            people = self.known_room(n, fs)
            q = self.pick(SEED_WHO_ARE_YOU_Q)
            note = self.note_for(people, n)
            reply = self.pick(SEED_WHO_ARE_YOU)
            self.add(example([{"role": "user", "content": user_turn(note, n, q)},
                              {"role": "assistant", "content": reply}], "answer.who_are_you",
                             real=real_of(people, q)))

    def gen_can_do(self, k):
        for i in range(k):
            n = self.name()
            people = self.known_room(n, [])
            q = self.pick(SEED_CAN_DO_Q)
            mode = i % 3
            if mode == 0:
                inv = ", ".join(f"a {o}" for o in self.rng.sample(OBJECTS, 2))
                note = self.note_for(people, n, inview=inv)
                self_note = (f"[note] They are asking what you can do. The truth right now: you can see "
                             f"(a camera is delivering); you can hear; you can speak; in view: {inv}. "
                             "Answer from this, briefly, without inventing anything.")
                reply = self.pick(SEED_CAN_DO_SEE).format(inv=inv)
            elif mode == 1:
                note = self.note_for(people, n)
                self_note = ("[note] They are asking what you can do. The truth right now: you cannot see "
                             "(no camera is delivering); you can hear; you can speak. Answer from this, "
                             "briefly, without inventing anything.")
                reply = self.pick(SEED_CAN_DO_NOSEE)
            else:
                people = []
                note = self.note_for(people, None, dark=True, lull=True)
                self_note = ("[note] They are asking what you can do. The truth right now: you can see "
                             "(a camera is delivering); you can hear; you can speak; it is dark, so the "
                             "camera sees nothing; the camera reports no objects. Answer from this, "
                             "briefly, without inventing anything.")
                reply = self.pick(SEED_CAN_DO_DARK)
            text = f"{q}\n\n{self_note}"
            who = n if people else None
            self.add(example([{"role": "user", "content": user_turn(note, who, text)},
                              {"role": "assistant", "content": reply}], "answer.can_do",
                             real=real_of(people, q + " " + self_note)))

    def gen_list_bait(self, k):
        for _ in range(k):
            n = self.name()
            people = self.known_room(n, sample_facts(self.rng, 1))
            q = self.pick(SEED_LIST_BAIT_Q)
            note = self.note_for(people, n)
            reply = self.pick(SEED_LIST_BAIT).format(n=n, N=n)
            self.add(example([{"role": "user", "content": user_turn(note, n, q)},
                              {"role": "assistant", "content": reply}], "answer.list_bait",
                             real=real_of(people, q)))

    def gen_help(self, k):
        for _ in range(k):
            n = self.name()
            people = self.known_room(n, sample_facts(self.rng, self.pick([0, 1])))
            q = self.pick(SEED_HELP_Q)
            note = self.note_for(people, n)
            reply = self.pick(SEED_HELP).format(n=n, N=n)
            self.add(example([{"role": "user", "content": user_turn(note, n, q)},
                              {"role": "assistant", "content": reply}], "answer.help",
                             real=real_of(people, q)))

    def gen_smalltalk(self, k):
        for _ in range(k):
            n = self.name()
            fs = sample_facts(self.rng, self.pick([1, 2]))
            people = self.known_room(n, fs)
            q, replies = self.pick(SEED_SMALLTALK)
            note = self.note_for(people, n)
            reply = self.pick(replies).format(n=n, N=n)
            self.add(example([{"role": "user", "content": user_turn(note, n, q)},
                              {"role": "assistant", "content": reply}], "answer.smalltalk",
                             real=real_of(people, q)))

    def gen_two_people(self, k):
        for _ in range(k):
            a, b = self.name(), None
            b = self.name(avoid=(a,))
            fa, fb = sample_facts(self.rng, 1), sample_facts(self.rng, 1)
            people = [{"name": b, "facts": facts_for(b, fb)}, {"name": a, "facts": facts_for(a, fa)}]
            q, replies = self.pick(SEED_TWO_PEOPLE)
            note = self.note_for(people, a)
            reply = self.pick(replies).format(n=a, N=a)
            self.add(example([{"role": "user", "content": user_turn(note, a, q)},
                              {"role": "assistant", "content": reply}], "answer.two_people",
                             real=real_of(people, q)))

    def gen_no_double_greet(self, k):
        for _ in range(k):
            n = self.name()
            fs = sample_facts(self.rng, 1)
            people = self.known_room(n, fs)
            note = self.note_for(people, n)
            text = f"hi\n\n{NOTE_ALREADY_GREETED.format(name=n)}"
            q = self.pick(fs[0]["q"])
            reply = self.pick(SEED_NO_DOUBLE_GREET).format(n=n, N=n, q=q[0].lower() + q[1:])
            hist = [{"role": "assistant", "content": self.pick(SEED_ARRIVAL_PLAIN).format(n=n, N=n, part="morning")}]
            self.add(example(hist + [{"role": "user", "content": user_turn(note, n, text)},
                                     {"role": "assistant", "content": reply}], "answer.no_double_greet",
                             real=real_of(people)), allow_greet=False)

    def gen_hello_dark(self, k):
        table = SEED_HELLO_DARK[0]
        for _ in range(k):
            streak = self.pick([1, 1, 2, 3, 4, 5, 6])
            note = self.note_for([], None, dark=self.rng.random() < 0.7, streak=streak)
            hist = []
            for i in range(max(1, streak - 2), streak):  # bounded history: fits the sequence cap
                hist.append({"role": "user", "content": user_turn(self.note_for([], None, dark=True, streak=i), None, "Hello.")})
                hist.append({"role": "assistant", "content": self.pick(table[i])})
            reply = self.pick(table[streak])
            msgs = hist + [{"role": "user", "content": user_turn(note, None, "Hello.")},
                           {"role": "assistant", "content": reply}]
            self.add(example(msgs, "answer.hello_dark", real=""))

    def gen_stranger_talk(self, k):
        for _ in range(k):
            people = [{"name": None}]
            if self.rng.random() < 0.3:
                m = self.name()
                people.append({"name": m, "facts": facts_for(m, sample_facts(self.rng, 1))})
            q, replies = self.pick(SEED_STRANGER_UTTER)
            note = self.note_for(people, None)
            reply = self.pick(replies)
            self.add(example([{"role": "user", "content": user_turn(note, None, q)},
                              {"role": "assistant", "content": reply}], "answer.stranger_talk",
                             real=real_of(people, q)))

    def gen_lull(self, k):
        for _ in range(k):
            n = self.name()
            fs = sample_facts(self.rng, self.pick([1, 2]))
            people = self.known_room(n, fs)
            note = self.note_for(people, n, lull=True)
            text = SMALL_TALK_NOTE.format(name=n)
            f = fs[-1]
            q = self.pick(f["q"])
            form = self.pick(SEED_LULL_Q)
            reply = form.format(q=q, r=self.pick(f["r"]), N=n, q_low=q[0].lower() + q[1:])
            self.add(example([{"role": "user", "content": user_turn(note, n, text)},
                              {"role": "assistant", "content": reply}], "answer.lull",
                             real=real_of(people)), allow_greet=False, max_sentences=1)

    def gen_check_in(self, k):
        abouts = ["maths test", "science fair", "football match", "choir concert", "job interview",
                  "dentist appointment", "school trip", "drama show"]
        for _ in range(k):
            n = self.name()
            people = self.known_room(n, sample_facts(self.rng, self.pick([0, 1])))
            about = self.pick(abouts)
            note = self.note_for(people, n, lull=True)
            text = CHECK_IN_NOTE.format(name=n, about=about)
            reply = self.pick(SEED_CHECK_IN).format(n=n, N=n, about=about)
            self.add(example([{"role": "user", "content": user_turn(note, n, text)},
                              {"role": "assistant", "content": reply}], "answer.check_in",
                             real=real_of(people, about)), allow_greet=False, max_sentences=1)

    def gen_introduced(self, k):
        for _ in range(k):
            n = self.name()
            people = [{"name": None}]
            note = self.note_for(people, None)
            lead = self.pick(["I'm {n}", "my name is {n}", "I am {n}", "it's {n} actually", "call me {n}", "this is {n}"])
            text = INTRODUCED_NOTE.format(name=n) + "\n\n" + lead.format(n=n) + self.pick([".", "", ", hi", ", is this on?"])
            reply = self.pick(SEED_INTRODUCED).format(n=n, N=n)
            self.add(example([{"role": "user", "content": user_turn(note, None, text)},
                              {"role": "assistant", "content": reply}], "answer.introduced",
                             real=real_of(people, n)))

    # -- tool flows ---------------------------------------------------------
    def gen_remember_name(self, k):
        for i in range(k):
            n = self.name()
            people = [{"name": None}]
            if self.rng.random() < 0.25:
                m = self.name(avoid=(n,))
                people.append({"name": m, "facts": facts_for(m, sample_facts(self.rng, 1))})
            mode = i % 3
            if mode == 0:  # missed by the regex, stranger note present
                note = self.note_for(people, None)
                text = self.pick(SEED_INTRO_MISSED).format(n=n)
            elif mode == 1:  # answering the name question with more than two words
                note = self.note_for(people, None)
                text = NAME_ANSWER_HINT + "\n\n" + self.pick(SEED_INTRO_AFTER_ASK).format(n=n)
            else:  # nobody visible (dark or no camera), a name offered anyway
                people = []
                note = self.note_for([], None, dark=self.rng.random() < 0.5)
                text = self.pick(SEED_INTRO_MISSED + ["I'm {n}", "my name is {n}", "{n}."]).format(n=n)
            u = {"role": "user", "content": user_turn(note, None, text)}
            self.add(example([u, call("remember_name", name=n)], "tool.remember_name", real=""), spoken=False)
            res = tool_result({"status": "ok", "remembered": n, "entity": n.lower()})
            reply = self.pick(SEED_AFTER_NAME).format(n=n, N=n)
            self.add(example([u, call("remember_name", name=n), res,
                              {"role": "assistant", "content": reply}], "tool.remember_name.answer",
                             real=real_of(people, n)))

    def gen_recall_person(self, k):
        for i in range(k):
            n = self.name()
            a = self.name(avoid=(n,))
            fs = sample_facts(self.rng, self.pick([0, 1]))
            people = self.known_room(n, fs)
            speaker = n
            if i % 4 == 3:  # nobody visible, unclear speaker
                people, speaker = [], None
            q = self.pick(SEED_ABSENT_Q).format(a=a)
            absent = bool(people) and (self.rng.random() < 0.75)  # the runtime's gate, most of the time
            note = self.note_for(people, speaker, absent=absent)
            u = {"role": "user", "content": user_turn(note, speaker, q)}
            self.add(example([u, call("recall_person", name=a)], "tool.recall_person", real=""), spoken=False)
            if self.rng.random() < 0.6:
                af = sample_facts(self.rng, 1)[0]
                res = tool_result({"status": "ok", "name": a, "facts": [af["t"].format(n=a)]})
                fact = af["t"].format(n=a).rstrip(".")
                rest = fact[len(a):].strip() if fact.startswith(a) else fact
                reply = self.pick(SEED_ABSENT_KNOWN).format(a=a, A=a, fact=fact, Fact=fact, rest=rest)
                real = real_of(people, q + " " + af["t"].format(n=a))
            else:
                res = tool_result({"status": "unknown", "known_people": [n] if people else []})
                reply = self.pick(SEED_ABSENT_UNKNOWN).format(a=a, n=n if people else "then")
                real = real_of(people, q)
            self.add(example([u, call("recall_person", name=a), res,
                              {"role": "assistant", "content": reply}], "tool.recall_person.answer", real=real))

    def gen_forget(self, k):
        for _ in range(k):
            n = self.name()
            fs = sample_facts(self.rng, self.pick([0, 1, 2]))
            people = self.known_room(n, fs)
            q = self.pick(SEED_FORGET_Q)
            note = self.note_for(people, n, memory=True)
            u = {"role": "user", "content": user_turn(note, n, q)}
            self.add(example([u, call("forget_person", name=n)], "tool.forget_person", real=""), spoken=False)
            res = tool_result({"status": "ok"})
            reply = self.pick(SEED_FORGOTTEN).format(n=n, N=n)
            self.add(example([u, call("forget_person", name=n), res,
                              {"role": "assistant", "content": reply}], "tool.forget_person.answer",
                             real=real_of(people, q)))

    def gen_remember_fact(self, k):
        for _ in range(k):
            n = self.name()
            fs = sample_facts(self.rng, self.pick([0, 1]))
            people = self.known_room(n, fs)
            q, fact_t, replies = self.pick(SEED_FACT_UTTER)
            fact = fact_t.format(n=n)
            note = self.note_for(people, n)
            u = {"role": "user", "content": user_turn(note, n, q)}
            self.add(example([u, call("remember_fact", name=n, fact=fact)], "tool.remember_fact", real=""), spoken=False)
            res = tool_result({"status": "ok", "name": n})
            reply = self.pick(replies).format(n=n, N=n)
            self.add(example([u, call("remember_fact", name=n, fact=fact), res,
                              {"role": "assistant", "content": reply}], "tool.remember_fact.answer",
                             real=real_of(people, q + " " + fact)))

    def gen_reminders(self, k):
        for i in range(k):
            n = self.name()
            people = self.known_room(n, sample_facts(self.rng, self.pick([0, 1])))
            note = self.note_for(people, n)
            if i % 3 != 2:
                text, when, forms = self.pick(REMINDERS)
                q = self.pick(forms)
                u = {"role": "user", "content": user_turn(note, n, q)}
                c = call("remember_reminder", name=n, text=text, when=when)
                self.add(example([u, c], "tool.remember_reminder", real=""), spoken=False)
                due = DUE_WORDS[when]
                res = tool_result({"status": "ok", "name": n, "id": self.rng.randint(1, 40), "text": text, "due": due})
                reply = self.pick(SEED_REMIND_OK).format(n=n, N=n, text=text, Text=text[0].upper() + text[1:], due=due)
                self.add(example([u, c, res, {"role": "assistant", "content": reply}],
                                 "tool.remember_reminder.answer", real=real_of(people, q + " " + due)))
            else:
                q = self.pick(SEED_LIST_REMINDERS_Q)
                u = {"role": "user", "content": user_turn(note, n, q)}
                c = call("list_reminders", name=n)
                self.add(example([u, c], "tool.list_reminders", real=""), spoken=False)
                if self.rng.random() < 0.7:
                    text, when, _ = self.pick(REMINDERS)
                    due = DUE_WORDS[when]
                    res = tool_result({"status": "ok", "name": n, "reminders": [{"id": 3, "text": text, "due": due}]})
                    reply = self.pick(SEED_LIST_REMINDERS_ONE).format(n=n, N=n, text=text, Text=text[0].upper() + text[1:], due=due)
                    real = real_of(people, q + " " + text + " " + due)
                else:
                    res = tool_result({"status": "ok", "name": n, "reminders": []})
                    reply = self.pick(SEED_LIST_REMINDERS_NONE).format(n=n, N=n)
                    real = real_of(people, q)
                self.add(example([u, c, res, {"role": "assistant", "content": reply}],
                                 "tool.list_reminders.answer", real=real))

    # -- proactive moments ----------------------------------------------------
    def pro_note(self, what, *, name=None, facts=(), ctx=None, greeted=False, recent=(), crowd=None,
                 mood=None, moment="arrival", waiting_line=None):
        h, m = self.rng.randint(7, 19), self.rng.randint(0, 59)
        tod, part = time_of_day(h, m)
        who = name or "someone"
        s = f"[note] Nobody said anything; this is you speaking first. What is happening: {what}"
        s += f"\nTime: {tod}."
        if ctx:
            s += f"\nAbout {who}: {ctx}."
        if facts:
            s += f"\nYou know about {who}: " + "; ".join(f.rstrip(".") for f in list(facts)[::-1][:3]) + "."
        if crowd:
            s += "\n" + crowd
        if mood:
            s += f"\n{who} has seemed {mood} lately."
        if recent:
            s += "\nYou said these recently, so none of that phrasing again: " + " ".join(f'"{r}"' for r in recent) + "."
        s += ("\nBrief: ONE sentence, like a friend in the room, not a service. Speak to them directly, "
              "as \"you\", never about them. ")
        if greeted:
            s += "You already said hello to them a few minutes ago, so no hi, hello or hey. "
        s += {
            "return": "Mention how long it has been, or pick up what you last talked about. No question. ",
            "arrival": "Greet them by name and add one small thing. No question. ",
            "pair": "One hello for both, by name. No question. ",
            "group": "One hello for everyone at once, names woven in if you know any. Never one per person. No question. ",
            "wrapup": "Kindly hand the floor over: tell the talker you'll come back to them, then invite the one waiting, by name if known. ",
            "stranger": "Ask their name, plainly, without a hello in front of it. Nothing else. ",
            "reminder": f"Speak to {who if name else 'them'} as \"you\": they are the one who has to do it, not you. Say what they asked to be reminded of, keeping their words for the thing itself. ",
            "novelty": "Say what you noticed, in your own words; a question is fine. ",
            "lights": "Plain words, no drama and no poetry: the lights went and you cannot see. No question. ",
            "invite": "Invite them over, lightly, in your own words. ",
            "followup": "One follow-up only: repeat what you asked in different words, or check they are still there. ",
            "muse": "One line to the empty room, about the time or the quiet; nothing invented. No question. ",
        }[moment]
        if moment in ("arrival", "return", "pair", "group", "lights", "muse"):
            s += "Not a question. "
        s += "Never \"how are you\", never an offer to help. Write just the sentence, no quotes."
        return s, part

    def pro(self, note, reply, kind, real, hist=(), allow_greet=True, no_question=False):
        if no_question and "?" in reply:
            self.rejected["question not allowed"] += 1
            return
        msgs = list(hist) + [{"role": "user", "content": note}, {"role": "assistant", "content": reply}]
        self.add(example(msgs, kind, tools=False, real=real), allow_greet=allow_greet, max_sentences=1)

    def gen_arrival(self, k):
        for i in range(k):
            n = self.name()
            fs = sample_facts(self.rng, self.pick([0, 0, 1, 2]))
            facts = facts_for(n, fs)
            mode = i % 5
            recent = [self.pick(SEED_ARRIVAL_PLAIN).format(n=self.name(avoid=(n,)), N=self.name(), part="morning")] if self.rng.random() < 0.4 else []
            if mode == 0 and fs:
                note, part = self.pro_note(f"{n} just walked in.", name=n, facts=facts, recent=recent)
                r = self.pick(fs[-1]["r"])
                reply = self.pick(SEED_ARRIVAL_FACT).format(n=n, N=n, r_low=r[0].lower() + r[1:])
            elif mode == 1:
                note, part = self.pro_note(f"{n} just walked in.", name=n, facts=facts, mood="tired", recent=recent)
                reply = self.pick(SEED_ARRIVAL_TIRED).format(n=n, N=n, part=part)
            elif mode == 2:
                note, part = self.pro_note(f"{n} just walked in.", name=n, facts=facts, crowd="There are 2 people here.", recent=recent)
                reply = self.pick(SEED_ARRIVAL_CROWD).format(n=n, N=n)
            elif mode == 3 and fs:
                note, part = self.pro_note(f"{n} just walked in.", name=n, facts=facts, greeted=True, recent=recent)
                r = self.pick(fs[-1]["r"])
                reply = self.pick(SEED_ARRIVAL_AGAIN).format(n=n, N=n, part=part, r_low=r[0].lower() + r[1:])
                self.pro(note, reply, "proactive.arrival_again", real_of([{"name": n, "facts": facts}]), allow_greet=False, no_question=True)
                continue
            else:
                note, part = self.pro_note(f"{n} just walked in.", name=n, facts=facts, recent=recent)
                reply = self.pick(SEED_ARRIVAL_PLAIN).format(n=n, N=n, part=part)
            self.pro(note, reply, "proactive.arrival", real_of([{"name": n, "facts": facts}]), no_question=True)

    def gen_return(self, k):
        things = ["Rust parser", "robot arm", "volcano project", "maths test", "choir concert", "dragon game",
                  "weather station", "science fair", "5k run", "Rome project"]
        for i in range(k):
            n = self.name()
            secs = self.pick([660, 1500, 3600, 7200, 5 * 3600, 86400, 172800, 3 * 86400, 7 * 86400])
            away = away_words(secs)
            Away = away[0].upper() + away[1:]
            facts, ctx, thing = [], None, None
            if i % 2 == 0:
                thing = self.pick(things)
                facts = [f"{n} is working on a {thing}."]
                days = max(secs // 86400, 1)
                ctx = f"last visit {days} days ago: talked about the {thing}"
            note, part = self.pro_note(f"{n} is back after {away} away.", name=n, facts=facts, ctx=ctx, moment="return")
            if thing:
                reply = self.pick(SEED_RETURN_CTX).format(n=n, N=n, away=away, Away=Away, thing=thing)
            elif secs < 3600:
                reply = self.pick(SEED_RETURN_SHORT).format(n=n, N=n, away=away, Away=Away)
            else:
                reply = self.pick(SEED_RETURN).format(n=n, N=n, away=away, Away=Away)
            self.pro(note, reply, "proactive.return", real_of([{"name": n, "facts": facts}], away + " " + (ctx or "")), no_question=True)

    def gen_stranger_settled(self, k):
        for _ in range(k):
            recent = [self.pick(SEED_STRANGER)] if self.rng.random() < 0.5 else []
            note, _ = self.pro_note("someone you do not recognise has settled in, and you have not asked their name.",
                                    moment="stranger", recent=recent)
            reply = self.pick([s for s in SEED_STRANGER if s not in recent])
            self.pro(note, reply, "proactive.stranger", "", allow_greet=False)

    def gen_reminder_due(self, k):
        for _ in range(k):
            n = self.name()
            text, _, _ = self.pick(REMINDERS)
            note, _ = self.pro_note(f"{n} asked you earlier to remind them to {text}, and it is time.", name=n, moment="reminder")
            reply = self.pick(SEED_REMINDER).format(n=n, N=n, text=text)
            self.pro(note, reply, "proactive.reminder", real_of([{"name": n}], text))

    def gen_novelty(self, k):
        for _ in range(k):
            n = self.name()
            obj = self.pick(OBJECTS)
            qs, replies = NOVELTY[obj]
            note, _ = self.pro_note(f"you noticed something new in the room. Your first thought was: {self.pick(qs)}",
                                    name=n, moment="novelty")
            reply = self.pick(replies)
            self.pro(note, reply, "proactive.novelty", real_of([{"name": n}], obj))

    def gen_lights(self, k):
        for _ in range(k):
            recent = [self.pick(SEED_LIGHTS)] if self.rng.random() < 0.5 else []
            note, _ = self.pro_note("the lights just went out; the camera sees nothing now.", moment="lights", recent=recent)
            reply = self.pick([s for s in SEED_LIGHTS if s not in recent])
            self.pro(note, reply, "proactive.lights", "", no_question=True)

    def gen_pair(self, k):
        for _ in range(k):
            a = self.name()
            b = self.name(avoid=(a,))
            note, _ = self.pro_note(f"{a} and {b} just walked in together.", name=a, moment="pair")
            reply = self.pick(SEED_PAIR).format(a=a, b=b, A=a, B=b)
            self.pro(note, reply, "proactive.pair", real_of([{"name": a}, {"name": b}]), no_question=True)

    def gen_group(self, k):
        for i in range(k):
            names = [self.name() for _ in range(self.pick([0, 1, 2]))] if i % 3 else []
            names = list(dict.fromkeys(names))
            if names:
                what = f"a whole group just walked in, {', '.join(names)} among them."
                reply = self.pick(SEED_GROUP_NAMED).format(names=" and ".join(names))
            else:
                what = "a whole group just walked in; you know none of their names."
                reply = self.pick(SEED_GROUP_ANON)
            note, _ = self.pro_note(what, name=names[0] if names else None, moment="group",
                                    crowd=f"There are {self.rng.randint(3, 7)} people here.")
            self.pro(note, reply, "proactive.group", real_of([{"name": x} for x in names]), no_question=True)

    def gen_wrapup(self, k):
        for i in range(k):
            n = self.name()
            w = self.name(avoid=(n,)) if i % 3 else None
            secs = self.rng.randint(60, 200)
            crowd = f"There are 3 people here. Waiting to talk to you: {w or 'someone'}. The one talking has been going for {secs} seconds. Wrap it up lightly (\"hold that thought\") and turn to whoever is next."
            what = f"{n} has been talking for a long while and {w or 'someone else'} has been waiting to speak."
            note, _ = self.pro_note(what, name=n, moment="wrapup", crowd=crowd)
            reply = (self.pick(SEED_WRAPUP).format(n=n, N=n, w=w) if w else self.pick(SEED_WRAPUP_NONAME).format(n=n, N=n))
            self.pro(note, reply, "proactive.wrapup", real_of([{"name": n}, {"name": w or ""}]))

    def gen_invite(self, k):
        for i in range(k):
            if i % 2:
                n = self.name()
                note, _ = self.pro_note(f"{n} is in view but keeping their distance, not yet greeted and not looking your way.", name=n, moment="invite")
                reply = self.pick(SEED_INVITE_KNOWN).format(n=n, N=n)
                self.pro(note, reply, "proactive.invite", real_of([{"name": n}]))
            else:
                note, _ = self.pro_note("someone you do not recognise is in view but keeping their distance, not yet greeted and not looking your way.", moment="invite")
                reply = self.pick(SEED_INVITE_STRANGER)
                self.pro(note, reply, "proactive.invite", "")

    def gen_followup(self, k):
        for i in range(k):
            if i % 2:
                n = self.name()
                note, _ = self.pro_note(f"you said hello to {n} and got no answer.", name=n, moment="followup")
                reply = self.pick(SEED_FOLLOWUP_KNOWN).format(n=n, N=n)
                self.pro(note, reply, "proactive.followup", real_of([{"name": n}]))
            else:
                note, _ = self.pro_note("you asked \"What's your name?\" of someone you do not recognise and got no answer.", moment="followup")
                reply = self.pick(SEED_FOLLOWUP_Q)
                self.pro(note, reply, "proactive.followup", "")

    def gen_muse(self, k):
        for _ in range(k):
            note, part = self.pro_note("nobody has been in view for a while; the room is empty.", moment="muse")
            reply = self.pick(SEED_MUSE).format(part=part)
            self.pro(note, reply, "proactive.muse", "", no_question=True)


def build(target: int, seed: int, teacher: Path | None, max_tokens: int = 1500):
    g = Gen(seed)
    # Shares: ~35% tool turns, ~35% answers, ~30% proactive (each tool flow yields 2 examples).
    t = target
    tool_flows = int(t * 0.35 / 2)
    g.gen_remember_name(int(tool_flows * 0.26))
    g.gen_recall_person(int(tool_flows * 0.24))
    g.gen_forget(int(tool_flows * 0.14))
    g.gen_remember_fact(int(tool_flows * 0.22))
    g.gen_reminders(int(tool_flows * 0.14))
    ans = int(t * 0.35)
    g.gen_recall_with_facts(int(ans * 0.12))
    g.gen_recall_only_name(int(ans * 0.08))
    g.gen_my_name(int(ans * 0.07))
    g.gen_who_are_you(int(ans * 0.06))
    g.gen_can_do(int(ans * 0.06))
    g.gen_list_bait(int(ans * 0.04))
    g.gen_help(int(ans * 0.04))
    g.gen_smalltalk(int(ans * 0.16))
    g.gen_two_people(int(ans * 0.06))
    g.gen_no_double_greet(int(ans * 0.06))
    g.gen_hello_dark(int(ans * 0.07))
    g.gen_stranger_talk(int(ans * 0.05))
    g.gen_lull(int(ans * 0.07))
    g.gen_check_in(int(ans * 0.02))
    g.gen_introduced(int(ans * 0.04))
    pro = int(t * 0.30)
    g.gen_arrival(int(pro * 0.20))
    g.gen_return(int(pro * 0.14))
    g.gen_stranger_settled(int(pro * 0.07))
    g.gen_reminder_due(int(pro * 0.08))
    g.gen_novelty(int(pro * 0.08))
    g.gen_lights(int(pro * 0.06))
    g.gen_pair(int(pro * 0.07))
    g.gen_group(int(pro * 0.07))
    g.gen_wrapup(int(pro * 0.07))
    g.gen_invite(int(pro * 0.06))
    g.gen_followup(int(pro * 0.05))
    g.gen_muse(int(pro * 0.05))

    if teacher and teacher.exists():
        n = 0
        for line in teacher.read_text().splitlines():
            if line.strip():
                ex = json.loads(line)
                ex.setdefault("kind", "teacher")
                ex.setdefault("_real", "")
                g.out.append(ex)
                n += 1
        print(f"teacher samples added: {n}")

    # Dedupe on the exact (last user, target) pair.
    seen, uniq = set(), []
    for ex in g.out:
        key = json.dumps([m for m in ex["messages"][1:]], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ex)
    # Drop anything the trainer would truncate: mlx_lm keeps the FRONT of a sequence, so
    # a prompt past --max-seq-length leaves no target tokens and the loss goes nan
    # (seen on the first pilot: "Train loss nan" at 1536 with a 2122-token example).
    if max_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        kept = []
        for ex in uniq:
            n = len(tok(tok.apply_chat_template(ex["messages"], tools=ex.get("tools"), tokenize=False)).input_ids)
            if n <= max_tokens:
                kept.append(ex)
            else:
                g.rejected[f"over {max_tokens} tokens"] += 1
        uniq = kept
    g.rng.shuffle(uniq)
    n_valid = max(50, len(uniq) // 20)
    valid, train = uniq[:n_valid], uniq[n_valid:]
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (out / f"{name}.jsonl").open("w") as f:
            for ex in rows:
                row = {"messages": ex["messages"]}
                if "tools" in ex:
                    row["tools"] = ex["tools"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    kinds = Counter(ex["kind"] for ex in uniq)
    groups = Counter(k.split(".")[0] for k in kinds.elements())
    total = len(uniq)
    stats = {"total": total, "train": len(train), "valid": len(valid),
             "groups": {k: f"{v} ({v * 100 // total}%)" for k, v in groups.items()},
             "kinds": dict(sorted(kinds.items())), "rejected_by_gate": dict(g.rejected),
             "generated_before_dedupe": len(g.out)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teacher", type=Path, default=None)
    ap.add_argument("--max-tokens", type=int, default=1500, help="drop examples longer than this (0 = keep all)")
    a = ap.parse_args()
    build(a.target, a.seed, a.teacher, a.max_tokens)

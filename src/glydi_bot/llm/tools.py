"""The tools that give Claude a memory of people.

Every one of these runs *after* the bot has already started speaking, or between
turns -- none of them sit between the user finishing a sentence and the first
audio coming back. Enrolment in particular is deliberately not on the fast path:
when the bot meets someone new it simply asks their name, which is an ordinary
conversational turn, and the gallery write happens while it is already talking.
"""

from __future__ import annotations

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from ..identity.client import IdentityClient


# The tool surface, independent of any handler. The local warm-up sends these
# too: Llama and Qwen templates put tool definitions ahead of the system prompt,
# so a warm-up without them primes a prefix the real turns never hit.
TOOL_SPECS: list[dict] = [
    {
        "name": "remember_name",
        "description": (
                    "Attach a name to the person you are currently talking to, so you "
                    "recognise their face and voice next time. Call this as soon as "
                    "someone tells you their name, but only if you do not already know "
                    "them."
                ),
        "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name the person gave you.",
                    }
                },
        "required": ["name"],
    },
    {
        "name": "remember_fact",
        "description": (
                    "Store something worth remembering about a person you already know "
                    "-- what they do, what they like, something they asked you to keep "
                    "track of. Do not store things they would not expect you to keep."
                ),
        "properties": {
                    "name": {"type": "string", "description": "Who the fact is about."},
                    "fact": {
                        "type": "string",
                        "description": "One short sentence, written in the third person.",
                    },
                },
        "required": ["name", "fact"],
    },
    {
        "name": "recall_person",
        "description": (
                    "Look up what you already know about someone by name. Use this when "
                    "you recognise a person and want to pick the conversation back up, "
                    "or when someone asks what you remember about them."
                ),
        "properties": {
                    "name": {"type": "string", "description": "The person's name."}
                },
        "required": ["name"],
    },
    {
        "name": "forget_person",
        "description": (
                    "Permanently delete a person and every stored face and voice sample "
                    "of them. Call this whenever someone asks you to forget them; treat "
                    "the request as final and confirm once it is done."
                ),
        "properties": {
                    "name": {"type": "string", "description": "The person to forget."}
                },
        "required": ["name"],
    },
]


def build_tools(identity: IdentityClient) -> ToolsSchema:
    async def remember_name(params: FunctionCallParams) -> None:
        name = str(params.arguments.get("name", "")).strip()
        result = await identity.request("enrol", name=name)
        if result.ok:
            await params.result_callback(
                {"status": "ok", "remembered": result.data.get("name", name)}
            )
        else:
            await params.result_callback({"status": "failed", "reason": result.error})

    async def remember_fact(params: FunctionCallParams) -> None:
        result = await identity.request(
            "remember",
            name=str(params.arguments.get("name", "")).strip(),
            fact=str(params.arguments.get("fact", "")).strip(),
        )
        await params.result_callback(
            {"status": "ok"} if result.ok else {"status": "failed", "reason": result.error}
        )

    async def recall_person(params: FunctionCallParams) -> None:
        name = str(params.arguments.get("name", "")).strip().lower()
        result = await identity.request("roster")
        if not result.ok:
            await params.result_callback({"status": "failed", "reason": result.error})
            return
        people = result.data.get("people", [])
        hit = next((p for p in people if p["name"].strip().lower() == name), None)
        if hit is None:
            await params.result_callback(
                {"status": "unknown", "known_people": [p["name"] for p in people]}
            )
            return
        await params.result_callback({"status": "ok", "name": hit["name"], "facts": hit["facts"]})

    async def forget_person(params: FunctionCallParams) -> None:
        result = await identity.request(
            "forget", name=str(params.arguments.get("name", "")).strip()
        )
        await params.result_callback(
            {"status": "ok"} if result.ok else {"status": "failed", "reason": result.error}
        )

    handlers = {
        "remember_name": remember_name,
        "remember_fact": remember_fact,
        "recall_person": recall_person,
        "forget_person": forget_person,
    }
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=spec["name"],
                description=spec["description"],
                properties=spec["properties"],
                required=spec["required"],
                handler=handlers[spec["name"]],
            )
            for spec in TOOL_SPECS
        ]
    )

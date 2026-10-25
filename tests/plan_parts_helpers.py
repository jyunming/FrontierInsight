"""For tests whose fake model used to answer a request to complete the plan with the whole plan.md: FI now asks for the
asked parts alone, as one JSON object shaped like the design block (``agents/plan_parts.md``)."""
from __future__ import annotations

import json
from typing import Any

from core import plan


def as_parts(whole_plan_text: str) -> str:
    """The design block of a whole-plan text as the JSON answer the plan's model gives now (what is not asked for is in it
    too, as a model may write it: FI keeps only the asked parts)."""
    block = plan.raw_design_block(whole_plan_text)
    assert isinstance(block, dict), "the fake's plan text has no readable design block"
    return json.dumps(block, default=str)


def parts_reply(obj: Any) -> str:
    return json.dumps(obj, default=str)


#: A repair FI asks of the plan (``Engine._revise_plan_parts``): the plan read-only and the parts it may change named by FI.
SCOPED_MARK = "Change only these parts of the plan"
_SHOWN_FROM = "# The plan as it stands (read it for context; do not copy it back)"


def is_scoped_ask(prompt: str) -> bool:
    return SCOPED_MARK in prompt and _SHOWN_FROM in prompt


def plan_shown(prompt: str) -> str:
    """The plan text a part ask shows the model."""
    return prompt.split(_SHOWN_FROM, 1)[1].split("# What is asked", 1)[0].strip()


def parts_of(edited_plan_text: str, *names: str) -> str:
    """The reply to a part ask: only the protocol parts ``names`` of the (edited) plan text, as one JSON object. A name
    written ``part.key`` is only that key of the part (``model.equations``)."""
    block = plan.raw_design_block(edited_plan_text) or {}
    protocol = block.get("protocol") if isinstance(block.get("protocol"), dict) else {}
    out: dict[str, Any] = {}
    for name in names:
        part, _, key = name.partition(".")
        if part not in protocol:
            continue
        if key:
            if isinstance(protocol[part], dict) and key in protocol[part]:
                out.setdefault(part, {})[key] = protocol[part][key]
        else:
            out[part] = protocol[part]
    return json.dumps({"protocol": out}, default=str)


def checks_reply(edited_plan_text: str, prompt: str) -> str:
    """The reply to a part ask about the checks (``Engine._revise_checks_only``): the checks and criteria of the edited plan,
    and the model's equations when the ask names the model."""
    names = ["oracles", "criteria"]
    if "`model`" in prompt.split(SCOPED_MARK, 1)[1]:
        names.append("model.equations")
    return parts_of(edited_plan_text, *names)

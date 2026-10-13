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

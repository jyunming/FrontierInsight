"""Skills — what FI knows about driving a piece of scientific software.

The unit of accumulation for a research engine that is meant to get better
by being used, rather than regenerating every experiment from scratch.

See ``base.py`` for the on-disk envelope and the status model,
``approval.py`` for the human gate, ``registry.py`` for discovery and
promotion, and ``scan.py`` for the static review a person reads before
approving one.
"""

from core.skills.base import (
    Kind,
    Maturity,
    Skill,
    SkillState,
    Status,
)
from core.skills.registry import (
    discover,
    evaluate,
    loadable_skills,
    run_selftest,
)
from core.skills import scan

__all__ = [
    "Kind",
    "Maturity",
    "Skill",
    "SkillState",
    "Status",
    "discover",
    "evaluate",
    "loadable_skills",
    "run_selftest",
    "scan",
]

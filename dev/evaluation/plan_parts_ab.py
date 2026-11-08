"""Ask a recorded quest's plan the contract's parts, and only those (no first plan call): which kind each result is, what
it changes with, what the undeclared names are, what is compared (core/contract_parts.py).

    python -m dev.evaluation.plan_parts_ab <quest folder> --out <folder> [--model gpt-5.6-terra] [--tag run1]

The plan is the quest's last ``plan.md`` design, with the frozen protocol in place of its protocol when the quest froze
one. The provider is the CLI one of the quest's own configuration with ``--model``; nothing is written into the quest. Output,
in ``<out>/<tag>/``: ``answers.json`` (each part: asks, validity, problems, the answer), ``design.json`` (the design after
the parts), ``contract.json`` (the study contract read from it), ``cost.json`` (calls, tokens, wall time).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


def load_design(quest: Path) -> dict[str, Any]:
    from core import frozen_protocol, plan

    design = plan.parse((quest / "plan.md").read_text(encoding="utf-8")).design
    if design is None:
        raise SystemExit(f"{quest}: plan.md has no usable design block")
    frozen = frozen_protocol.protocol_of(quest)
    return {**design, "protocol": frozen} if frozen else design


async def run(quest: Path, out: Path, model: str, tag: str) -> dict[str, Any]:
    from core import contract
    from core.config import Config, ProviderConfig
    from core.engine import Engine

    cfg = Config.from_yaml(quest / "config.yaml")
    cfg.provider = ProviderConfig(name="codex_cli", model=model, node_cli_timeout_s={"plan_contract": 1800})
    cfg.output.output_dir = out
    cfg.knowledge.enabled = False
    engine = Engine(cfg, resume_quest_id=tag)
    design = load_design(quest)
    started = time.time()
    await engine._connect_llm()
    try:
        after, notes = await engine._ask_contract_parts(design)
    finally:
        await engine._client.aclose()
    wall = time.time() - started
    folder = engine.quest_root
    record = json.loads((engine.fi_dir / "contract_parts.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in (engine.fi_dir / "model_calls.jsonl").read_text(encoding="utf-8").splitlines() if line]
    usage = [c.get("usage") or {} for c in calls]
    cost = {"calls": len(calls), "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage), "wall_s": round(wall, 1)}
    c = contract.from_design(after)
    (folder / "answers.json").write_text(json.dumps({**record, "notes": notes}, indent=1, default=str), encoding="utf-8")
    (folder / "design.json").write_text(json.dumps(after, indent=1, default=str), encoding="utf-8")
    (folder / "contract.json").write_text(c.model_dump_json(indent=1), encoding="utf-8")
    (folder / "cost.json").write_text(json.dumps(cost), encoding="utf-8")
    return {"form": record["form"], **cost}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    ap = argparse.ArgumentParser()
    ap.add_argument("quest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--tag", default="run1")
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a.quest.resolve(), a.out.resolve(), a.model, a.tag))))


if __name__ == "__main__":
    main()

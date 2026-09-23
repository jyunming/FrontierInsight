"""A script that ignores ``FI_REPLICATE_SEED`` is repaired once, after implement.

Four of six recent codex quests hard-coded their own seed (``RNG_SEED = ...``,
``SEED = 20260919``, ``MASTER_SEED``) and never read the variable the engine
sets on every run. Every replicate was the same run, ``execute`` published no
aggregate, and the paper could report one measurement with no interval. The
implement prompts already say not to do that; a model that does it anyway is
asked once more, right after implement, with the one change to make.

The chat client is a fake that records every call. The replicate runs are real
subprocesses of THIS interpreter (the scripts are stdlib only), so "the
replicates are then independent" is observed rather than assumed.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine, _script_reads_replicate_seed

# The shape the four quests shipped: an RNG seeded from a constant the script
# wrote down, with the engine's variable nowhere in the file. Padded past the
# 8000 characters ``execute_reflect`` cuts its previous code to, with a marker
# on the last line: a repair must be handed the whole script.
TAIL = "# END-OF-SCRIPT-MARKER"
IGNORING = (
    "import json\n"
    "import random\n"
    "# " + "x" * 9000 + "\n"
    "RNG_SEED = 20260919\n"
    "rng = random.Random(RNG_SEED)\n"
    'print("RESULT_JSON: " + json.dumps({"draw": rng.random()}))\n'
    + TAIL  # no final newline: a fenced block is read back without one
)
READING = IGNORING.replace(
    "RNG_SEED = 20260919\n", 'import os\nRNG_SEED = int(os.environ.get("FI_REPLICATE_SEED", 0))\n',
)
# Rewritten, but the seed is still the script's own.
STILL_IGNORING = IGNORING.replace("20260919", "12345")
# Names the variable, but is cut off mid-statement: not a script.
TRUNCATED = 'import os\nseed = int(os.environ.get("FI_REPLICATE_SEED", 0)\nrng = ('

DESIGN = {"hypothesis": "h", "dependencies": []}


def _implement_reply(code: str, deps: str = "numpy") -> str:
    return f"```python\n{code}\n```\nDEPS: {deps}"


def _repair_reply(code: str, **extra: Any) -> str:
    return json.dumps({"code": code, "deps": [], "patch_summary": "read FI_REPLICATE_SEED", **extra})


class FakeClient:
    """Answers each chat call from ``replies`` in order and remembers it."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []  # (node, prompt)
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.calls.append((kw.get("node", ""), messages[-1]["content"]))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def nodes(self) -> list[str]:
        return [node for node, _ in self.calls]


def _engine(
    tmp_path: Path, replies: list[Any], *, replicates: int = 3, background_jobs: bool = False,
    rigor_profile: str = "default",
) -> tuple[Engine, FakeClient, list[tuple[int, str]]]:
    cfg = Config(
        topic="replicates", title="seed-repair",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
            execute_replicates=replicates, pilot_run=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, background_jobs=background_jobs),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    if rigor_profile != "default":
        # model_copy, not the constructor: this file's fixtures deliberately skip the profile's
        # other forced settings (a fresh venv per quest, etc.) to keep these tests fast and
        # focused on the seed-repair path alone -- only rigor_profile itself needs to be "research"
        # for _repair_ignored_replicate_seed's own check to see it.
        cfg = cfg.model_copy(update={"rigor_profile": rigor_profile})
    eng = Engine(cfg)
    client = FakeClient(replies)
    eng._client = client  # type: ignore[assignment]
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    logged: list[tuple[int, str]] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logged.append((record.levelno, record.getMessage()))

    eng._log.addHandler(_Capture())
    return eng, client, logged


def _implement_state(**extra: Any) -> dict[str, Any]:
    return {"topic": "t", "title": "x", "design": DESIGN, "clarify_answers": {}, **extra}


def _code_path(eng: Engine) -> Path:
    return eng.quest_root / "code" / "experiment.py"


def _run_with_this_python(eng: Engine) -> None:
    """Run each replicate for real, in this interpreter, with the environment
    the engine hands it."""

    async def run(argv: list[str], *, cwd: Path, timeout_s: float, env: dict[str, str] | None = None):
        done = subprocess.run(
            [sys.executable, *argv[1:]], cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
        )
        return SimpleNamespace(
            returncode=done.returncode, stdout=done.stdout, stderr=done.stderr,
            duration_s=1.0, timed_out=False,
        )

    eng.executor.execute = run  # type: ignore[method-assign,assignment]
    eng.executor.install = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(returncode=0, stderr=""))


def _warnings(logged: list[tuple[int, str]]) -> list[str]:
    return [msg for level, msg in logged if level >= logging.WARNING]


def _cost_nodes(eng: Engine) -> list[str]:
    lines = (eng.fi_dir / "cost.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["node"] for line in lines]


# --- the repair --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_ignoring_script_is_repaired_once_and_its_replicates_are_then_independent(
    tmp_path: Path,
) -> None:
    eng, client, logged = _engine(
        tmp_path, [_implement_reply(IGNORING), _repair_reply(READING, deps=["scipy"])],
    )

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    # One call more than before, under its own node name; the second script is
    # the one written, returned and used.
    assert client.nodes == ["implement", "implement_seed"]
    assert _code_path(eng).read_text(encoding="utf-8") == READING
    assert patch["code"] == READING
    assert _script_reads_replicate_seed(_code_path(eng))
    # Not a repair iteration: the node hands back the script and its packages, nothing else,
    # so the exec_reflect budget is untouched.
    assert set(patch) == {"code", "deps"}
    assert patch["deps"] == ["numpy", "scipy"]
    assert not [w for w in _warnings(logged)], _warnings(logged)

    # The replicates the repair was made for: three real runs, three seeds, three draws.
    _run_with_this_python(eng)
    executed = await eng._node_execute({"deps": []})  # type: ignore[arg-type]
    replicates = executed["result_json_replicates"]
    assert [r["_seed"] for r in replicates] == [0, 1, 2]
    assert len({r["draw"] for r in replicates}) == 3
    assert not executed.get("result_json_replicate_seed_ignored")


@pytest.mark.asyncio
async def test_without_the_repair_the_same_script_is_reported_as_one_measurement(
    tmp_path: Path,
) -> None:
    """The control: what the repair is for. Same script, same runs, no repair."""
    eng, _, _ = _engine(tmp_path, [])
    _code_path(eng).write_text(IGNORING, encoding="utf-8")
    _run_with_this_python(eng)

    executed = await eng._node_execute({"deps": []})  # type: ignore[arg-type]

    assert executed["result_json_replicate_seed_ignored"] is True
    assert executed["result_json_replicates"] == []


@pytest.mark.asyncio
async def test_the_repair_prompt_carries_the_whole_script_and_the_one_change(tmp_path: Path) -> None:
    eng, client, _ = _engine(tmp_path, [_implement_reply(IGNORING), _repair_reply(READING)])

    await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    prompt = client.calls[1][1]
    assert TAIL in prompt, "the script was cut short, so a returned script would be too"
    assert "This script has NOT been run" in prompt
    assert "FI_REPLICATE_SEED (default 0 when it is unset)" in prompt
    assert "Keep everything else in the script unchanged" in prompt
    assert '"hypothesis": "h"' in prompt, "the design is in front of the model, as for any repair"
    assert "$previous_code" not in prompt and "$stdout_tail" not in prompt, "a slot was left unfilled"


@pytest.mark.asyncio
async def test_the_repair_is_logged_under_its_own_node_name(tmp_path: Path) -> None:
    eng, _, logged = _engine(tmp_path, [_implement_reply(IGNORING), _repair_reply(READING)])

    await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert _cost_nodes(eng) == ["implement", "implement_seed"]
    assert any(
        "[implement] the script now seeds every generator it builds" in m for _, m in logged
    )


@pytest.mark.asyncio
async def test_a_fenced_answer_is_accepted_too(tmp_path: Path) -> None:
    eng, client, logged = _engine(
        tmp_path, [_implement_reply(IGNORING), _implement_reply(READING, deps="scipy")],
    )

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert patch["code"] == READING  # a fence is read the way implement's is
    assert patch["deps"] == ["numpy", "scipy"]
    assert client.nodes == ["implement", "implement_seed"]
    assert not _warnings(logged), _warnings(logged)


# --- when it must not spend a call -----------------------------------------------------

@pytest.mark.asyncio
async def test_a_script_that_already_reads_the_seed_costs_no_extra_call(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [_implement_reply(READING)])

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert client.nodes == ["implement"]
    assert patch["code"] == READING
    assert _code_path(eng).read_text(encoding="utf-8") == READING
    assert _cost_nodes(eng) == ["implement"]
    assert not _warnings(logged)


@pytest.mark.asyncio
@pytest.mark.parametrize("why", ["one replicate", "background job", "no_simulation", "survey"])
async def test_no_call_when_no_replicate_runs_would_be_made(tmp_path: Path, why: str) -> None:
    eng, client, _ = _engine(
        tmp_path, [_implement_reply(IGNORING)],
        replicates=1 if why == "one replicate" else 3,
        background_jobs=why == "background job",
    )
    state = _implement_state(
        no_simulation_resolved=why == "no_simulation", survey_mode_resolved=why == "survey",
    )

    patch = await eng._node_implement(state)  # type: ignore[arg-type]

    assert client.nodes == ["implement"], f"{why}: made a repair call"
    assert patch["code"] == IGNORING


@pytest.mark.asyncio
async def test_no_call_to_seed_the_stub_written_when_implement_returned_nothing(tmp_path: Path) -> None:
    eng, client, _ = _engine(tmp_path, ["I could not write this."])

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert client.nodes == ["implement"]
    assert patch["code"] == 'print("RESULT_JSON: {}")'


# --- when the repair does not work -------------------------------------------------------

@pytest.mark.asyncio
async def test_a_repair_that_still_ignores_the_seed_warns_once_and_the_quest_proceeds(
    tmp_path: Path,
) -> None:
    eng, client, logged = _engine(tmp_path, [_implement_reply(IGNORING), _repair_reply(STILL_IGNORING)])

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    # Asked once, not twice; the script as written stays.
    assert client.nodes == ["implement", "implement_seed"]
    assert patch["code"] == IGNORING
    assert _code_path(eng).read_text(encoding="utf-8") == IGNORING
    warned = _warnings(logged)
    assert len(warned) == 1, warned
    assert "[implement]" in warned[0] and "still does not read it" in warned[0]
    assert "single measurement" in warned[0]

    # ...and the fallback that already existed handles what is left.
    _run_with_this_python(eng)
    executed = await eng._node_execute({"deps": []})  # type: ignore[arg-type]
    assert executed["result_json_replicate_seed_ignored"] is True


@pytest.mark.asyncio
async def test_under_research_profile_an_unrepairable_seed_pauses_instead_of_proceeding(
    tmp_path: Path,
) -> None:
    """The re-audit's P1-3: outside research profile this failure only warns (the test above) --
    under it, the quest must not silently run on results it just warned are unreproducible."""
    eng, client, logged = _engine(
        tmp_path, [_implement_reply(IGNORING), _repair_reply(STILL_IGNORING)], rigor_profile="research",
    )
    paused: dict[str, Any] = {}

    def fake_pause(self, **kwargs):  # noqa: ANN001
        paused.update(kwargs)
        raise RuntimeError("paused")

    with mock.patch.object(Engine, "_pause_for_human", fake_pause):
        with pytest.raises(RuntimeError, match="paused"):
            await eng._node_implement(_implement_state())  # type: ignore[arg-type]
    assert paused["kind"] == "replicate_seed_unrepairable" and paused["interaction"] == "supply"
    # The script is not silently kept and run on -- the quest stopped before execute ever saw it.
    assert _code_path(eng).read_text(encoding="utf-8") == IGNORING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply", [_repair_reply(TRUNCATED), "{}", "no script, sorry", _repair_reply("")],
    ids=["truncated", "no-code", "prose", "empty-code"],
)
async def test_a_repair_that_is_not_a_script_is_never_written_over_the_original(
    tmp_path: Path, reply: str,
) -> None:
    eng, client, logged = _engine(tmp_path, [_implement_reply(IGNORING), reply])

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert client.nodes == ["implement", "implement_seed"]
    assert patch["code"] == IGNORING
    assert _code_path(eng).read_text(encoding="utf-8") == IGNORING
    assert any("keeping the script as written" in w for w in _warnings(logged))


@pytest.mark.asyncio
async def test_a_failing_repair_call_does_not_stop_the_quest(tmp_path: Path) -> None:
    eng, client, logged = _engine(
        tmp_path, [_implement_reply(IGNORING), RuntimeError("provider timed out")],
    )

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert patch["code"] == IGNORING
    assert _code_path(eng).read_text(encoding="utf-8") == IGNORING
    warned = _warnings(logged)
    assert len(warned) == 1 and "the repair call failed" in warned[0], warned


def test_the_repair_node_has_a_timeout_that_fits_a_whole_script() -> None:
    """A full-script rewrite on a slow provider outruns the 300 s CLI default,
    and a call killed at that wall is killed again on each retry."""
    cfg = ProviderConfig(name="codex_cli")
    # Every node whose reply is a whole script (or the whole plan) gets execute_reflect's budget: a live kimi-k3 campaign
    # timed out four times on the ones this table had left at the base (implement_oracle, implement_protocol, plan,
    # plan_revise).
    for node in ("implement_seed", "implement_oracle", "implement_protocol", "plan", "plan_revise"):
        assert cfg.node_cli_timeout_s[node] >= cfg.node_cli_timeout_s["execute_reflect"], node
        assert cfg.node_http_timeout_s[node] >= cfg.node_http_timeout_s["execute_reflect"], node


# --- the other shape: it reads the seed, and the seed reaches nothing ---------------
#
# A graded quest shipped this: FI_REPLICATE_SEED read and handed to numpy's
# legacy global, and every trajectory drawn from a Generator built with no seed
# ninety lines earlier. The name is in the file, so the repair above never
# fired; the replicates differ, so ``execute`` published an aggregate; and not
# one of the runs could be reproduced.

ENTROPY = (
    "import json\n"
    "import os\n"
    "import random\n"
    "# " + "x" * 9000 + "\n"
    'random.seed(int(os.environ.get("FI_REPLICATE_SEED", 0)))\n'
    "rng = random.Random()\n"
    'print("RESULT_JSON: " + json.dumps({"draw": rng.random()}))\n'
    + TAIL
)
SEEDED = ENTROPY.replace(
    "rng = random.Random()\n",
    'rng = random.Random(int(os.environ.get("FI_REPLICATE_SEED", 0)))\n',
)
# Rewritten, and the generator is still built from entropy.
STILL_ENTROPY = ENTROPY.replace("random.Random()", "random.Random(None)")


@pytest.mark.asyncio
async def test_a_script_whose_generator_the_seed_cannot_reach_is_repaired_once(
    tmp_path: Path,
) -> None:
    eng, client, logged = _engine(tmp_path, [_implement_reply(ENTROPY), _repair_reply(SEEDED)])

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert client.nodes == ["implement", "implement_seed"]
    assert patch["code"] == SEEDED
    assert _code_path(eng).read_text(encoding="utf-8") == SEEDED
    assert not _warnings(logged), _warnings(logged)

    # The prompt says which line, and that seeding the global is not enough.
    prompt = client.calls[1][1]
    assert TAIL in prompt, "the script was cut short, so a returned script would be too"
    assert "line 6: random.Random()" in prompt
    assert "np.random.seed) does NOT seed a Generator" in prompt
    assert "$previous_code" not in prompt and "$stdout_tail" not in prompt

    # And the point of the repair: the same seed now gives the same draw twice.
    _run_with_this_python(eng)
    executed = await eng._node_execute({"deps": []})  # type: ignore[arg-type]
    draws = [r["draw"] for r in executed["result_json_replicates"]]
    assert len(set(draws)) == len(draws), draws
    again = await eng._node_execute({"deps": []})  # type: ignore[arg-type]
    assert [r["draw"] for r in again["result_json_replicates"]] == draws


@pytest.mark.asyncio
async def test_a_repair_that_still_draws_from_entropy_is_dropped_and_says_so(
    tmp_path: Path,
) -> None:
    eng, client, logged = _engine(tmp_path, [_implement_reply(ENTROPY), _repair_reply(STILL_ENTROPY)])

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert client.nodes == ["implement", "implement_seed"]
    assert patch["code"] == ENTROPY
    assert _code_path(eng).read_text(encoding="utf-8") == ENTROPY
    warned = _warnings(logged)
    assert len(warned) == 1, warned
    assert "still builds a generator without a seed" in warned[0]
    # The consequence is not the one the other shape has: these replicates DO
    # differ, so the aggregate stands and only reproducibility is lost.
    assert "will not reproduce these numbers" in warned[0]
    assert "single measurement" not in warned[0]


@pytest.mark.asyncio
async def test_execute_says_in_the_log_that_the_runs_cannot_be_reproduced(
    tmp_path: Path,
) -> None:
    """The script reaches ``execute`` anyway -- the repair is offered once, and
    may fail. ``run.log`` has to be readable on its own."""
    eng, _, logged = _engine(tmp_path, [_implement_reply(ENTROPY), _repair_reply(STILL_ENTROPY)])
    await eng._node_implement(_implement_state())  # type: ignore[arg-type]
    logged.clear()

    _run_with_this_python(eng)
    executed = await eng._node_execute({"deps": []})  # type: ignore[arg-type]

    assert len(executed["result_json_replicates"]) == 3, "independent samples, so they aggregate"
    assert executed["result_json_replicate_seed_ignored"] is False
    warned = [w for w in _warnings(logged) if "[execute]" in w]
    assert len(warned) == 1, warned
    assert "draws from OS entropy (line 6: random.Random())" in warned[0]
    assert "not reproducible" in warned[0]


@pytest.mark.asyncio
async def test_a_guarded_fallback_is_not_sent_back(tmp_path: Path) -> None:
    """``if rng is None: rng = random.Random()`` is how a script says its
    caller hands it a seeded generator. Rewriting a reproducible script costs a
    call and risks the script."""
    guarded = (
        "import json\n"
        "import os\n"
        "import random\n"
        "def draw(rng=None):\n"
        "    if rng is None:\n"
        "        rng = random.Random()\n"
        "    return rng.random()\n"
        'print("RESULT_JSON: " + json.dumps('
        '{"draw": draw(random.Random(int(os.environ.get("FI_REPLICATE_SEED", 0))))}))'
    )
    eng, client, logged = _engine(tmp_path, [_implement_reply(guarded)])

    patch = await eng._node_implement(_implement_state())  # type: ignore[arg-type]

    assert client.nodes == ["implement"]
    assert patch["code"] == guarded
    assert not _warnings(logged)

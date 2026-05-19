"""Multi-model ensemble primitive — N parallel LLM calls + a merger.

Used by:
* in-quest nodes (``ideate``, ``analyze``, ``cross_check``) when the
  YAML carries ``provider.node_ensemble[node]``, and
* standalone tools (``--proposal``, ``--critique``) when the user passes
  ``--proposal-ensemble`` / ``--critique-ensemble``.

The design rests on three primitives:

1. ``fanout_chat`` — fire N calls in parallel via ``asyncio.gather``,
   each with a per-call ``model`` override. Lenient error semantics:
   per-model failures are captured as ``FanoutResponse.error`` rather
   than raised; the merger sees only the survivors.
2. Three mergers (``merge_tournament``, ``merge_synthesize``,
   ``merge_vote``) that consume the survivor list and produce one
   final output.
3. ``cost.jsonl`` integration — every ensembled call is logged with
   ``ensemble=True`` and a ``role`` field (``fanout`` / ``moderator``)
   so the cost tool can break down ensemble spend later.

Failure semantics — by design:
* All N models fail → raise ``EnsembleError`` (no survivors, nothing to merge).
* Some succeed → proceed with survivors; log a warning naming the failed models.
* The moderator call itself fails → the merger raises; engine catches and
  decides whether to fall back to the first survivor or hard-fail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

_log = logging.getLogger("frontier_insight.ensemble")


# Callable that performs one chat call. The engine plugs in its own
# ``self._client.chat`` here (which already routes through the bridge
# transport and honors per-call ``model`` overrides). Keeping the
# primitive transport-agnostic means tests can pass a pure-Python
# fake without spinning up a real bridge.
ChatFn = Callable[..., Awaitable[str]]


class EnsembleError(RuntimeError):
    """All N fan-out calls failed — nothing to merge."""


@dataclass(frozen=True)
class FanoutResponse:
    model: str
    text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class EnsembleResult:
    """Output of ``fanout_chat + merger``.

    ``merged`` is the final answer the caller plugs back into the
    pipeline. ``raw`` carries every per-model response (success +
    failure) so the cost tool + run.log can show what ran.
    ``disagreement_score`` is 0.0 when every model agreed (e.g. all
    voted the same way), 1.0 when every model disagreed. Only the
    ``vote`` merger fills this in; the other two leave it at 0.0
    since their notion of "agreement" is fuzzy.
    """
    merged: str | dict[str, Any]
    raw: list[FanoutResponse]
    disagreement_score: float = 0.0
    notes: str = ""
    # Set by tournament/synthesize when a moderator was called. Lets
    # cost_jsonl_entries attribute the moderator row to a concrete model
    # so per-model cost accounting stays consistent across all rows.
    moderator_model: str | None = None


async def fanout_chat(
    messages: list[dict[str, str]],
    models: list[str],
    *,
    chat_fn: ChatFn,
    node: str,
    temperature: float = 0.2,
) -> list[FanoutResponse]:
    """Fire ``len(models)`` parallel chat calls. Each call uses the
    same messages + temperature but a different ``model`` override.

    Returns one ``FanoutResponse`` per requested model, in the same
    order. Per-model failures are captured (not raised); callers check
    ``.ok`` / ``.error`` to count survivors.

    The ``node`` argument is logged into each call's cost.jsonl entry
    with an ``ensemble.role="fanout"`` tag so per-quest spend
    breakdown is straightforward to compute.
    """
    if not models:
        raise EnsembleError("fanout_chat called with empty models list")

    async def _one(model: str) -> FanoutResponse:
        try:
            text = await chat_fn(
                messages, temperature=temperature, model=model,
                node=f"{node}.ensemble[{model}]",
            )
            return FanoutResponse(model=model, text=text)
        except Exception as e:  # noqa: BLE001  — every transport raises differently
            _log.warning(
                "[ensemble] %s: model %r failed: %s", node, model, e,
            )
            return FanoutResponse(model=model, error=f"{type(e).__name__}: {e}")

    results = await asyncio.gather(*(_one(m) for m in models))
    n_ok = sum(1 for r in results if r.ok)
    if n_ok == 0:
        msgs = "; ".join(f"{r.model}: {r.error}" for r in results)
        raise EnsembleError(f"all {len(models)} ensemble calls failed at {node}: {msgs}")
    if n_ok < len(models):
        _log.warning(
            "[ensemble] %s: %d/%d models succeeded; proceeding with survivors",
            node, n_ok, len(models),
        )
    return results


def _survivors(raw: list[FanoutResponse]) -> list[FanoutResponse]:
    return [r for r in raw if r.ok]


# ---------------------------------------------------------------------------
# Merger 1: tournament — pick the best response.
# ---------------------------------------------------------------------------


async def merge_tournament(
    raw: list[FanoutResponse],
    *,
    moderator_model: str | None,
    chat_fn: ChatFn,
    node: str,
    prompt_summary: str = "",
) -> EnsembleResult:
    """Tournament: a moderator LLM picks the single best response.

    Each candidate is presented to the moderator with a stable
    `[A]` / `[B]` / `[C]` label; the moderator returns the winning
    label + a one-line rationale. We parse the label and return the
    corresponding candidate verbatim (no rewriting — preserves the
    chosen model's voice).

    ``prompt_summary`` is a one-line description of what was asked;
    the moderator uses it to know what "best" means. Empty string
    is OK — the moderator falls back to the candidate content alone.
    """
    survivors = _survivors(raw)
    if not survivors:
        # Defensive: fanout_chat already raises EnsembleError when every
        # call fails, but a caller could construct a result and pass it
        # here directly. Fail loudly instead of IndexError'ing on [0].
        raise EnsembleError("merge_tournament: no surviving responses")
    if len(survivors) == 1:
        return EnsembleResult(
            merged=survivors[0].text, raw=raw,
            notes=f"single-survivor (other {len(raw)-1} failed)" if len(raw) > 1 else "",
        )
    # Single-survivor or no moderator → return first candidate.
    if not moderator_model:
        return EnsembleResult(merged=survivors[0].text, raw=raw,
                              notes="no moderator configured — taking first survivor")

    if len(survivors) > 26:
        # The label scheme is single A-Z; 26 is the hard cap. Real
        # ensembles fan out 2-5 ways — anything beyond 26 is a config
        # bug, not a legitimate ask.
        raise EnsembleError(
            f"merge_tournament: too many candidates ({len(survivors)}); "
            f"max 26 to fit single-letter labels A-Z",
        )
    labels = [chr(ord("A") + i) for i in range(len(survivors))]
    candidate_blocks = "\n\n".join(
        f"--- Candidate [{labels[i]}] (model={s.model}) ---\n{s.text}"
        for i, s in enumerate(survivors)
    )
    instruction = (
        f"You are a moderator. {len(survivors)} expert models independently produced "
        f"answers to the same task. Read each and pick the single BEST candidate.\n\n"
        f"Task summary: {prompt_summary or '(see candidate content)'}\n\n"
        f"{candidate_blocks}\n\n"
        f"Reply with EXACTLY this format on the first line:\n"
        f"  WINNER: [<letter>]\n"
        f"Then on the next line a one-sentence rationale. Nothing else.\n"
        f"Valid letters: {', '.join('['+x+']' for x in labels)}.\n"
    )
    try:
        mod_text = await chat_fn(
            [{"role": "user", "content": instruction}],
            temperature=0.0, model=moderator_model,
            node=f"{node}.ensemble.moderator",
        )
    except Exception as e:  # noqa: BLE001  — every transport raises differently
        # Moderator call itself failed (timeout / rate-limit / transport).
        # The fan-out work isn't wasted: fall back to the first survivor
        # and tell the caller. Same shape as the unparseable-output
        # fallback below so engine integration is uniform.
        _log.warning(
            "[ensemble] %s: moderator call failed (%s); using first survivor",
            node, e,
        )
        return EnsembleResult(
            merged=survivors[0].text, raw=raw,
            notes=f"moderator call failed ({type(e).__name__}: {e}); took first survivor",
        )
    pick = _parse_winner_label(mod_text, labels)
    if pick is None:
        # Moderator didn't follow the format — fall back to the first
        # survivor + flag in notes. Better than crashing the node.
        return EnsembleResult(
            merged=survivors[0].text, raw=raw,
            moderator_model=moderator_model,
            notes=f"moderator output unparseable, took first survivor; raw: {mod_text[:200]}",
        )
    winner_idx = labels.index(pick)
    return EnsembleResult(
        merged=survivors[winner_idx].text, raw=raw,
        moderator_model=moderator_model,
        notes=f"moderator picked [{pick}] (model={survivors[winner_idx].model})",
    )


def _parse_winner_label(mod_text: str, valid: list[str]) -> str | None:
    """Find ``WINNER: [X]`` or ``WINNER: X`` on any line; case-insensitive
    on the prefix, case-sensitive on the letter."""
    pattern = re.compile(r"WINNER\s*:\s*\[?([A-Z])\]?", re.IGNORECASE)
    for line in mod_text.splitlines():
        m = pattern.search(line)
        if m:
            label = m.group(1).upper()
            if label in valid:
                return label
    return None


# ---------------------------------------------------------------------------
# Merger 2: synthesize — moderator merges N reports + flags disagreement.
# ---------------------------------------------------------------------------


async def merge_synthesize(
    raw: list[FanoutResponse],
    *,
    moderator_model: str | None,
    chat_fn: ChatFn,
    node: str,
    prompt_summary: str = "",
) -> EnsembleResult:
    """Synthesize: a moderator LLM merges N analyses into one coherent
    answer, EXPLICITLY flagging any disagreement between candidates.

    Used for ``analyze`` and ``--critique`` ensembles — where every
    candidate's perspective is potentially load-bearing and silently
    picking one would lose the disagreement signal that's the whole
    point of running N models.
    """
    survivors = _survivors(raw)
    if not survivors:
        # Defensive: matches merge_tournament/merge_vote, so callers see
        # one consistent failure contract regardless of merger choice.
        raise EnsembleError("merge_synthesize: no surviving responses")
    if len(survivors) == 1:
        return EnsembleResult(
            merged=survivors[0].text, raw=raw,
            notes=f"single-survivor (other {len(raw)-1} failed)" if len(raw) > 1 else "",
        )
    if not moderator_model:
        # No moderator → concatenate with separators. Crude but truthful.
        joined = "\n\n---\n\n".join(
            f"### {s.model}\n\n{s.text}" for s in survivors
        )
        return EnsembleResult(merged=joined, raw=raw,
                              notes="no moderator configured — raw concatenation")

    blocks = "\n\n".join(
        f"--- Analysis from model={s.model} ---\n{s.text}" for s in survivors
    )
    instruction = (
        f"You are a moderator. {len(survivors)} expert models independently produced "
        f"analyses of the same data / question. Merge them into ONE coherent answer "
        f"that preserves all factual content.\n\n"
        f"CRITICAL: where the models DISAGREE on a finding or judgement, do NOT "
        f"silently pick one side. Write a sentence like:\n"
        f"  '⚠ Disagreement: model A says <X>; model B says <not-X>. <Brief note on which is more defensible and why.>'\n"
        f"Disagreement is signal, not noise — it tells the user where to look closer.\n\n"
        f"Task summary: {prompt_summary or '(see analysis blocks)'}\n\n"
        f"{blocks}\n\n"
        f"Output: one merged analysis in markdown, with disagreement-flag lines "
        f"inline where applicable. No JSON, no preamble, no surrounding fence."
    )
    try:
        merged = await chat_fn(
            [{"role": "user", "content": instruction}],
            temperature=0.2, model=moderator_model,
            node=f"{node}.ensemble.moderator",
        )
    except Exception as e:  # noqa: BLE001  — every transport raises differently
        # Moderator failed → fall back to raw concatenation (preserves
        # every survivor's analysis verbatim, no information loss) and
        # flag in notes. Engine integration handles this uniformly via
        # the same fallback path as the no-moderator case.
        _log.warning(
            "[ensemble] %s: moderator call failed (%s); concatenating survivors",
            node, e,
        )
        joined = "\n\n---\n\n".join(
            f"### {s.model}\n\n{s.text}" for s in survivors
        )
        return EnsembleResult(
            merged=joined, raw=raw,
            notes=f"moderator call failed ({type(e).__name__}: {e}); raw concatenation",
        )
    # Count disagreement markers — gives the engine a numeric handle to
    # log + display in the cost report. Single pattern so a line like
    # "⚠ Disagreement:" counts once, not twice.
    n_disagree = len(_DISAGREEMENT_RE.findall(merged))
    # disagreement_score: 0..1, clamped at 1.0 once we see 5+ flags.
    score = min(1.0, n_disagree / 5.0)
    return EnsembleResult(
        merged=merged, raw=raw, disagreement_score=score,
        moderator_model=moderator_model,
        notes=f"moderator-synthesized; {n_disagree} disagreement flag(s)",
    )


# One disagreement flag per match: either the ⚠-prefixed form OR a
# plain "Disagreement:" line. The leading sentinel (⚠ or line-start)
# prevents double-counting when the moderator writes "⚠ Disagreement:".
_DISAGREEMENT_RE = re.compile(
    r"(?:⚠\s*Disagreement|(?:^|\n)\s*Disagreement)\b\s*:?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Merger 3: vote — pure tally, no LLM call. For structured outputs.
# ---------------------------------------------------------------------------


def merge_vote(
    raw: list[FanoutResponse], *, key: str,
) -> EnsembleResult:
    """Majority vote on a structured field (e.g. ``verdict``,
    ``classification``) parsed from each model's JSON response.

    Used by ``cross_check`` — per-finding, each model classifies the
    finding as supporting/conflicting/neutral; majority wins, ties
    surfaced in ``notes``.

    Returns the winning value in ``merged[key]``. ``merged`` is a
    dict so the engine can attach per-vote metadata without changing
    the contract.

    No LLM call — pure tally. Fastest of the three mergers.
    """
    survivors = _survivors(raw)
    if not survivors:
        raise EnsembleError("merge_vote: no surviving responses")

    parsed: list[Any] = []
    for s in survivors:
        try:
            obj = json.loads(s.text.strip()) if s.text.strip().startswith("{") else None
        except json.JSONDecodeError:
            obj = None
        if obj is None or key not in obj:
            # Tolerate non-JSON / missing-key by sniffing the raw text
            # for common verdict tokens. Engine callers can post-filter.
            tokens = re.findall(r"\b(supporting|conflicting|neutral|accept|revise|reject|yes|no)\b",
                                s.text, re.IGNORECASE)
            obj = {key: tokens[0].lower() if tokens else "unknown"}
        parsed.append(obj[key])

    tally = Counter(parsed)
    if not tally:
        raise EnsembleError("merge_vote: no parseable responses")
    most_common = tally.most_common()
    winner, win_count = most_common[0]
    # disagreement: ratio of dissenting votes to total
    n = len(parsed)
    disagree = 1.0 - (win_count / n) if n > 0 else 0.0
    tie = len(most_common) > 1 and most_common[1][1] == win_count
    notes = (f"vote tally: {dict(tally)}; "
             + ("⚠ TIE — picked first occurrence" if tie else f"winner = {winner!r} ({win_count}/{n})"))
    return EnsembleResult(
        merged={key: winner, "tally": dict(tally), "tie": tie},
        raw=raw, disagreement_score=disagree, notes=notes,
    )


# ---------------------------------------------------------------------------
# Cost-tracking helper.
# ---------------------------------------------------------------------------


def cost_jsonl_entries(
    result: EnsembleResult, *, base_node: str, merge_strategy: str,
) -> list[dict[str, Any]]:
    """Return rows ready for ``<quest_root>/.fi/cost.jsonl``. Engine
    appends each one with its own timestamp; this helper just builds
    the shape.

    Schema (per row):
      {"node": "<base>.ensemble[<model>]", "model": "<model>",
       "ensemble": True, "role": "fanout" | "moderator",
       "merge": "tournament|synthesize|vote",
       "ok": True/False, "error": "..." | None}

    The engine's existing _log_chat_cost writes per-call usage/USD;
    these rows are the structural breadcrumb for the cost tool to
    aggregate by ``node.startswith("<base>.ensemble")``.
    """
    rows: list[dict[str, Any]] = []
    for r in result.raw:
        rows.append({
            "node": f"{base_node}.ensemble[{r.model}]",
            "model": r.model,
            "ensemble": True,
            "role": "fanout",
            "merge": merge_strategy,
            "ok": r.ok,
            "error": r.error,
        })
    # Moderator row only emitted when a moderator actually ran.
    # Tournament + synthesize have early-return paths (single survivor,
    # moderator_model=None, moderator-call failure) where no LLM call
    # was made — we gate on the moderator_model field, which the mergers
    # only set after a successful moderator call. Vote has no moderator
    # at all (pure tally). The moderator row carries the same
    # {model, ok, error} keys as the fan-out rows so the cost tool can
    # attribute spend per model without special-casing role.
    if merge_strategy in ("tournament", "synthesize") and result.moderator_model:
        rows.append({
            "node": f"{base_node}.ensemble.moderator",
            "model": result.moderator_model,
            "ensemble": True,
            "role": "moderator",
            "merge": merge_strategy,
            "ok": True,
            "error": None,
            "disagreement_score": result.disagreement_score,
        })
    return rows


async def run_singleshot_with_ensemble(
    *,
    prompt: str,
    chat_fn: ChatFn,
    ensemble: "Any",  # NodeEnsembleConfig | None
    default_merge: str,
    node: str,
    temperature: float = 0.2,
    prompt_summary: str = "",
) -> str:
    """Shared helper for the standalone single-shot tools (digest /
    portfolio / summarize / critique). Either fires one chat call
    (when ``ensemble`` is None) or fans out across the configured
    model trio and merges per ``ensemble.merge`` (or ``default_merge``
    when the config didn't pin one).

    Falls back to the single-call path with a logged warning when
    every fan-out call fails (``EnsembleError``) — same lenient
    semantics the engine uses for per-node ensemble. The moderator
    failing inside the merger still raises so the caller can decide
    whether to surface it as a hard error.

    ``default_merge`` differs by tool: ``synthesize`` for critique
    (preserve disagreement), ``tournament`` for proposal / summarize /
    portfolio (pick the strongest single answer); ``digest`` defaults
    to ``synthesize`` because the WeekDiff narrative benefits from
    multiple model perspectives on the same set of quests.
    """
    if ensemble is None:
        return await chat_fn(
            [{"role": "user", "content": prompt}],
            temperature=temperature, node=node,
        )
    merge_kind = ensemble.merge or default_merge
    # ``vote`` is only meaningful for structured-output nodes
    # (cross_check) where every model emits the same JSON shape and a
    # majority can be tallied on a named field. Single-shot tools emit
    # free-form markdown — no field to vote on. Reject UP-FRONT,
    # outside the all-fanout-failed fallback path below, so the user
    # sees a clear configuration error instead of a silent fallback to
    # single-call.
    if merge_kind == "vote":
        raise EnsembleError(
            "merge='vote' is only supported for structured-output "
            "nodes (cross_check). For single-shot tools "
            "(digest/portfolio/summarize/critique/proposal) pick "
            "'tournament' (one canonical answer) or 'synthesize' "
            "(merge with disagreement flags) instead."
        )
    summary = prompt_summary or node
    try:
        raw = await fanout_chat(
            [{"role": "user", "content": prompt}],
            ensemble.models, chat_fn=chat_fn, node=node,
            temperature=temperature,
        )
        moderator = ensemble.moderator or ensemble.models[0]
        if merge_kind == "tournament":
            result = await merge_tournament(
                raw, moderator_model=moderator,
                chat_fn=chat_fn, node=node, prompt_summary=summary,
            )
        else:
            result = await merge_synthesize(
                raw, moderator_model=moderator,
                chat_fn=chat_fn, node=node, prompt_summary=summary,
            )
        return result.merged if isinstance(result.merged, str) else str(result.merged)
    except EnsembleError as e:
        _log.warning(
            "ensemble all-failed at %s (%s); falling back to single-call",
            node, e,
        )
        return await chat_fn(
            [{"role": "user", "content": prompt}],
            temperature=temperature, node=node,
        )

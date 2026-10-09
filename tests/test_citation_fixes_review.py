"""Four ways the citation check could have damaged a paper or run out of calls, each with its scenario
(core/citation_support.py): a reworded sentence that changes the study's own number, a fallback that cuts the study's own
result, an edit that would change a heading or a figure and took every other fix down with it, and a paper with more
citations than one reply holds."""
from __future__ import annotations

import asyncio
import json
import math
import re

from core import citation_support as cs

FULL = "filler words " * 400


def _word(i: int) -> str:
    return "".join(chr(97 + int(d)) for d in str(i)) + "x"


def _src(label: str, text: str) -> cs.Source:
    return cs.Source(label=label, title=f"Source {label}", text=FULL + text, full=True)


def _run(paper: str, sources: dict[str, cs.Source], *, verdict, fixes=None, own=(), claim_unsupported=lambda s: False):
    """``verdict(sentence, label) -> bool | str`` (a quote means supported); ``fixes(id, sentence) -> dict | None``."""
    asked: list[int] = []
    saved: list[dict] = []

    async def judge(prompt: str) -> str:
        rows = re.findall(r"^(\S+) \| cites \[(\w+)\] \| (.*)$", prompt, re.M)
        asked.append(len(rows))
        out = []
        for pid, label, sentence in rows:
            v = verdict(sentence, label)
            out.append({"id": pid, "supported": bool(v), "quote": v if isinstance(v, str) else "", "why": "scripted"})
        return json.dumps({"verdicts": out})

    async def repair(prompt: str) -> str:
        out = []
        for sid, sentence in re.findall(r"^Item (\S+) \(\w+\): (.*)$", prompt, re.M):
            fix = (fixes or (lambda i, s: None))(sid, sentence)
            if fix:
                out.append({"id": sid, **fix})
        return json.dumps({"fixes": out})

    record = cs.new_record(paper, None)
    result = asyncio.run(cs.check_paper(
        paper, sources, own_numbers=cs.OwnNumbers(own), quote_ok=lambda q, s: q in s, judge=judge, repair=repair,
        record=record, save=lambda r: saved.append(json.loads(json.dumps(r))),
        judge_prompt=lambda a, srcs: "\n".join(f"{p} | cites [{lb}] | {' '.join(s.text.split())}" for p, s, lb in a),
        repair_prompt=lambda ts: "\n".join(f"Item {t['id']} ({t['kind']}): {' '.join(t['sentence'].split())}" for t in ts),
        claim_unsupported=claim_unsupported))
    return result, asked, record


def test_a_reworded_sentence_may_not_change_the_studys_own_number() -> None:
    paper = "# T\n\nWe achieved 95%, beating the baseline [1].\n"
    sources = {"1": _src("1", " the baseline reached 92% in the earlier work, so beating it is hard. ")}
    quote = "the baseline reached 92% in the earlier work"

    def fixes(sid, sentence):  # the fixer makes our result the source's number, and quotes the source for it
        return {"action": "reword", "source": "1", "quote": quote,
                "text": "We achieved 92%, beating the baseline [1]."}

    def verdict(sentence, label):
        return quote if ">" in sentence or "92%" in sentence else False

    result, _asked, _rec = _run(paper, sources, verdict=lambda s, lb: False, fixes=fixes, own=["95"])
    assert "92" not in result.paper and "95%" in result.paper, "the study's own result is kept as it is"
    assert result.counts["reworded"] == 0 and result.counts["citation_dropped"] == 1
    # a reword that keeps the number is allowed to stand (the reader then confirms it)
    ok = lambda sid, sentence: {"action": "reword", "source": "1", "quote": quote,
                                "text": "We achieved 95%, beating a baseline that reached 92% earlier [1]."}
    # a reword that keeps the number and adds a grounded source number may stand (the reader then confirms it)
    assert cs._validate_fix(cs.find_sites(paper)[0], {"labels": ["1"], "kept": []}, {**ok("", ""), "source": "1"},
                            sources, cs.OwnNumbers(["95"]), lambda q, s: q in s) is not None
    # one that drops it, or adds a second result of the study's, does not
    for text in ("We beat the baseline [1].", "We achieved 95% and 97%, beating the baseline [1]."):
        assert cs._validate_fix(cs.find_sites(paper)[0], {"labels": ["1"], "kept": []},
                                {"action": "reword", "source": "1", "quote": quote, "text": text},
                                sources, cs.OwnNumbers(["95", "97"]), lambda q, s: q in s) is None


def test_the_study_s_own_result_is_never_cut_out_of_a_sentence_that_also_states_a_source_number() -> None:
    sources = {"1": _src("1", " nothing about the topic is said here at all. ")}
    own = ["0.95"]
    # a clause that holds only the source's number can be cut cleanly
    paper = "# T\n\nThe coverage was 0.95, against 62% reported in earlier work [1].\n"
    result, _a, _r = _run(paper, sources, verdict=lambda s, lb: False, own=own)
    assert "The coverage was 0.95." in result.paper and "62" not in result.paper and "[1]" not in result.paper
    assert result.counts["clause_dropped"] == 1 and result.counts["sentence_dropped"] == 0
    # one that cannot be cut cleanly keeps the result, loses the citation, and says what is left in the paper
    paper = "# T\n\nCoverage 0.95 exceeded the 62% of earlier work [1].\n"
    result, _a, _r = _run(paper, sources, verdict=lambda s, lb: False, own=own)
    assert "0.95" in result.paper and "[1]" not in result.paper
    assert result.counts["sentence_dropped"] == 0 and result.counts["left_in_place"] == 1
    assert "62" in result.rows[0].note
    md = cs.render_md(cs.ledger(result, reader="r", same_model=False))
    assert "note: left in the paper" in md and "62" in md
    # a sentence with no result of the study and a source number still goes whole
    paper = "# T\n\nEarlier work reported 62% of cases [1].\n"
    result, _a, _r = _run(paper, sources, verdict=lambda s, lb: False, own=own)
    assert result.counts["sentence_dropped"] == 1 and "62" not in result.paper


def test_an_edit_that_would_change_a_heading_or_a_figure_does_not_take_the_other_fixes_down() -> None:
    paper = ("# T\n\nEarlier work reported 62% of cases [1]. A method detail with no number is attributed to a source [2].\n")
    sources = {"1": _src("1", " nothing here. "), "2": _src("2", " nothing here either. ")}

    def fixes(sid, sentence):
        if "62%" in sentence:  # this fix would put a figure into the paper
            return {"action": "reword", "source": "", "quote": "", "text": "Earlier work reported cases ![x](figures/x.png)."}
        return {"action": "drop_citation"}

    result, _a, _r = _run(paper, sources, verdict=lambda s, lb: False, fixes=fixes)
    assert "![x]" not in result.paper
    assert "A method detail with no number is attributed to a source." in result.paper, "the fix that was fine stands"
    assert "62%" not in result.paper, "the sentence whose fix was refused is settled by the rule, not left with a bare citation removed"
    assert result.counts["sentence_dropped"] == 1 and result.counts["citation_dropped"] == 1


def test_a_paper_with_many_citations_is_read_whole_within_the_calls_allowed() -> None:
    n = 130
    sentences = [f"Claim {_word(i)} about the subject is stated here [1]." for i in range(n)]
    paper = "# T\n\n" + "\n".join(f"- {s}" for s in sentences) + "\n"
    quote = "the subject is described in full here"
    sources = {"1": _src("1", f" {quote} ")}
    result, asked, record = _run(paper, sources, verdict=lambda s, lb: quote)
    assert sum(asked) >= n and max(asked) <= cs.BATCH_PAIRS
    assert len(asked) == math.ceil(n / cs.BATCH_PAIRS) and record["judge_calls"] == len(asked)
    assert result.counts["supported"] == n and result.counts["unchecked"] == 0 and result.clean
    assert result.paper == paper


def test_pairs_a_reply_leaves_out_are_asked_again_without_holding_up_the_others() -> None:
    n = cs.BATCH_PAIRS + 5
    paper = "# T\n\n" + "\n".join(f"- Claim {_word(i)} about the subject is stated here [1]." for i in range(n)) + "\n"
    quote = "the subject is described in full here"
    sources = {"1": _src("1", f" {quote} ")}
    first = {"done": False}

    def verdict(sentence, label):
        return quote

    # the reader answers only half of the first batch
    asked_ids: list[list[str]] = []

    async def judge(prompt: str) -> str:
        rows = re.findall(r"^(\S+) \| cites", prompt, re.M)
        asked_ids.append(rows)
        rows = rows[: len(rows) // 2] if not first["done"] else rows
        first["done"] = True
        return json.dumps({"verdicts": [{"id": r, "supported": True, "quote": quote} for r in rows]})

    async def repair(prompt: str) -> str:
        return "{}"

    record = cs.new_record(paper, None)
    result = asyncio.run(cs.check_paper(
        paper, sources, own_numbers=cs.OwnNumbers(), quote_ok=lambda q, s: q in s, judge=judge, repair=repair,
        record=record, save=lambda r: None,
        judge_prompt=lambda a, srcs: "\n".join(f"{p} | cites [{lb}] | x" for p, s, lb in a), repair_prompt=lambda ts: ""))
    assert result.counts["supported"] == n and result.counts["unchecked"] == 0
    assert len(asked_ids) <= 3


def test_a_fixer_may_not_drop_a_sentence_that_carries_the_study_s_own_result() -> None:
    paper = "# T\n\nWe achieved 95%, beating the baseline [1].\n"
    sources = {"1": _src("1", " an unrelated text. ")}
    site = cs.find_sites(paper)[0]
    drop = {"action": "drop_sentence", "source": "", "quote": "", "text": ""}
    assert cs._validate_fix(site, {"labels": ["1"], "kept": []}, drop, sources, cs.OwnNumbers(["95"]),
                            lambda q, s: q in s) is None  # refused: the rule settles it and keeps the result
    assert cs._validate_fix(site, {"labels": ["1"], "kept": []}, drop, sources, cs.OwnNumbers(),
                            lambda q, s: q in s) == {"action": "drop_sentence", "text": ""}  # no own result: may go
    result, _asked, _rec = _run(paper, sources, verdict=lambda s, lb: False,
                                fixes=lambda sid, sentence: drop, own=["95"])
    assert "95%" in result.paper

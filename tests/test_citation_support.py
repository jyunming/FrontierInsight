"""The parts of the citation check that need no model (core/citation_support.py): which sentences cite which sources, the
rules that only ever say "unsupported", how a sentence's brackets are rewritten, and the bounds on the calls."""
from __future__ import annotations

import asyncio
import json

from core import citation_support as cs

PAPER = """# Title

## Introduction
First, a sentence about a method [1, 3]. A second one cites a range [2-4], and a web page [W2]. Code like `x[1]` and
math like $a_{[2]}$ and a link [text](http://x.org) and an interval [0, 1] and a year [1995] are not citations.

- A list item that cites [5].

| a | b |
|---|---|
| cell [6] | x |

## References

1. Somebody (2000). A paper.
"""


def _src(label: str, title: str, text: str, *, full: bool = False) -> cs.Source:
    return cs.Source(label=label, title=title, text=text, full=full)


def test_every_sentence_that_cites_is_listed_with_every_source_it_names() -> None:
    sites = cs.find_sites(PAPER)
    assert [s.labels for s in sites] == [["1", "3"], ["2", "3", "4", "W2"], ["5"], ["6"]]
    assert [s.kind for s in sites] == ["prose", "prose", "list", "table"]
    assert all(PAPER[s.start:s.end] == s.text for s in sites), "a site is an exact span of the paper"
    # nothing in the reference list is a site
    assert all(s.end <= cs.references_start(PAPER) for s in sites)


def test_the_same_sentence_twice_is_two_sites() -> None:
    sites = cs.find_sites("A sentence [1]. Another one here. A sentence [1].\n")
    assert len(sites) == 2 and sites[0].key != sites[1].key


def test_brackets_are_rewritten_without_touching_the_rest() -> None:
    assert cs.rewrite_brackets("A claim [1, 5, 6].", remove={"5", "6"}) == "A claim [1]."
    assert cs.rewrite_brackets("A claim [3].", remove={"3"}) == "A claim."
    assert cs.rewrite_brackets("A claim [3].", remove={"3"}, add=["7"]) == "A claim [7]."
    assert cs.rewrite_brackets("A claim.", remove=set(), add=["7"]) == "A claim [7]."
    assert cs.rewrite_brackets("A claim [1] and `x[1]`.", remove={"1"}) == "A claim and `x[1]`."
    assert cs.rewrite_brackets("A claim [2, 3, 4, W1].", remove={"3"}) == "A claim [2, 4, W1]."


def test_numbers_a_sentence_states_are_the_ones_a_source_must_show() -> None:
    nums = cs._numbers("It rose to 4,824.8 J/mol (95% CI) in 1995, with 3 sites and $x = 12$ [4].")
    assert nums == ["4824.8", "95"]
    own = cs.OwnNumbers(["4824.8", "0.9503"])
    assert "4824.8" in own and "95" in own and "12.5" not in own


def test_a_retracted_or_textless_source_supports_nothing() -> None:
    site = cs.Site(0, 0, "It is known [1].", ["1"], "prose")
    assert "retracted" in cs.pre_problem(site, "1", cs.Source("1", "T", "text", retracted=True), cs.OwnNumbers())
    assert "no text" in cs.pre_problem(site, "1", cs.Source("1", "T", "", full=False), cs.OwnNumbers())
    assert "not one of the paper's sources" in cs.pre_problem(site, "9", None, cs.OwnNumbers())


def test_a_source_of_a_title_only_is_title_only_and_a_full_text_is_not_judged_by_its_shape() -> None:
    assert _src("1", "A Title", "A Title").kind() == "title only"
    assert _src("1", "A Title", "A Title\n\nA short blurb.").kind() == "title only"
    assert _src("1", "A Title", "x " * 4000, full=True).kind() == "full text"
    abstract = "A Title\n\n" + " ".join(f"This is sentence number {i} of an abstract that says something real." for i in range(5))
    assert _src("1", "A Title", abstract).kind() == "abstract"
    contents = "A Title\n\nPreface. Introduction. Simple Linear Regression. Multiple Regression. Model Adequacy. References. Index."
    assert _src("1", "A Title", contents + " " + contents).kind() == "snippet or contents list"


def test_a_prose_abstract_is_not_refused_for_its_wording() -> None:
    abstract = "A Title\n\n" + " ".join(
        f"The method estimates the rate constant and reports its uncertainty in case {i} of the study." for i in range(5))
    src = _src("1", "A Title", abstract)
    site = cs.Site(0, 0, "Estimating rate constants carefully requires reporting uncertainty intervals alongside [1].", ["1"], "prose")
    assert cs.pre_problem(site, "1", src, cs.OwnNumbers()) == ""


def test_replies_are_read_leniently_and_only_for_the_ids_asked() -> None:
    reply = '```json\n{"verdicts": [{"id": "a", "supported": "yes", "quote": "x y"}, {"id": "b", "supported": "maybe"}, {"id": "zz", "supported": true}]}\n```'
    got = cs.parse_verdicts(reply, {"a", "b"})
    assert set(got) == {"a"} and got["a"]["supported"] is True
    fixes = cs.parse_fixes('{"fixes": [{"id": "s", "action": "Drop_Citation"}, {"id": "t", "action": "delete everything"}]}', {"s", "t"})
    assert set(fixes) == {"s"} and fixes["s"]["action"] == "drop_citation"
    assert cs.parse_verdicts("not json", {"a"}) == {} and cs.parse_fixes("", {"a"}) == {}


def _check(paper: str, sources: dict[str, cs.Source], judge, repair, record: dict, saved: list, **kw):
    return asyncio.run(cs.check_paper(
        paper, sources, own_numbers=kw.get("own", cs.OwnNumbers()), quote_ok=lambda q, s: q in s, judge=judge, repair=repair,
        record=record, save=lambda r: saved.append(json.loads(json.dumps(r))),
        judge_prompt=lambda asked, srcs: "\n".join(f"{p} | cites [{l}] | {s.text}" for p, s, l in asked),
        repair_prompt=lambda targets: json.dumps([t["id"] for t in targets]),
        claim_unsupported=kw.get("claim_unsupported", lambda s: False)))


FULL = "x " * 3000


def test_a_reply_that_leaves_a_pair_out_is_asked_once_more_and_then_the_citation_is_unchecked() -> None:
    paper = "A first claim about it [1]. A second claim about it [2].\n"
    sources = {"1": _src("1", "One", FULL + "the first passage is here", full=True),
               "2": _src("2", "Two", FULL + "the second passage is here", full=True)}
    asked: list[str] = []

    async def judge(prompt: str) -> str:
        asked.append(prompt)
        pid = prompt.splitlines()[0].split(" | ")[0]
        return json.dumps({"verdicts": [{"id": pid, "supported": True, "quote": "the first passage is here"}]}) if "[1]" in prompt.splitlines()[0] else "{}"

    async def repair(prompt: str) -> str:
        raise AssertionError("nothing was found unsupported")

    saved: list[dict] = []
    rec = cs.new_record(paper, None)
    result = _check(paper, sources, judge, repair, rec, saved)
    assert len(asked) == cs.MAX_JUDGE_CALLS, "the pair it left out was asked about once more, and no more"
    assert result.counts["supported"] == 1 and result.counts["unchecked"] == 1 and not result.clean
    assert result.paper == paper, "an unchecked citation is not removed on a guess"


def test_the_calls_are_counted_before_they_are_made_and_capped_for_the_quest() -> None:
    paper = "A claim about it [1].\n"
    sources = {"1": _src("1", "One", FULL + "no", full=True)}
    seen: list[int] = []
    saved: list[dict] = []

    async def judge(prompt: str) -> str:
        seen.append(saved[-1]["judge_calls"])
        raise RuntimeError("down")

    async def repair(prompt: str) -> str:
        raise AssertionError

    rec = cs.new_record(paper, None)
    result = _check(paper, sources, judge, repair, rec, saved)
    assert seen == [1] and result.judge_failed and result.counts["unchecked"] == 1 and result.paper == paper
    # a quest that has spent its calls on other papers asks for none
    spent = cs.new_record(paper, {"calls_in_quest": cs.MAX_CALLS_IN_QUEST})
    calls: list[str] = []

    async def judge2(prompt: str) -> str:
        calls.append(prompt)
        return "{}"

    result = _check(paper, sources, judge2, repair, spent, [])
    assert calls == [] and result.counts["unchecked"] == 1


def test_a_passage_that_is_not_in_the_source_does_not_back_the_sentence() -> None:
    paper = "A claim about it [1].\n"
    sources = {"1": _src("1", "One", FULL + "the real words", full=True)}

    async def judge(prompt: str) -> str:
        pid = prompt.split(" | ")[0]
        return json.dumps({"verdicts": [{"id": pid, "supported": True, "quote": "words the source never had"}]})

    async def repair(prompt: str) -> str:
        return json.dumps({"fixes": []})

    result = _check(paper, sources, judge, repair, cs.new_record(paper, None), [])
    assert result.counts["found_unsupported"] == 1 and "[1]" not in result.paper
    assert result.rows[0].problem == "the passage named is not in the retrieved text of [1]"


def _drop_run(paper: str) -> cs.Result:
    sources = {"1": _src("1", "One", FULL + "something else entirely", full=True)}

    async def judge(prompt: str) -> str:
        pid = prompt.splitlines()[0].split(" | ")[0]
        return json.dumps({"verdicts": [{"id": pid, "supported": False, "why": "not there"}]})

    async def repair(prompt: str) -> str:
        return json.dumps({"fixes": [{"id": i, "action": "drop_sentence"} for i in json.loads(prompt)]})

    return _check(paper, sources, judge, repair, cs.new_record(paper, None), [])


def test_a_sentence_taken_out_does_not_leave_the_next_one_pointing_at_nothing() -> None:
    paper = ("Intro sentence.\n\nHolm and adaptive procedures would close the gap [1]. Neither was run, so we claim nothing. "
             "Another point.\n")
    out = _drop_run(paper)
    assert "Holm" not in out.paper and "Neither" not in out.paper, out.paper
    assert "Another point." in out.paper and "Intro sentence." in out.paper


def test_a_sentence_something_refers_back_to_stays_when_the_referrer_cannot_go() -> None:
    paper = "Holm would close the gap [1]. Neither was run, and the gap is 0.25 here.\n"
    out = _drop_run(paper)
    assert "Neither was run, and the gap is 0.25 here." in out.paper
    assert "[1]" not in out.paper, "only the citation went"


def test_a_sentence_nothing_refers_back_to_is_still_taken_out_alone() -> None:
    out = _drop_run("Holm would close the gap [1]. The study ran BH only.\n")
    assert out.paper.strip() == "The study ran BH only."


_DANGLING = ("BH's proof of FDR control for independent statistics is in [2]. "
             "We did not verify the equality itself against the text of [3]. BH power has no reference line.\n")


def _drop_citation_run(paper: str, action: str) -> cs.Result:
    sources = {"2": _src("2", "Two", FULL, full=True), "3": _src("3", "Three", FULL + "something else", full=True)}

    async def judge(prompt: str) -> str:
        rows = [ln.split(" | ") for ln in prompt.splitlines() if " | cites [" in ln]
        return json.dumps({"verdicts": [
            {"id": r[0], "supported": "[2]" in r[1] and "proof" in r[2], "quote": FULL[:40] if "proof" in r[2] else "",
             "why": "the sentence makes no claim taken from the source"} for r in rows]})

    async def repair(prompt: str) -> str:
        return json.dumps({"fixes": [{"id": i, "action": action} for i in json.loads(prompt)]})

    return _check(paper, sources, judge, repair, cs.new_record(paper, None), [])


def test_a_sentence_that_cannot_stand_without_its_citation_is_not_left_cut_off() -> None:
    """Round s6 delivered 'We did not verify the equality itself against the text of.' (the source named by the citation
    was the object of the sentence)."""
    for action in ("drop_citation", "none"):
        out = _drop_citation_run(_DANGLING, action)
        assert "text of." not in out.paper and "against the text" not in out.paper, (action, out.paper)
        assert "BH power has no reference line." in out.paper and "[3]" not in out.paper


def test_a_cited_sentence_that_reads_whole_without_it_keeps_its_words() -> None:
    out = _drop_citation_run("BH power exceeds Bonferroni power in this run [3]. Next.\n", "drop_citation")
    assert out.paper.strip() == "BH power exceeds Bonferroni power in this run. Next."

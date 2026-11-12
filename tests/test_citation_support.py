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


def _drop_citation_run(paper: str, action: str, own: Any = None) -> cs.Result:
    sources = {"2": _src("2", "Two", FULL, full=True), "3": _src("3", "Three", FULL + "something else", full=True)}

    async def judge(prompt: str) -> str:
        rows = [ln.split(" | ") for ln in prompt.splitlines() if " | cites [" in ln]
        return json.dumps({"verdicts": [
            {"id": r[0], "supported": "[2]" in r[1] and "proof" in r[2], "quote": FULL[:40] if "proof" in r[2] else "",
             "why": "the sentence makes no claim taken from the source"} for r in rows]})

    async def repair(prompt: str) -> str:
        return json.dumps({"fixes": [{"id": i, "action": action} for i in json.loads(prompt)]})

    return _check(paper, sources, judge, repair, cs.new_record(paper, None), [], **({"own": own} if own else {}))


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


def test_a_cut_off_sentence_inside_a_bullet_goes_but_a_bullet_that_is_only_it_stays_a_bullet() -> None:
    """The s6 sentence stood in a bullet, after two sentences of the same bullet."""
    inside = ("- The BH FDR reference is a bound. BH's proof of FDR control is in [2]. We did not verify the equality "
              "itself against the text of [3].\n- BH power has no reference line.\n")
    out = _drop_citation_run(inside, "drop_citation")
    assert "text of." not in out.paper and "- BH power has no reference line." in out.paper
    alone = "Intro [2].\n- We did not verify the equality itself against the text of [3].\n- Next point.\n"
    out2 = _drop_citation_run(alone, "drop_citation")
    assert "text of." not in out2.paper and "- Next point." in out2.paper and "\n- \n" not in out2.paper


def test_a_sentence_that_holds_a_result_and_ends_on_the_citations_words_stays_a_whole_sentence() -> None:
    """The citation was the object of the last words ('above the rounded 22 degrees of [3]') and the sentence holds a result
    of the study, so it cannot go whole: it used to stay as 'the rounded 22 degrees of.'"""
    paper = ("The proof of the error bound for independent trials is in [2]. "
             "The crossing is therefore somewhat above the rounded 22 degrees of [3]. Next point.\n")
    sources = {"2": _src("2", "Two", FULL, full=True), "3": _src("3", "Three", FULL + "something else", full=True)}

    async def judge(prompt: str) -> str:
        rows = [ln.split(" | ") for ln in prompt.splitlines() if " | cites [" in ln]
        return json.dumps({"verdicts": [{"id": r[0], "supported": "[2]" in r[1], "quote": FULL[:40] if "[2]" in r[1] else "",
                                         "why": "not shown"} for r in rows]})

    async def repair(prompt: str) -> str:
        return json.dumps({"fixes": [{"id": i, "action": "drop_citation"} for i in json.loads(prompt)]})

    out = _check(paper, sources, judge, repair, cs.new_record(paper, None), [], own=cs.OwnNumbers(["22"]))
    assert "the rounded 22 degrees." in out.paper and "degrees of." not in out.paper and "[3]" not in out.paper
    assert cs.uncut("Result is.") == "" and cs.uncut("It reads whole.") == "It reads whole."


# --- FI's own source lists are never prose -----------------------------------------------------------------------------

WEB_ONLY = """# Title

## Introduction
A claim from a page [W1]. Another one [W2].

## Further reading

- [W1] (2001). First page title. https://example.org/a
- [W2] Second page title. https://example.org/b
"""


def test_a_further_reading_list_without_references_is_not_read_as_sentences() -> None:
    sites = cs.find_sites(WEB_ONLY)
    assert [s.labels for s in sites] == [["W1"], ["W2"]]
    assert all(s.end <= cs.cited_text_end(WEB_ONLY) for s in sites)


def test_a_check_that_drops_every_citation_leaves_the_list_labels() -> None:
    sources = {"W1": _src("W1", "First page title", ""), "W2": _src("W2", "Second page title", "")}

    async def never(_prompt: str) -> str:
        return "{}"

    res = asyncio.run(cs.check_paper(
        WEB_ONLY, sources, own_numbers=set(), quote_ok=lambda q, t: False, judge=never, repair=never, record={},
        save=lambda r: None, judge_prompt=lambda c, s: "x", repair_prompt=lambda t: "x"))
    assert res.changed, "the body citations were dropped"
    assert res.paper.partition("## Further reading")[2] == WEB_ONLY.partition("## Further reading")[2]
    assert "- [W1] (2001). First page title." in res.paper


def test_a_paper_with_both_lists_has_the_same_sites_as_before() -> None:
    both = PAPER + "\n## Further reading\n\n- [W1] A page. https://example.org\n"
    assert [s.labels for s in cs.find_sites(both)] == [["1", "3"], ["2", "3", "4", "W2"], ["5"], ["6"]]
    assert cs.cited_text_end(both) == cs.references_start(both)


_PROOF = "The proof of the bound for independent inputs is in [2]. "


def test_a_citation_that_is_the_object_of_a_word_mid_sentence_does_not_leave_the_word_hanging() -> None:
    """Delivered papers read 'a series of follows', 'not a contradiction of, because', 'the statements in and with ...':
    the dropped citation was the object of the word before it, and only a sentence ENDING on such a word was mended."""
    for action in ("drop_citation", "none"):
        for sentence, broken in (
                ("The leading series of [3] follows the curve at small values.", "series of follows"),
                ("This is not a contradiction of [3], because its data differ from ours.", "contradiction of,"),
                ("This agrees with the statements in [3] and with the proof in [2].", "statements in and")):
            out = _drop_citation_run(_PROOF + sentence + " Next point.\n", action)
            assert broken not in out.paper and "[3]" not in out.paper, (action, sentence, out.paper)
            assert "Next point." in out.paper


def test_a_mid_sentence_object_citation_in_a_sentence_with_a_result_stays_and_says_so() -> None:
    paper = _PROOF + "The yield is 22 grams, above the value of [3], so it is larger. Next point.\n"
    out = _drop_citation_run(paper, "drop_citation", own=cs.OwnNumbers(["22"]))
    assert "22 grams" in out.paper and "[3]" not in out.paper
    row = next(r for r in out.rows if r.label == "3")
    assert "nothing after it" in row.note


def test_a_citation_after_a_complete_clause_keeps_its_sentence() -> None:
    out = _drop_citation_run(_PROOF + "The result holds [3]. Next point.\n", "drop_citation")
    assert "The result holds. Next point." in out.paper
    out2 = _drop_citation_run(_PROOF + "The result holds as shown [3], and the rest follows. Next.\n", "drop_citation")
    assert "The result holds as shown, and the rest follows." in out2.paper


def test_object_dropped_reads_the_word_before_a_bracket_that_goes_whole() -> None:
    assert cs.object_dropped("A contradiction of [3], because x.", {"3"})
    assert not cs.object_dropped("A contradiction of [3], because x.", {"4"})
    assert not cs.object_dropped("A contradiction of [3, 4], because x.", {"3"})
    assert not cs.object_dropped("The result holds [3].", {"3"})


# --- a citation is read against its own part of the sentence ---------------------------------------------------------

_SNIPPET = ("Queue Waits in Small Shops | Request PDF  A survey of shop counters. Waits grow quickly with the arrival "
            "rate of customers, which ...")
_NO_RESULTS = cs.OwnNumbers()


def _snippet() -> cs.Source:
    src = _src("1", "Queue Waits in Small Shops", _SNIPPET)
    assert src.kind() == "snippet or contents list"
    return src


def _full(label: str, text: str) -> cs.Source:
    return _src(label, "Batching", FULL + text, full=True)


def _site(sentence: str) -> cs.Site:
    return cs.Site(0, len(sentence), sentence, cs.labels_in_text(sentence), "prose")


TWO_PARTS = ("Waits grow with the arrival rate [1], and batching orders lowers staffing cost considerably overall "
             "by 37% [2].")


def test_each_source_is_read_against_the_part_of_the_sentence_its_bracket_closes() -> None:
    site = _site(TWO_PARTS)
    src2 = _full("2", "batching orders lowers staffing cost considerably overall by 37%")
    # the words and the number of the second part are [2]'s claim, not [1]'s
    assert cs.pre_problem(site, "1", _snippet(), _NO_RESULTS) == ""
    assert cs.pre_problem(site, "2", src2, _NO_RESULTS) == ""
    # read against the whole sentence, as before, [1] is refused for them
    assert "37" in cs.pre_problem(site, "1", _snippet(), _NO_RESULTS, whole_sentence=True)
    # a number planted in a part is still read against the source that closes it, and only that source
    planted = _site("Waits grow with the arrival rate by 0.81 [1], and batching orders lowers staffing cost "
                    "considerably overall by 37% [2].")
    assert "0.81" in cs.pre_problem(planted, "1", _snippet(), _NO_RESULTS)
    assert cs.pre_problem(planted, "2", src2, _NO_RESULTS) == ""
    planted2 = _site("Waits grow with the arrival rate [1], and batching orders lowers staffing cost by 0.81 [2].")
    assert cs.pre_problem(planted2, "1", _snippet(), _NO_RESULTS) == ""
    assert "0.81" in cs.pre_problem(planted2, "2", src2, _NO_RESULTS)
    # a finding planted in the snippet source's own part is still refused
    finding = _site("Staffing models fail badly whenever weekend demand doubles sharply [1], and batching helps [2].")
    assert "only a snippet" in cs.pre_problem(finding, "1", _snippet(), _NO_RESULTS)


def test_a_bracket_in_a_list_of_citations_shares_the_claim_before_it_and_the_tail_goes_to_every_source() -> None:
    src2 = _snippet()
    src2.label = "2"
    src2.text = "Unrelated Title | Request PDF  Something else entirely about a different matter, which ..."
    src2.title = "Unrelated Title"
    # ", [2]" names no claim of its own: the claim before it is read against [2] too
    listed = _site("Waits grow with the arrival rate across busy counters [1], [2].")
    assert "says more than it shows" in cs.pre_problem(listed, "2", src2, _NO_RESULTS)
    # a claim written after the last bracket is read against each source
    tail = _site("Earlier work [1], [2] showed staffing models fail badly whenever weekend demand doubles sharply.")
    assert "only a snippet" in cs.pre_problem(tail, "1", _snippet(), _NO_RESULTS)
    # a label cited in two brackets is read against both of its parts
    twice = _site("Waits grow with the arrival rate [1], and staffing models fail badly whenever weekend demand "
                  "doubles sharply [1, 3].")
    assert "only a snippet" in cs.pre_problem(twice, "1", _snippet(), _NO_RESULTS)


def test_a_bracket_that_is_not_a_citation_does_not_split_the_sentence() -> None:
    for sentence in (
            "Batching lowers staffing cost within load share [0, 1] for shops [1].",
            "Batching lowers staffing cost within load share $x_{[2]}$ for shops [1].",
            "Batching lowers staffing cost within load share [2](http://x.org) for shops [1].",
            "Batching lowers staffing cost within load share [1995] for shops [1]."):
        site = _site(sentence)
        assert cs.pre_problem(site, "1", _snippet(), _NO_RESULTS) == cs.pre_problem(
            site, "1", _snippet(), _NO_RESULTS, whole_sentence=True) != "", sentence


def test_a_sentence_with_one_bracket_is_read_exactly_as_before() -> None:
    src2 = _full("2", "a passage with 12.5 in it")
    for sentence in ("Waits grow with the arrival rate [1].",
                     "Staffing models fail badly whenever weekend demand doubles sharply [1].",
                     "Earlier work [1] showed staffing models fail badly whenever weekend demand doubles sharply.",
                     "It rose by 12.5 and then by 44 overall [1, 2]."):
        site = _site(sentence)
        for lb, src in (("1", _snippet()), ("2", src2)):
            assert cs.pre_problem(site, lb, src, _NO_RESULTS) == cs.pre_problem(
                site, lb, src, _NO_RESULTS, whole_sentence=True), (sentence, lb)


def test_a_reword_that_keeps_each_source_on_its_own_part_can_stand() -> None:
    sources = {"1": _snippet(), "2": _full("2", "batching orders lowers staffing cost considerably overall by 37%")}
    site = _site(TWO_PARTS)
    fix = {"action": "reword", "source": "1", "quote": "Waits grow quickly with the arrival rate",
           "text": "Waits grow quickly with the arrival rate [1], and batching orders lowers staffing cost "
                   "considerably overall by 37% [2]."}
    plan = cs._validate_fix(site, {"labels": ["1"], "kept": ["2"]}, fix, sources, _NO_RESULTS, lambda q, t: q in t)
    assert plan and plan["action"] == "reword"


def _two_part_check(judge) -> cs.Result:
    paper = "# Title\n\n" + TWO_PARTS + " Next point.\n\n## References\n"
    sources = {"1": _snippet(), "2": _full("2", "batching orders lowers staffing cost considerably overall by 37%")}

    async def repair(prompt: str) -> str:
        return "{}"

    return _check(paper, sources, judge, repair, cs.new_record(paper, None), [])


def test_a_freed_citation_goes_to_the_reader_and_stays_when_it_names_a_passage() -> None:
    asked: list[str] = []

    async def judge(prompt: str) -> str:
        asked.append(prompt)
        quotes = {"1": "Waits grow quickly with the arrival rate", "2": "batching orders lowers staffing cost"}
        return json.dumps({"verdicts": [{"id": line.split(" | ")[0], "supported": True,
                                         "quote": quotes[line.split(" | ")[0][-1]]} for line in prompt.splitlines()]})

    out = _two_part_check(judge)
    assert any(":1 |" in p for p in asked), "the freed pair was asked"
    assert "[1]" in out.paper and "[2]" in out.paper
    assert {r.label: r.status for r in out.rows} == {"1": "supported", "2": "supported"}


def test_a_freed_citation_the_reader_does_not_answer_is_settled_by_the_whole_sentence() -> None:
    async def answers_only_two(prompt: str) -> str:
        return json.dumps({"verdicts": [{"id": line.split(" | ")[0], "supported": True,
                                         "quote": "batching orders lowers staffing cost"}
                                        for line in prompt.splitlines() if line.split(" | ")[0].endswith(":2")]})

    async def down(prompt: str) -> str:
        raise RuntimeError("down")

    for judge in (answers_only_two, down):
        out = _two_part_check(judge)
        rows = {r.label: r for r in out.rows}
        assert rows["1"].status in ("citation_dropped", "sentence_dropped") and "37" in rows["1"].problem, judge.__name__
        assert "[1]" not in out.paper
    # a pair the whole-sentence rule also passes is left unchecked, as before
    assert {r.label: r.status for r in _two_part_check(down).rows}["2"] == "unchecked"

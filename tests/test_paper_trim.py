"""A draft a little over its page limit loses a few sentences instead of being written again.

The review used to answer an over-long draft with a whole-paper rewrite, then a claim check
and a review of the rewrite (about 43,000 tokens on a stored quest, and the step that changes
correct citations). Now the model is shown the draft with a number before each sentence that
may go and answers with numbers; the engine takes those sentences out, renders the draft, and
only a draft that still does not fit is written again. ``core/paper_trim.py`` holds the parts
that need no model; ``Engine._trim_body_to_fit`` and ``Engine._page_limit_review`` the rest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core import paper_trim
from core.engine import Engine, _grounding_after_trim, _page_limit_hit
from core.paper_patch import PatchError
from tests.test_page_limit import _engine, _quiet_log, _review


def _paper(*, discussion: str = "", with_lists: bool = True) -> str:
    """A paper with the sections a trim may and may not take from."""
    return (
        "# Outbreaks in Small Populations\n\n"
        "## Abstract\n\n"
        "We compare two kinds of model. The chance of an outbreak was 0.25 in the smallest population.\n\n"
        "**Keywords:** outbreak, stochastic\n\n"
        "## Introduction\n\n"
        "Epidemics of infectious diseases spread through contact between the people of a population [1]. "
        "Deterministic models describe the average course of such an epidemic in a large population. "
        "Stochastic models add the chance events that matter when only a few people are infected [2]. "
        "A further sentence of general background that nothing later in the paper relies on at all. "
        "Public health agencies use models of both kinds to plan vaccination campaigns and hospital capacity.\n\n"
        "Let $R_0$ denote the basic reproduction number of the disease in the population under study. "
        "This paper compares the two kinds of model on the probability of an outbreak, shown in Figure 1. "
        "We hypothesize that the two kinds of model agree once the population is large enough. "
        "The comparison uses 300 simulated epidemics for each population size in the study. "
        "Simulation is the usual way to study such processes when no closed form is available.\n\n"
        "## Methods\n\n"
        "We simulated the epidemic 300 times for each population size. "
        "Every simulation started from a single infected person in a fully susceptible population.\n\n"
        "## Results\n\n"
        "The chance of an outbreak was 0.25 in the smallest population and 0.03 in the largest one (Figure 1). "
        "The large populations followed the deterministic curve closely in every simulated setting we tried.\n\n"
        "![**Figure 1.** The chance of an outbreak by population size.](figures/outbreak.png)\n\n"
        "## Discussion\n\n"
        + (discussion or (
            "The simulations agree with the classical threshold theory of epidemics [1]. "
            "Small populations may still experience minor outbreaks because of random fluctuations in contact. "
            "Farms and schools are settings where stochastic models matter, since few people are infected at first [2]. "
            "An average, however carefully it is computed, cannot show the range of outcomes a single epidemic has. "
            "Planning for a single outbreak therefore needs more than the expected course of the disease. "
            "Models are simplified pictures of reality, and their value lies in showing which assumptions matter most "
            "for the question that was asked. "
            "Whether a simple model is enough depends on the decision that has to be made with its help."
        ))
        + "\n\n### Limitations\n\n"
        "The study used one contact structure. Other structures may change the result in ways that were not examined. "
        "A further limitation of the study is that it did not vary the length of the infectious period. "
        "Real epidemics also differ in how strongly the people who are infected change how they behave.\n\n"
        + ("- A list item of background that should stay because taking it out would leave its bullet.\n\n"
           if with_lists else "")
        + "## Conclusion\n\n"
        "Stochastic models are needed for small populations. "
        "Deterministic models describe large populations well enough for planning.\n\n"
        "## References\n\n1. Kermack (1927). A contribution.\n2. Whittle (1955). The outcome.\n"
    )


def _cands(paper: str) -> list[paper_trim.Sentence]:
    return paper_trim.candidates(paper, paper.index("## References"))


def _texts(paper: str) -> list[str]:
    return [s.text for s in _cands(paper)]


# --- which sentences may be chosen -------------------------------------------------------

def test_only_background_and_discussion_sentences_are_offered() -> None:
    texts = _texts(_paper())
    joined = " ".join(texts)
    for stays in (
        "We compare two kinds of model",  # Abstract
        "We simulated the epidemic 300 times",  # Methods
        "The large populations followed the deterministic curve",  # Results
        "Every simulation started from a single infected person",  # Methods
        "**Keywords:**",
    ):
        assert stays not in joined, stays
    assert "Deterministic models describe the average course" in joined
    assert "Small populations may still experience minor outbreaks" in joined
    assert "A further limitation of the study" in joined
    assert "Deterministic models describe large populations well enough" in joined


def test_the_sentence_that_opens_a_paragraph_is_never_offered_even_under_a_heading() -> None:
    texts = _texts(_paper())
    # First sentences of their paragraphs: the Introduction's, the Discussion's (right after its
    # heading, in the same block of lines), the Limitations' and the Conclusion's.
    for opener in (
        "Epidemics of infectious diseases spread",
        "The simulations agree with the classical threshold theory",
        "The study used one contact structure",
        "Stochastic models are needed for small populations",
    ):
        assert not any(t.startswith(opener) for t in texts), opener


def test_a_sentence_with_a_figure_a_definition_or_a_list_bullet_is_not_offered() -> None:
    joined = " ".join(_texts(_paper()))
    assert "shown in Figure 1" not in joined  # refers to a figure
    assert "Let $R_0$ denote" not in joined  # defines a symbol (and opens its paragraph)
    assert "A list item of background" not in joined  # a bullet would be left behind


def test_a_number_the_paper_states_nowhere_else_keeps_its_sentence() -> None:
    joined = " ".join(_texts(_paper()))
    # "300" is in Methods too, so the Introduction sentence that says it may go; "0.25" and "0.03"
    # are in the Results too, but a Discussion sentence stating a number only it states may not.
    assert "The comparison uses 300 simulated epidemics" in joined
    only_here = _paper(discussion=(
        "Opening sentence of the paragraph about the results. "
        "In the largest population the chance of an outbreak fell to 0.017 in the simulations that were run. "
        "Small populations may still experience minor outbreaks because of random fluctuations in contact."
    ))
    joined = " ".join(_texts(only_here))
    assert "fell to 0.017" not in joined
    assert "Small populations may still experience" in joined


def test_a_paper_whose_sections_are_not_background_or_discussion_offers_nothing() -> None:
    paper = (
        "# T\n\n## Method details\n\nOpening. A second sentence here that is long enough to count as one. "
        "A third one that is also long enough to count as a sentence.\n\n## Findings\n\nOpening. "
        "A sentence in the findings that is long enough to count as one of them.\n"
    )
    assert paper_trim.candidates(paper, len(paper)) == []


def test_a_sentence_about_the_papers_own_work_is_never_offered() -> None:
    """On six of six stored papers a model asked for the least valuable sentences took the Introduction's
    first ones, and with them the statement of what the study sets out to do."""
    joined = " ".join(_texts(_paper()))
    assert "We hypothesize that the two kinds of model agree" not in joined
    assert "This paper compares the two kinds" not in joined
    only_own = _paper(discussion=(
        "Opening sentence of the paragraph about the results. "
        "Our simulations show that small populations differ from large ones in the way described. "
        "Small populations may still experience minor outbreaks because of random fluctuations in contact."
    ))
    joined = " ".join(_texts(only_own))
    assert "Our simulations show" not in joined and "Small populations may still experience" in joined


def test_the_sentence_that_says_why_the_work_is_needed_stays_in_an_introduction_only() -> None:
    gap = "However, real epidemics are inherently stochastic in the early stages of an outbreak."
    intro = (
        "# T\n\n## Introduction\n\nOpening sentence of the paragraph is here to be skipped over now. "
        f"Models of epidemics are used to plan the response to a disease in a population. {gap} "
        "Simulation is the usual way to study such processes when no closed form is available.\n\n## References\n\n"
    )
    assert not any(gap in c.text for c in paper_trim.candidates(intro, intro.index("## References")))
    discussion = intro.replace("## Introduction", "## Discussion")
    assert any(gap in c.text for c in paper_trim.candidates(discussion, discussion.index("## References")))


def test_a_sentence_the_next_one_points_back_at_is_not_offered_but_the_pointer_is() -> None:
    paper = (
        "# T\n\n## Introduction\n\nOpening sentence of the paragraph is here to be skipped over now. "
        "A background claim that the next sentence continues from where it stopped. "
        "Moreover, the claim holds only for populations that are large enough to count. "
        "A separate general remark about how such processes are usually studied in practice. "
        "This remark is followed by a sentence that begins with a pointer as well.\n\n## References\n\n"
    )
    texts = [c.text for c in paper_trim.candidates(paper, paper.index("## References"))]
    assert not any(t.startswith("A background claim") for t in texts)  # "Moreover, ..." leans on it
    assert not any(t.startswith("A separate general remark") for t in texts)  # so does "This remark ..."
    assert any(t.startswith("Moreover, the claim holds") for t in texts)  # the pointer itself may go
    assert any(t.startswith("This remark is followed") for t in texts)


def test_the_marked_body_is_the_paper_with_a_marker_before_each_offered_sentence() -> None:
    paper = _paper()
    end = paper.index("## References")
    cands = _cands(paper)
    marked = paper_trim.marked_body(paper, end, cands)
    assert re.sub(r"<<\d+: \d+ words>> ", "", marked) == paper[:end]
    assert marked.count("<<") == len(cands)
    assert f"<<1: {cands[0].words} words>> {cands[0].text}" in marked
    assert "References" not in marked


# --- reading the answer -----------------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ('{"delete": [3, 1, 2]}', [3, 1, 2]),
    ("[3, 1, 2]", [3, 1, 2]),
    ('```json\n{"delete": [4, "2", "<<7>>", 4, 99, 0, true]}\n```', [4, 2, 7]),
    ('Here you go: {"delete": [5, 6]}', [5, 6]),
])
def test_the_ranking_is_read_from_a_reply(reply: str, expected: list[int]) -> None:
    assert paper_trim.parse_ranking(reply, 9) == expected


@pytest.mark.parametrize("reply", ["", "I cannot choose.", '{"delete": []}', '{"delete": [40]}', '{"other": [1]}', "3"])
def test_a_reply_that_is_not_a_list_of_offered_numbers_is_refused(reply: str) -> None:
    with pytest.raises(PatchError):
        paper_trim.parse_ranking(reply, 9)


# --- taking them out --------------------------------------------------------------------

def test_the_sentences_are_taken_in_the_models_order_until_the_words_are_gone() -> None:
    paper = _paper()
    end = paper.index("## References")
    cands = _cands(paper)
    order = [c.n for c in cands[:5]][::-1]
    got = paper_trim.choose(paper, end, cands, order, 30)
    assert got is not None and got.words >= 30
    taken = [s.n for s in got.taken]
    assert taken == order[:len(taken)]  # the model's order, a prefix of its list
    assert got.words - got.taken[-1].words < 30  # and no more than it takes to reach the target
    # Nothing but those sentences is gone: the words left are the paper's words minus theirs.
    without = paper
    for s in sorted(got.taken, key=lambda s: -s.start):
        without = without[:s.start] + without[s.end:]
    assert got.text.split() == without.split()
    for s in got.taken:
        assert s.text not in got.text
    # The headings, the figure and the lists are untouched.
    for keep in ("## Abstract", "## Methods", "![**Figure 1.**", "## References", "- A list item of background"):
        assert keep in got.text


def test_the_spaces_around_a_sentence_taken_out_go_with_it() -> None:
    text = "Alpha one is here. Beta two is here. Gamma three is here.\nDelta four is here."
    spans = [(text.index("Beta"), text.index("Beta") + len("Beta two is here."))]
    assert paper_trim._delete(text, spans) == "Alpha one is here. Gamma three is here.\nDelta four is here."
    last = [(text.index("Gamma"), text.index("Gamma") + len("Gamma three is here."))]
    assert paper_trim._delete(text, last) == "Alpha one is here. Beta two is here.\nDelta four is here."
    both = spans + last
    assert paper_trim._delete(text, both) == "Alpha one is here.\nDelta four is here."


def test_the_only_citation_of_a_source_and_the_only_statement_of_a_number_are_never_taken() -> None:
    paper = (
        "# T\n\n## Introduction\n\nOpening sentence of the paragraph is here. "
        "One background sentence that cites a source found nowhere else in the paper [3]. "
        "Another background sentence that cites the first source again in the same way [1]. "
        "A third background sentence that also cites the first source once more [1]. "
        "A fourth sentence about the number 7.5 which the paper states only here.\n\n## References\n\n"
    )
    end = paper.index("## References")
    cands = paper_trim.candidates(paper, end)
    # The 7.5 sentence is never offered, and the [3] one is offered but never taken.
    assert all("7.5" not in c.text for c in cands)
    order = [c.n for c in cands]
    got = paper_trim.choose(paper, end, cands, order, 10)
    assert got is not None
    assert all("[3]" not in s.text for s in got.taken)
    # Both sentences citing [1] cannot go: the second would leave it cited nowhere.
    assert sum("[1]" in s.text for s in got.taken) == 1


def test_a_list_that_cannot_reach_the_words_is_refused_and_so_is_a_target_over_the_cap() -> None:
    paper = _paper()
    end = paper.index("## References")
    cands = _cands(paper)
    assert paper_trim.choose(paper, end, cands, [cands[0].n], 400) is None
    assert paper_trim.choose(paper, end, cands, [c.n for c in cands], paper_trim.MAX_WORDS + 50) is None


# --- the review: a draft a few sentences over ------------------------------------------------

def _put(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "paper" / "paper.md"
    path.write_text(text, encoding="utf-8")
    return path


def _render_by_words(monkeypatch, *, fits_at: int, over: int = 5, last_page_empty: float = 0.9) -> list[str]:  # noqa: ANN001
    """Stand in for the render: a draft of ``fits_at`` words or fewer is 4 pages, one of more is
    ``over``. Returns the text of every draft rendered."""
    seen: list[str] = []

    async def fake(self, paper_path, state):  # noqa: ANN001, ARG001
        text = Path(paper_path).read_text(encoding="utf-8")
        seen.append(text)
        return {
            "pages": 4 if len(text.split()) <= fits_at else over,
            "words": len(text.split()), "last_page_lines": 3, "last_page_empty": last_page_empty,
        }

    monkeypatch.setattr(Engine, "_measure_draft_pages", fake)
    return seen


def _with_chat(eng: Engine, trim_reply: str | None) -> list[tuple[str, str]]:
    """The engine's chat: ``trim_reply`` for the trim, an accepting review for everything else."""
    calls: list[tuple[str, str]] = []

    async def fake(prompt, *, node=None):  # noqa: ANN001
        calls.append((str(node), prompt))
        if node == "write.trim":
            if trim_reply is None:
                raise RuntimeError("provider down")
            return trim_reply
        if node == "review_moderator":
            return json.dumps({"rationale": "fine"})
        return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})

    eng._chat = fake  # type: ignore[method-assign]
    return calls


def _numbered(paper: str) -> dict[str, int]:
    return {c.text[:40]: c.n for c in _cands(paper)}


@pytest.mark.parametrize("panel", [False, True])
def test_a_draft_a_few_sentences_over_the_limit_loses_them_and_is_not_sent_back(
    tmp_path: Path, monkeypatch, panel: bool,
) -> None:
    eng = _engine(tmp_path, panel=panel)
    written = _paper(with_lists=False)
    paper = _put(tmp_path, written)
    cands = _cands(written)
    total = len(written.split())
    # One trim's worth over: the draft fits once about 100 words are gone.
    rendered = _render_by_words(monkeypatch, fits_at=total - 100)
    calls = _with_chat(eng, json.dumps({"delete": [c.n for c in cands][::-1]}))
    warnings = _quiet_log(eng)

    patch = _review(eng, tmp_path)

    review = patch["review"]
    assert review["must_flag_hits"] == []  # no over_page_limit, nothing to answer
    assert "iteration" not in patch and "page_limit_rewrites" not in patch
    record = review["page_limit"]
    assert record["pages"] == 4 and record["limit"] == 4 and "words_to_cut" not in record
    cut = record["trimmed"]
    assert cut["words"] >= 100 and len(cut["sentences"]) >= 2
    now = paper.read_text(encoding="utf-8")
    assert len(now.split()) <= total - 100
    for sentence in cut["sentences"]:
        assert sentence not in now and sentence in written
    # Only background and discussion lost anything: the rest is character for character as written.
    for keep in (
        "## Abstract\n\nWe compare two kinds of model.", "## Methods", "## Results",
        "The large populations followed the deterministic curve closely", "![**Figure 1.**", "## References",
    ):
        assert keep in now
    assert now.count("[1]") + now.count("[2]") >= 2 and "[1]" in now and "[2]" in now  # both sources still cited
    # The model was called once, for the trim, and saw a numbered paper with no reference list.
    trim = [prompt for node, prompt in calls if node == "write.trim"]
    assert len(trim) == 1 and "<<1:" in trim[0]
    assert "between **100 and 140 words**" in trim[0]  # the words to cut, and how far a list may go over
    assert "## References" not in trim[0].split("<paper>")[1]
    assert [node for node, _p in calls].count("write") == 0
    assert not (tmp_path / ".fi" / "page_check_trial.md").exists()
    assert any("took" in w and "instead of writing the paper again" in w for w in warnings)
    assert rendered[0] == written  # measured as written first, then the trial(s)


def test_the_trim_is_tried_after_the_shortening_rewrites_are_used_up(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    written = _paper(with_lists=False)
    _put(tmp_path, written)
    _render_by_words(monkeypatch, fits_at=len(written.split()) - 100)
    _with_chat(eng, json.dumps({"delete": [c.n for c in _cands(written)]}))
    patch = _review(eng, tmp_path, page_limit_rewrites=2)
    record = patch["review"]["page_limit"]
    assert record.get("trimmed") and "exceeded_after_rewrites" not in record
    assert patch["review"]["must_flag_hits"] == []


def test_a_list_that_falls_short_is_asked_for_again_on_the_paper_as_it_now_stands(
    tmp_path: Path, monkeypatch,
) -> None:
    """The model lists two short sentences (about 30 words) for a hundred: the engine takes them and asks
    again for the rest, on the paper without them, and only then renders."""
    eng = _engine(tmp_path)
    written = _paper(with_lists=False)
    paper = _put(tmp_path, written)
    first = [c.n for c in _cands(written)][:2]
    rendered = _render_by_words(monkeypatch, fits_at=len(written.split()) - 100)
    asked: list[str] = []

    async def chat(prompt, *, node=None):  # noqa: ANN001
        if node != "write.trim":
            return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})
        asked.append(prompt)
        numbers = [int(n) for n in re.findall(r"<<(\d+): \d+ words>>", prompt)]
        return json.dumps({"delete": first if len(asked) == 1 else numbers})

    eng._chat = chat  # type: ignore[method-assign]
    patch = _review(eng, tmp_path)
    assert len(asked) == 2 and patch["review"]["must_flag_hits"] == []
    # The second call was told how many words were left, on a paper that no longer has the first two.
    left = 100 - sum(c.words for c in _cands(written)[:2])
    assert f"between **{left} and {left + 40} words**" in asked[1]
    assert asked[1].count("<<") < asked[0].count("<<")
    cut = patch["review"]["page_limit"]["trimmed"]
    assert len(cut["sentences"]) > 2 and cut["words"] >= 100
    assert len(rendered) == 2  # the draft as written, then one render of the whole trim
    assert len(paper.read_text(encoding="utf-8").split()) <= len(written.split()) - 100


def test_a_draft_still_over_after_the_first_round_is_trimmed_again(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    written = _paper(with_lists=False)
    paper = _put(tmp_path, written)
    total = len(written.split())
    # The first hundred words are not enough: the draft fits only once 150 are gone.
    _render_by_words(monkeypatch, fits_at=total - 150)
    asked: list[str] = []

    async def chat(prompt, *, node=None):  # noqa: ANN001
        if node != "write.trim":
            return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})
        asked.append(prompt)
        return json.dumps({"delete": [int(n) for n in re.findall(r"<<(\d+): \d+ words>>", prompt)]})

    eng._chat = chat  # type: ignore[method-assign]
    patch = _review(eng, tmp_path)
    assert patch["review"]["must_flag_hits"] == []
    assert len(asked) == 2 and patch["review"]["page_limit"]["trimmed"]["words"] >= 150
    assert len(paper.read_text(encoding="utf-8").split()) <= total - 150


@pytest.mark.parametrize("reply", ["I would rather not.", '{"delete": []}', '{"delete": [999]}', None])
def test_a_reply_that_cannot_be_used_leaves_the_paper_and_the_rewrite_as_before(
    tmp_path: Path, monkeypatch, reply: str | None,
) -> None:
    eng = _engine(tmp_path)
    written = _paper(with_lists=False)
    paper = _put(tmp_path, written)
    _render_by_words(monkeypatch, fits_at=len(written.split()) - 100)
    _with_chat(eng, reply)
    warnings = _quiet_log(eng)
    patch = _review(eng, tmp_path)
    assert paper.read_text(encoding="utf-8") == written
    hits = patch["review"]["must_flag_hits"]
    assert hits == [_page_limit_hit(5, 4, 100)]
    assert "trimmed" not in patch["review"]["page_limit"] and patch["page_limit_rewrites"] == 1
    assert any("[page_limit]" in w and "sentences to take out" in w for w in warnings)


def test_a_trim_that_does_not_bring_the_draft_within_the_limit_leaves_it_as_written(
    tmp_path: Path, monkeypatch,
) -> None:
    eng = _engine(tmp_path)
    written = _paper(with_lists=False)
    paper = _put(tmp_path, written)
    rendered = _render_by_words(monkeypatch, fits_at=10)  # nothing the trim can take out is enough
    _with_chat(eng, json.dumps({"delete": [c.n for c in _cands(written)]}))
    patch = _review(eng, tmp_path)
    assert paper.read_text(encoding="utf-8") == written
    assert patch["review"]["must_flag_hits"][0].startswith("over_page_limit")
    assert len(rendered) <= 1 + 3  # the draft, then at most three trials
    assert not (tmp_path / ".fi" / "page_check_trial.md").exists()


def test_a_draft_far_over_the_limit_is_not_trimmed_and_the_model_is_not_asked(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    written = _paper(with_lists=False)
    paper = _put(tmp_path, written)
    _render_by_words(monkeypatch, fits_at=10, over=6, last_page_empty=0.0)  # a page and a half over
    calls = _with_chat(eng, json.dumps({"delete": [1]}))
    patch = _review(eng, tmp_path)
    assert [node for node, _p in calls].count("write.trim") == 0
    assert paper.read_text(encoding="utf-8") == written
    assert patch["review"]["must_flag_hits"][0].startswith("over_page_limit")


def test_the_trim_node_answers_at_temperature_zero_and_uses_the_write_model() -> None:
    from core.engine import _temperature_for_node
    from core.provider import model_for_node, node_budget

    assert _temperature_for_node("write.trim") == 0.0
    assert model_for_node({"write": "big"}, "write.trim") == "big"
    assert model_for_node({"write": "big", "write.trim": "small"}, "write.trim") == "small"
    assert node_budget({"write": 600.0}, "write.trim", 60.0) == 600.0


# --- the claims follow the paper ----------------------------------------------------------------

def _grounding(*claims: tuple[str, str]) -> dict:
    listed = [{"claim": c, "basis": b, "citation_index": None, "quote": "", "evidence": ""} for c, b in claims]
    unsupported = [c["claim"] for c in listed if c["basis"] == "unsupported"]
    return {"claims": listed, "summary": "s", "total": len(listed), "grounded": len(listed) - len(unsupported),
            "unsupported": unsupported}


def test_a_claim_whose_sentence_was_taken_out_leaves_the_grounding() -> None:
    gone = "Small populations may still experience minor outbreaks because of random fluctuations in contact."
    grounding = _grounding(
        ("Small populations may experience minor outbreaks because of random fluctuations", "unsupported"),
        ("Stochastic models are needed for small populations", "experiment"),
    )
    paper = "# T\n\n## Conclusion\n\nStochastic models are needed for small populations.\n"
    pruned = _grounding_after_trim(grounding, [gone], paper)
    assert pruned is not None
    assert [c["claim"] for c in pruned["claims"]] == ["Stochastic models are needed for small populations"]
    assert pruned["total"] == 1 and pruned["grounded"] == 1 and pruned["unsupported"] == []
    assert grounding["total"] == 2  # the original is not changed


def test_a_claim_the_paper_still_says_elsewhere_stays_and_nothing_changes_without_a_match() -> None:
    gone = "Small populations may still experience minor outbreaks because of random fluctuations in contact."
    grounding = _grounding(("Small populations may experience minor outbreaks because of random fluctuations", "unsupported"))
    still = f"# T\n\n## Discussion\n\nOpening sentence. {gone}\n"
    assert _grounding_after_trim(grounding, [gone], still) is None
    assert _grounding_after_trim(_grounding(("An unrelated claim about something else entirely", "experiment")),
                                 [gone], "# T\n") is None


@pytest.mark.parametrize("panel", [False, True])
def test_the_review_reads_the_grounding_of_the_paper_it_reads(tmp_path: Path, monkeypatch, panel: bool) -> None:
    eng = _engine(tmp_path, panel=panel)
    written = _paper(with_lists=False)
    _put(tmp_path, written)
    cands = _cands(written)
    target = next(c for c in cands if c.text.startswith("Small populations may still"))
    _render_by_words(monkeypatch, fits_at=len(written.split()) - 100)
    prompts: list[str] = []
    # The claim's sentence first, then enough others to reach the hundred words to cut.
    calls = _with_chat(eng, json.dumps({"delete": [target.n] + [c.n for c in cands if c.n != target.n]}))
    accepting = eng._chat

    async def spy(prompt, *, node=None):  # noqa: ANN001
        prompts.append(prompt)
        return await accepting(prompt, node=node)

    eng._chat = spy  # type: ignore[method-assign]
    claim = "Small populations may still experience minor outbreaks because of random fluctuations in contact"
    grounding = _grounding((claim, "unsupported"), ("Stochastic models are needed for small populations", "experiment"))
    patch = _review(eng, tmp_path, claim_grounding=grounding)
    assert patch["review"]["page_limit"]["trimmed"]["sentences"][0] == target.text
    pruned = patch["claim_grounding"]
    assert pruned["unsupported"] == [] and pruned["total"] == 1
    # The reviewer is not told to flag a claim the paper no longer makes.
    reviews = [p for p in prompts if "substantive claims trace" in p]
    assert reviews and all(
        "UNSUPPORTED CLAIMS (" not in p and "No unsupported claims were detected" in p and claim not in p
        for p in reviews
    )
    ledger = json.loads((tmp_path / "paper" / "claims.json").read_text(encoding="utf-8"))
    assert ledger["total"] == 1 and ledger["unsupported"] == []
    assert [node for node, _p in calls].count("write.trim") == 1

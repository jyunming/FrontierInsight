"""Replay of a real paper that was delivered with four of its five citations not backed by the retrieved text of their
sources (tests/fixtures/replay/citations_unsupported/): an equation cited to a source whose retrieved text is the German
opening of a paper, a linearised form cited to a book's table of contents, a method cited to a contents list plus a login
page, and a comparison cited to an abstract about another model. The claim check had found three of them and the
rounds were spent; it never listed the fourth. The citation check runs on the final text with the reader's replies
scripted: no model is called."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import yaml

from core import citation_support as cs
from core.engine import _quote_in_source

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "citations_unsupported"
ARUNTHONG_QUOTE = ("It is because the linearization and the inverse transformation to the original space in the linear "
                   "least square method that give rise to the uncertainty.")
REWEIGHTED_QUOTE = ("we use an iteratively reweighted least squares (IRLS) algorithm to obtain the maximum likelihood "
                    "estimates of the model parameters and the error variances specific for the different species in a model")


def _sources() -> dict[str, cs.Source]:
    out: dict[str, cs.Source] = {}
    for f in sorted((FIXTURE / "sources").glob("*.md")):
        label = f.name.split("_")[0]
        _, front, body = f.read_text(encoding="utf-8").split("---", 2)
        meta = yaml.safe_load(front)
        out[label] = cs.Source(label=label, title=str(meta["title"]), text=body.strip(),
                               full=meta.get("content_quality") == "full_text")
    return out


def _paper() -> str:
    return (FIXTURE / "paper.md").read_text(encoding="utf-8")


def _own(paper: str) -> cs.OwnNumbers:
    return cs.OwnNumbers(re.findall(r"\d+(?:\.\d+)?", paper))


def _key(paper: str, needle: str) -> str:
    return next(s.key for s in cs.find_sites(paper) if needle in s.text)


class Scripted:
    """The reader and the fixer, answering from a script; keeps what they were asked and what was saved before each call."""

    def __init__(self, paper: str, *, fixes: dict[str, dict] | None = None, unsupported_3b: bool = True) -> None:
        self.paper = paper
        self.fixes = fixes or {}
        self.unsupported_3b = unsupported_3b
        self.judge_prompts: list[str] = []
        self.repair_prompts: list[str] = []
        self.saved: list[dict] = []
        self.spent_at_call: list[int] = []

    def save(self, record: dict) -> None:
        self.saved.append(json.loads(json.dumps(record)))

    async def judge(self, prompt: str) -> str:
        self.judge_prompts.append(prompt)
        self.spent_at_call.append(self.saved[-1]["judge_calls"] + self.saved[-1].get("verify_calls", 0))
        verdicts = []
        for pid, label, sentence in re.findall(r"^(\S+) \| cites \[(\w+)\] \| (.*)$", prompt, re.M):
            if ">" in pid and label == "5":
                v = {"supported": True, "quote": REWEIGHTED_QUOTE}
            elif ">" in pid and label == "3":
                v = {"supported": True, "quote": ARUNTHONG_QUOTE}
            elif label == "3" and "bias issues" in sentence:
                v = {"supported": not self.unsupported_3b, "quote": ARUNTHONG_QUOTE if not self.unsupported_3b else "",
                     "why": "the abstract is about another model and nothing was compared"}
            elif label == "3":
                v = {"supported": True, "quote": ARUNTHONG_QUOTE}
            else:
                v = {"supported": False, "quote": "", "why": "scripted: nothing in the retrieved text"}
            verdicts.append({"id": pid, **v})
        return json.dumps({"verdicts": verdicts})

    async def repair(self, prompt: str) -> str:
        self.repair_prompts.append(prompt)
        asked = re.findall(r"^Item (\S+) \(", prompt, re.M)
        return json.dumps({"fixes": [{"id": i, **self.fixes[i]} for i in asked if i in self.fixes]})

    def judge_prompt(self, asked, sources) -> str:  # noqa: ANN001
        lines = [f"{pid} | cites [{label}] | {' '.join(site.text.split())}" for pid, site, label in asked]
        return "\n".join(lines) + "\n\n" + "\n".join(sources[label].title for _p, _s, label in asked)

    def repair_prompt(self, targets) -> str:  # noqa: ANN001
        return "\n".join(f"Item {t['id']} ({t['kind']}): {t['sentence']}\n" + "\n".join(
            f"  other source [{lb}]: {p[:60]}" for lb, p in t["others"]) for t in targets)

    def run(self, paper: str | None = None, *, claim_unsupported=lambda s: False, record: dict | None = None,
            own: cs.OwnNumbers | None = None):
        paper = paper if paper is not None else self.paper
        record = record if record is not None else cs.new_record(paper, None)
        return asyncio.run(cs.check_paper(
            paper, _sources(), own_numbers=own if own is not None else _own(paper), quote_ok=_quote_in_source,
            judge=self.judge, repair=self.repair, record=record, save=self.save, judge_prompt=self.judge_prompt,
            repair_prompt=self.repair_prompt, claim_unsupported=claim_unsupported))


def test_the_evidence_is_what_the_real_paper_held() -> None:
    sources = _sources()
    assert sources["1"].kind() == "snippet or contents list" and "Leitf" in sources["1"].text
    assert sources["2"].kind() == "snippet or contents list" and sources["2"].text.count("Regression") >= 2
    assert sources["4"].kind() == "snippet or contents list" and "Infoscience" in sources["4"].text
    assert sources["3"].kind() == "abstract" and ARUNTHONG_QUOTE in sources["3"].text
    claims = json.loads((FIXTURE / "claims.json").read_text(encoding="utf-8"))
    assert claims["total"] == 15 and len(claims["unsupported"]) == 3
    assert not any("[4]" in c["claim"] or c.get("citation_index") == 4 for c in claims["claims"]), "[4] was never listed"
    sites = cs.find_sites(_paper())
    assert [s.labels for s in sites] == [["1"], ["2"], ["3"], ["4"], ["3"]]


def test_the_four_unsupported_citations_are_found_and_the_supported_one_is_kept_with_its_passage() -> None:
    paper = _paper()
    script = Scripted(paper)
    result = script.run()
    rows = {(r.label, "bias issues" in r.sentence): r for r in result.rows}
    # [1], [2], [4] are refused by the rules that need no model: the reader is never asked about them.
    asked = "\n".join(script.judge_prompts[0].splitlines())
    assert "cites [1]" not in asked and "cites [2]" not in asked and "cites [4]" not in asked
    assert "retrieved text of [1] is only a snippet" in rows[("1", False)].problem
    assert "retrieved text of [2] is only a snippet" in rows[("2", False)].problem
    assert "retrieved text of [4] is only a snippet" in rows[("4", False)].problem
    # the second use of [3] is found by the reader, the first is kept with its passage
    assert rows[("3", True)].problem and rows[("3", True)].status != "supported"
    kept = rows[("3", False)]
    assert kept.status == "supported" and kept.passage == ARUNTHONG_QUOTE and kept.passage in _sources()["3"].text
    assert result.counts["supported"] == 1 and result.counts["found_unsupported"] == 4
    # the kept sentence is untouched
    assert "linearization can introduce bias or misrepresent uncertainty if not handled carefully [3]." in result.paper


def test_each_unsupported_citation_is_fixed_by_one_of_the_four_ways() -> None:
    paper = _paper()
    fixes = {
        _key(paper, "regressed against"): {"action": "cite_other", "source": "5", "quote": REWEIGHTED_QUOTE},
        _key(paper, "bias issues"): {"action": "reword", "source": "3", "quote": ARUNTHONG_QUOTE,
                                      "text": "A linearised fit can carry the uncertainty of the transformation into the "
                                              "estimate [3]."},
        _key(paper, "linearizing the Arrhenius"): {"action": "drop_citation"},
        # the sentence about the Arrhenius equation gets no reply: the rule that needs no model settles it
    }
    script = Scripted(paper, fixes=fixes)
    result = script.run()
    c = result.counts
    assert (c["replaced"], c["reworded"], c["citation_dropped"], c["sentence_dropped"], c["unchecked"]) == (1, 1, 2, 0, 0)
    text = result.paper
    assert "using weighted linear regression [5]." in text            # (a) another retrieved source
    assert "A linearised fit can carry the uncertainty of the transformation into the estimate [3]." in text  # (b)
    assert "bias issues" not in text
    assert "into $\\ln k = \\ln A - (E_a/R)(1/T)$." in text or "(1/T)$." in text  # (c) the citation is gone, the sentence stays
    assert "[2]" not in text and "[1]" not in text and "[4]" not in text
    # the fixes were read once more by the reader before they stood
    assert len(script.judge_prompts) == 2 and ">" in script.judge_prompts[1]
    verified = [r for r in result.rows if r.action.startswith("added")]
    assert {r.label for r in verified} == {"5", "3"} and all(r.passage for r in verified)
    # the ledger says, for every citation, the passage that backs it or what was done instead
    for r in result.rows:
        assert r.passage or r.problem
    assert result.changed and result.clean


def test_a_sentence_the_study_does_not_back_either_is_dropped_when_nothing_fixes_it() -> None:
    paper = _paper()
    script = Scripted(paper)  # the fixer answers nothing
    result = script.run(claim_unsupported=lambda s: "bias issues" in s)
    assert result.counts["sentence_dropped"] == 1
    assert "bias issues" not in result.paper and "mentioned in prior work" not in result.paper
    assert "\n\n\n" not in result.paper
    # the other three are settled by the rule: the citation goes, the sentence stays
    assert result.counts["citation_dropped"] == 3
    assert "weighted linear regression." in result.paper


def test_a_fix_the_reader_does_not_confirm_does_not_stand() -> None:
    paper = _paper()
    fixes = {_key(paper, "regressed against"): {"action": "cite_other", "source": "5", "quote": REWEIGHTED_QUOTE}}
    script = Scripted(paper, fixes=fixes)

    async def refuse(prompt: str) -> str:
        script.judge_prompts.append(prompt)
        if "> " in prompt or re.search(r"^\S+>\w+ \|", prompt, re.M):
            return json.dumps({"verdicts": [{"id": i, "supported": False, "quote": ""} for i in
                                            re.findall(r"^(\S+>\w+) \|", prompt, re.M)]})
        return await Scripted.judge(script, prompt)

    script.spent_at_call = []
    script.judge = refuse  # type: ignore[method-assign]
    result = script.run()
    assert "weighted linear regression [5]" not in result.paper
    assert result.counts["replaced"] == 0 and result.counts["citation_dropped"] >= 1


def test_a_citation_a_later_rewrite_adds_is_caught() -> None:
    paper = _paper()
    first = Scripted(paper).run()
    # a rewrite after the check adds a sentence citing a source whose retrieved text does not say it
    later = first.paper.replace("## Limitations", "Reaction rates rise with temperature in every system studied [6].\n\n## Limitations")
    assert "[6]" in later
    second = Scripted(later)
    result = second.run()
    row = next(r for r in result.rows if r.label == "6")
    assert row.status == "citation_dropped" and row.problem
    assert "every system studied." in result.paper and "[6]" not in result.paper


def test_a_snippet_cannot_support_a_number_or_a_method() -> None:
    sources = _sources()
    own = cs.OwnNumbers(["0.95", "1200"])
    number = cs.Site(0, 0, "The activation energy was 4824.8 J/mol [2].", ["2"], "prose")
    why = cs.pre_problem(number, "2", sources["2"], own)
    assert "4824.8" in why and "neither in the retrieved text" in why
    # a number that is the study's own result is not the citation's to show
    assert cs.pre_problem(number, "2", sources["2"], cs.OwnNumbers(["4824.8"])) == ""
    # a method detail a contents list does not show
    method = cs.Site(0, 0, "The weights were set to the inverse variance of each fitted value by an iterative scheme [4].",
                     ["4"], "prose")
    assert "only a snippet or contents list" in cs.pre_problem(method, "4", sources["4"], own)
    # what a contents list does show: the subject
    studied = cs.Site(0, 0, "Linear regression analysis is the subject of a textbook [2].", ["2"], "prose")
    assert cs.pre_problem(studied, "2", sources["2"], own) == ""

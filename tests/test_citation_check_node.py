"""The citation check as a node of the graph (core/citation_support.py, ``Engine._node_citation_check``): it runs last, on
the paper that is delivered, with the reader's replies scripted. Every model call is counted on disk before it is made."""
from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path

import yaml
from langgraph.graph import END

from core import citation_support as cs
from core import receipts as _receipts
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine, _finalize_after_citation_edits

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "citations_unsupported"
REWEIGHTED = ("we use an iteratively reweighted least squares (IRLS) algorithm to obtain the maximum likelihood "
              "estimates of the model parameters and the error variances specific for the different species in a model")
ARUNTHONG = ("It is because the linearization and the inverse transformation to the original space in the linear least "
             "square method that give rise to the uncertainty.")


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.config.pauses.review = "off"  # type: ignore[assignment]
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


def _literature() -> list[dict]:
    out = []
    for f in sorted((FIXTURE / "sources").glob("*.md")):
        _, front, body = f.read_text(encoding="utf-8").split("---", 2)
        meta = yaml.safe_load(front)
        meta["authors"] = ast.literal_eval(meta["authors"]) if isinstance(meta.get("authors"), str) else meta.get("authors")
        out.append({"content": body.strip(), "metadata": meta})
    return out


def _paper(tmp_path: Path) -> Path:
    path = tmp_path / "paper" / "paper.md"
    path.parent.mkdir(parents=True)
    # The replayed paper is from before FI's blocks left the paper: as FI leaves a paper now, without them (the checks
    # take them out before they read it).
    from core.disclosure import without_block

    path.write_text(without_block((FIXTURE / "paper.md").read_text(encoding="utf-8")), encoding="utf-8")
    return path


class Replies:
    def __init__(self, *, fix: bool = True, fail_judge: bool = False) -> None:
        self.calls: list[str] = []
        self.fix = fix
        self.fail_judge = fail_judge
        self.on_disk_at_call: list[int] = []
        self.fi_dir: Path | None = None

    async def __call__(self, prompt: str, *, node: str = "") -> str:
        if "**Citation Reader**" in prompt:
            self.calls.append(node)
            if self.fi_dir is not None:
                rec = json.loads((self.fi_dir / cs.FILE).read_text(encoding="utf-8"))
                self.on_disk_at_call.append(rec["judge_calls"] + rec.get("verify_calls", 0))
            if self.fail_judge:
                raise RuntimeError("UNAVAILABLE (code 503): No capacity available")
            verdicts = []
            for pid, label, sentence in re.findall(r"^(\S+) \| cites \[(\w+)\] \| (.*)$", prompt, re.M):
                if ">" in pid:
                    verdicts.append({"id": pid, "supported": True, "quote": REWEIGHTED if label == "5" else ARUNTHONG})
                elif label == "3" and "bias issues" not in sentence:
                    verdicts.append({"id": pid, "supported": True, "quote": ARUNTHONG})
                else:
                    verdicts.append({"id": pid, "supported": False, "quote": "", "why": "nothing in the retrieved text"})
            return json.dumps({"verdicts": verdicts})
        if "**Citation Fixer**" in prompt:
            self.calls.append(node)
            fixes = []
            for item in re.findall(r"^Item (\S+) \(", prompt, re.M):
                block = prompt.split(f"Item {item} (", 1)[1].split("\nItem ", 1)[0]
                if "regressed against" in block and self.fix:
                    fixes.append({"id": item, "action": "cite_other", "source": "5", "quote": REWEIGHTED})
                elif "bias issues" in block and self.fix:
                    fixes.append({"id": item, "action": "reword", "source": "3", "quote": ARUNTHONG,
                                  "text": "A linearised fit can carry the uncertainty of the transformation into the estimate [3]."})
            return json.dumps({"fixes": fixes})
        if "Claim Grounding" in prompt:
            self.calls.append("claim_check")
            return json.dumps({"claims": [{"claim": "the study ran 1200 trials", "basis": "experiment", "evidence": "protocol"}]})
        raise AssertionError(f"unexpected prompt for node {node}")


def _run(eng: Engine, paper: Path, replies: Replies, **state):
    eng._chat = replies  # type: ignore[assignment,method-assign]
    replies.fi_dir = eng.fi_dir
    base = {"topic": "t", "paper_md": str(paper), "literature": _literature(),
            "review": {"verdict": "accept", "must_flag_hits": []}, "iteration": 2, **state}
    return base, asyncio.run(eng._node_citation_check(base))  # type: ignore[arg-type]


def test_the_review_leaves_through_the_citation_check_and_it_leaves_for_the_end_or_a_person(tmp_path: Path) -> None:
    g = _engine(tmp_path)._build_graph()
    review = next(iter(g.branches["review"].values()))
    assert review.ends["done"] == "citation_check" and review.ends["human_feedback"] == "citation_check"
    after = next(iter(g.branches["citation_check"].values()))
    assert after.ends == {"done": END, "human_feedback": "human_feedback"}


def test_the_last_node_fixes_the_paper_numbers_the_sources_again_and_leaves_receipts_for_the_final_text(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    paper = _paper(tmp_path)
    replies = Replies()
    _state, out = _run(eng, paper, replies)
    final = paper.read_text(encoding="utf-8")
    assert out["citation_next"] == "done"
    # the sources the final text cites are 1 and 2 in the order it cites them; References lists exactly those
    refs = final.split("## References", 1)[1].split("## Further reading")[0]
    assert re.findall(r"^\d+\. ", refs, re.M) == ["1. ", "2. "]
    assert "Arunthong" in refs.split("\n2. ")[0] and "Gao" in refs.split("\n2. ")[1]
    assert "weighted linear regression [2]." in final
    assert "A linearised fit can carry the uncertainty of the transformation into the estimate [1]." in final
    assert [i["metadata"]["title"] for i in out["literature"][:2]] == [
        "Uncertainty analysis of linear least square fitting apply to non-linear model",
        "Improving uncertainty analysis in kinetic evaluations using iteratively reweighted least squares"]
    # the counts, the ledger, the receipts
    assert out["citation_check"]["found_unsupported"] == 4 and out["citation_check"]["supported"] == 1
    ledger = json.loads((paper.parent / "citations.json").read_text(encoding="utf-8"))
    assert len(ledger["citations"]) == 7 and (paper.parent / "CITATIONS.md").is_file()
    for check in ("citation_check", "claim_check"):
        _status, record, problem = _receipts.read(tmp_path, check)
        assert not problem and record["input_hashes"]["paper"] == _receipts.sha256(paper.read_bytes()), check
    assert _receipts.read(tmp_path, "citation_check")[0] == "pass"
    # one reading, one fix, one look at the fix (and the claim check on the paper as delivered)
    assert [c for c in replies.calls if c != "claim_check"] == ["oracle_review.citations", "write.citations", "oracle_review.citations"]
    assert replies.on_disk_at_call == [1, 2], "each call is counted on disk before it is made"


def test_a_second_pass_over_the_paper_it_left_asks_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    paper = _paper(tmp_path)
    _state, out = _run(eng, paper, Replies())
    again = Replies()
    _state, out2 = _run(eng, paper, again, literature=out["literature"])
    assert again.calls == [] and out2["citation_check"]["found_unsupported"] == 4


def test_a_reader_that_cannot_be_had_leaves_the_unread_citations_alone_and_stops_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    paper = _paper(tmp_path)
    before = paper.read_text(encoding="utf-8")
    replies = Replies(fail_judge=True)
    _state, out = _run(eng, paper, replies)
    final = paper.read_text(encoding="utf-8")
    assert out["citation_check"]["judge_failed"] and out["citation_check"]["unchecked"] == 2
    # [3] was not read, so it stays (it is the only source left, so it is numbered [1] now); the three the rules refuse
    # go, with no model, and References lists only the one that is left
    body, refs = final.split("## References", 1)
    assert body.count("[1]") == before.split("## References")[0].count("[3]") == 2
    assert "[2]" not in body and "[3]" not in body and "[4]" not in body
    assert len(re.findall(r"^\d+\. ", refs.split("## Further reading")[0], re.M)) == 1 and "Arunthong" in refs
    assert _receipts.read(tmp_path, "citation_check")[0] == "unknown"
    rec = json.loads((eng.fi_dir / cs.FILE).read_text(encoding="utf-8"))
    assert rec["judge_calls"] <= cs.MAX_JUDGE_CALLS and len([c for c in replies.calls if c != "claim_check"]) == rec["judge_calls"]


def test_the_calls_have_a_cap_and_a_resume_does_not_spend_them_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    paper = _paper(tmp_path)
    text = paper.read_text(encoding="utf-8")
    eng.fi_dir.mkdir(parents=True)
    spent = cs.new_record(text, None)
    spent.update(judge_calls=cs.MAX_JUDGE_CALLS, repair_calls=cs.MAX_REPAIR_CALLS)
    (eng.fi_dir / cs.FILE).write_text(json.dumps(spent), encoding="utf-8")
    replies = Replies()
    _state, out = _run(eng, paper, replies)
    assert [c for c in replies.calls if c != "claim_check"] == [], "a paper whose calls were spent before the stop asks for no more"
    assert out["citation_check"]["unchecked"] == 2


def test_the_sources_are_numbered_again_in_the_order_the_text_cites_them() -> None:
    literature = _literature()
    paper = (FIXTURE / "paper.md").read_text(encoding="utf-8").replace("weighted linear regression [4]", "weighted linear regression [5]")
    text, ordered = _finalize_after_citation_edits(paper, literature, "external")
    body, refs = text.split("## References", 1)
    first_cited = [int(n) for n in dict.fromkeys(re.findall(r"\[(\d+)\]", body))]
    assert first_cited == sorted(first_cited) and first_cited[0] == 1
    assert text.endswith("\n") and "## Further reading" in text
    assert [i["metadata"]["title"] for i in ordered[:5]][0].startswith("Über die Reaktions")


def test_the_ledger_names_the_citations_by_the_numbers_of_the_paper_as_delivered(tmp_path: Path) -> None:
    """Round s6: the check judged the Holm sentence as [8]; the sources were then numbered again and the paper delivered
    it as [7], while the ledger (and the quality report built from it) still said [8] and listed [7] as another claim."""
    eng = _engine(tmp_path)
    paper = _paper(tmp_path)
    _run(eng, paper, Replies())
    final = paper.read_text(encoding="utf-8")
    body = final.split("## References", 1)[0]
    ledger = json.loads((paper.parent / "citations.json").read_text(encoding="utf-8"))
    kept = [r for r in ledger["citations"] if r["status"] == "supported"]
    assert kept
    for r in kept:
        assert r["sentence"] in body, "a backed citation is shown as the sentence stands in the delivered paper"
        assert f"[{r['label']}]" in r["sentence"] or r["label"] in re.findall(r"\[([^\]]+)\]", r["sentence"])
    for r in ledger["citations"]:
        if r["status"] != "supported":
            assert r["label"].startswith("was "), "a citation that was fixed is named as it was before the fix"
            if r.get("new_sentence"):
                assert r["new_sentence"] in body

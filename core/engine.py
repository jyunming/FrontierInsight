"""Frontier Insight research engine — async LangGraph DAG.

Phase B: real LLM-driven nodes, code generation + execution in a per-quest
venv, Axon-backed knowledge retrieval, and SQLite-checkpointed state for
resumability after stalls (LLM quotas, OS sleep, manual interrupt).

Each Engine instance is stateless w.r.t. process globals; N instances must
coexist in one process for the fleet runner.
"""

from __future__ import annotations

import json
import logging
import re
import string
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from .config import Config
from .execution import ExecutionResult, make_executor
from .knowledge import Knowledge, RetrievedDoc
from .provider import (
    LLMClient,
    ProxySupervisor,
    _PROXY_PROVIDERS,
    resolve_endpoint_async,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents"

_FIGURE_SUFFIXES = frozenset({".png", ".svg", ".jpg", ".jpeg", ".pdf"})


@dataclass
class QuestArtifacts:
    quest_id: str
    quest_root: Path
    paper_md: Path | None = None
    paper_pdf: Path | None = None
    figures_dir: Path | None = None
    bundle_manifest: Path | None = None
    raw_state: dict[str, Any] = field(default_factory=dict)


class QuestState(TypedDict, total=False):
    topic: str
    title: str
    iteration: int
    ideas: list[dict[str, Any]]
    chosen_idea: dict[str, Any]
    literature: list[dict[str, Any]]
    design: dict[str, Any]
    code: str
    deps: list[str]
    exec_result: dict[str, Any]
    figures: list[str]
    result_json: dict[str, Any]
    analysis: dict[str, Any]
    paper_md: str
    review: dict[str, Any]


class Engine:
    """Owns one quest's research graph, executor, knowledge layer, and LLM client."""

    def __init__(self, config: Config, *, supervisor: ProxySupervisor | None = None) -> None:
        self.config = config
        self.quest_id = _new_quest_id(config.title or config.topic)
        self.quest_root: Path = config.output.output_dir / self.quest_id
        self.fi_dir: Path = self.quest_root / ".fi"
        self.supervisor = supervisor or ProxySupervisor()
        self.executor = make_executor(
            config.execution.sandbox,
            python_version=config.execution.python_version,
            docker_image=config.execution.docker_image,
        )
        self.knowledge = Knowledge(config.knowledge)
        self._log = _quest_logger(self.quest_id, self.fi_dir)
        self._prompts = _load_prompts()
        self._client: LLMClient | None = None

    async def run(self) -> QuestArtifacts:
        self.fi_dir.mkdir(parents=True, exist_ok=True)
        (self.quest_root / "figures").mkdir(parents=True, exist_ok=True)
        (self.quest_root / "code").mkdir(parents=True, exist_ok=True)
        (self.quest_root / "paper").mkdir(parents=True, exist_ok=True)
        self._log.info("starting quest %s", self.quest_id)
        await self.executor.setup(self.quest_root)

        endpoint = await resolve_endpoint_async(self.config.provider, self.supervisor)
        self._log.info(
            "provider %s -> %s (%s)",
            self.config.provider.name, endpoint.base_url, endpoint.model,
        )
        self._client = LLMClient(endpoint)

        checkpoint_path = self.fi_dir / "state.sqlite"
        try:
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
                graph = self._build_graph().compile(checkpointer=saver)
                initial: QuestState = {
                    "topic": self.config.topic,
                    "title": self.config.title or _slugify(self.config.topic)[:60],
                    "iteration": 0,
                }
                run_config = {"configurable": {"thread_id": self.quest_id}}
                final_state: QuestState = await graph.ainvoke(initial, config=run_config)
        finally:
            await self._client.aclose()
            if self.config.provider.name in _PROXY_PROVIDERS:
                await self.supervisor.release(self.config.provider.name)

        artifacts = self._collect_artifacts(final_state)
        self._write_back_knowledge(artifacts, final_state)
        self._log.info("quest %s reached terminal state", self.quest_id)
        return artifacts

    # ---- graph topology --------------------------------------------------

    def _build_graph(self) -> StateGraph:
        # Phase G hook: subclassing `Engine` and overriding `_build_graph`
        # is the supported way to ship a domain-specific pipeline (e.g.,
        # a lithography graph) without forking the full Engine class.
        # The QuestState TypedDict is the contract — keep field names
        # backwards-compatible if you add a graph here.
        g: StateGraph[QuestState] = StateGraph(QuestState)
        g.add_node("ideate", self._node_ideate)
        g.add_node("literature", self._node_literature)
        g.add_node("design", self._node_design)
        g.add_node("implement", self._node_implement)
        g.add_node("execute", self._node_execute)
        g.add_node("analyze", self._node_analyze)
        g.add_node("write", self._node_write)
        g.add_node("review", self._node_review)

        g.add_edge(START, "ideate")
        g.add_edge("ideate", "literature")
        g.add_edge("literature", "design")
        g.add_edge("design", "implement")
        g.add_edge("implement", "execute")
        g.add_edge("execute", "analyze")
        g.add_edge("analyze", "write")
        g.add_edge("write", "review")
        g.add_conditional_edges(
            "review",
            self._route_after_review,
            {"revise": "design", "done": END},
        )
        return g

    def _route_after_review(self, state: QuestState) -> str:
        if not self.config.engine.review_loop:
            return "done"
        review = state.get("review") or {}
        verdict = review.get("verdict", "accept")
        if verdict == "revise" and state.get("iteration", 0) < self.config.engine.max_iterations:
            return "revise"
        return "done"

    # ---- nodes -----------------------------------------------------------

    async def _node_ideate(self, state: QuestState) -> QuestState:
        self._log.info("[ideate] topic=%s", state["topic"][:80].replace("\n", " "))
        # Pull a few related items from the knowledge base to ground ideation.
        seeded = self.knowledge.search(state["topic"], top_k=3)
        prompt = self._prompts["ideate"].substitute(
            topic=state["topic"],
            literature_block=_format_lit(seeded),
        )
        text = await self._chat(prompt)
        parsed = _parse_json_lenient(text) or {}
        ideas = parsed.get("ideas") or []
        chosen = parsed.get("chosen") or (ideas[0] if ideas else {"title": "fallback", "rationale": ""})
        return {"ideas": ideas, "chosen_idea": chosen}

    async def _node_literature(self, state: QuestState) -> QuestState:
        chosen = state.get("chosen_idea") or {}
        query = (chosen.get("title") or "") + " " + state["topic"][:200]
        docs = self.knowledge.search(query.strip(), top_k=self.config.knowledge.top_k)
        self._log.info("[literature] retrieved %d docs from Axon", len(docs))
        return {
            "literature": [
                {"content": d.content[:2000], "metadata": d.metadata} for d in docs
            ]
        }

    async def _node_design(self, state: QuestState) -> QuestState:
        iteration = state.get("iteration", 0)
        self._log.info("[design] iteration=%d", iteration)
        review_feedback = ""
        if iteration > 0:
            review_feedback = json.dumps(state.get("review", {}), indent=2)
        prompt = self._prompts["design"].substitute(
            topic=state["topic"],
            chosen_idea=json.dumps(state.get("chosen_idea") or {}, indent=2),
            literature_block=_format_lit_from_state(state),
            review_feedback=review_feedback or "(none — first iteration)",
            timeout_s=str(self.config.execution.timeout_s),
        )
        text = await self._chat(prompt)
        design = _parse_json_lenient(text) or {"hypothesis": "(parse failed)", "dependencies": []}
        return {"design": design}

    async def _node_implement(self, state: QuestState) -> QuestState:
        self._log.info("[implement] generating experiment code")
        prompt = self._prompts["implement"].substitute(
            design_block=json.dumps(state.get("design") or {}, indent=2),
            timeout_s=str(self.config.execution.timeout_s),
        )
        text = await self._chat(prompt)
        parsed = _parse_json_lenient(text) or {}
        code = parsed.get("code") or 'print("RESULT_JSON: {}")'
        deps = parsed.get("deps") or []
        # Defensive: if the design listed deps and the impl skipped them, union.
        design_deps = (state.get("design") or {}).get("dependencies") or []
        deps = sorted({*deps, *design_deps})

        code_path = self.quest_root / "code" / "experiment.py"
        code_path.write_text(code, encoding="utf-8")
        self._log.info("[implement] wrote %s (%d bytes)", code_path, len(code))
        return {"code": code, "deps": deps}

    async def _node_execute(self, state: QuestState) -> QuestState:
        deps = state.get("deps") or []
        if deps:
            self._log.info("[execute] pip install %s", deps)
            install = await self.executor.install(deps, quest_root=self.quest_root)
            if install.returncode != 0:
                self._log.warning(
                    "[execute] pip install rc=%d stderr_tail=%s",
                    install.returncode, install.stderr[-400:],
                )

        py = self.executor.python_path(self.quest_root)
        code_path = self.quest_root / "code" / "experiment.py"
        # Run from quest_root so figures/ is the relative target.
        result: ExecutionResult = await self.executor.execute(
            [str(py), str(code_path)],
            cwd=self.quest_root,
            timeout_s=self.config.execution.timeout_s,
        )
        figures = sorted(
            p.name for p in (self.quest_root / "figures").iterdir()
            if p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
        ) if (self.quest_root / "figures").is_dir() else []
        result_json = _extract_result_json(result.stdout)
        self._log.info(
            "[execute] rc=%d duration=%.1fs figures=%d result_json=%s",
            result.returncode, result.duration_s, len(figures), bool(result_json),
        )
        return {
            "exec_result": {
                "returncode": result.returncode,
                "duration_s": result.duration_s,
                "timed_out": result.timed_out,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            },
            "figures": figures,
            "result_json": result_json or {},
        }

    async def _node_analyze(self, state: QuestState) -> QuestState:
        self._log.info("[analyze] interpreting results")
        exec_result = state.get("exec_result") or {}
        prompt = self._prompts["analyze"].substitute(
            design_block=json.dumps(state.get("design") or {}, indent=2),
            returncode=str(exec_result.get("returncode")),
            duration_s=f"{exec_result.get('duration_s', 0):.1f}",
            timed_out=str(exec_result.get("timed_out", False)),
            stdout_tail=exec_result.get("stdout_tail", "")[:2000],
            stderr_tail=exec_result.get("stderr_tail", "")[:1000],
            result_json=json.dumps(state.get("result_json") or {}, indent=2),
            figure_list="\n".join(f"- figures/{f}" for f in state.get("figures", [])) or "(none)",
        )
        text = await self._chat(prompt)
        analysis = _parse_json_lenient(text) or {"summary": "(parse failed)", "key_findings": []}
        return {"analysis": analysis}

    async def _node_write(self, state: QuestState) -> QuestState:
        self._log.info("[write] authoring IMRAD paper.md")
        prompt = self._prompts["write"].substitute(
            topic=state["topic"],
            title=state.get("title", "Untitled"),
            design_block=json.dumps(state.get("design") or {}, indent=2),
            analysis_block=json.dumps(state.get("analysis") or {}, indent=2),
            literature_block=_format_lit_from_state(state),
            figure_list="\n".join(f"- figures/{f}" for f in state.get("figures", [])) or "(none)",
        )
        markdown = await self._chat(prompt)
        # The model may wrap with a fence; strip it.
        markdown = _strip_outer_fence(markdown)
        paper_path = self.quest_root / "paper" / "paper.md"
        paper_path.write_text(markdown, encoding="utf-8")
        self._log.info("[write] wrote %s (%d bytes)", paper_path, len(markdown))
        return {"paper_md": str(paper_path)}

    async def _node_review(self, state: QuestState) -> QuestState:
        self._log.info("[review] judging paper")
        paper_md = ""
        paper_path = state.get("paper_md")
        if paper_path:
            try:
                paper_md = Path(paper_path).read_text(encoding="utf-8")
            except OSError:
                paper_md = ""
        prompt = self._prompts["review"].substitute(
            topic=state["topic"],
            design_block=json.dumps(state.get("design") or {}, indent=2),
            analysis_block=json.dumps(state.get("analysis") or {}, indent=2),
            paper_md=paper_md[:8000],
        )
        text = await self._chat(prompt)
        review = _parse_json_lenient(text) or {"verdict": "accept", "score": 3, "suggestions": []}
        update: QuestState = {"review": review}
        if review.get("verdict") == "revise":
            update["iteration"] = state.get("iteration", 0) + 1
            self._log.info("[review] verdict=revise -> iteration %d", update["iteration"])
        else:
            self._log.info("[review] verdict=%s score=%s", review.get("verdict"), review.get("score"))
        return update

    # ---- helpers ---------------------------------------------------------

    async def _chat(self, prompt: str) -> str:
        assert self._client is not None
        messages = [{"role": "user", "content": prompt}]
        return await self._client.chat(messages, temperature=0.2)

    def _collect_artifacts(self, state: QuestState) -> QuestArtifacts:
        paper_md = self.quest_root / "paper" / "paper.md"
        figures = self.quest_root / "figures"
        manifest = self.quest_root / "paper" / "paper_bundle_manifest.json"
        figures_present = (
            figures.is_dir()
            and any(
                p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
                for p in figures.iterdir()
            )
        )
        return QuestArtifacts(
            quest_id=self.quest_id,
            quest_root=self.quest_root,
            paper_md=paper_md if paper_md.exists() else None,
            paper_pdf=None,
            figures_dir=figures if figures_present else None,
            bundle_manifest=manifest if manifest.exists() else None,
            raw_state=dict(state),
        )

    def _write_back_knowledge(self, artifacts: QuestArtifacts, state: QuestState) -> None:
        if not self.knowledge.enabled or not self.config.knowledge.write_back_quests:
            return
        if artifacts.paper_md is None:
            return
        summary_parts: list[str] = []
        analysis = state.get("analysis") or {}
        if analysis.get("summary"):
            summary_parts.append(str(analysis["summary"]))
        for kf in analysis.get("key_findings", []) or []:
            summary_parts.append(f"- {kf}")
        summary = "\n".join(summary_parts)
        ok = self.knowledge.add_quest_artifacts(
            quest_id=self.quest_id,
            paper_md_path=artifacts.paper_md,
            summary=summary,
        )
        self._log.info("[write-back] axon ingest=%s", ok)


# ---- module-level helpers ------------------------------------------------


def _load_prompts() -> dict[str, string.Template]:
    names = ("ideate", "design", "implement", "analyze", "write", "review")
    out: dict[str, string.Template] = {}
    for n in names:
        path = PROMPTS_DIR / f"{n}.md"
        out[n] = string.Template(path.read_text(encoding="utf-8"))
    return out


def _format_lit(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(no prior work surfaced from the knowledge base)"
    lines: list[str] = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title") or d.metadata.get("source") or f"item-{i}"
        lines.append(f"[{i}] {title}\n{d.content[:600]}")
    return "\n\n".join(lines)


def _format_lit_from_state(state: QuestState) -> str:
    items = state.get("literature") or []
    if not items:
        return "(no prior work surfaced from the knowledge base)"
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        meta = item.get("metadata") or {}
        title = meta.get("title") or meta.get("source") or f"item-{i}"
        lines.append(f"[{i}] {title}\n{item.get('content', '')[:600]}")
    return "\n\n".join(lines)


_FENCE_RE = re.compile(r"^```(?:\w+)?\n(.*)\n```$", re.DOTALL)


def _strip_outer_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def _parse_json_lenient(text: str) -> dict[str, Any] | None:
    """Find and parse a JSON object inside arbitrary LLM output.

    LLMs frequently wrap JSON in markdown fences, prose, or trailing
    commentary. This tries strict parse first; on failure, slices from
    the first `{` to the last `}` and parses that. The slice is NOT
    balance-aware — if the LLM emits two top-level objects in one
    response, the slice will span both and parsing will fail.
    """
    if not text:
        return None
    candidate = _strip_outer_fence(text).strip()
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    # Find the first '{' and last '}' and try the slice.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            result = json.loads(candidate[start : end + 1])
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None
    return None


_RESULT_LINE_RE = re.compile(r"RESULT_JSON:\s*(\{.*\})\s*$", re.MULTILINE)


def _extract_result_json(stdout: str) -> dict[str, Any] | None:
    if not stdout:
        return None
    matches = list(_RESULT_LINE_RE.finditer(stdout))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].group(1))
    except json.JSONDecodeError:
        return None


def _new_quest_id(seed: str) -> str:
    base = _slugify(seed)[:32] or "quest"
    return f"{int(time.time())}-{base}-{uuid.uuid4().hex[:6]}"


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "untitled"


def _quest_logger(quest_id: str, fi_dir: Path) -> logging.Logger:
    fi_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"frontier_insight.{quest_id}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(fi_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f"[{quest_id[:24]}] %(message)s"))
    logger.addHandler(sh)
    return logger

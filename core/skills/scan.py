"""Static review of a skill's contents, for a person about to approve it.

A skill is the one thing in FI that arrives from outside and goes straight
into two privileged places: its ``SKILL.md`` is injected verbatim into the
``design`` and ``implement`` prompts, and its ``selftest.py`` is executed as
a subprocess. Import accepts any filesystem path, and entry-point discovery
loads whatever an installed package advertises.

The promotion gate already asks two questions — *does the self-test pass*,
and *did a person approve this exact content*. Neither asks whether the
content is trying to do something to FI. Human approval was the only answer
to that, and reading an injection payload out of plausible prose is not
something a person reliably does.

**What this is for.** Not to certify a skill. To put the specific lines a
person should look at in front of them at the moment they are deciding. The
decision stays theirs; this only makes sure they are looking at the right
place.

Three rules follow from that, and each is load-bearing:

**Parse, never import or execute.** Reviewing hostile code by running it is
self-defeating. Python is walked as an AST, everything else by pattern. Note
that ``scaffold.py`` deliberately does the opposite — it imports a library
to read its real signatures — which is safe there because the person named
the module and it is already installed. Here the file is the untrusted
input, so nothing in this module may import, exec, or run it.

**Findings inform; they never quarantine.** Every rule here is a heuristic,
and a legitimate skill mentions ``subprocess`` in prose or ships a script
that genuinely needs the network. QUARANTINED stays reserved for a self-test
that actually failed — something FI observed rather than guessed.

**Never report "safe".** This module reports how many findings came out of
how many rules, and nothing else. A scanner that prints "clean" manufactures
exactly the false confidence the promotion gate exists to catch: a payload
this does not have a rule for is not absent, only unfound.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from core.skills.base import Skill

# --- severities -------------------------------------------------------------
#
# Three levels, not six. A person reads this list at approval time, and a
# taxonomy finer than "look at this / note this / be aware" spends their
# attention on grading rather than on the lines.

HIGH = "high"
MEDIUM = "medium"
INFO = "info"

_ORDER = {INFO: 0, MEDIUM: 1, HIGH: 2}


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    path: str
    line: int
    detail: str

    def render(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.severity:<6}] {self.rule}  {where}\n           {self.detail}"


# --- prose rules ------------------------------------------------------------

#: Phrases that try to redirect the model reading the skill. A skill's job is
#: to describe software; none of these belong in that description, which is
#: what makes them worth flagging even though the wording varies endlessly.
_INJECTION = [
    (r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier|the\s+above)\s+"
     r"(?:instructions?|prompts?|rules?|directions?)",
     "attempts to discard earlier instructions"),
    # The determiner and the adjective are independently optional: real
    # payloads say "disregard the above rules" as readily as "disregard
    # previous instructions", and an early version matched only the second.
    (r"disregard\s+(?:all\s+)?(?:the\s+)?"
     r"(?:(?:previous|prior|above|earlier)\s+)?"
     r"(?:instructions?|system|prompts?|rules?|directions?)",
     "attempts to discard earlier instructions"),
    (r"do\s+not\s+(?:tell|inform|show|mention|reveal|report)\s+(?:the\s+)?"
     r"(?:user|human|operator)",
     "instructs the model to conceal something from the user"),
    # "From now on" is usually followed by a subject before the verb
    # ("from now on you must act as"), which an earlier version required to
    # be adjacent and so missed.
    (r"(?:you\s+(?:must|should|will)\s+now|from\s+now\s+on)[\s,]+"
     r"(?:you\s+(?:must|should|will|are)\s+)?"
     r"(?:act|behave|pretend|respond|ignore|forget|only)",
     "attempts to redefine the model's role"),
    (r"new\s+(?:instructions?|system\s+prompt|rules?)\s*:",
     "declares replacement instructions"),
    (r"</?(?:system|assistant|human|user)>",
     "fake conversation-role tag, which can forge turn boundaries"),
    (r"reveal\s+(?:your|the)\s+(?:system\s+prompt|instructions?)",
     "attempts to extract the system prompt"),
]

#: Invisible characters. Cheap to check and high-signal: legitimate technical
#: prose has no reason to carry zero-width or direction-override codepoints,
#: and they are how a payload hides from the person reading the file.
_INVISIBLE = re.compile(
    "["
    "​-‏"   # zero-width space/joiners, LTR/RTL marks
    "‪-‮"   # bidirectional embedding and override
    "⁠-⁤"   # word joiner, invisible operators
    "﻿"          # zero-width no-break space
    "]"
)

_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.S)
_IMPERATIVE_IN_COMMENT = re.compile(
    r"\b(?:ignore|must|do\s+not|instruct|you\s+are|system)\b", re.I
)
_MD_LINK = re.compile(r"\[([^\]]{4,})\]\((https?://[^)\s]+)\)")
_ALLOWED_TOOLS = re.compile(r"^allowed-tools\s*:\s*(.+)$", re.M | re.I)


def _scan_text(rel: str, text: str) -> list[Finding]:
    out: list[Finding] = []
    lines = text.splitlines()

    for i, line in enumerate(lines, 1):
        for pattern, why in _INJECTION:
            if re.search(pattern, line, re.I):
                out.append(Finding(
                    "INJ001", HIGH, rel, i,
                    f"{why}: {line.strip()[:120]}",
                ))
                break
        m = _INVISIBLE.search(line)
        if m:
            out.append(Finding(
                "INJ002", HIGH, rel, i,
                f"invisible character U+{ord(m.group()):04X} — text can be "
                f"hidden from a reader while still reaching the model",
            ))
        if _BASE64_BLOB.search(line):
            out.append(Finding(
                "OBF001", MEDIUM, rel, i,
                "long base64-like blob; decode it before approving",
            ))

    for m in _HTML_COMMENT.finditer(text):
        body = m.group(1)
        if _IMPERATIVE_IN_COMMENT.search(body):
            out.append(Finding(
                "INJ003", MEDIUM, rel, text[:m.start()].count("\n") + 1,
                f"HTML comment carrying an instruction — invisible when the "
                f"markdown is rendered, but not to the model: "
                f"{' '.join(body.split())[:100]}",
            ))

    for m in _MD_LINK.finditer(text):
        label, href = m.group(1), m.group(2)
        # Only flag when the label itself looks like a different URL or host —
        # ordinary descriptive link text is not a mismatch.
        if re.match(r"^(?:https?://|www\.)", label.strip(), re.I):
            label_host = re.sub(r"^https?://", "", label.strip()).split("/")[0]
            href_host = re.sub(r"^https?://", "", href).split("/")[0]
            if label_host.lower().lstrip("www.") != href_host.lower().lstrip("www."):
                out.append(Finding(
                    "LNK001", MEDIUM, rel, text[:m.start()].count("\n") + 1,
                    f"link text says {label_host} but points at {href_host}",
                ))

    for m in _ALLOWED_TOOLS.finditer(text):
        out.append(Finding(
            "MET001", INFO, rel, text[:m.start()].count("\n") + 1,
            f"declares allowed-tools: {m.group(1).strip()} — the capabilities "
            f"this skill expects to be given",
        ))

    return out


# --- python rules -----------------------------------------------------------

_NETWORK_MODULES = {
    "socket", "ssl", "urllib", "urllib2", "urllib3", "http", "httplib",
    "requests", "httpx", "aiohttp", "ftplib", "telnetlib", "smtplib",
    "paramiko", "websockets", "websocket",
}
_EXEC_NAMES = {"eval", "exec", "compile", "__import__"}
_SERIALISATION = {"pickle", "marshal", "dill", "shelve"}
_PROCESS_MODULES = {"subprocess", "pty", "multiprocessing"}


def _module_root(name: str) -> str:
    return (name or "").split(".")[0]


class _PyVisitor(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.found: list[Finding] = []

    def _add(self, rule: str, sev: str, node: ast.AST, detail: str) -> None:
        self.found.append(
            Finding(rule, sev, self.rel, getattr(node, "lineno", 0), detail)
        )

    # imports

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(_module_root(alias.name), node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._check_module(_module_root(node.module or ""), node)
        self.generic_visit(node)

    def _check_module(self, root: str, node: ast.AST) -> None:
        if root in _NETWORK_MODULES:
            self._add(
                "NET001", HIGH, node,
                f"imports {root} — this skill can reach the network. A "
                f"self-test should probe a local tool, not call out.",
            )
        elif root == "ctypes":
            self._add(
                "EXE002", HIGH, node,
                "imports ctypes — can call arbitrary native code, which no "
                "static rule below will see",
            )
        elif root in _SERIALISATION:
            self._add(
                "EXE003", MEDIUM, node,
                f"imports {root} — deserialising untrusted data executes code",
            )
        elif root in _PROCESS_MODULES:
            self._add(
                "PRC001", MEDIUM, node,
                f"imports {root} — starts other processes; check what it runs",
            )

    # calls

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _EXEC_NAMES:
            self._add(
                "EXE001", HIGH, node,
                f"calls {fn.id}() — executes code assembled at run time, so "
                f"what actually runs is not visible in this file",
            )
        elif isinstance(fn, ast.Attribute):
            owner = fn.value.id if isinstance(fn.value, ast.Name) else ""
            if owner == "os" and fn.attr in {"system", "popen", "execv", "execve"}:
                self._add(
                    "PRC002", MEDIUM, node,
                    f"calls os.{fn.attr}() — runs a shell command",
                )
            elif owner == "subprocess" and fn.attr in {
                "run", "call", "check_call", "check_output", "Popen"
            }:
                self._add(
                    "PRC003", MEDIUM, node,
                    f"calls subprocess.{fn.attr}() — check the command and "
                    f"whether shell=True is used",
                )
        self.generic_visit(node)

    # environment

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in {"environ", "getenv"}
        ):
            self._add(
                "ENV001", MEDIUM, node,
                "reads the environment — credentials and API keys live there",
            )
        self.generic_visit(node)


def _scan_python(rel: str, text: str) -> list[Finding]:
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        # Not a finding about intent — but a file that cannot be parsed also
        # cannot be reviewed by the rules below, and saying so is better than
        # returning an empty list that reads as "nothing found".
        return [Finding(
            "PAR001", MEDIUM, rel, e.lineno or 0,
            f"could not be parsed ({e.msg}), so no Python rule was applied "
            f"to it — review this file by hand",
        )]
    v = _PyVisitor(rel)
    v.visit(tree)
    return v.found


# --- shell rules ------------------------------------------------------------

_SHELL_RULES = [
    (re.compile(r"\b(?:curl|wget)\b"), "NET002", HIGH,
     "fetches over the network"),
    (re.compile(r"\bnc\s|\bncat\b|\btelnet\b"), "NET003", HIGH,
     "opens a raw network connection"),
    (re.compile(r"base64\s+(?:-d|--decode)"), "OBF002", HIGH,
     "decodes base64 — commonly how a payload is hidden from review"),
    (re.compile(r"\brm\s+-[rf]{1,2}\b"), "DES001", MEDIUM,
     "recursive or forced delete"),
    (re.compile(r"\bchmod\s+(?:\+x|777)"), "PRC004", MEDIUM,
     "makes a file executable or world-writable"),
]


def _scan_shell(rel: str, text: str) -> list[Finding]:
    out: list[Finding] = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, rule, sev, why in _SHELL_RULES:
            if pattern.search(line):
                out.append(Finding(rule, sev, rel, i, f"{why}: {stripped[:100]}"))
    return out


# --- entry point ------------------------------------------------------------

#: Roughly the number of distinct checks applied, reported alongside the
#: findings so "no findings" is always qualified by how much was looked for.
RULE_COUNT = len(_INJECTION) + len(_SHELL_RULES) + 14

_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
_SHELL_SUFFIXES = {".sh", ".bash", ".zsh", ".ps1"}
_MAX_BYTES = 2_000_000


def scan(skill: Skill) -> list[Finding]:
    """Review everything in the skill directory. Never imports or runs it.

    Ordered worst-first so a person reading a truncated list sees the
    findings that matter.
    """
    out: list[Finding] = []
    for rel, path in skill.hashable_files():
        try:
            if path.stat().st_size > _MAX_BYTES:
                out.append(Finding(
                    "SIZ001", INFO, rel, 0,
                    "too large to review here; check it by hand",
                ))
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        suffix = path.suffix.lower()
        if suffix == ".py":
            out.extend(_scan_python(rel, text))
        elif suffix in _SHELL_SUFFIXES:
            out.extend(_scan_shell(rel, text))
        elif suffix in _TEXT_SUFFIXES or not suffix:
            out.extend(_scan_text(rel, text))

    out.sort(key=lambda f: (-_ORDER.get(f.severity, 0), f.path, f.line))
    return out


def worst(findings: list[Finding]) -> str | None:
    """The highest severity present, or None when there are no findings."""
    if not findings:
        return None
    return max(findings, key=lambda f: _ORDER.get(f.severity, 0)).severity


def summarise(findings: list[Finding]) -> str:
    """One line for a listing.

    Says how many rules were applied even when nothing was found, because
    "no findings" and "safe" are different claims and only the first one is
    ours to make.
    """
    if not findings:
        return f"no findings from {RULE_COUNT} rules"
    counts = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    parts = [f"{counts[s]} {s}" for s in (HIGH, MEDIUM, INFO) if s in counts]
    return f"{len(findings)} finding(s) from {RULE_COUNT} rules: {', '.join(parts)}"


def render(findings: list[Finding], *, limit: int = 0) -> str:
    """Full report for a person deciding whether to approve."""
    if not findings:
        return (
            f"No findings from {RULE_COUNT} static rules.\n"
            "That is not a clean bill of health: these rules are heuristics, "
            "and a payload none of them describes would not appear here. "
            "Read SKILL.md yourself before approving."
        )
    shown = findings[:limit] if limit else findings
    lines = [summarise(findings), ""]
    lines += [f.render() for f in shown]
    if limit and len(findings) > limit:
        lines.append(f"... and {len(findings) - limit} more")
    lines += [
        "",
        "These are heuristics, not verdicts. A skill that legitimately drives "
        "a network tool will flag NET001; the point is that you saw it.",
    ]
    return "\n".join(lines)

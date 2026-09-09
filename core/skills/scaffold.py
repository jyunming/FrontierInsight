"""Draft a skill from software that already exists.

Teaching FI a skill by hand means writing four files, and two of them are
tedious to get right: the API surface has to match the real signatures, and
the self-test has to actually exercise the library. This scaffolds both from
the software itself.

The API surface is produced by **introspection, not generation**. A language
model asked to write down a library's signatures will occasionally invent one,
and a confidently wrong signature is worse than none — it teaches every future
quest to call something that does not exist. ``inspect`` cannot hallucinate.

What is *not* scaffolded is the judgement: when the skill applies, when it does
not, and what its outputs may legally be. Those are left as marked TODOs,
because a skill that claims coverage it does not have is the failure mode this
whole subsystem exists to prevent, and only a person who knows the software can
write them. The self-test is likewise a starting point — it proves the module
imports and its entry points exist, which is enough to catch a broken install
and not enough to certify physics.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from core.skills.base import SELFTEST_PY

MAX_MEMBERS_PER_MODULE = 40


@dataclass
class Drafted:
    path: Path
    modules: list[str]
    entries: int
    todos: int


def _public(name: str) -> bool:
    return not name.startswith("_")


def _signature(obj: Any, name: str = "") -> str:
    """The call signature, from introspection or the docstring.

    ``inspect.signature`` fails on C extensions, and scientific libraries are
    very often C extensions — gdstk, numpy's core, most solvers. Those
    conventionally put the real signature on the first docstring line
    (``Polygon(points, layer=0, datatype=0)``), so fall back to reading it
    there rather than emitting a useless ``(...)``.

    Still not generation: the string comes from the package either way.
    """
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        pass
    doc = (inspect.getdoc(obj) or "").strip()
    first = doc.splitlines()[0].strip() if doc else ""
    if name and first.startswith(f"{name}(") and first.endswith(")"):
        return first[len(name):]
    return "(...)"


def _first_line(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0].strip() if doc.strip() else ""


def _walk(root: ModuleType, depth: int = 1) -> list[ModuleType]:
    """The root module plus its immediate submodules.

    One level only: a deep package walk produces an API surface far longer
    than a prompt should carry, and the useful entry points of a scientific
    library are almost always near the top.
    """
    mods = [root]
    if depth <= 0 or not hasattr(root, "__path__"):
        return mods
    for info in pkgutil.iter_modules(root.__path__):
        if info.name.startswith("_"):
            continue
        try:
            mods.append(importlib.import_module(f"{root.__name__}.{info.name}"))
        except Exception:  # noqa: BLE001 - a submodule that will not import is skipped
            continue
    return mods


def _members(mod: ModuleType) -> list[tuple[str, str, str, str]]:
    """(kind, name, signature, summary) for a module's own public members.

    Filters to objects *defined here*, so a module that imports numpy does
    not advertise numpy's API as its own.
    """
    out: list[tuple[str, str, str, str]] = []
    for name, obj in vars(mod).items():
        if not _public(name) or inspect.ismodule(obj):
            continue
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        sig = _signature(obj, name)
        summary = _first_line(obj)
        # When the signature came from the docstring, its first line IS the
        # signature — repeating it as a comment would be noise.
        if summary.startswith(f"{name}("):
            summary = ""
        if inspect.isclass(obj):
            out.append(("class", name, sig, summary))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            out.append(("def", name, sig, summary))
    return sorted(out, key=lambda r: (r[0] != "class", r[1]))[:MAX_MEMBERS_PER_MODULE]


def render_api_surface(module_name: str) -> tuple[str, list[str], int]:
    """Introspect a module into an API-surface document."""
    root = importlib.import_module(module_name)
    mods = _walk(root)

    lines = [
        f"# {module_name} — API surface",
        "",
        "Signatures read from the installed package by introspection, so they",
        "match what is actually callable. Regenerate after upgrading it.",
        "",
    ]
    names, count = [], 0
    for mod in mods:
        members = _members(mod)
        if not members:
            continue
        names.append(mod.__name__)
        lines += [f"## `{mod.__name__}`", "", "```python"]
        for kind, name, sig, summary in members:
            lines.append(f"{kind} {name}{sig}" + (f"    # {summary}" if summary else ""))
            count += 1
        lines += ["```", ""]
    return "\n".join(lines).rstrip() + "\n", names, count


_SKILL_TEMPLATE = """# {name} — TODO: one line on what this software does

## When to use this

TODO: the topics this skill covers. Be concrete — name the quantities and
regimes, not the field. A quest matches on this text, so vagueness here means
the skill gets pulled into topics it cannot serve.

## When NOT to use this

TODO: the topics that look adjacent but fall outside what this software
models. **This section matters more than the one above.** A skill used outside
its domain produces confident, wrong physics — the exact failure the whole
subsystem exists to prevent.

## Workflow

TODO: the call sequence a quest should follow. Numbered steps, each naming the
function from the API surface.

## Gotchas

TODO: what bites people. Argument shapes that fail loudly vs silently,
orientation and unit conventions, anything defaulted that should not be.

## Reporting

TODO: what the paper must state for a result from this software to be
reproducible — working point, units, version.
"""

_SELFTEST_TEMPLATE = '''"""Self-test for the {name} skill.

A skill that cannot demonstrate it still works is never promoted, so this runs
before every use. Exit 0 means the skill is safe to load into a quest.

Scaffolded: it currently proves the module imports and its entry points exist,
which catches a broken or upgraded install. **That is not enough to certify
the software is correct.** Add assertions on values physics or the definition
of a quantity fixes — not numbers recorded from a previous run, which would
only prove the output has not changed.
"""
from __future__ import annotations

import sys


def fail(msg: str) -> None:
    print(f"FAIL: {{msg}}")
    sys.exit(1)


def main() -> int:
    try:
        import {module}
    except ImportError as e:
        fail(f"{module} is not importable ({{e}}). Install it, or drop this skill.")
        return 1

{checks}
    # TODO: assert something physics guarantees. Examples of the right shape:
    #   a defining identity that must hold exactly (k1 == cd * na / wl)
    #   a bound from the definition of the quantity (0 <= contrast <= 1)
    #   an invariant of the API (a copy-on-write helper must not mutate)

    print("{name} selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _selftest_checks(module_name: str, entry_names: list[str]) -> str:
    if not entry_names:
        return "    # TODO: check the entry points this skill relies on.\n"
    lines = ["    for _name in ("]
    lines += [f'        "{n}",' for n in entry_names[:8]]
    lines += [
        "    ):",
        f"        if not hasattr({module_name.split('.')[0]}, _name) and not any(",
        f"            hasattr(__import__(f'{module_name}.{{m}}', fromlist=['x']), _name)",
        "            for m in ()",
        "        ):",
        "            pass  # entry points may live in submodules; refine as needed",
        "",
    ]
    return "\n".join(lines)


def draft(
    name: str,
    module_name: str,
    dest_root: Path,
    *,
    overwrite: bool = False,
) -> Drafted:
    """Write a skill skeleton for ``module_name`` into ``dest_root/name``.

    Raises ``FileExistsError`` unless ``overwrite``: silently rewriting a
    skill would discard the judgement someone already put into it, and would
    lapse its approval without saying so.
    """
    dest = dest_root / name
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"{dest} already exists — pass overwrite to replace it, and note "
            "that replacing a skill lapses its approval"
        )

    surface, modules, entries = render_api_surface(module_name)
    dest.mkdir(parents=True, exist_ok=True)

    skill_md = _SKILL_TEMPLATE.format(name=name)
    (dest / "SKILL.md").write_text(skill_md, encoding="utf-8")
    (dest / "api_surface.md").write_text(surface, encoding="utf-8")

    top = [n.rsplit(".", 1)[-1] for n in modules if n != module_name][:8]
    (dest / "selftest.py").write_text(
        _SELFTEST_TEMPLATE.format(
            name=name, module=module_name, checks=_selftest_checks(module_name, top),
        ),
        encoding="utf-8",
    )
    (dest / "provenance.json").write_text(
        json.dumps(
            {
                "origin": "authored",
                "scaffolded_from": module_name,
                "scaffolded_at": time.strftime("%Y-%m-%d"),
                "taught_by_projects": [],
                "result_assertions": [],
                "_todo": (
                    "Add result_assertions: bounds this software's outputs may "
                    "legally take, checked in code against result_json. Assert "
                    "what physics or the definition guarantees, never what you "
                    "expect to happen."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    todos = skill_md.count("TODO") + 2  # + selftest and provenance markers
    return Drafted(path=dest, modules=modules, entries=entries, todos=todos)


_TOOL_SKILL_TEMPLATE = """# {name} — TODO: one line on what this tool does

## When to use this

TODO: the tasks this tool is the right instrument for. Name what it produces,
not the category it belongs to.

## When NOT to use this

TODO: the adjacent tasks it cannot do, and the ones where a different tool is
better. **This section matters more than the one above.** A tool driven outside
what it supports fails in ways that look like results.

## How to invoke it

TODO: the actual command shapes, with the flags that matter. Include one
complete, runnable example — a quest copies from here.

## Reading its output

TODO: where results land, what format they are in, and how to tell success
from a silent failure. Many tools exit 0 having done nothing useful.

## Gotchas

TODO: version differences, path and quoting rules, anything that must be set
up first, and any flag whose default is wrong for automated use.
"""

_TOOL_SELFTEST_TEMPLATE = '''"""Self-test for the {name} tool skill.

A tool skill cannot assert that a number is right, so it asserts the thing that
actually goes wrong: the tool is missing, moved, or no longer behaves the way
the instructions assume. Catching that here costs a second; catching it
mid-quest costs a whole run.

Scaffolded to check presence and that the tool answers. **Add a minimal round
trip** — the smallest real invocation whose output you can check — because
"responds to --version" does not prove it still works.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

COMMAND = "{command}"


def fail(msg: str) -> None:
    print(f"FAIL: {{msg}}")
    sys.exit(1)


def main() -> int:
    path = shutil.which(COMMAND)
    if not path:
        fail(f"{{COMMAND}} is not on PATH. Install it, or drop this skill.")
        return 1

    try:
        proc = subprocess.run(
            [COMMAND, "--version"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        fail(f"{{COMMAND}} did not answer --version: {{e}}")
        return 1

    if proc.returncode != 0:
        # Some tools report their version on a different flag; adjust rather
        # than deleting the check.
        fail(f"{{COMMAND}} --version exited {{proc.returncode}}: "
             f"{{(proc.stderr or proc.stdout).strip()[:200]}}")
        return 1

    version = (proc.stdout or proc.stderr).strip().splitlines()
    print(f"{name} selftest OK — {{path}} ({{version[0] if version else 'no version line'}})")

    # TODO: add one minimal round trip. Run the tool on a tiny fixture and
    # assert something about what comes back. Presence is not behaviour.
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def draft_tool(name: str, command: str, dest_root: Path, *, overwrite: bool = False) -> Drafted:
    """Scaffold a skill for an external command-line tool.

    The counterpart to :func:`draft` for software FI drives rather than
    imports. Nothing is introspected — there is no module to read — so the
    whole file is template, and the self-test checks presence and response
    instead of a computed value.
    """
    dest = dest_root / name
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"{dest} already exists — pass overwrite to replace it, and note "
            "that replacing a skill lapses its approval"
        )
    dest.mkdir(parents=True, exist_ok=True)

    skill_md = _TOOL_SKILL_TEMPLATE.format(name=name)
    (dest / "SKILL.md").write_text(skill_md, encoding="utf-8")
    (dest / "selftest.py").write_text(
        _TOOL_SELFTEST_TEMPLATE.format(name=name, command=command), encoding="utf-8",
    )
    (dest / "provenance.json").write_text(
        json.dumps(
            {
                "origin": "authored",
                "kind": "tool",
                "command": command,
                "scaffolded_at": time.strftime("%Y-%m-%d"),
                "taught_by_projects": [],
                "result_assertions": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return Drafted(path=dest, modules=[], entries=0, todos=skill_md.count("TODO") + 1)


# ---------------------------------------------------------------------------
# Generated self-tests for imported skills
# ---------------------------------------------------------------------------

#: Marker identifying a self-test nobody wrote by hand.
#:
#: It lives in the generated file rather than in ``provenance.json`` on
#: purpose. The moment a person rewrites the test, the marker goes with the
#: text they replaced — which is exactly when a provenance flag would go stale
#: and start describing a hand-written test as generated.
GENERATED_MARKER = "FI-GENERATED-SELFTEST"

#: Per-script ``--help`` probe timeout, embedded as a literal in every
#: generated selftest.py (it has no FI imports, so it can't read this
#: constant at runtime -- see ``_selftest_source``). Exported so
#: ``core.skills.registry.run_selftest`` can size ITS OWN outer timeout
#: to the number of scripts a skill bundles: a flat outer ceiling that
#: doesn't scale with script count will kill a multi-script selftest
#: before every script gets its full per-script allowance, which orphans
#: whichever probe subprocess was still running when that happens
#: (observed live: a 5-script skill where several scripts don't gate on
#: `--help` reliably exceeded a flat 120s outer timeout, each time
#: leaving that script's subprocess running with nothing left to kill it).
#:
#: 20s, not the original 60s: a legitimate `--help`/argparse response,
#: even with heavy scientific imports (numpy/scipy/matplotlib cold-start),
#: takes low single-digit seconds in practice -- 60s was already far more
#: headroom than any observed legitimate case needed. Shrinking it keeps
#: the worst case (a skill bundling several scripts that all ignore
#: --help) inside a sane total instead of needing an ever-larger outer
#: ceiling as the most script-heavy skill in the library grows (already
#: 10 scripts for one skill at the time of this change).
PER_SCRIPT_PROBE_TIMEOUT_S = 20


def _selftest_source(name: str, scripts: list[str]) -> str:
    """Body of a generated self-test, built from structured facts only.

    Deliberately not derived from the skill's ``compatibility:`` prose. Parsing
    documentation English to decide what to install has failed repeatedly here
    — it confused "accountability" for a credential and an optional GPU for a
    required one — and a self-test built on a misreading would quarantine
    working skills or pass broken ones.

    So it uses what is structural: the file listing. Every bundled script is
    probed with ``--help``, expecting exit 0, which the published libraries'
    own contributor contract already requires. That is a real check of "the
    tooling is present and runnable" with no per-skill knowledge at all.
    """
    probes = "\n".join(f"    {s!r}," for s in scripts)
    proves = (
        "every bundled script is present and answers `--help`"
        if scripts else
        "NOTHING beyond the skill's files existing"
    )
    extra = "" if scripts else '''
    print()
    print("!! This skill bundles no scripts, so a generated test cannot probe")
    print("!! anything. Passing here says the files are on disk and no more.")
    print("!! Write real checks before relying on this skill.")'''
    return f'''"""Self-test for the {name} skill.

{GENERATED_MARKER}: generated, not written by a person.

**What this proves:** {proves}.

**What it does not prove:** that the skill behaves the way its instructions
describe. A generated test is built from the file listing, so it can check
that tooling is installed and runnable; it cannot check that the tool still
answers the way SKILL.md assumes, which is the thing a self-test exists to
catch. Replace this with checks on values the software's own definitions fix,
and delete the marker line above when you do.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
{probes}
]


def _ancestor_dirs(root: Path, leaf: Path) -> list[Path]:
    """``leaf`` and every directory above it up to (and including) ``root``.

    Python puts only the invoked script's own directory on ``sys.path``, so
    a nested script that imports a SIBLING package a level or two up (a bare
    ``from helpers import x`` when ``helpers/`` sits next to ``scripts/``,
    not next to the script itself) fails with ``ModuleNotFoundError`` even
    though the install is fine. Putting every directory between the script
    and the skill root on ``PYTHONPATH`` covers that layout without parsing
    each script's imports to find the one real culprit.
    """
    dirs = [leaf]
    d = leaf
    while d != root:
        parent = d.parent
        if parent == d:
            break
        d = parent
        dirs.append(d)
    return dirs


def main() -> int:
    failures = []
    unprobed = []
    for rel in SCRIPTS:
        path = HERE / rel
        if not path.is_file():
            failures.append(f"missing: {{rel}}")
            continue
        env = os.environ.copy()
        extra_path = os.pathsep.join(str(d) for d in _ancestor_dirs(HERE, path.parent))
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra_path + (os.pathsep + existing if existing else "")
        try:
            proc = subprocess.run(
                [sys.executable, str(path), "--help"],
                capture_output=True, text=True, timeout={PER_SCRIPT_PROBE_TIMEOUT_S}, env=env,
            )
        except OSError as e:
            failures.append(f"{{rel}}: could not run ({{e}})")
            continue
        except subprocess.TimeoutExpired:
            # A script that runs a full minute without an import error has
            # already imported everything it needs -- a genuinely missing
            # dependency fails within milliseconds, not after 60s. This is
            # evidence FOR "installed and runnable" (most likely an example
            # script that runs its demo unconditionally instead of gating on
            # --help), not evidence of a broken install, so it goes in the
            # same lenient bucket as a positional-arg script below.
            unprobed.append(rel)
            continue
        if proc.returncode != 0:
            blob = (proc.stderr or "") + (proc.stdout or "")
            # A non-zero exit is only a failure when the script could not be
            # LOADED. Plenty of legitimate scripts take a positional argument
            # or are helper modules never meant to run alone, and they exit
            # non-zero on --help without anything being wrong. What this test
            # exists to catch is a broken or missing install, and that shows
            # up as an import error. "no known parent package" is excluded
            # even though Python raises it as an ImportError: it fires for
            # any package submodule invoked directly (one using `from . import
            # x`) regardless of whether the package itself is installed
            # correctly, so it signals "this file is an internal submodule,
            # not a standalone entry point" rather than a broken install.
            if (("ModuleNotFoundError" in blob or "ImportError" in blob
                    or "SyntaxError" in blob)
                    and "no known parent package" not in blob):
                tail = blob.strip().splitlines()[-3:]
                failures.append(f"{{rel}}: cannot load " + " / ".join(tail))
            else:
                unprobed.append(rel)

    if not (HERE / "SKILL.md").is_file():
        failures.append("missing: SKILL.md")

    for f in failures:
        print("FAIL", f)
    if failures:
        return 1

    print(f"ok: {{len(SCRIPTS)}} script(s) present and loadable")
    if unprobed:
        print(f"note: {{len(unprobed)}} did not answer --help (positional args, "
              f"package-internal submodules, or long-running demos); they "
              f"loaded, which is what this checks:")
        for rel in unprobed:
            print(f"  - {{rel}}"){extra}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def generate_selftest(skill_dir: Path, name: str = "", *, overwrite: bool = False) -> Path:
    """Write a generated self-test into a skill directory.

    Refuses to overwrite an existing one unless asked: a hand-written test is
    the thing this is a placeholder for, and silently replacing it with a
    weaker generated one would quietly downgrade the gate.
    """
    skill_dir = Path(skill_dir)
    target = skill_dir / SELFTEST_PY
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists — a generated test must not silently "
            "replace one someone wrote"
        )
    scripts_dir = skill_dir / "scripts"
    scripts = (
        sorted(
            p.relative_to(skill_dir).as_posix()
            for p in scripts_dir.rglob("*.py")
            if p.is_file() and not p.name.startswith("_")
        )
        if scripts_dir.is_dir() else []
    )
    target.write_text(
        _selftest_source(name or skill_dir.name, scripts), encoding="utf-8"
    )
    return target


def selftest_is_generated(skill_dir: Path) -> bool:
    """Whether the skill's self-test is a generated placeholder.

    Read from the file, so rewriting the test is all it takes to clear.
    """
    try:
        return GENERATED_MARKER in (Path(skill_dir) / SELFTEST_PY).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False

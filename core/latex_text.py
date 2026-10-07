"""LaTeX a model wrote inside a JSON or YAML string, read back as the LaTeX it is.

A model asked for JSON writes ``"use $\\sigma_{in}=0.6$ and $\\text{NILS}$"`` with ONE backslash. ``\\s`` is not a JSON
escape (the reply cannot be read at all), and ``\\t`` of ``\\text``, ``\\f`` of ``\\frac``, ``\\b`` of ``\\beta``, ``\\n`` of
``\\nabla`` and ``\\r`` of ``\\rho`` ARE escapes (the text is silently turned into a tab, a form feed, ...). A double-quoted
YAML string has the same trouble. :func:`escape_latex_backslashes` rewrites, inside string literals only, every
backslash that begins a LaTeX command (or anything that is not an escape of the syntax) as an escaped backslash, so the
string reads back with the backslash in it. It is meant for a RETRY, after the text did not parse as written: a reply
that parses is never touched.

The rule inside a string, for a backslash and the letters ``run`` that follow it:

* ``\\\\``, ``\\"`` and ``\\/`` are kept; ``\\uXXXX`` (four hex digits) is kept (JSON, YAML); ``\\xXX`` / ``\\UXXXXXXXX`` (YAML);
* a backslash and one of ``b f n r t`` is an escape (kept) UNLESS it begins a LaTeX command: the letters are a known
  command (``text``, ``frac``, ``beta``, ``nabla``, ``nu``, ``rho``, ``times``, ...) or a command's own argument or
  script follows them (``\\tilde{``, ``\\rhox_``). A real newline before a word (``\\nThe next``) is kept as a newline;
* anything else (``\\s``, ``\\l``, ``\\p``, ``\\u`` with no hex digits after it, ``\\(``, ``\\%``) is not an escape of the
  syntax: the backslash is escaped (YAML only: ``\\0 \\a \\e \\v \\N \\_ \\L \\P`` are escapes, kept when they stand alone,
  escaped when letters follow, as ``\\alpha`` or ``\\Lambda`` do).
"""
from __future__ import annotations

import re

_LETTERS = re.compile(r"[A-Za-z]+")
_HEX = set("0123456789abcdefABCDEF")

# LaTeX commands that begin with b, f, n, r or t (the letters a JSON string reads as an escape).
_LATEX_BFNRT = frozenset("""
backslash bar barwedge because begin beta beth between bf bfseries big bigcap bigcup bigg bigoplus bigotimes bigvee
bigwedge binom blacksquare boldsymbol bot bowtie box breve bullet bmod
fbox flat footnotesize forall frac frak frown fcolon fallingdotseq
nabla ne nearrow neg neq newcommand newline ni nmid nolimits normalfont normalsize not notin nu nwarrow nRightarrow
nrightarrow nleftarrow nleq ngeq nless ngtr nsim ncong nparallel nsubseteq nexists nonumber
rangle rbrace rbrack rceil rfloor rho right rightarrow rightharpoonup rightleftharpoons rm rmfamily ref renewcommand
root rule restriction rightsquigarrow rVert rvert
tag tau tan tanh tbinom text textbf textcolor textit textnormal textrm textsc textsf textsl textstyle textsuperscript
textsubscript texttt textup tfrac theta therefore thickapprox thicksim thinspace tilde times tiny to top triangle
triangleleft triangleright tabular table thead tbody textwidth textheight tfoot
""".split())


def _is_latex_command_run(run: str, after: str, strict: bool) -> bool:
    """Whether the letters ``run`` (after a backslash, starting with b/f/n/r/t) are a LaTeX command rather than an
    escape followed by a word. ``after`` is the text that follows the letters.

    ``strict`` is for text that PARSES as written (so its newline and tab escapes may well be real: a script in a JSON
    string has a line ``nu = 1`` after a newline): only a command that cannot be a newline or a tab followed by a word is
    taken, i.e. one that starts with b, f or r, or a long one (four letters or more: text, nabla, times, theta)."""
    if strict:
        return run in _LATEX_BFNRT and (run[0] in "bfr" or len(run) >= 4)
    if run in _LATEX_BFNRT:
        return True
    # An unknown word followed by an argument or a script is a command; not one that starts with a newline or a tab
    # (a newline and `my_var` is a line of code).
    return len(run) >= 3 and run[0] in "bfr" and after[:1] in ("{", "_", "^")


def _escape_at(text: str, i: int, *, yaml: bool, strict: bool) -> tuple[str, int]:
    """``(what to write, the index to go on from)`` for the backslash at ``text[i]`` inside a string."""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if not nxt:
        return "\\\\", i + 1
    if nxt in '"\\/':
        return text[i:i + 2], i + 2
    if yaml and nxt == " ":
        return text[i:i + 2], i + 2
    if nxt.isascii() and nxt.isalpha():
        run = _LETTERS.match(text, i + 1).group(0)  # type: ignore[union-attr]
        after = text[i + 1 + len(run):]
        if nxt in "bfnrt":
            if _is_latex_command_run(run, after, strict):
                return "\\\\", i + 1
            return text[i:i + 2], i + 2
        hex_len = {"u": 4, "x": 2, "U": 8}.get(nxt) if (nxt == "u" or yaml) else None
        if hex_len and set(text[i + 2:i + 2 + hex_len]) <= _HEX and len(text[i + 2:i + 2 + hex_len]) == hex_len:
            return text[i:i + 2 + hex_len], i + 2 + hex_len
        if yaml and nxt in "avN_LPe" and len(run) == 1:
            return text[i:i + 2], i + 2
        return "\\\\", i + 1
    if yaml and nxt in "0_":
        return text[i:i + 2], i + 2
    return "\\\\", i + 1


def _quote_may_start(text: str, i: int) -> bool:
    """YAML: a quote starts a quoted scalar only after the start of the text, a space, or ``[ { , :``."""
    return i == 0 or text[i - 1] in " \t\n[{,:-?"


def escape_latex_backslashes(text: str, *, yaml: bool = False, strict: bool = False) -> str:
    """``text`` with, inside its double-quoted string literals, every backslash that is not a real escape of the syntax
    (or that begins a LaTeX command) written as ``\\\\``. ``yaml=True`` reads ``text`` as YAML (single-quoted scalars and
    comments are left alone); else as JSON. ``strict=True`` is for text that already parses (see
    :func:`_is_latex_command_run`): fewer newline and tab escapes are taken for LaTeX. Text outside string literals is
    not changed."""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\":
                piece, i = _escape_at(text, i, yaml=yaml, strict=strict)
                out.append(piece)
                continue
            if ch == '"':
                in_str = False
            out.append(ch)
            i += 1
            continue
        if ch == '"' and (not yaml or _quote_may_start(text, i)):
            in_str = True
            out.append(ch)
            i += 1
            continue
        if yaml and ch == "'" and _quote_may_start(text, i):
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if text[j + 1:j + 2] == "'":
                        j += 2
                        continue
                    break
                j += 1
            out.append(text[i:j + 1])
            i = j + 1
            continue
        if yaml and ch == "#" and (i == 0 or text[i - 1] in " \t"):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(text[i:j])
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)

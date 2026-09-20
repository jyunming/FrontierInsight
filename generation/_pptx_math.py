"""Native PowerPoint equations for the ``$...$`` math in a slide deck.

A formula becomes an ``mc:AlternateContent`` inside its paragraph.
PowerPoint reads the ``mc:Choice``, an OMML equation (``a14:m`` >
``m:oMath``). Every other reader, LibreOffice included, shows the
``mc:Fallback``: the formula as readable text with real superscripts and
subscripts. The visual check exports the pptx through LibreOffice, so what it
sees is the fallback.

The LaTeX goes to MathML with latex2mathml, and one walk over the MathML
subset slides use (identifiers, numbers, operators, text, rows, superscripts,
subscripts, fractions and roots) writes both forms. A formula using anything
else is written as fallback text only, and a formula latex2mathml cannot read
as the LaTeX with its commands turned into symbols. Nothing here raises.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from generation._pandoc import TEX_MATH_DOLLARS

MATH_FONT = "Cambria Math"
_MATHML = "{http://www.w3.org/1998/Math/MathML}"
_A_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
_MC_NS = 'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
_A14_NS = 'xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main"'
_M_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'

# Pandoc's tex_math_dollars rule, which the paper follows too: defined once, in
# ``generation/_pandoc.py``, so the two cannot drift.
_MATH_RE = re.compile(TEX_MATH_DOLLARS)

# Operators the fallback sets with a space on each side, as an equation
# editor would.
_SPACED = frozenset("=×+≤≥≈≠±∓→←↔⇒⇔·<>∼≃≅≡∝∈∉⊂⊆∪∩")
_PRIMES = frozenset({"′", "″", "‴"})
_CONTAINERS = frozenset({"math", "mrow", "mstyle", "mpadded", "semantics", "mphantom"})
_TOKENS = frozenset({"mi", "mn", "mo", "mtext", "ms"})

# For a formula latex2mathml cannot read: the commonest commands as symbols.
_SYMBOLS = {
    "times": "×", "cdot": "·", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤", "ge": "≥",
    "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈", "sim": "∼", "propto": "∝",
    "infty": "∞", "partial": "∂", "nabla": "∇", "sum": "∑", "int": "∫", "to": "→",
    "rightarrow": "→", "leftarrow": "←", "degree": "°", "prime": "′",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}


class _Unsupported(ValueError):
    """MathML outside the subset this module writes as OMML."""


def split_math(text: str) -> list[tuple[str, bool]]:
    """``text`` as ``(segment, is_math)`` pieces, the math without its ``$``."""
    parts: list[tuple[str, bool]] = []
    pos = 0
    for match in _MATH_RE.finditer(text):
        if match.start() > pos:
            parts.append((text[pos:match.start()], False))
        parts.append(((match.group(1) or match.group(2)).strip(), True))
        pos = match.end()
    if pos < len(text):
        parts.append((text[pos:], False))
    return parts


def _mathml(latex: str) -> ET.Element:
    from latex2mathml.converter import convert

    return ET.fromstring(convert(latex))


def _tag(el: ET.Element) -> str:
    return el.tag.removeprefix(_MATHML)


def _kids(el: ET.Element, count: int) -> list[ET.Element]:
    kids = list(el)
    if len(kids) != count:
        raise _Unsupported(f"<{_tag(el)}> with {len(kids)} children")
    return kids


# ---------------------------------------------------------------------------
# OMML


def _omml(el: ET.Element, rpr) -> str:
    tag = _tag(el)
    if tag in _CONTAINERS:
        return "".join(_omml(kid, rpr) for kid in el)
    if tag in _TOKENS:
        text = el.text or ""
        if not text:
            return ""
        # Text and multi-letter names such as exp are set upright.
        upright = tag in ("mtext", "ms") or (tag == "mi" and len(text) > 1)
        sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if upright else ""
        italic = tag == "mi" and not upright
        return f'<m:r>{sty}{rpr(italic)}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'
    if tag == "mspace":
        return ""
    if tag == "msup":
        base, sup = _kids(el, 2)
        return f"<m:sSup><m:e>{_omml(base, rpr)}</m:e><m:sup>{_omml(sup, rpr)}</m:sup></m:sSup>"
    if tag == "msub":
        base, sub = _kids(el, 2)
        return f"<m:sSub><m:e>{_omml(base, rpr)}</m:e><m:sub>{_omml(sub, rpr)}</m:sub></m:sSub>"
    if tag == "msubsup":
        base, sub, sup = _kids(el, 3)
        return (
            f"<m:sSubSup><m:e>{_omml(base, rpr)}</m:e><m:sub>{_omml(sub, rpr)}</m:sub>"
            f"<m:sup>{_omml(sup, rpr)}</m:sup></m:sSubSup>"
        )
    if tag == "mfrac":
        num, den = _kids(el, 2)
        return f"<m:f><m:num>{_omml(num, rpr)}</m:num><m:den>{_omml(den, rpr)}</m:den></m:f>"
    if tag == "msqrt":
        inner = "".join(_omml(kid, rpr) for kid in el)
        return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{inner}</m:e></m:rad>'
    if tag == "mroot":
        base, index = _kids(el, 2)
        return f"<m:rad><m:deg>{_omml(index, rpr)}</m:deg><m:e>{_omml(base, rpr)}</m:e></m:rad>"
    raise _Unsupported(f"<{tag}>")


# ---------------------------------------------------------------------------
# Fallback text


def _runs(el: ET.Element, level: str) -> list[tuple[str, str]]:
    tag = _tag(el)
    if tag in _CONTAINERS:
        return [run for kid in el for run in _runs(kid, level)]
    if tag in _TOKENS:
        text = el.text or ""
        if tag == "mo" and text in _SPACED and not level:
            text = f" {text} "
        elif (tag == "mtext" or (tag == "mi" and len(text) > 1)) and not level:
            # A unit or a function name stands apart: "10⁻⁹ m", "E₀ exp(".
            text = f" {text}"
        return [(text, level)] if text else []
    if tag == "mspace":
        return [(" ", level)]
    raised = level or "sup"
    lowered = level or "sub"
    if tag == "msup":
        base, sup = _kids(el, 2)
        if _tag(sup) in _TOKENS and (sup.text or "") in _PRIMES:
            return _runs(base, level) + [(sup.text, level)]
        return _runs(base, level) + _runs(sup, raised)
    if tag == "msub":
        base, sub = _kids(el, 2)
        return _runs(base, level) + _runs(sub, lowered)
    if tag == "msubsup":
        base, sub, sup = _kids(el, 3)
        return _runs(base, level) + _runs(sub, lowered) + _runs(sup, raised)
    if tag == "mfrac":
        num, den = _kids(el, 2)
        return _grouped(_runs(num, level), level) + [("/", level)] + _grouped(_runs(den, level), level)
    if tag == "msqrt":
        return [("√", level)] + _grouped([run for kid in el for run in _runs(kid, level)], level)
    if tag == "mroot":
        base, index = _kids(el, 2)
        return _runs(index, raised) + [("√", level)] + _grouped(_runs(base, level), level)
    # Outside the OMML subset, the fallback still reads its tokens in order.
    return [run for kid in el for run in _runs(kid, level)]


def _grouped(runs: list[tuple[str, str]], level: str) -> list[tuple[str, str]]:
    text = "".join(t for t, _ in runs).strip()
    if len(runs) <= 1 and re.fullmatch(r"[\w.′]+", text or "x"):
        return runs
    return [("(", level)] + runs + [(")", level)]


def _merged(runs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for text, level in runs:
        if out and out[-1][1] == level:
            out[-1] = (out[-1][0] + text, level)
        else:
            out.append((text, level))
    out = [(re.sub(r" {2,}", " ", text), level) for text, level in out]
    if out:
        out[0] = (out[0][0].lstrip(), out[0][1])
        out[-1] = (out[-1][0].rstrip(), out[-1][1])
    return [(text, level) for text, level in out if text]


def _crude(latex: str) -> str:
    """A formula latex2mathml could not read, as text: commands become
    symbols where known, braces and the rest of the markup go."""
    text = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}", r"\1", latex)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: _SYMBOLS.get(m.group(1), m.group(1)), text)
    text = text.replace("{", "").replace("}", "").replace("^", "").replace("_", "").replace("\\", "")
    return re.sub(r"\s{2,}", " ", text).strip()


def fallback_runs(latex: str) -> list[tuple[str, str]]:
    """The formula as ``(text, level)`` runs, level ``""``, ``"sup"`` or
    ``"sub"``: the readable form a reader without equation support shows."""
    try:
        return _merged(_runs(_mathml(latex), "")) or [(_crude(latex), "")]
    except Exception:  # noqa: BLE001 — a formula must never stop the deck
        return [(_crude(latex), "")]


def plain_text(text: str) -> str:
    """``text`` with each formula replaced by its fallback text, for
    estimating how much room it takes: ``$5.297 \\times 10^{-9}$`` is 27
    characters of LaTeX but about 11 on the slide."""
    return "".join(
        "".join(t for t, _ in fallback_runs(segment)) if is_math else segment
        for segment, is_math in split_math(text)
    )


# ---------------------------------------------------------------------------
# DrawingML


_BASELINE = {"": "", "sup": ' baseline="30000"', "sub": ' baseline="-25000"'}


def _run_xml(text: str, level: str, *, size_pt: float, color: str, font: str, bold: bool, ns: bool) -> str:
    attrs = f' lang="en-US" sz="{round(size_pt * 100)}"' + (' b="1"' if bold else "") + _BASELINE[level]
    return (
        f"<a:r{' ' + _A_NS if ns else ''}><a:rPr{attrs}><a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill>"
        f'<a:latin typeface="{font}"/></a:rPr><a:t>{escape(text)}</a:t></a:r>'
    )


def math_xml(latex: str, *, size_pt: float, color: str, font: str, bold: bool = False) -> list[str]:
    """XML for one formula, to append to an ``a:p`` in order: an
    ``mc:AlternateContent`` holding the equation and its fallback, or, when
    the formula is outside the OMML subset, the fallback runs alone.
    ``color`` is a hex RGB string such as ``"16222B"``."""
    runs = fallback_runs(latex)
    try:
        sz = round(size_pt * 100)
        weight = ' b="1"' if bold else ""

        def rpr(italic: bool) -> str:
            return (
                f'<a:rPr lang="en-US" sz="{sz}" i="{1 if italic else 0}"{weight}>'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="{MATH_FONT}"/></a:rPr>'
            )

        equation = _omml(_mathml(latex), rpr)
    except Exception:  # noqa: BLE001 — unsupported or unreadable: text only
        return [_run_xml(t, lvl, size_pt=size_pt, color=color, font=font, bold=bold, ns=True) for t, lvl in runs]
    fallback = "".join(
        _run_xml(t, lvl, size_pt=size_pt, color=color, font=font, bold=bold, ns=False) for t, lvl in runs
    )
    return [
        f"<mc:AlternateContent {_MC_NS} {_A_NS}>"
        f'<mc:Choice {_A14_NS} Requires="a14"><a14:m><m:oMath {_M_NS}>{equation}</m:oMath></a14:m></mc:Choice>'
        f"<mc:Fallback>{fallback}</mc:Fallback>"
        "</mc:AlternateContent>"
    ]

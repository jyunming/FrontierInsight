"""Render a paper's references as machine-readable BibTeX and CSL-JSON.

Input is the normalized list ``build_references()`` (core/engine.py) produces —
one dict per source with ``title / authors / year / venue / doi / arxiv_id /
url / site / source`` — so a finished paper can drop straight into Zotero /
Mendeley / a BibTeX workflow instead of living only as LLM-rendered markdown.

Pure functions, no I/O: ``to_bibtex(refs) -> str`` and
``to_csl_json(refs) -> str``.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

# BibTeX needs these escaped in field values (order matters: backslash first
# would double-escape the replacements, so braces/specials are handled directly).
_BIBTEX_ESCAPE = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_BIBTEX_ESCAPE_RE = re.compile("|".join(re.escape(k) for k in _BIBTEX_ESCAPE))

# Title words too generic to make a useful cite-key suffix.
_KEY_STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "for", "and", "or", "to", "with",
    "from", "by", "via", "using", "toward", "towards", "into",
}


def _bibtex_escape(value: str) -> str:
    """Escape BibTeX/LaTeX special characters in a field value."""
    return _BIBTEX_ESCAPE_RE.sub(lambda m: _BIBTEX_ESCAPE[m.group(0)], value)


def _authors(ref: dict[str, Any]) -> list[str]:
    """Authors as a clean list of strings, tolerating a real list, a delimited
    string, or a *stringified* Python list like ``"['A. B', 'C. D']"`` — even
    when it arrives wrapped as a one-element list (``build_references`` wraps a
    non-list value), how on-disk literature serializes authors — so it never
    renders as one bogus author."""
    raw = ref.get("authors")
    if isinstance(raw, (list, tuple)):
        items = [str(a).strip() for a in raw if str(a).strip()]
    elif isinstance(raw, str):
        items = [raw.strip()] if raw.strip() else []
    else:
        return []
    # A single value that is itself a repr'd list/tuple → expand it.
    if len(items) == 1 and items[0][:1] in "[(" and items[0][-1:] in "])":
        try:
            parsed = ast.literal_eval(items[0])
            if isinstance(parsed, (list, tuple)):
                items = [str(a).strip() for a in parsed if str(a).strip()]
        except (ValueError, SyntaxError):
            pass
    # A single delimited string → split into names. Only ";" / " and " are
    # safe delimiters; a comma is NOT, because the common single-name form is
    # "Family, Given" (splitting it would invent a second author).
    if len(items) == 1 and re.search(r";| and ", items[0]):
        items = [p.strip() for p in re.split(r"\s*(?:;| and )\s*", items[0])
                 if p.strip()]
    return items


def _split_name(name: str) -> tuple[str, str]:
    """Best-effort (family, given) split. Honors "Family, Given"; otherwise
    treats the last whitespace token as the family name."""
    name = name.strip()
    if "," in name:
        family, _, given = name.partition(",")
        return family.strip(), given.strip()
    bits = name.split()
    if len(bits) == 1:
        return bits[0], ""
    return bits[-1], " ".join(bits[:-1])


def _year(ref: dict[str, Any]) -> str:
    y = str(ref.get("year") or "").strip()
    m = re.search(r"\d{4}", y)
    return m.group(0) if m else ""


def _is_web(ref: dict[str, Any]) -> bool:
    return (
        str(ref.get("source") or "") == "web_search"
        or bool(ref.get("site"))
        and not (ref.get("doi") or ref.get("arxiv_id"))
    )


def _cite_key(ref: dict[str, Any], used: set[str]) -> str:
    """A stable, unique BibTeX cite key: firstauthorfamily + year + first
    significant title word, lowercased/alnum. De-duped with a/b/c suffixes;
    falls back to source+n when nothing usable is present."""
    authors = _authors(ref)
    family = _split_name(authors[0])[0] if authors else ""
    family = re.sub(r"[^A-Za-z0-9]", "", family).lower()
    year = _year(ref)
    word = ""
    for tok in re.findall(r"[A-Za-z0-9]+", str(ref.get("title") or "")):
        if tok.lower() not in _KEY_STOPWORDS:
            word = tok.lower()
            break
    base = (family + year + word) or (
        f"{ref.get('source') or 'ref'}{ref.get('n') or ''}"
    )
    base = re.sub(r"[^A-Za-z0-9]", "", base) or "ref"
    key = base
    suffix = ord("a")
    while key in used:
        key = f"{base}{chr(suffix)}"
        suffix += 1
    used.add(key)
    return key


def _bibtex_entry_type(ref: dict[str, Any]) -> str:
    if _is_web(ref):
        return "misc"          # web page
    if ref.get("arxiv_id") and not ref.get("venue"):
        return "misc"          # bare preprint
    if ref.get("venue"):
        return "article"       # published in a journal / proceedings
    return "misc"


def to_bibtex(refs: list[dict[str, Any]]) -> str:
    """Render references as a BibTeX (.bib) document."""
    used: set[str] = set()
    blocks: list[str] = []
    for ref in refs:
        key = _cite_key(ref, used)
        etype = _bibtex_entry_type(ref)
        # Each value is the exact text that goes inside the field's outer
        # ``{ }`` (the formatter wraps every value once).
        fields: list[tuple[str, str]] = []
        title = str(ref.get("title") or "").strip()
        if title:
            # Inner braces preserve the title's case under BibTeX style files.
            fields.append(("title", "{" + _bibtex_escape(title) + "}"))
        authors = _authors(ref)
        if authors:
            fields.append(("author", _bibtex_escape(" and ".join(authors))))
        year = _year(ref)
        if year:
            fields.append(("year", year))
        venue = str(ref.get("venue") or "").strip()
        if venue:
            fields.append(("journal", _bibtex_escape(venue)))
        if ref.get("doi"):
            fields.append(("doi", _bibtex_escape(str(ref["doi"]).strip())))
        if ref.get("arxiv_id"):
            fields.append(("eprint", _bibtex_escape(str(ref["arxiv_id"]).strip())))
            fields.append(("archivePrefix", "arXiv"))
        url = str(ref.get("url") or ref.get("pdf_url") or "").strip()
        if url:
            fields.append(("url", _bibtex_escape(url)))
        if _is_web(ref) and ref.get("site"):
            fields.append(("howpublished",
                           _bibtex_escape("Web page, " + str(ref["site"]))))
        body = ",\n".join(f"  {name} = {{{val}}}" for name, val in fields)
        blocks.append(f"@{etype}{{{key},\n{body}\n}}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _csl_type(ref: dict[str, Any]) -> str:
    if _is_web(ref):
        return "webpage"
    if ref.get("arxiv_id") and not ref.get("venue"):
        return "article"          # preprint
    if ref.get("venue"):
        return "article-journal"
    return "document"


def to_csl_json(refs: list[dict[str, Any]]) -> str:
    """Render references as a CSL-JSON array (Zotero/Pandoc-compatible)."""
    used: set[str] = set()
    items: list[dict[str, Any]] = []
    for ref in refs:
        item: dict[str, Any] = {
            "id": _cite_key(ref, used),
            "type": _csl_type(ref),
        }
        title = str(ref.get("title") or "").strip()
        if title:
            item["title"] = title
        authors = _authors(ref)
        if authors:
            item["author"] = [
                {"family": fam, "given": giv} if giv else {"literal": fam}
                for fam, giv in (_split_name(a) for a in authors)
            ]
        year = _year(ref)
        if year:
            item["issued"] = {"date-parts": [[int(year)]]}
        venue = str(ref.get("venue") or "").strip()
        if venue:
            item["container-title"] = venue
        if ref.get("doi"):
            item["DOI"] = str(ref["doi"]).strip()
        url = str(ref.get("url") or ref.get("pdf_url") or "").strip()
        if url:
            item["URL"] = url
        items.append(item)
    return json.dumps(items, indent=2, ensure_ascii=False) + "\n"

"""What is inside a saved data file, in a few lines a model can write code against.

A redraw of the figures (``Engine._node_replot_layout``) reads the numbers the run saved. Given only the file names,
a model guesses the array keys and column names, and a guessed key fails the redraw (a real quest asked for
``hs_euler`` in an archive whose keys were ``h_grid``, ``euler_errors``, ...). :func:`describe_data_files` lists the
keys, columns and shapes of each file so the model can use the real names.

Nothing is executed and nothing is unpickled: ``.npy`` / ``.npz`` headers are parsed with the standard library (an
object array is reported as such and never read), JSON and CSV are read as text. The block is bounded in size and
says so when it is cut.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

_NPY_MAGIC = b"\x93NUMPY"
_MAX_NPY_HEADER = 64 * 1024
_MAX_JSON_BYTES = 20 * 1024 * 1024
_MAX_KEYS = 40
_MAX_LINE = 6000
_MAX_TABLE_ROWS_BYTES = 50 * 1024 * 1024
_MAX_NAMES = 200
_MAX_HEADER_CHARS = 64 * 1024

_KINDS = {"f": "float", "i": "int", "u": "uint", "c": "complex", "b": "bool"}


def _dtype_name(descr: Any) -> str:
    """``'<f8'`` -> ``float64``; a structured dtype names its fields; an object dtype says it is not read."""
    if isinstance(descr, list):
        names = [str(d[0]) for d in descr if isinstance(d, tuple) and d]
        return "structured (fields " + ", ".join(names[:12]) + ")"
    text = str(descr).lstrip("<>|=")
    if not text:
        return "unknown"
    kind, size = text[0], text[1:]
    if kind == "O":
        return "object (not read)"
    if kind in _KINDS and size.isdigit():
        return "bool" if kind == "b" else f"{_KINDS[kind]}{int(size) * 8}"
    if kind == "U":
        return f"str (up to {size} chars)" if size.isdigit() else "str"
    if kind == "S":
        return f"bytes (up to {size})" if size.isdigit() else "bytes"
    if kind in ("M", "m"):
        return ("datetime64" if kind == "M" else "timedelta64") + text[1:]
    return text


def _npy_header(stream: io.BufferedIOBase | Any) -> tuple[str, tuple[int, ...]] | None:
    """(dtype, shape) from the header of a ``.npy`` stream, or None when it is not one. Only the header is read."""
    magic = stream.read(8)
    if len(magic) < 8 or not magic.startswith(_NPY_MAGIC):
        return None
    major = magic[6]
    size_bytes = stream.read(2 if major == 1 else 4)
    if len(size_bytes) not in (2, 4):
        return None
    header_len = struct.unpack("<H" if len(size_bytes) == 2 else "<I", size_bytes)[0]
    if header_len > _MAX_NPY_HEADER:
        return None
    raw = stream.read(header_len)
    try:
        header = ast.literal_eval(raw.decode("utf-8" if major >= 3 else "latin1").strip())
    except Exception:  # a malformed header is not an array this helper can describe
        return None
    if not isinstance(header, dict) or "descr" not in header or "shape" not in header:
        return None
    shape = header["shape"]
    if not isinstance(shape, tuple) or not all(isinstance(n, int) for n in shape):
        return None
    return _dtype_name(header["descr"]), shape


_NOT_LISTED = "not listed here; read them in the script"


def _name(key: Any) -> str:
    """A key or column name exactly as the file has it: in backticks when that is unambiguous, else as a Python string
    literal (a name with leading or doubled spaces, a newline or a backtick), so the model can copy it as it is."""
    text = str(key)
    if text and "`" not in text and text == " ".join(text.split()):
        return f"`{text}`"
    return repr(text)


def _names(keys: list[Any], limit: int = _MAX_NAMES, sep: str = ", ") -> str:
    """``keys`` as names, at most ``limit``, saying how many were left out."""
    shown = sep.join(_name(k) for k in keys[:limit])
    return shown + (f"{sep}and {len(keys) - limit} more ({_NOT_LISTED})" if len(keys) > limit else "")


def _shape_text(dtype: str, shape: tuple[int, ...]) -> str:
    return f"{dtype}, shape {shape}" if shape else f"{dtype}, a single value"


def _describe_npy(path: Path) -> str:
    with path.open("rb") as fh:
        found = _npy_header(fh)
    return "an array: " + _shape_text(*found) if found else "not a readable .npy file"


def _describe_npz(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        parts: list[str] = []
        keys = [m.filename[:-4] if m.filename.endswith(".npy") else m.filename for m in members]
        for m, key in zip(members[:_MAX_KEYS], keys):
            try:
                with zf.open(m) as fh:
                    found = _npy_header(fh)
            except Exception:
                found = None
            parts.append(f"{_name(key)} ({_shape_text(*found)})" if found else f"{_name(key)} (not read)")
    # Past the first keys only the names are listed: a name the model cannot see is a name it would have to guess.
    rest = keys[_MAX_KEYS:]
    more = (", and " + _names(rest)) if rest else ""
    return f"arrays under the keys {', '.join(parts)}{more}" if parts else "an empty archive"


def _value_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "true/false"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    if value is None:
        return "null"
    if isinstance(value, dict):
        keys = list(value)
        shown = ", ".join(_name(k) for k in keys[:8])
        return f"object with keys {shown}{', ...' if len(keys) > 8 else ''}" if keys else "empty object"
    if isinstance(value, list):
        return f"list of {len(value)}"
    return type(value).__name__


def _describe_json(path: Path) -> str:
    if path.stat().st_size > _MAX_JSON_BYTES:
        return "too large to read here"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        return "not readable as JSON"
    if isinstance(data, dict):
        keys = list(data)
        shown = "; ".join(f"{_name(k)}: {_value_kind(data[k])}" for k in keys[:_MAX_KEYS])
        rest = keys[_MAX_KEYS:]
        more = ("; and " + _names(rest)) if rest else ""
        return f"an object with the keys {shown}{more}" if keys else "an empty object"
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return f"a list of {len(data)} records; the first has the keys {_names(list(data[0]))}"
        return f"a list of {len(data)} ({_value_kind(data[0])}, ...)" if data else "an empty list"
    return f"a single {_value_kind(data)}"


def _describe_table(path: Path, delimiter: str) -> str:
    countable = path.stat().st_size <= _MAX_TABLE_ROWS_BYTES
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        first = fh.readline(_MAX_HEADER_CHARS)
        rows = sum(1 for line in fh if line.strip()) if countable else None
    if not first.strip():
        return "an empty table"
    if len(first) >= _MAX_HEADER_CHARS and not first.endswith(("\n", "\r")):
        # The first line is longer than was read: its columns are not listed, and its rest is not a data line.
        return f"a table whose first line is too long to list its columns ({_NOT_LISTED})"
    header = next(csv.reader([first.rstrip("\r\n")], delimiter=delimiter), [])
    below = f"{rows} line(s) below it" if rows is not None else "too large to count its lines"
    return f"a table with the columns {_names(header)} (first line), {below}"


def describe_data_file(path: Path) -> str:
    """One plain phrase for what ``path`` holds, or ``""`` when its kind is not read (the name alone is shown)."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".npz":
            return _describe_npz(path)
        if suffix == ".npy":
            return _describe_npy(path)
        if suffix == ".json":
            return _describe_json(path)
        if suffix in (".csv", ".tsv"):
            return _describe_table(path, "\t" if suffix == ".tsv" else ",")
    except Exception as exc:  # a file that cannot be described is named, never a failed redraw
        return f"could not be read ({type(exc).__name__})"
    return ""


def describe_data_files(root: Path, rel_paths: list[str], *, max_chars: int = 8000) -> str:
    """A bullet per file (relative to ``root``) with what it holds, at most ``max_chars`` long; a cut list says how
    many files it left out."""
    lines: list[str] = []
    used = 0
    for i, rel in enumerate(rel_paths):
        what = describe_data_file(root / rel)
        line = f"- {rel}: {what}" if what else f"- {rel}"
        if len(line) > _MAX_LINE:
            line = line[: _MAX_LINE - 60].rstrip() + f" ... (cut: more names, {_NOT_LISTED})"
        if lines and used + len(line) + 1 > max_chars:
            lines.append(f"- ... and {len(rel_paths) - i} more file(s), not described here (the list was cut to fit)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)

"""``fi demo``: the first thing to run after installing FI.

It writes a small example quest (``fi-demo.yaml``) into the folder it is run from, checks at no cost that the model
it names can be used (``core.provider_readiness.preflight``), and asks before running it for real. The example's
text lives here, not in ``examples/``: a pip-installed FI has no ``examples/`` folder (only the source checkout and
the sdist carry it), and a YAML file outside a package is not shipped in the wheel either.

It never overwrites a file: an existing ``fi-demo.yaml`` makes the next one ``fi-demo-2.yaml``, and so on.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DEMO_FILE", "PROVIDER_PREFERENCE", "demo_yaml", "new_demo_path", "set_up_providers", "write_demo_config"]

DEMO_FILE = "fi-demo.yaml"

#: The order ``fi demo`` picks a provider in, when more than one is set up on this computer: an API key first (it
#: needs nothing else), then the signed-in CLIs, then a local Ollama.
PROVIDER_PREFERENCE: tuple[str, ...] = ("openai", "gemini", "claude_cli", "codex_cli", "gemini_cli", "ollama")

_TEMPLATE = """\
# A small example quest, written by `fi demo`. Run it with:  fi --config {name}
#
# It compares three numerical integrators on a damped oscillator, so it needs no data. It is set up as an
# exploration (result_use: explore): the cheapest kind of quest, whose paper is marked preliminary.

topic: |
  Compare three numerical integrators (forward Euler, 4th-order Runge-Kutta and Velocity-Verlet) on a 1-D damped
  harmonic oscillator m x'' + c x' + k x = 0 with m = k = 1, c = 0.1, x(0) = 1, x'(0) = 0. Measure the trajectory
  error against the analytical solution for step sizes h in {{0.5, 0.1, 0.05, 0.01}}, and the long-time energy drift
  over t in [0, 200]. Relate the result to the literature on symplectic and non-symplectic methods.

title: fi-demo-integrators

# What the result is for: explore (a cheap, preliminary draft), research or decision (every check, slower).
result_use: explore

# The one thing to change if you want another model: provider.name (openai, gemini, claude_cli, codex_cli,
# gemini_cli, ollama or vscode_extension) and, if you like, provider.model.{provider_note}
provider:
  name: {provider}{model_line}

engine:
  max_iterations: 1

execution:
  sandbox: venv
  timeout_s: 1800

# FI normally stops twice for you (to add paywalled papers by hand, and to accept the paper); the demo does not.
pauses:
  papers: false
  review: "off"

output:
  kinds:
    - paper_md
  output_dir: ./outputs
"""


def demo_yaml(name: str, provider: str, model: str | None = None, provider_note: str = "") -> str:
    """The example quest's YAML, naming ``provider`` (and ``model``, when given)."""
    note = f"\n# {provider_note}" if provider_note else ""
    model_line = f"\n  model: {model}" if model else ""
    return _TEMPLATE.format(name=name, provider=provider, model_line=model_line, provider_note=note)


def new_demo_path(folder: Path) -> Path:
    """``fi-demo.yaml`` in ``folder``, or the first ``fi-demo-N.yaml`` that does not exist yet."""
    first = folder / DEMO_FILE
    if not first.exists():
        return first
    n = 2
    while (folder / f"fi-demo-{n}.yaml").exists():
        n += 1
    return folder / f"fi-demo-{n}.yaml"


def write_demo_config(folder: Path, provider: str, model: str | None = None, provider_note: str = "") -> Path:
    """Write the example into a new file in ``folder`` (never over an existing one); its path."""
    path = new_demo_path(folder)
    text = demo_yaml(path.name, provider, model, provider_note)
    # "x": fail rather than overwrite, should the file appear between the check above and this write.
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def set_up_providers() -> list[str]:
    """The providers of :data:`PROVIDER_PREFERENCE` set up on this computer (a key set, a CLI or Ollama installed),
    in that order, by the local check alone. ``fi demo`` checks them in turn and names the first that passes."""
    from core.provider_readiness import check_local

    return [name for name in PROVIDER_PREFERENCE if check_local(name).state in ("installed", "key_present")]

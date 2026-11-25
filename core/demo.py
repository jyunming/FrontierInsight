"""``fi demo``: the first thing to run after installing FI.

It writes a small example study into the folder it is run from (``fi-demo-commission.yaml``, the study FI is asked
to run, and ``fi-demo.yaml``, the config that names it), checks at no cost that the model it names can be used (``core.provider_readiness.preflight``), and asks before running it for real. The example's
text lives here, not in ``examples/``: a pip-installed FI has no ``examples/`` folder (only the source checkout and
the sdist carry it), and a YAML file outside a package is not shipped in the wheel either.

It never overwrites a file: an existing ``fi-demo.yaml`` makes the next pair ``fi-demo-2.yaml`` and
``fi-demo-2-commission.yaml``, and so on.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DEMO_FILE", "PROVIDER_PREFERENCE", "commission_path", "demo_yaml", "new_demo_path",
           "set_up_providers", "write_demo_config"]

DEMO_FILE = "fi-demo.yaml"

#: The order ``fi demo`` picks a provider in, when more than one is set up on this computer: an API key first (it
#: needs nothing else), then the signed-in CLIs, then a local Ollama.
PROVIDER_PREFERENCE: tuple[str, ...] = ("openai", "gemini", "claude_cli", "codex_cli", "gemini_cli", "ollama")

_TEMPLATE = """\
# A small example study, written by `fi demo`. Run it with:  fi --config {name}
#
# The study itself is in {commission} (the question, what it covers, the time allowed and what counts as
# supported); this file says how FI runs it. It needs no data. It is set up as an exploration (result_use: explore):
# the cheapest kind of study, whose paper is marked preliminary.

commission: {commission}

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

# FI normally stops for you to accept the paper; the demo does not.
pauses:
  review: "off"

output:
  kinds:
    - paper_md
  output_dir: ./outputs
"""

#: The example's commission (core/commission.py): the study FI is asked to run. It gives no known value, so its
#: results have no outside value to agree with (``no_known_values: true``).
_COMMISSION = """\
# The study `fi demo` asks FI to run (a commission). `fi check {name}` checks it; `fi new` writes an empty one.

question: >-
  How do the midpoint method, 4th-order Runge-Kutta and Velocity-Verlet compare in trajectory error and long-time
  energy drift on a 1-D damped harmonic oscillator m x'' + c x' + k x = 0 with m = k = 1, c = 0.1, x(0) = 1, x'(0) = 0?

scope: >-
  The three integrators on that one oscillator: the trajectory error against the analytical solution for step sizes
  h in {{0.5, 0.1, 0.05, 0.01}}, and the energy drift over t in [0, 200]. No other methods, forms or parameters.

limits:
  hours: 1
  model_calls: 60

known_values: []
no_known_values: true

success:
  - condition: The trajectory error of each method falls as h shrinks, at about its order of accuracy.
    verify: The slope of log(error) against log(h) for each method, over the four step sizes.
failure:
  - condition: A method's error does not fall as h shrinks, or its energy grows without bound at h = 0.01.
    verify: The error at each step size, and the energy at t = 200 against its value at t = 0.

doubts: []
"""


def demo_yaml(name: str, provider: str, model: str | None = None, provider_note: str = "",
              commission: str = "fi-demo-commission.yaml") -> str:
    """The example's config, naming ``provider`` (and ``model``, when given) and its commission file."""
    note = f"\n# {provider_note}" if provider_note else ""
    model_line = f"\n  model: {model}" if model else ""
    return _TEMPLATE.format(name=name, provider=provider, model_line=model_line, provider_note=note,
                            commission=commission)


def commission_path(config: Path) -> Path:
    """The commission paired with a demo config: ``fi-demo-2.yaml`` -> ``fi-demo-2-commission.yaml``."""
    return config.with_name(f"{config.stem}-commission.yaml")


def new_demo_path(folder: Path) -> Path:
    """``fi-demo.yaml`` in ``folder``, or the first ``fi-demo-N.yaml`` that does not exist yet (nor its commission)."""
    def free(path: Path) -> bool:
        return not (path.exists() or commission_path(path).exists())

    first = folder / DEMO_FILE
    if free(first):
        return first
    n = 2
    while not free(folder / f"fi-demo-{n}.yaml"):
        n += 1
    return folder / f"fi-demo-{n}.yaml"


def write_demo_config(folder: Path, provider: str, model: str | None = None, provider_note: str = "") -> Path:
    """Write the example's commission and its config into new files in ``folder`` (never over an existing one); the
    config's path."""
    path = new_demo_path(folder)
    commission = commission_path(path)
    # "x": fail rather than overwrite, should a file appear between the check above and this write.
    with open(commission, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(_COMMISSION.format(name=commission.name))
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(demo_yaml(path.name, provider, model, provider_note, commission=commission.name))
    return path


def set_up_providers() -> list[str]:
    """The providers of :data:`PROVIDER_PREFERENCE` set up on this computer (a key set, a CLI or Ollama installed),
    in that order, by the local check alone. ``fi demo`` checks them in turn and names the first that passes."""
    from core.provider_readiness import check_local

    return [name for name in PROVIDER_PREFERENCE if check_local(name).state in ("installed", "key_present")]

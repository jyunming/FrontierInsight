"""LaTeX a model writes inside a JSON or YAML string (single backslashes: ``$\\sigma$``, ``\\text{NILS}``) is read back as
LaTeX, not as an unreadable reply or as a tab, a form feed or a backspace."""
from __future__ import annotations

import json

import pytest
import yaml

from core import plan as _plan
from core.engine import _parse_json_lenient, _top_level_json_objects
from core.latex_text import escape_latex_backslashes

BS = "\\"


def _reply(objection: str) -> str:
    return ('{"objections": [{"check": "oracle", "objection": "' + objection + '", "fix": "x"}], '
            '"amended_design": {"a": 1}}')


def test_a_backslash_sigma_reply_is_read() -> None:
    reply = _reply("use $" + BS + "sigma_{in}=0.6$")
    with pytest.raises(json.JSONDecodeError):
        json.loads(reply)
    parsed = _parse_json_lenient(reply, node="t")
    assert parsed is not None
    assert parsed["objections"][0]["objection"] == "use $" + BS + "sigma_{in}=0.6$"
    assert parsed["amended_design"] == {"a": 1}


@pytest.mark.parametrize("latex", [
    BS + "text{NILS}", BS + "frac{a}{b}", BS + "beta", BS + "nabla", BS + "nu", BS + "rho", BS + "right)", BS + "times",
    BS + "tau_0", BS + "theta", BS + "upsilon", BS + "lambda", BS + "mathrm{dose}", BS + "approx", BS + "pi", BS + "cdot",
    BS + "sqrt{2}", BS + "left(", BS + "rightarrow", BS + "tilde{x}", BS + "bar{x}", BS + "forall",
])
def test_latex_commands_survive_as_literal_latex(latex: str) -> None:
    # with `\sigma` beside it the reply does not parse as written: every command is read as LaTeX
    parsed = _parse_json_lenient('{"k": "value $' + BS + 'sigma ' + latex + '$ end"}', node="t")
    assert parsed == {"k": "value $" + BS + "sigma " + latex + "$ end"}


@pytest.mark.parametrize("latex", [
    BS + "text{NILS}", BS + "frac{a}{b}", BS + "beta", BS + "nabla", BS + "rho", BS + "right)", BS + "times",
    BS + "theta", BS + "rightarrow", BS + "tilde{x}", BS + "bar{x}", BS + "forall",
])
def test_a_command_that_parses_as_a_tab_or_a_form_feed_is_read_as_latex_alone(latex: str) -> None:
    reply = '{"k": "value $' + latex + '$ end"}'
    assert json.loads(reply) != {"k": "value $" + latex + "$ end"}  # as written it is a tab, a form feed, ...
    assert _parse_json_lenient(reply, node="t") == {"k": "value $" + latex + "$ end"}


def test_code_in_a_valid_json_reply_is_never_read_as_latex() -> None:
    code = "import numpy as np\nnu = 1\nto = 2\ntau = 3\nne = 4\nni = 5\n\ttop = 6\nmy_var = 7\n"
    reply = json.dumps({"code": code, "deps": ["numpy"]})
    assert _parse_json_lenient(reply) == {"code": code, "deps": ["numpy"]}
    assert _parse_json_lenient("```json\n" + reply + "\n```", want=("code",)) == {"code": code, "deps": ["numpy"]}


def test_latex_survives_with_want_and_in_a_fence_and_nested() -> None:
    reply = ("Here you go:\n```json\n" + '{"design": {"hypothesis": "NILS $' + BS + 'text{NILS}>2$", '
             '"variables": [{"name": "' + BS + 'sigma", "range": "0.3 to 0.9"}]}, "plan": {"in_short": "x"}}'
             + "\n```\n")
    parsed = _parse_json_lenient(reply, node="t", want=("design", "plan"))
    assert parsed is not None
    assert parsed["design"]["hypothesis"] == "NILS $" + BS + "text{NILS}>2$"
    assert parsed["design"]["variables"][0]["name"] == BS + "sigma"
    plain = _parse_json_lenient(reply, node="t")
    assert plain == parsed


def test_the_second_object_of_two_is_found_with_latex_in_it() -> None:
    first = '{"hypothesis": "h"}'
    second = '{"objections_addressed": [{"objection": "$' + BS + 'sigma$"}]}'
    found = _parse_json_lenient(first + "\nand then\n" + second, want=("objections_addressed",))
    assert found == {"objections_addressed": [{"objection": "$" + BS + "sigma$"}]}
    assert _top_level_json_objects(first + second)[-1]["objections_addressed"][0]["objection"] == "$" + BS + "sigma$"


def test_a_real_newline_and_tab_in_a_string_stay_what_they_are() -> None:
    reply = '{"k": "line one' + BS + 'nThe second line' + BS + 'tTabbed ' + BS + 'sigma and ' + BS + 'nabla"}'
    assert _parse_json_lenient(reply) == {"k": "line one\nThe second line\tTabbed " + BS + "sigma and " + BS + "nabla"}
    # a newline or tab before a non-letter, or at the end of the string
    reply = '{"k": "a' + BS + 'n' + BS + 'n1' + BS + 't' + BS + 'sigma' + BS + 'n"}'
    assert _parse_json_lenient(reply) == {"k": "a\n\n1\t" + BS + "sigma\n"}


def test_a_degree_sign_escape_still_decodes_and_a_bare_u_is_latex() -> None:
    reply = '{"k": "90' + BS + 'u00b0 and ' + BS + 'upsilon"}'
    assert _parse_json_lenient(reply) == {"k": "90° and " + BS + "upsilon"}


def test_a_valid_reply_is_parsed_exactly_as_before() -> None:
    valid = '{"k": "a' + BS + 'ntext' + BS + 'tframe", "m": "' + BS + BS + 'sigma and ' + BS + '"q' + BS + '""}'
    assert _parse_json_lenient(valid) == json.loads(valid)
    assert _parse_json_lenient(valid)["k"] == "a\ntext\tframe"
    assert escape_latex_backslashes("no strings here " + BS + "sigma") == "no strings here " + BS + "sigma"


def test_a_reply_that_is_not_json_is_still_none() -> None:
    assert _parse_json_lenient("I could not do that " + BS + "sigma", node="t") is None
    assert _parse_json_lenient('{"k": "unterminated ' + BS + 'sigma', node="t") is None


# --- YAML (the design block of plan.md) ---------------------------------------------------------------------------

def test_a_double_quoted_yaml_value_with_latex_is_repaired_to_the_same_text() -> None:
    block = ('hypothesis: "use $' + BS + 'sigma_{in}$ and ' + BS + 'text{NILS}"\n'
             'note: \'single ' + BS + 'quoted stays\'\n'
             'variables:\n  - name: "' + BS + 'lambda"\n    values: [1, 2]  # a ' + BS + 'comment "q"\n')
    assert _plan.yaml_problem(block) is not None
    fixed, notes = _plan.repair_block(block)
    assert fixed is not None and notes
    loaded = yaml.safe_load(fixed)
    assert loaded["hypothesis"] == "use $" + BS + "sigma_{in}$ and " + BS + "text{NILS}"
    assert loaded["note"] == "single " + BS + "quoted stays"
    assert loaded["variables"][0]["name"] == BS + "lambda"


def test_a_plan_the_engine_writes_with_latex_reads_back_unchanged() -> None:
    design = {"hypothesis": "NILS $" + BS + "text{NILS}$ rises with " + BS + "sigma", "dependencies": ["numpy"],
              "variables": {"independent": [BS + "beta"], "dependent": ["NILS"], "controls": []}}
    block = yaml.safe_dump(design, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
    assert yaml.safe_load(block) == design
    text = "## " + _plan.DESIGN_HEADING + "\n\n```yaml\n" + block + "```\n"
    assert _plan.parse(text).design["hypothesis"] == design["hypothesis"]

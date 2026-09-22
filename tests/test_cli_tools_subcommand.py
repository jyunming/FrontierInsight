"""``fi tools <name> ...`` (launch.py:_expand_tools_argv) and the short-vs-full ``--help`` split.

103 top-level flags in one listing was unreadable; the ~15 rarer, one-shot utilities (a weekly digest, a
cross-quest portfolio, an adversarial critique, ...) move under `fi tools <name>`, which is exactly the
existing flag under a friendlier name — nothing is removed, and every old script or doc that already says
`--digest` keeps working unchanged."""

from __future__ import annotations

import pytest

from launch import _TOOL_SUBCOMMANDS, parse_args


def test_every_tool_subcommand_parses_the_same_as_its_old_flag(tmp_path):
    cases = [
        (["tools", "digest", "--days", "3"], ["--digest", "--days", "3"]),
        (["tools", "summarize", str(tmp_path)], ["--summarize", str(tmp_path)]),
        (["tools", "critique", "q-1"], ["--critique", "q-1"]),
        (["tools", "ingest", "a.pdf", "b.pdf"], ["--ingest", "a.pdf", "b.pdf"]),
        (["tools", "fleet", "a.yaml", "b.yaml"], ["--fleet", "a.yaml", "b.yaml"]),
        (["tools", "install-tectonic"], ["--install-tectonic"]),
        (["tools", "install-tectonic-from", str(tmp_path)], ["--install-tectonic-from", str(tmp_path)]),
        (["tools", "list-drafts"], ["--list-drafts"]),
        (["tools", "export-models", str(tmp_path)], ["--export-models", str(tmp_path)]),
    ]
    for via_tools, via_flag in cases:
        assert vars(parse_args(via_tools)) == vars(parse_args(via_flag)), via_tools


def test_every_declared_subcommand_is_reachable_and_a_real_flag() -> None:
    """The map is the only source of truth for both `fi tools --help` and the dispatch: catches a typo the moment one
    is added, rather than a dead entry nobody notices."""
    import launch

    p = launch  # module, for attribute access below via getattr on argparse actions is overkill; just parse --help-all
    help_text = __import__("io").StringIO()
    import contextlib

    with contextlib.redirect_stdout(help_text):
        with pytest.raises(SystemExit):
            parse_args(["--help-all"])
    full_help = help_text.getvalue()
    for name, (flag, summary) in _TOOL_SUBCOMMANDS.items():
        assert flag in full_help, f"{name} claims {flag}, which --help-all does not list"
        assert f"`fi tools {name}" in summary or f"`fi tools {name.split('-')[0]}" in summary


def test_an_unknown_tool_name_says_so_and_lists_the_real_ones(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        parse_args(["tools", "nonsense"])
    assert stop.value.code == 2
    out = capsys.readouterr().out
    assert "no tool named 'nonsense'" in out and "digest" in out and "portfolio" in out


@pytest.mark.parametrize("bare", [["tools"], ["tools", "-h"], ["tools", "--help"]])
def test_tools_with_no_name_or_asking_for_help_lists_every_tool(bare: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        parse_args(bare)
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert all(name in out for name in _TOOL_SUBCOMMANDS)
    assert "fi --digest" in out, "says the old flag still works"


def test_a_flag_after_a_tool_still_belongs_to_the_tool_not_swallowed_by_the_rewrite() -> None:
    args = parse_args(["tools", "critique", "q-1", "--critique-provider", "gemini"])
    assert args.critique == "q-1" and args.critique_provider == "gemini"


# --- the short vs. full help --------------------------------------------------------------------------------------


def test_bare_help_is_short_and_does_not_demand_a_mode(capsys: pytest.CaptureFixture[str]) -> None:
    """`-h` / `--help` must exit 0 with the short list even though no mode (--config, --new, ...) was given — the
    opposite of every other bare invocation, which `_check_mode` correctly refuses."""
    for flag in ("-h", "--help"):
        with pytest.raises(SystemExit) as stop:
            parse_args([flag])
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert "--config" in out and "fi tools --help" in out
        assert out.count("\n") < 40, "the short help is short"
        assert "--critique-provider" not in out, "qualifier flags belong to --help-all, not the short form"


def test_help_all_has_every_flag_including_the_deep_qualifiers(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        parse_args(["--help-all"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "--critique-provider" in out and "--digest-ensemble" in out and out.count("\n") > 60


def test_no_arguments_at_all_still_asks_for_a_mode_not_the_help(capsys: pytest.CaptureFixture[str]) -> None:
    """An empty command line is a mistake, not a request for help: it must still fail the way it always has."""
    with pytest.raises(SystemExit) as stop:
        parse_args([])
    assert stop.value.code == 2


# --- fixes from the agy review of this PR --------------------------------------------------------------------------


def test_help_short_circuits_even_past_an_unrecognized_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`-h` wins over everything else on the command line, the standard convention — even a typo elsewhere must not
    make argparse refuse the line before ever reaching the help check."""
    with pytest.raises(SystemExit) as stop:
        parse_args(["--digest", "--this-flag-does-not-exist", "-h"])
    assert stop.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_tools_help_all_means_the_top_level_help_all(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        parse_args(["tools", "--help-all"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "--digest-provider" in out and "no tool named" not in out


def test_help_inside_a_tool_shows_that_tools_own_qualifiers(capsys: pytest.CaptureFixture[str]) -> None:
    """`fi tools digest --help` used to fall through to the generic short list, which knows nothing about
    `--digest-provider` / `--digest-model` / `--digest-ensemble`; it now shows the full listing instead."""
    for tail in (["--help"], ["-h"], ["--bad-flag", "-h"]):
        with pytest.raises(SystemExit) as stop:
            parse_args(["tools", "digest", *tail])
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert "--digest-provider" in out and "--digest-ensemble" in out


def test_a_global_flag_before_tools_is_a_documented_limitation_not_a_crash_surprise() -> None:
    """`tools` must be the first word — this is a real, deliberate limit of the new spelling (see
    `_expand_tools_argv`'s docstring), not something silently different from what the old flag would have done: the
    OLD flag (`--digest`) is completely unaffected and still order-independent."""
    with pytest.raises(SystemExit) as stop:
        parse_args(["--memory-cap-mb", "4096", "tools", "digest"])
    assert stop.value.code == 2  # a clear argparse error, not a silent wrong parse
    assert parse_args(["--digest", "--memory-cap-mb", "4096"]).digest is True  # the old flag form is unaffected

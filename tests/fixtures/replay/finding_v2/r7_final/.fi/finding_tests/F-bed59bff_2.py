import types


def test_main_computes_dose_to_size_with_deterministic_root_finding():
    try:
        import experiment
    except Exception as exc:
        assert False, f"experiment must import before dose-to-size can be checked: {exc}"

    def code_names(code):
        names = set(code.co_names)
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                names.update(code_names(constant))
        return names

    root_finders = {"brentq", "bisect", "newton", "root_scalar"}
    assert code_names(experiment.main) & root_finders, (
        "dose_to_size_mJ_cm2 must be computed by deterministic root finding "
        "against the target developed CD, not reported as a hardcoded value"
    )

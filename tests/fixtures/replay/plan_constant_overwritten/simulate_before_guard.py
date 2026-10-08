# RECONSTRUCTED from .fi/run.log (the text before 16:39 was not kept): "simulate.py line 290: reads the setting
# `reference_dose` or `dose_to_size` (and gives 80.0 when it is not there)". The surrounding lines are the real ones
# (simulate_after_guard.py), with only the read of the reference dose as the log describes it.
def run_trial(cell, trial_id=0):
    dose = float(cell["exposure_dose"])
    resist = cell["resist_type"]
    chemistry = cell["chemistry"]
    reference_dose = float(cell.get("reference_dose", cell.get("dose_to_size", 80.0)))
    spare = 0.2  # a spare constant: the test turns this line into a bad read
    threshold = .55 * reference_dose * 1.0
    return {"printed_fraction": float(dose > threshold)}

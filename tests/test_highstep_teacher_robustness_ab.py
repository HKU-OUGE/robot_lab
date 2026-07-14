from tools.highstep_teacher_robustness_ab import (
    aggregate_teacher,
    build_matrix,
    improvement_decision,
)


def _rows(*, nominal=3, inward_levels=(9, 9, 9, 0), impulse_levels=(18, 18, 0)):
    rows = []
    for row in build_matrix():
        item = dict(row)
        index = 0
        if row["family"] == "nominal":
            passed = index < nominal
        elif row["family"] == "inward":
            widths = (0.34, 0.30, 0.26, 0.22)
            level = widths.index(row["initial_rear_width_m"])
            siblings = [x for x in build_matrix() if x["family"] == "inward" and x["initial_rear_width_m"] == widths[level]]
            index = [x["run_id"] for x in siblings].index(row["run_id"])
            passed = index < inward_levels[level]
        elif row["family"] == "impulse":
            levels = (0.10, 0.20, 0.30)
            level = levels.index(row["impulse_delta_v_mps"])
            siblings = [x for x in build_matrix() if x["family"] == "impulse" and x["impulse_delta_v_mps"] == levels[level]]
            index = [x["run_id"] for x in siblings].index(row["run_id"])
            passed = index < impulse_levels[level]
        else:
            passed = True
        item.update(valid=True, full_climb=passed, rear_hold=passed, recovery=passed, fell=False, centerline_crossed=False)
        rows.append(item)
    return rows


def test_matrix_is_exactly_102_unique_rows_and_fixed_family_sizes():
    rows = build_matrix()
    assert len(rows) == len({row["run_id"] for row in rows}) == 102
    assert sum(row["family"] == "nominal" for row in rows) == 3
    assert sum(row["family"] == "inward" for row in rows) == 36
    assert sum(row["family"] == "impulse" for row in rows) == 54
    assert sum(row["family"] == "combined" for row in rows) == 9


def test_improvement_requires_all_four_prefixed_conditions():
    old = aggregate_teacher(_rows(inward_levels=(9, 9, 0, 0), impulse_levels=(18, 0, 0)))
    new = aggregate_teacher(_rows(inward_levels=(9, 9, 9, 0), impulse_levels=(18, 18, 0)))
    decision = improvement_decision(old, new)
    assert decision["clearly_improved"] is True
    assert decision["status"] == "new_teacher_robustness_clearly_improved"


def test_nominal_regression_fails_closed_even_when_levels_improve():
    old = aggregate_teacher(_rows(inward_levels=(9, 9, 0, 0), impulse_levels=(18, 0, 0)))
    new_rows = _rows(inward_levels=(9, 9, 9, 0), impulse_levels=(18, 18, 0))
    nominal = next(row for row in new_rows if row["family"] == "nominal")
    nominal["full_climb"] = False
    decision = improvement_decision(old, aggregate_teacher(new_rows))
    assert decision["nominal_not_regressed"] is False
    assert decision["clearly_improved"] is False


def test_missing_matrix_row_is_rejected():
    rows = _rows()
    try:
        aggregate_teacher(rows[:-1])
    except ValueError as error:
        assert "matrix mismatch" in str(error)
    else:
        raise AssertionError("missing row was accepted")

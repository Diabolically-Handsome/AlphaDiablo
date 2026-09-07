"""Named optional-null compatibility only; no model, rollout, environment or updates."""
from copy import deepcopy

import pytest

import analyze_first_update as analysis

KEY = "worker_descend_escrow_readiness_table"
BASE = {"implementation_sha256": "same", "algorithm_recipe": {"clip_range": 0.2},
        "resource_service_policy": "sustain-v6"}


@pytest.mark.parametrize("before_extra,after_extra,allowed", [
    ({}, {}, True), ({KEY: None}, {KEY: None}, True),
    ({}, {KEY: None}, True), ({KEY: None}, {}, True),
    ({}, {KEY: "v2"}, False), ({KEY: "v2"}, {KEY: "v2"}, False),
    ({}, {KEY: "v1"}, False), ({}, {KEY: False}, False), ({KEY: 0}, {}, False),
    ({}, {"other_optional_key": None}, False),
    ({"algorithm_recipe": {"clip_range": 0.2}}, {"algorithm_recipe": {"clip_range": 0.3}}, False),
    ({"resource_service_policy": "sustain-v5"}, {}, False),
    ({"implementation_sha256": "changed"}, {}, False),
])
def test_only_named_absent_null_equivalence(before_extra, after_extra, allowed):
    initial, capture = dict(BASE, **before_extra), dict(BASE, **after_extra)
    saved = deepcopy((initial, capture))
    if allowed:
        result = analysis.initialization_contract_equivalence(initial, capture)
        assert result["rule"] == "only-worker-descend-escrow-readiness-table-absent-or-null-v1"
        assert result["field"] == KEY
        assert result["applied"] == (analysis.json_bytes(initial) != analysis.json_bytes(capture))
        assert result["initialization_contract_sha256"] == analysis.sha(analysis.json_bytes(initial))
        assert result["capture_contract_sha256"] == analysis.sha(analysis.json_bytes(capture))
        assert result["initialization_field_present"] == (KEY in initial)
        assert result["capture_field_present"] == (KEY in capture)
    else:
        with pytest.raises(ValueError, match="initialization contract/lineage mismatch"):
            analysis.initialization_contract_equivalence(initial, capture)
    assert (initial, capture) == saved


def test_normalization_keeps_distinct_input_identity():
    initial = deepcopy(BASE)
    captured = dict(BASE, **{KEY: None})
    evidence = analysis.initialization_contract_equivalence(initial, captured)
    assert evidence["initialization_contract_sha256"] != evidence["capture_contract_sha256"]
    assert KEY not in initial and captured[KEY] is None

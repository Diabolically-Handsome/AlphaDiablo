"""Deterministic, fail-closed paired statistics for the R7 campaign.

The evaluation archive contract proves *what* was evaluated and that every
aggregate is reproduced by its rows.  This module answers the separate
selection question: whether paired candidate-minus-baseline results provide
enough evidence to pass a pre-registered gate.

There is deliberately no SciPy dependency:

* continuous/count metrics use a one-sided paired Student-t lower bound;
* an exact one-sided sign test protects the mean from a small set of outliers;
* paired deaths use a conservative risk-difference upper bound made from two
  Bonferroni-adjusted one-sided Clopper-Pearson bounds.

Official callers should first load both inputs through
``eval_contract.read_eval_archive`` and then call
``analyze_paired_archives`` with the SHA-256 of the exact archive bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Literal, Mapping, Sequence


R7_STATISTICS_SCHEMA = "diablogym-r7-paired-statistics/1"
R7_METHOD_REVISION = "paired-t-sign-death-cp/1"
SUPPORTED_EVAL_ARCHIVE_SCHEMA = 5
MIN_DEVELOPMENT_PAIRS = 128
MIN_FINAL_PAIRS = 256

_PHASE_MINIMUMS = {
    "development": MIN_DEVELOPMENT_PAIRS,
    "final": MIN_FINAL_PAIRS,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MANAGER_IDENTITY_KEYS = frozenset({
    "kind",
    "path",
    "sha256",
    "num_timesteps",
    "gate_report_sha256",
})
_MANAGER_CONTENT_IDENTITY_KEYS = (
    "kind",
    "sha256",
    "num_timesteps",
    "gate_report_sha256",
)


class StatisticsContractError(ValueError):
    """The paired inputs or pre-registered statistical rule are invalid."""


@dataclass(frozen=True, slots=True)
class MetricRule:
    """One pre-registered paired metric constraint.

    ``minimum_effect`` is expressed in the beneficial direction.  For a
    ``higher`` metric it is candidate minus baseline; for a ``lower`` metric
    it is baseline minus candidate.
    """

    key: str
    direction: Literal["higher", "lower"] = "higher"
    minimum_effect: float = 0.0
    require_sign_test: bool = True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StatisticsContractError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be a number",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a,b), using a stable continued fraction."""
    _require(a > 0.0 and b > 0.0, "beta shape parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Modified Lentz algorithm from the standard incomplete-beta continued
    # fraction.  The symmetry branch avoids slow convergence near x=1.
    def continued_fraction(left: float, right: float, point: float) -> float:
        maximum_iterations = 10_000
        epsilon = 3e-14
        tiny = 1e-300
        qab = left + right
        qap = left + 1.0
        qam = left - 1.0
        c = 1.0
        d = 1.0 - qab * point / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        result = d
        for iteration in range(1, maximum_iterations + 1):
            even = 2 * iteration
            coefficient = (
                iteration
                * (right - iteration)
                * point
                / ((qam + even) * (left + even))
            )
            d = 1.0 + coefficient * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + coefficient / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            result *= d * c

            coefficient = (
                -(left + iteration)
                * (qab + iteration)
                * point
                / ((left + even) * (qap + even))
            )
            d = 1.0 + coefficient * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + coefficient / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) <= epsilon:
                return result
        raise StatisticsContractError("incomplete-beta continued fraction did not converge")

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        result = front * continued_fraction(a, b, x) / a
    else:
        result = 1.0 - front * continued_fraction(b, a, 1.0 - x) / b
    # Roundoff in the symmetry branch can escape the mathematical range by a
    # few ulps; clamping keeps inversion monotone and JSON-safe.
    return min(1.0, max(0.0, result))


def _beta_quantile(probability: float, a: float, b: float) -> float:
    _require(0.0 <= probability <= 1.0, "beta probability must be in [0,1]")
    _require(a > 0.0 and b > 0.0, "beta shape parameters must be positive")
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(160):
        middle = (low + high) / 2.0
        if _regularized_beta(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    _require(
        _is_int(degrees_of_freedom) and degrees_of_freedom > 0,
        "Student-t degrees of freedom must be positive",
    )
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail_twice = _regularized_beta(
        x, degrees_of_freedom / 2.0, 0.5
    )
    return 1.0 - 0.5 * tail_twice if value > 0.0 else 0.5 * tail_twice


def _student_t_upper_critical(alpha: float, degrees_of_freedom: int) -> float:
    """Return t such that P(T_df > t) == alpha."""
    _require(0.0 < alpha < 0.5, "one-sided alpha must be in (0,0.5)")
    target = 1.0 - alpha
    low, high = 0.0, 1.0
    while _student_t_cdf(high, degrees_of_freedom) < target:
        high *= 2.0
        _require(high <= 1e12, "Student-t critical value search diverged")
    for _ in range(120):
        middle = (low + high) / 2.0
        if _student_t_cdf(middle, degrees_of_freedom) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _exact_sign_p_value(wins: int, non_ties: int) -> float:
    """P(Binomial(non_ties, 0.5) >= wins), exactly accumulated as integers."""
    _require(
        _is_int(wins)
        and _is_int(non_ties)
        and 0 <= wins <= non_ties,
        "sign-test counts are invalid",
    )
    if non_ties == 0:
        return 1.0
    numerator = sum(
        math.comb(non_ties, value)
        for value in range(wins, non_ties + 1)
    )
    return numerator / (1 << non_ties)


def _clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    if successes == 0:
        return 0.0
    return _beta_quantile(alpha, successes, trials - successes + 1)


def _clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    if successes == trials:
        return 1.0
    return _beta_quantile(
        1.0 - alpha, successes + 1, trials - successes
    )


def _validated_rules(metric_rules: Iterable[MetricRule]) -> tuple[MetricRule, ...]:
    rules = tuple(metric_rules)
    _require(bool(rules), "at least one metric rule is required")
    _require(
        all(isinstance(rule, MetricRule) for rule in rules),
        "metric_rules must contain MetricRule values",
    )
    keys = [rule.key for rule in rules]
    _require(
        all(isinstance(key, str) and bool(key) for key in keys),
        "metric rule keys must be non-empty strings",
    )
    _require(len(keys) == len(set(keys)), "metric rule keys must be unique")
    for rule in rules:
        _require(
            rule.direction in {"higher", "lower"},
            f"invalid direction for {rule.key}",
        )
        _finite_number(rule.minimum_effect, f"{rule.key}.minimum_effect")
        _require(
            isinstance(rule.require_sign_test, bool),
            f"{rule.key}.require_sign_test must be bool",
        )
    # Call order must not alter frozen bytes or hashes.
    return tuple(sorted(rules, key=lambda rule: rule.key))


def _manager_content_identity(
    value: Any,
    label: str,
) -> dict[str, Any]:
    """Return the path-independent identity of a validated v5 manager.

    Each evaluation tag gets its own immutable staging directory, so the
    absolute ``path`` is intentionally deployment-local.  Every other manager
    field remains part of the scientific identity and must compare exactly.
    """
    _require(isinstance(value, Mapping), f"{label} manager identity must be a mapping")
    _require(
        set(value) == _MANAGER_IDENTITY_KEYS,
        f"{label} manager identity fields are invalid",
    )
    _require(
        value["kind"] == "numpy_policy",
        f"{label} manager kind must be numpy_policy",
    )
    _require(
        isinstance(value["path"], str) and bool(value["path"]),
        f"{label} manager path must be a non-empty string",
    )
    _require(
        isinstance(value["sha256"], str)
        and _SHA256_RE.fullmatch(value["sha256"]) is not None,
        f"{label} manager SHA-256 must be complete and lowercase",
    )
    _require(
        value["num_timesteps"] is None,
        f"{label} numpy manager num_timesteps must be null",
    )
    _require(
        value["gate_report_sha256"] is None,
        f"{label} numpy manager gate_report_sha256 must be null",
    )
    return {
        key: value[key]
        for key in _MANAGER_CONTENT_IDENTITY_KEYS
    }


def _archive_pair_inputs(
    baseline_archive: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    baseline_archive_sha256: str,
    candidate_archive_sha256: str,
) -> tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], dict[str, Any]]:
    for label, document in (
        ("baseline", baseline_archive),
        ("candidate", candidate_archive),
    ):
        _require(isinstance(document, Mapping), f"{label} archive must be a mapping")
        _require(
            set(document) == {"schema_version", "meta", "agg", "rows"},
            f"{label} archive must be a validated current-schema archive",
        )
        _require(
            isinstance(document["meta"], Mapping)
            and isinstance(document["rows"], list),
            f"{label} archive meta/rows are invalid",
        )
    _require(
        baseline_archive["schema_version"]
        == candidate_archive["schema_version"]
        == SUPPORTED_EVAL_ARCHIVE_SCHEMA,
        "baseline/candidate evaluation schema must both be supported v5",
    )
    baseline_meta = baseline_archive["meta"]
    candidate_meta = candidate_archive["meta"]
    for field in ("protocol", "worker", "manager", "runtime"):
        _require(field in baseline_meta and field in candidate_meta, f"missing meta.{field}")
    _require(
        baseline_meta["protocol"] == candidate_meta["protocol"],
        "baseline/candidate protocols or seed lists differ",
    )
    baseline_manager_identity = _manager_content_identity(
        baseline_meta["manager"], "baseline")
    candidate_manager_identity = _manager_content_identity(
        candidate_meta["manager"], "candidate")
    _require(
        baseline_manager_identity == candidate_manager_identity,
        "baseline/candidate manager content identities differ",
    )
    _require(
        baseline_meta["runtime"] == candidate_meta["runtime"],
        "baseline/candidate runtime/content identities differ",
    )
    for label, digest in (
        ("baseline_archive_sha256", baseline_archive_sha256),
        ("candidate_archive_sha256", candidate_archive_sha256),
    ):
        _require(
            isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
            f"{label} must be a complete lowercase SHA-256",
        )
    _require(
        baseline_archive_sha256 != candidate_archive_sha256,
        "baseline/candidate archive SHA-256 values must differ",
    )
    protocol = baseline_meta["protocol"]
    _require(
        isinstance(protocol, Mapping)
        and isinstance(protocol.get("seeds"), list)
        and protocol.get("deterministic") is True,
        "validated deterministic protocol seed list is missing",
    )
    seeds = protocol["seeds"]
    baseline_rows = baseline_archive["rows"]
    candidate_rows = candidate_archive["rows"]
    _require(
        all(isinstance(row, Mapping) for row in baseline_rows)
        and all(isinstance(row, Mapping) for row in candidate_rows),
        "validated archive rows must all be mappings",
    )
    _require(
        [row.get("seed") for row in baseline_rows] == seeds
        and [row.get("seed") for row in candidate_rows] == seeds,
        "row order must exactly match the shared protocol seed list",
    )

    try:
        baseline_worker_sha = baseline_meta["worker"]["sha256"]
        candidate_worker_sha = candidate_meta["worker"]["sha256"]
        manager_sha = baseline_meta["manager"]["sha256"]
        protocol_bundle_sha = baseline_meta["runtime"]["python_protocol"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise StatisticsContractError(
            "validated archive identity fields are missing"
        ) from exc
    for label, digest in (
        ("baseline worker", baseline_worker_sha),
        ("candidate worker", candidate_worker_sha),
        ("manager", manager_sha),
        ("protocol bundle", protocol_bundle_sha),
    ):
        _require(
            isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
            f"{label} identity must be a complete lowercase SHA-256",
        )
    source = {
        "eval_schema_version": baseline_archive["schema_version"],
        "baseline_archive_sha256": baseline_archive_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "baseline_worker_sha256": baseline_worker_sha,
        "candidate_worker_sha256": candidate_worker_sha,
        "manager_sha256": manager_sha,
        "protocol_bundle_sha256": protocol_bundle_sha,
    }
    return baseline_rows, candidate_rows, source


def analyze_paired_archives(
    baseline_archive: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    *,
    baseline_archive_sha256: str,
    candidate_archive_sha256: str,
    phase: Literal["development", "final"],
    metric_rules: Iterable[MetricRule],
    death_key: str = "died",
    death_noninferiority_margin: float,
    familywise_alpha: float = 0.05,
    minimum_pairs: int | None = None,
    require_observed_deaths_not_higher: bool = True,
) -> dict[str, Any]:
    """Analyze two already validated, seed-paired evaluation archives.

    The result is timestamp-free and canonically ordered, so identical inputs
    and rules always produce identical JSON data.  ``minimum_pairs`` may raise
    a phase floor but can never lower the built-in 128/256 requirements.
    """
    _require(phase in _PHASE_MINIMUMS, "phase must be development or final")
    rules = _validated_rules(metric_rules)
    _require(
        isinstance(death_key, str) and bool(death_key),
        "death_key must be a non-empty string",
    )
    _require(
        death_key not in {rule.key for rule in rules},
        "death_key cannot also be a numeric metric rule",
    )
    margin = _finite_number(
        death_noninferiority_margin, "death_noninferiority_margin"
    )
    _require(0.0 <= margin < 1.0, "death noninferiority margin must be in [0,1)")
    alpha = _finite_number(familywise_alpha, "familywise_alpha")
    _require(0.0 < alpha < 0.5, "familywise_alpha must be in (0,0.5)")
    _require(
        isinstance(require_observed_deaths_not_higher, bool),
        "require_observed_deaths_not_higher must be bool",
    )

    phase_floor = _PHASE_MINIMUMS[phase]
    if minimum_pairs is None:
        required_pairs = phase_floor
    else:
        _require(
            _is_int(minimum_pairs) and minimum_pairs >= phase_floor,
            f"minimum_pairs cannot be below the {phase} floor {phase_floor}",
        )
        required_pairs = minimum_pairs

    baseline_rows, candidate_rows, source = _archive_pair_inputs(
        baseline_archive,
        candidate_archive,
        baseline_archive_sha256,
        candidate_archive_sha256,
    )
    _require(
        len(baseline_rows) == len(candidate_rows) >= required_pairs,
        f"{phase} analysis requires at least {required_pairs} paired rows",
    )
    n_pairs = len(baseline_rows)
    seeds = [row.get("seed") for row in baseline_rows]
    _require(
        all(_is_int(seed) and seed >= 0 for seed in seeds),
        "paired seeds must be non-negative integers",
    )
    _require(len(seeds) == len(set(seeds)), "paired seeds must be unique")

    constraint_count = len(rules) + 1  # metric rules plus death risk.
    per_constraint_alpha = alpha / constraint_count
    t_critical = _student_t_upper_critical(
        per_constraint_alpha, n_pairs - 1
    )
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    paired_hash_rows = []

    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
        _require(
            isinstance(baseline_row, Mapping)
            and isinstance(candidate_row, Mapping),
            "paired rows must be mappings",
        )
        _require(
            baseline_row.get("seed") == candidate_row.get("seed"),
            "baseline/candidate row seeds differ",
        )
        hash_row = {
            "seed": baseline_row["seed"],
            "baseline": {},
            "candidate": {},
        }
        for rule in rules:
            _finite_number(
                baseline_row.get(rule.key),
                f"baseline seed {baseline_row['seed']} {rule.key}",
            )
            _finite_number(
                candidate_row.get(rule.key),
                f"candidate seed {candidate_row['seed']} {rule.key}",
            )
            hash_row["baseline"][rule.key] = baseline_row[rule.key]
            hash_row["candidate"][rule.key] = candidate_row[rule.key]
        _require(
            isinstance(baseline_row.get(death_key), bool)
            and isinstance(candidate_row.get(death_key), bool),
            f"{death_key} must be bool in every paired row",
        )
        hash_row["baseline"][death_key] = baseline_row[death_key]
        hash_row["candidate"][death_key] = candidate_row[death_key]
        paired_hash_rows.append(hash_row)

    for rule in rules:
        baseline_values = [
            float(row[rule.key]) for row in baseline_rows
        ]
        candidate_values = [
            float(row[rule.key]) for row in candidate_rows
        ]
        try:
            raw_deltas = [
                candidate - baseline
                for baseline, candidate in zip(baseline_values, candidate_values)
            ]
            improvement_values = (
                raw_deltas
                if rule.direction == "higher"
                else [-value for value in raw_deltas]
            )
            baseline_mean = math.fsum(baseline_values) / n_pairs
            candidate_mean = math.fsum(candidate_values) / n_pairs
            raw_delta_mean = math.fsum(raw_deltas) / n_pairs
            improvement_mean = math.fsum(improvement_values) / n_pairs
            squared = math.fsum(
                (value - improvement_mean) ** 2
                for value in improvement_values
            )
            sample_stddev = math.sqrt(squared / (n_pairs - 1))
            standard_error = sample_stddev / math.sqrt(n_pairs)
            lower_bound = improvement_mean - t_critical * standard_error
        except (OverflowError, ValueError) as exc:
            raise StatisticsContractError(
                f"{rule.key} paired arithmetic overflowed"
            ) from exc
        for label, value in (
            ("improvement_mean", improvement_mean),
            ("baseline_mean", baseline_mean),
            ("candidate_mean", candidate_mean),
            ("raw_delta_mean", raw_delta_mean),
            ("sample_stddev", sample_stddev),
            ("standard_error", standard_error),
            ("lower_confidence_bound", lower_bound),
        ):
            _require(math.isfinite(value), f"{rule.key}.{label} must be finite")
        mean_passed = lower_bound > float(rule.minimum_effect)

        centered = [
            value - float(rule.minimum_effect)
            for value in improvement_values
        ]
        _require(
            all(math.isfinite(value) for value in centered),
            f"{rule.key} sign-test residuals must be finite",
        )
        wins = sum(value > 0.0 for value in centered)
        losses = sum(value < 0.0 for value in centered)
        ties = n_pairs - wins - losses
        sign_p = _exact_sign_p_value(wins, wins + losses)
        sign_passed = sign_p <= per_constraint_alpha
        metric_passed = mean_passed and (
            sign_passed if rule.require_sign_test else True
        )
        checks[f"{rule.key}.mean_lcb"] = mean_passed
        if rule.require_sign_test:
            checks[f"{rule.key}.exact_sign"] = sign_passed
        metrics[rule.key] = {
            "direction": rule.direction,
            "minimum_effect": float(rule.minimum_effect),
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "raw_candidate_minus_baseline_mean": raw_delta_mean,
            "improvement_mean": improvement_mean,
            "sample_stddev": sample_stddev,
            "standard_error": standard_error,
            "one_sided_t_critical": t_critical,
            "lower_confidence_bound": lower_bound,
            "mean_lcb_passed": mean_passed,
            "sign_test": {
                "required": rule.require_sign_test,
                "wins_above_minimum_effect": wins,
                "losses_below_minimum_effect": losses,
                "ties_at_minimum_effect": ties,
                "non_ties": wins + losses,
                "one_sided_exact_p_value": sign_p,
                "passed": sign_passed,
            },
            "passed": metric_passed,
        }

    baseline_deaths = sum(bool(row[death_key]) for row in baseline_rows)
    candidate_deaths = sum(bool(row[death_key]) for row in candidate_rows)
    candidate_only_deaths = sum(
        (not bool(baseline[death_key])) and bool(candidate[death_key])
        for baseline, candidate in zip(baseline_rows, candidate_rows)
    )
    baseline_only_deaths = sum(
        bool(baseline[death_key]) and (not bool(candidate[death_key]))
        for baseline, candidate in zip(baseline_rows, candidate_rows)
    )
    # The two component one-sided bounds each spend alpha/2.  By the union
    # bound their difference is an at-least (1-alpha) upper confidence bound
    # even though the discordant categories are dependent.
    component_alpha = per_constraint_alpha / 2.0
    candidate_only_upper = _clopper_pearson_upper(
        candidate_only_deaths, n_pairs, component_alpha
    )
    baseline_only_lower = _clopper_pearson_lower(
        baseline_only_deaths, n_pairs, component_alpha
    )
    death_difference_upper = candidate_only_upper - baseline_only_lower
    observed_not_higher = candidate_deaths <= baseline_deaths
    death_bound_passed = death_difference_upper <= margin
    checks["deaths.observed_not_higher"] = (
        observed_not_higher
        if require_observed_deaths_not_higher
        else True
    )
    checks["deaths.noninferiority_upper_bound"] = death_bound_passed
    death_report = {
        "key": death_key,
        "method": "bonferroni-one-sided-clopper-pearson-risk-difference",
        "noninferiority_margin": margin,
        "require_observed_not_higher": require_observed_deaths_not_higher,
        "baseline_deaths": baseline_deaths,
        "candidate_deaths": candidate_deaths,
        "baseline_death_rate": baseline_deaths / n_pairs,
        "candidate_death_rate": candidate_deaths / n_pairs,
        "observed_candidate_minus_baseline_risk": (
            (candidate_deaths - baseline_deaths) / n_pairs
        ),
        "candidate_only_deaths": candidate_only_deaths,
        "baseline_only_deaths": baseline_only_deaths,
        "concordant_pairs": n_pairs - candidate_only_deaths - baseline_only_deaths,
        "component_alpha": component_alpha,
        "candidate_only_risk_upper_bound": candidate_only_upper,
        "baseline_only_risk_lower_bound": baseline_only_lower,
        "candidate_minus_baseline_risk_upper_bound": death_difference_upper,
        "observed_not_higher_passed": observed_not_higher,
        "noninferiority_passed": death_bound_passed,
        "passed": (
            death_bound_passed
            and (
                observed_not_higher
                if require_observed_deaths_not_higher
                else True
            )
        ),
    }

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "schema_version": R7_STATISTICS_SCHEMA,
        "method_revision": R7_METHOD_REVISION,
        "phase": phase,
        "n_pairs": n_pairs,
        "required_pairs": required_pairs,
        "familywise_alpha": alpha,
        "simultaneous_confidence": 1.0 - alpha,
        "constraint_count": constraint_count,
        "per_constraint_alpha": per_constraint_alpha,
        "source": source,
        "seeds_sha256": _canonical_sha256(seeds),
        "paired_data_sha256": _canonical_sha256(paired_hash_rows),
        "rules": [
            {
                "key": rule.key,
                "direction": rule.direction,
                "minimum_effect": float(rule.minimum_effect),
                "require_sign_test": rule.require_sign_test,
            }
            for rule in rules
        ],
        "metrics": metrics,
        "death_noninferiority": death_report,
        "verdict": {
            "status": "PASS" if not failed_checks else "FAIL",
            "checks": checks,
            "failed_checks": failed_checks,
        },
    }
    # This is also a final guard against accidental NaN/Infinity in a frozen
    # result.  The digest itself is intentionally left to the caller/file
    # archive layer, just like the existing official analysis files.
    json.dumps(result, allow_nan=False)
    return result


__all__ = [
    "MIN_DEVELOPMENT_PAIRS",
    "MIN_FINAL_PAIRS",
    "MetricRule",
    "R7_METHOD_REVISION",
    "R7_STATISTICS_SCHEMA",
    "SUPPORTED_EVAL_ARCHIVE_SCHEMA",
    "StatisticsContractError",
    "analyze_paired_archives",
]

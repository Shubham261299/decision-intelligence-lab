"""Explicit metric denominators; missing measurements remain null."""

from collections import defaultdict

from decisionlab.confidence import selection_confidence

from decisionlab.evaluation.models import (
    CaseEvaluation, ConfidenceBin, ConfidenceCoverage, ConfidenceSummary,
    MetricSummary, Rate, ReferenceConfidence,
)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def confidence_summary(results: list[CaseEvaluation]) -> ConfidenceSummary:
    samples: list[tuple[float, bool]] = []
    for result in results:
        rec = result.observation.run.recommendation
        estimate = selection_confidence(rec)
        if result.outcome == "selected" and estimate.probability is not None:
            samples.append((estimate.probability, result.outcome_correct))
    bins: list[ConfidenceBin] = []
    for index in range(5):
        lower, upper = index / 5, (index + 1) / 5
        group = [
            (confidence, correct) for confidence, correct in samples
            if min(int(confidence * 5), 4) == index
        ]
        group_mean = mean([p for p, _ in group])
        accuracy = mean([float(y) for _, y in group])
        bins.append(ConfidenceBin(
            lower=lower, upper=upper, count=len(group),
            mean_confidence=group_mean, accuracy=accuracy,
            fraction=len(group) / len(samples) if samples else None,
            reliability_gap=abs(group_mean - accuracy)
            if group_mean is not None and accuracy is not None else None,
        ))
    # Predeclared diagnostic thresholds, never learned from either split.
    curve: list[ConfidenceCoverage] = []
    for threshold in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        retained = [(p, y) for p, y in samples if p >= threshold]
        curve.append(ConfidenceCoverage(
            threshold=threshold,
            coverage=Rate(numerator=len(retained), denominator=len(results)),
            retention=Rate(numerator=len(retained), denominator=len(samples)),
            accuracy=Rate(numerator=sum(y for _, y in retained), denominator=len(retained)),
        ))
    selected = [r for r in results if r.outcome == "selected"]
    selected_accuracy = mean([float(r.outcome_correct) for r in selected])
    gaps = [b.reliability_gap for b in bins if b.reliability_gap is not None]
    return ConfidenceSummary(
        sample_count=len(samples), mean_confidence=mean([p for p, _ in samples]),
        brier_score=mean([(p - float(y)) ** 2 for p, y in samples]), bins=bins,
        availability="available" if samples else "unavailable",
        selected_count=len(selected), missing_selected_count=len(selected) - len(samples),
        excluded_abstention_confidence_count=sum(
            r.outcome == "abstained" and r.observation.run.recommendation is not None
            and r.observation.run.recommendation.confidence is not None for r in results
        ),
        excluded_failure_count=sum(r.outcome == "failure" for r in results),
        calibration_error=sum(b.count * (b.reliability_gap or 0) for b in bins)
        / len(samples) if samples else None,
        maximum_reliability_gap=max(gaps) if gaps else None,
        coverage_curve=curve,
        fixed_reference=ReferenceConfidence(
            sample_count=len(selected), accuracy=selected_accuracy,
            brier_score=0.25 if selected else None,
            calibration_error=abs(0.5 - selected_accuracy)
            if selected_accuracy is not None else None,
        ),
    )


def summarize(results: list[CaseEvaluation]) -> MetricSummary:
    n = len(results)
    selected = [r for r in results if r.outcome == "selected"]
    abstained = [r for r in results if r.outcome == "abstained"]
    expected_selection = [r for r in results if not r.expected_abstention]
    expected_abstention = [r for r in results if r.expected_abstention]
    correct_selected = sum(r.outcome_correct for r in selected)
    correct_abstained = sum(r.outcome_correct for r in abstained)
    tp = sum(r.system_review is True and r.expected_review for r in results)
    fp = sum(r.system_review is True and not r.expected_review for r in results)
    fn = sum(r.system_review is not True and r.expected_review for r in results)
    tn = sum(r.system_review is False and not r.expected_review for r in results)
    observations = [r.observation for r in results]
    schema_samples = [o for o in observations if o.schema_valid is not None]
    reference_samples = [o for o in observations if o.references_valid is not None]
    evidence_samples = [o for o in observations if o.evidence_references_valid is not None]
    candidate_selected = [
        r for r in results if r.observation.candidate is not None
        and r.observation.candidate.status == "selected"
    ]
    violations = sum(bool(r.hard_constraint_violations) for r in results)
    refs = sum(o.evidence_reference_count for o in observations)
    bad_refs = sum(o.invalid_evidence_reference_count for o in observations)
    counts = {
        "total": n, "successful_runs": sum(r.outcome != "failure" for r in results),
        "failed_runs": sum(r.outcome == "failure" for r in results),
        "selected": len(selected), "abstained": len(abstained),
        "expected_selection": len(expected_selection), "expected_abstention": len(expected_abstention),
        "schema_invalid": sum(o.schema_valid is False for o in observations),
        "references_invalid": sum(o.references_valid is False for o in observations),
        "hard_constraint_violating_outputs": violations,
        "review_true_positive": tp, "review_false_positive": fp,
        "review_false_negative": fn, "review_true_negative": tn,
        "review_unavailable": sum(r.system_review is None for r in results),
        "evidence_references": refs, "invalid_evidence_references": bad_refs,
        "outputs_without_evidence_references": sum(o.candidate is not None and not o.evidence_reference_count for o in observations),
    }

    def rate(numerator: int, denominator: int) -> Rate:
        return Rate(numerator=numerator, denominator=denominator)

    rates = {
        "outcome_correctness": rate(correct_selected + correct_abstained, n),
        "selection_correctness": rate(correct_selected, len(expected_selection)),
        "abstention_correctness": rate(correct_abstained, len(expected_abstention)),
        "selected_output_accuracy": rate(correct_selected, len(selected)),
        "decision_coverage": rate(len(selected), n),
        "abstention_precision": rate(correct_abstained, len(abstained)),
        "execution_failure_rate": rate(counts["failed_runs"], n),
        "schema_validity": rate(sum(o.schema_valid is True for o in schema_samples), len(schema_samples)),
        "reference_validity": rate(sum(o.references_valid is True for o in reference_samples), len(reference_samples)),
        "evidence_reference_validity": rate(sum(o.evidence_references_valid is True for o in evidence_samples), len(evidence_samples)),
        "evidence_citation_validity": rate(refs - bad_refs, refs),
        "hard_constraint_violation_rate": rate(violations, len(candidate_selected)),
        "human_review_accuracy": rate(sum(r.review_correct for r in results), n),
        "human_review_precision": rate(tp, tp + fp),
        "human_review_recall": rate(tp, tp + fn),
        "strategy_review_accuracy": rate(sum(r.strategy_review is not None and r.strategy_review == r.expected_review for r in results), n),
    }
    groups: dict[str, list[CaseEvaluation]] = defaultdict(list)
    for result in results:
        groups[result.case_id].append(result)
    repeated = [group for group in groups.values() if len(group) > 1]
    agreement = 0
    for group in repeated:
        keys = set()
        for r in group:
            rec = r.observation.run.recommendation
            keys.add((r.outcome, rec.selected_option_id if rec else None, r.system_review))
        agreement += len(keys) == 1 and all(r.outcome != "failure" for r in group)
    rates["repeated_decision_agreement"] = rate(agreement, len(repeated))
    latency = [r.observation.run.latency_ms for r in results if r.observation.run.latency_ms is not None]
    costs = [r.observation.run.usage.cost_usd for r in results if r.observation.run.usage is not None and r.observation.run.usage.cost_usd is not None]
    input_tokens = [
        r.observation.run.usage.input_tokens for r in results
        if r.observation.run.usage is not None and r.observation.run.usage.input_tokens is not None
    ]
    output_tokens = [
        r.observation.run.usage.output_tokens for r in results
        if r.observation.run.usage is not None and r.observation.run.usage.output_tokens is not None
    ]
    return MetricSummary(
        counts=counts, rates=rates, confidence=confidence_summary(results),
        latency_sample_count=len(latency), mean_latency_ms=mean(latency),
        cost_sample_count=len(costs), total_cost_usd=sum(costs) if costs else None,
        input_token_sample_count=len(input_tokens), total_input_tokens=sum(input_tokens) if input_tokens else None,
        output_token_sample_count=len(output_tokens), total_output_tokens=sum(output_tokens) if output_tokens else None,
    )

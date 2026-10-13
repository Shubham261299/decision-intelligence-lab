"""Research presentation from saved reports and explicitly labelled documentation."""

import hashlib
from pathlib import Path

import streamlit as st

from app import research_data as data
from app.presentation import render_table
from app.release_scope import research_is_withheld
from decisionlab.evaluation.benchmark import Manifest
from decisionlab.evaluation.human_review import HumanReviewRecord
from decisionlab.representation_experiment import PairedAnalysis, RepresentationReport

INTERPRETATIONS = {
    "Strategy comparison": "The original pipelines differed in prompts, output representation, parsing and schema enforcement. Their score gap does not isolate a causal benefit of any one component.",
    "JSON enforcement": "No observed decision or contract benefit from enforcement under this shared JSON prompt. This does not establish universal equivalence or rule out an interaction with another prompt.",
    "Output representation": "Output instructions and strict parser compliance affected usable outcomes in this dev setup. Agreement among successful paired targets does not establish better JSON reasoning or full semantic equivalence.",
}


def render_pairs(root: Path, experiment: str) -> None:
    if experiment == "Strategy comparison":
        st.caption("A matched-pipeline ablation was not performed in Experiment 1.")
        return
    path = data.DIRECTORIES[experiment] + "/paired-analysis.json"
    st.caption(f"Source: {path}")
    if experiment == "JSON enforcement":
        loaded_counts = data.read_model(root, path, data.EnforcementPairedAnalysis)
        if loaded_counts.data is None:
            st.warning(loaded_counts.error)
            return
        counts = loaded_counts.data.paired_counts
        render_table([{"Paired measure": name.replace("_", " "), "Count": value}
                      for name, value in counts.items()])
        st.caption("Complete candidate identity includes schema-valid candidates rejected for references; joint-valid agreement uses a narrower denominator.")
    else:
        loaded = data.read_model(root, path, PairedAnalysis)
        if loaded.data is None:
            st.warning(loaded.error)
            return
        paired = loaded.data
        st.write("Decision-target agreement among jointly successful pairs: "
                 + data.rate_text(paired.decision_agreement))
        render_table([
            {"Paired outcome": label, "Count": count}
            for label, count in [("Both correct", paired.both_correct),
                                 ("Labelled text only correct", paired.labelled_only_correct),
                                 ("JSON only correct", paired.json_only_correct),
                                 ("Neither correct", paired.neither_correct)]
        ])
        with st.expander("Per-case paired comparisons"):
            render_table([item.model_dump() for item in paired.pairs])


def render_diagnosis(root: Path) -> None:
    st.subheader("Prospective diagnosis - separate from primary scoring")
    path = "runs/representation/dev-20261009/experiment.json"
    loaded = data.read_model(root, path, RepresentationReport)
    if loaded.data is None:
        st.warning(f"{path}: {loaded.error}")
        return
    report = loaded.data
    rows = []
    for condition, summary in report.diagnostics.items():
        rows.append({"Condition": condition, "Failed responses": summary.failed_responses,
                     "Acceptable explicit targets rejected": summary.apparent_correct_rejected_targets,
                     "Incorrect explicit targets": summary.explicit_incorrect_targets,
                     "Ambiguous targets": summary.ambiguous_targets,
                     "Full semantic correctness": "Not assessed"})
    render_table(rows)
    st.caption("Target matching is narrower than semantic correctness. Diagnostic labels do not repair responses or change primary scores.")
    st.write(f"Recorded attempts: {len(report.observations)}. Returned raw responses: "
             f"{sum(item.raw_response.text is not None for item in report.observations)}.")
    with st.expander("Raw responses and mechanical diagnoses"):
        identifiers = [f"{item.scheduled.sequence}: {item.scheduled.case_id} / "
                       f"{item.scheduled.condition} / repeat {item.scheduled.repetition}"
                       for item in report.observations]
        if identifiers:
            chosen = st.selectbox("Recorded observation", identifiers, key="raw-observation")
            item = report.observations[identifiers.index(chosen)]
            if item.raw_response.text is None:
                st.warning("Raw model response unavailable for this attempt; it has not been reconstructed.")
            else:
                st.code(item.raw_response.text, language="text")
            st.json(item.diagnosis.model_dump(mode="json"))


def render_review(root: Path) -> None:
    path = "docs/reviews/experiment-3-first-pass-v1.json"
    loaded = data.read_model(root, path, HumanReviewRecord)
    st.subheader("Limited first-pass review")
    st.caption(f"Source: {path}")
    if loaded.data is None:
        st.warning(loaded.error)
        return
    st.warning("User-accepted, assistant-assisted first-pass assessments; not independent blind expert validation.")
    render_table([
        {"Item": item.review_id, "Decision target": item.decision_target,
         "Reasoning/evidence": item.reasoning_evidence, "Policy": item.policy_compliance,
         "Review label": item.review_requirement, "Review note": item.review_requirement_note}
        for item in loaded.data.reviewed_items
    ])
    st.caption("Primary score effect: none. Selected responses and repeated observations are not independent judgments or a representative semantic review.")


def render_dashboard(root: Path | None = None) -> None:
    root = root if root is not None else data.ROOT
    st.caption("DECISION INTELLIGENCE LAB / SAVED RESEARCH")
    st.title("Research & Experiments")
    st.write("How can LLM-based decision making become more measurable, structured, explainable and reliable?")
    st.info("Descriptive results on a small synthetic service-routing benchmark. No general model superiority, statistical significance or production reliability is established.")
    manifest = data.read_model(root, "evals/manifest.json", Manifest)
    if manifest.data:
        m = manifest.data
        st.caption(f"Benchmark {m.benchmark} v{m.version} | {m.case_count} synthetic cases | "
                   + " / ".join(f"{split}: {len(ids)}" for split, ids in m.splits.items()))
    else:
        st.warning(manifest.error)
    st.caption("Selection, abstention, execution failure, policy, review and confidence are separate dimensions. Held-out labels are public: this is a development holdout, not a blind external test.")
    if research_is_withheld(root):
        st.info("The public app does not load local raw reports, saved experiment results, "
                "diagnoses or review packets. Findings and accepted annotations already committed "
                "in the existing repository history remain unchanged; this display restriction "
                "does not remove them.")
        st.subheader("Available experiment software")
        st.write("Compare strategies on the same frozen inputs; vary schema enforcement with matched JSON prompts; "
                 "compare output representations with prospective diagnostics. These are implemented workflows, "
                 "not published outcome claims.")
        st.caption("Use the Decision Playground for offline examples. Benchmark schemas, policy and metric definitions "
                   "are included in the source and documentation. The local raw archive remains "
                   "unapproved for publication; existing committed findings and annotations are preserved.")
        st.warning("Structural/reference validity does not prove semantic support or factual correctness.")
        return
    experiment = st.selectbox("Experiment", list(data.COMPARISONS), key="research-experiment")
    questions = {
        "Strategy comparison": "Experiment 1 - RuleBased vs DirectLLM vs StructuredLLM. Compare complete strategy pipelines on dev and held-out separately.",
        "JSON enforcement": "Experiment 2 - Same JSON prompt, schema description, model, settings and local validation; vary only Ollama schema enforcement. Dev only.",
        "Output representation": "Experiment 3 - Shared policy/input/settings and no schema enforcement; vary labelled-text vs JSON instructions and strict parsing. Dev only, counterbalanced order.",
    }
    st.subheader(questions[experiment].split(" - ")[0])
    st.write(questions[experiment].split(" - ", 1)[1])
    split = st.selectbox("Recorded split", list(data.COMPARISONS[experiment]), key=f"split:{experiment}")
    relative = data.COMPARISONS[experiment][split]
    loaded = data.load_comparison(root, relative)
    st.caption(f"Artifact-derived metrics | Source: {relative}")
    if loaded.data is None:
        st.warning(loaded.error)
        st.caption("Saved runs are ignored by Git and may be absent from a checkout. No experiment is rerun and no substitute scores are generated.")
    else:
        report = loaded.data
        st.caption(f"{len(report.case_ids)} cases x {report.repeats} repetitions x "
                   f"{len(report.columns)} conditions = "
                   f"{sum(c.report.summary.counts['total'] for c in report.columns.values())} recorded strategy attempts. "
                   "RuleBased attempts, where present, are non-LLM.")
        chart = data.correctness_chart(report)
        if chart:
            st.bar_chart(chart, x="Condition", y="Correctness (%)", color="#8ca9c9")
        render_table(data.metric_rows(report))
        st.caption("Each rate retains its recorded numerator/denominator. Failures are included in overall correctness. Citation validity checks IDs, not semantic support; policy violations are assessable only for suitable candidates.")
        st.subheader("Failures, verification and confidence")
        for name, column in report.columns.items():
            with st.expander(name):
                st.write("Failure codes", column.failure_codes)
                st.write("Known invalid output rate: " + data.rate_text(column.invalid_output_rate))
                st.write("Wrong accepted selections: " + str(column.wrong_selected_decisions))
                st.write("Verifier dispositions", column.report.verification_summary.model_dump())
                confidence = column.report.summary.confidence
                st.write(f"Reported confidence: {confidence.status}; availability: {confidence.availability}; samples: {confidence.sample_count}")
                st.write({"mean": confidence.mean_confidence, "Brier": confidence.brier_score,
                          "calibration error": confidence.calibration_error})
                st.caption("Only available selected-output confidence is scored; failures, abstentions and missing confidence limit interpretation. Confidence is not correctness or calibrated reliability.")
                render_table([{"Provider": profile.provider or "None (deterministic)",
                               "Model": profile.model_id or "Not applicable", "Strategy version": profile.strategy_version,
                               "Prompt version": profile.prompt_version or "Not applicable"}
                              for profile in column.metadata_profiles])
        with st.expander("Per-case primary outcomes"):
            render_table(data.case_rows(report))
        st.subheader("Paired outcomes")
        render_pairs(root, experiment)
        for warning in report.comparability_warnings:
            st.warning("Recorded comparability warning: " + warning)
    st.subheader("Interpretation, not a new measurement")
    st.write(INTERPRETATIONS[experiment])
    st.caption("Source: docs/research-findings.md. These statements are documented interpretations, not replacement metrics when reports are missing.")
    st.write("Untested hypothesis: representation-induced generation and instruction effects may interact with parser rigidity. These experiments do not isolate a parser-only causal effect.")
    if experiment == "Output representation":
        render_diagnosis(root)
        render_review(root)
    else:
        st.warning("Prospectively preserved raw responses are unavailable for this experiment. Structured candidates and provider metadata are not raw model text.")
    st.subheader("Evidence & reproducibility")
    directory = data.DIRECTORIES[experiment]
    protocol, protocol_error = data.read_document(root, directory + "/protocol.json")
    if protocol_error:
        st.warning(protocol_error)
    else:
        assert protocol is not None
        st.json({key: protocol[key] for key in ["recorded_at", "benchmark_version", "manifest_sha256",
                 "split", "repeats", "planned_generations", "independent_variable", "controlled_variables"] if key in protocol})
        config = protocol.get("configuration")
        if isinstance(config, dict):
            st.json({key: config[key] for key in ["requested_model", "provider", "model_id", "temperature",
                     "max_output_tokens", "timeout_seconds", "max_retries"] if key in config})
        if loaded.data and protocol.get("manifest_sha256") != loaded.data.manifest_sha256:
            st.warning("Protocol/report benchmark hashes differ or are unavailable; inspect provenance before attribution.")
    if loaded.data and manifest.data:
        current_hash = hashlib.sha256((root / "evals/manifest.json").read_bytes()).hexdigest()
        if loaded.data.manifest_sha256 != current_hash:
            st.warning("Saved report refers to a different benchmark manifest; it is historical evidence, not current benchmark validation.")
    paths = [relative, directory + "/protocol.json", directory + "/audit.json"]
    if experiment != "Strategy comparison":
        paths.append(directory + "/paired-analysis.json")
    if experiment == "Output representation":
        paths += [directory + "/experiment.json", directory + "/observations.jsonl",
                  directory + "/human-review-packet.json", directory + "/human-review-finalization-v1.json"]
    render_table([{"Artifact": path, "Availability": "Present (not a full integrity audit)"
                   if (root / path).is_file() else "Missing"} for path in paths])
    audit, audit_error = data.read_document(root, directory + "/audit.json")
    if audit_error:
        st.warning(audit_error)
    elif audit:
        st.json({key: audit[key] for key in ["generation_attempts", "returned_model_responses", "execution_failures",
                 "provider_timeouts", "condition_first_pair_counts", "excluded_partial_attempt",
                 "raw_response_and_envelope_hashes_valid", "human_review_status"] if key in audit})
        st.caption("Audit values are recorded claims, not checks rerun by the dashboard. Historical pending review status is superseded only by the separate accepted annotation record.")
    st.warning("Structural/reference validation and verifier acceptance do not establish factual correctness or semantic evidence support. Related cases and repetitions are not independent samples; dev has been examined extensively.")
    with st.expander("Documented research findings (source text)"):
        try:
            st.markdown((root / "docs/research-findings.md").read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError):
            st.warning("Unavailable evidence: docs/research-findings.md")

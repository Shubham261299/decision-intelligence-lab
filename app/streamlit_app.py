"""Launch with python -m streamlit run app/streamlit_app.py."""

from html import escape

import streamlit as st

from app.playground import EXAMPLES, decide_json, example_json
from decisionlab.evaluation.models import Observation


def validation_label(value: bool | None) -> str:
    return "Not assessed" if value is None else "Valid" if value else "Invalid"


def plain_markdown(value: str) -> str:
    """Display untrusted text literally in Markdown-capable Streamlit elements."""
    # Escape HTML/autolinks first, then Markdown delimiters (backslash first).
    value = escape(value)
    for character in "\\`*_{}[]()#+-.!|~":
        value = value.replace(character, "\\" + character)
    return value


def render_result(result: Observation) -> None:
    run = result.run
    recommendation = run.recommendation
    verification = result.verification
    st.subheader("Decision result")
    summary = [
        ("Execution", run.status.title()),
        ("Decision", recommendation.selected_option_id
         if recommendation and recommendation.selected_option_id
         else "Abstained" if recommendation else "Unavailable"),
        ("Policy Review", "Required" if run.policy_result and run.policy_result.requires_human_review
         else "Not required" if run.policy_result else "Unavailable"),
        ("Verification", verification.disposition.replace("_", " ").title()
         if verification else "Unavailable"),
    ]
    cards = "".join(
        f'<div class="summary-card"><div class="summary-label">{escape(label)}</div>'
        f'<div class="summary-value">{escape(value)}</div></div>'
        for label, value in summary
    )
    st.markdown(f'<div class="decision-summary">{cards}</div>', unsafe_allow_html=True)
    st.caption(plain_markdown(f"Strategy: {run.metadata.strategy} · version {run.metadata.strategy_version}"))
    if run.error:
        st.error(plain_markdown(f"Execution failure · {run.error.code}: {run.error.message}"))
    if recommendation:
        st.write(plain_markdown(recommendation.explanation))
        st.subheader("Option eligibility")
        if recommendation.assessments:
            rows = "".join(
                '<tr>'
                f'<td>{escape(item.option_id)}</td>'
                f'<td><span class="eligibility {item.eligibility}">{escape(item.eligibility.title())}</span></td>'
                '<td>' + '<br>'.join(escape(reason) for reason in item.reasons) + '</td>'
                f'<td>{escape(", ".join(item.evidence_ids) or "None")}<br>'
                f'<span class="table-detail">Constraints: {escape(", ".join(item.constraint_ids) or "None")}</span></td>'
                '</tr>'
                for item in recommendation.assessments
            )
            st.markdown(
                '<div class="eligibility-scroll"><table class="eligibility-table">'
                '<caption>Strategy-reported eligibility and supporting references</caption>'
                '<thead><tr><th scope="col">Option</th><th scope="col">Eligibility</th>'
                '<th scope="col">Reasons</th><th scope="col">References</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>', unsafe_allow_html=True,
            )
        else:
            st.info("No option assessments supplied.")
        st.subheader("Unknown or unresolved inputs")
        if not recommendation.unknowns:
            st.caption("No unknowns reported by the strategy.")
        for unknown in recommendation.unknowns:
            if unknown.material_to_decision:
                st.warning(plain_markdown(unknown.description))
            else:
                st.write(plain_markdown(unknown.description))
            st.caption(plain_markdown(f"Material: {unknown.material_to_decision} · affected options: "
                       + (", ".join(unknown.affected_option_ids) or "None specified")))
    policy_column, verification_column = st.columns(2)
    with policy_column:
        st.subheader("Policy findings")
        if run.policy_result:
            if run.policy_result.requires_human_review:
                st.warning("Human review required by policy.")
            else:
                st.caption("Human review is not required by policy.")
            st.write(plain_markdown("Review reasons: " + (", ".join(run.policy_result.review_reasons) or "None")))
            if run.policy_result.detected_violations:
                st.error(plain_markdown("Violations: " + ", ".join(run.policy_result.detected_violations)))
            else:
                st.caption("No explicit policy violations detected.")
        else:
            st.caption("Policy outcome unavailable; execution did not produce an accepted result.")
    with verification_column:
        st.subheader("Verifier findings")
        if verification and verification.disposition == "human_review":
            st.warning("Human Review - this recommendation needs review.")
        st.write(f"Structure: {validation_label(result.schema_valid)}")
        st.write(f"Contextual references: {validation_label(result.references_valid)}")
        st.write(f"Evidence references: {validation_label(result.evidence_references_valid)}")
        st.caption("Semantic evidence support: not assessed. Verification does not prove factual correctness.")
        if result.verification_error:
            st.error("Verifier execution failed; the original decision remains unchanged.")
        elif verification:
            for finding in verification.findings:
                st.write(plain_markdown(f"{finding.kind.replace('_', ' ')} · {finding.code}"))
            if not verification.findings:
                st.caption("No verifier findings.")
        else:
            st.caption("No verification result available.")
    with st.expander("Full structured observation"):
        st.json(result.model_dump(mode="json"))


def main() -> None:
    st.set_page_config(page_title="DecisionLab", page_icon="◈", layout="wide")
    st.markdown("""<style>
    .stApp { background: #10151f; color: #e6edf6; }
    [data-testid="stHeader"] { background: #10151f; }
    .block-container { max-width: 1200px; padding-top: 2.5rem; }
    .decision-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem; margin: .5rem 0 1.5rem; }
    .summary-card { background: #1b2433; padding: 1.25rem; border: 1px solid #303d50;
        border-radius: 12px; }
    .summary-label { font-size: .85rem; color: #b8c4d6; margin-bottom: .65rem; }
    .summary-value { font-size: clamp(1.25rem, 2vw, 1.7rem); font-weight: 600;
        line-height: 1.35; white-space: normal; overflow-wrap: anywhere; }
    .eligibility-scroll { overflow-x: auto; margin-bottom: 1.5rem; }
    .eligibility-table { width: 100%; table-layout: fixed; border-collapse: collapse;
        font-size: .95rem; min-width: 520px; }
    .eligibility-table caption { text-align: left; color: #b8c4d6; padding-bottom: .6rem; }
    .eligibility-table th, .eligibility-table td { text-align: left; vertical-align: top;
        white-space: normal; overflow-wrap: anywhere; padding: .9rem .7rem;
        border-bottom: 1px solid #303d50; }
    .eligibility-table th:nth-child(1) { width: 15%; }
    .eligibility-table th:nth-child(2) { width: 18%; }
    .eligibility-table th:nth-child(3) { width: 42%; }
    .eligibility-table th:nth-child(4) { width: 25%; }
    .eligibility { display: inline-block; padding: .2rem .5rem; border-radius: 5px;
        background: #243247; color: #e6edf6; font-weight: 600; }
    .eligibility.ineligible { background: #303540; color: #c9cfd8; }
    .eligibility.unknown { background: #463b25; color: #f2d596; }
    .table-detail { color: #b8c4d6; font-size: .85rem; }
    @media (max-width: 950px) { .decision-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 480px) { .decision-summary { grid-template-columns: 1fr; } }
    </style>""", unsafe_allow_html=True)
    from app.presentation import TABLE_STYLE

    st.markdown(TABLE_STYLE + """<style>
    .block-container { padding-bottom: 3rem; }
    h1 { line-height: 1.12; letter-spacing: -.025em; }
    h2 { margin-top: 1.1rem; }
    [data-testid="stMarkdownContainer"] p { line-height: 1.65; overflow-wrap: anywhere; }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
        min-height: 2.8rem; white-space: normal; }
    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.5rem; }
        .summary-card { padding: 1rem; }
        .eligibility-table { min-width: 600px; }
    }
    </style>""", unsafe_allow_html=True)
    st.sidebar.title("DecisionLab")
    page = st.sidebar.selectbox("Workspace", ["Overview", "Decision Playground", "Research Dashboard"], key="page")
    st.sidebar.caption("Synthetic decisions. Reproducible research. Offline by default.")
    if page != "Overview":
        from app.overview import navigate

        st.sidebar.button("Return to Overview", on_click=navigate, args=("Overview",), key="return-overview")
    if page == "Overview":
        from app.overview import render_overview

        render_overview()
        return
    if page == "Research Dashboard":
        from app.research_dashboard import render_dashboard

        render_dashboard()
        return
    st.caption("DECISION INTELLIGENCE LAB / OFFLINE PLAYGROUND")
    st.title("Inspect a decision. Understand its boundaries.")
    st.write("Run a synthetic service-routing problem through the existing rules, policy and verifier.")
    st.info("Synthetic examples, not operational advice. This playground reports one decision; "
            "it does not measure benchmark accuracy or run an LLM.")
    example = st.radio("Example", list(EXAMPLES), horizontal=True)
    st.caption("Routine routing should select routine. Missing urgency should abstain and require review.")
    try:
        initial = example_json(example)
    except ValueError as error:
        st.error(str(error))
        return
    with st.form("decision-input"):
        st.caption("Offline strategy: RuleBased v1. Run the selected scenario or edit its inputs below.")
        with st.expander("Inspect or edit input JSON", expanded=False):
            text = st.text_area("DecisionProblem JSON", value=initial, height=280, key=f"input:{example}")
            st.caption("Use synthetic inputs only. Changes take effect when you run the decision.")
        submitted = st.form_submit_button("Run verified decision", type="primary")
    if submitted:
        try:
            result = decide_json(text)
        except ValueError as error:
            st.error(str(error))
        else:
            render_result(result)


if __name__ == "__main__":
    main()

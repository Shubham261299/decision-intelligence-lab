# DecisionLab Portfolio Demo

The local Streamlit demo opens on Overview, alongside the Decision Playground and Research Dashboard.
The playground is a presentation layer over DecisionLab's existing
`run_verified_decision` path. It runs RuleBasedStrategy only: no provider
configuration, credentials, live generation, external service or benchmark scoring.

## Local Setup and Launch

Use Python 3.11+ and run from the project root. In your activated environment,
install the optional demo extra when needed (not required for normal CLI use):

```powershell
python -m pip install -e ".[demo]"
$env:PYTHONPATH = 'src;.'
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false --theme.base dark
```

For Bash, use `PYTHONPATH=src:.` before the same launch command. Open the local
URL printed by Streamlit. Binding to loopback keeps this unauthenticated demo local;
the command also disables Streamlit usage telemetry.

## Walkthrough

Overview introduces the research question, framework, methodology and limitations.
The public profile does not load saved local reports or interpretations, even
if their files exist locally. Existing findings and accepted annotations remain in
repository history; this presentation restriction does not remove them. Use
**Open Decision Playground**, **Explore Research Dashboard**, or the sidebar
workspace selector to navigate. Return to **Overview** through the sidebar.
Navigation uses Streamlit-native callbacks; it never invokes a model or experiment.
The compact **Return to Overview** sidebar button is available in both workspaces.
Research tables wrap full labels and values instead of truncating cells. On narrow
screens, use the labelled table region's horizontal scrollbar (or keyboard scrolling
after focusing it) to see every column. Longer tables also scroll vertically;
print styling expands them for PDF capture. Rates and missing-data states are unchanged.
The GitHub link is intentionally a manual-confirmation placeholder until a repository
URL is verified in project configuration or documentation.

Select a scenario; input JSON is available in the collapsed **Inspect or edit input
JSON** section. The primary **Run verified decision** button remains visible.
Choose **Routine routing** and click **Run verified decision**: execution succeeds,
`routine` is selected, no policy review is required and the verifier accepts.
Choose **Missing urgency** and run: execution succeeds with abstention, policy
requires human review and the verifier returns `human_review`.

The editable JSON comes from `examples/decision-routine.json` and
`examples/decision-missing-urgency.json`. Inputs are validated as `DecisionProblem`;
invalid inputs do not execute. Results expose eligibility, reasons, unknowns,
policy violations, reference validation and separate verifier findings. Execution
failures remain failures, never substituted abstentions. No results are saved.
The result summary wraps important states such as Human Review. Eligibility reasons
wrap in a table, with separate sections for unknown inputs, policy and verification.

These are synthetic routing examples. Structural validity and resolvable evidence
IDs do not establish semantic support or factual truth. Verifier acceptance is not
an accuracy score, production certification or permission to execute routing.
The playground does not read benchmark expectations or saved research results.

## Research & Experiments Dashboard

Choose **Research Dashboard** in the sidebar workspace selector; switch back to
**Decision Playground** to run synthetic examples. Launch instructions are unchanged.
The dashboard presents benchmark scope and the implemented experimental approach.
Local raw reports, responses, diagnostics and annotations are not loaded or
reconstructed. Existing committed findings/review records remain separately readable. No strategy is
executed from this workspace. See [release scope](software-first-release.md).

The public presentation profile is declared in the root `software-first-release.json`.
Its old export inventory is superseded by the single-repository plan. Missing
or malformed configuration also withholds evidence. A private local-review mode
is separate from publication approval; it is never enabled by the public manifest.

## Checks and Boundaries

```powershell
python -m pytest -q tests/test_demo.py tests/test_research_dashboard.py tests/test_overview.py tests/test_presentation.py --basetemp .pytest_cache/demo-tests
python -m pytest -q --basetemp .pytest_cache/all-tests
```

Integration tests run without Streamlit; rendered-app tests run only when the demo
extra is available. Local `.streamlit/` configuration and secrets are ignored,
alongside environment files and caches. There is no authentication or live-provider
selector, and this phase does not deploy the app.

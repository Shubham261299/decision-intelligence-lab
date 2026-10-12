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
Its experiment summaries read saved reports, keeping observations separate from
documented interpretations. Missing evidence remains visibly unavailable. Use
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
The dashboard only reads saved files; it never invokes a strategy or re-evaluates
the benchmark. Each experiment has its own split selector, primary metric table,
correctness chart, per-case outcomes, failure codes, confidence diagnostics and
provenance. Rates retain recorded numerators and denominators.

Evidence sources:

- Experiment 1: `runs/ollama-first/{dev,held-out}/comparison.json`, root protocol/audit.
- Experiment 2: `runs/matched-json/dev/comparison.json`, root paired analysis/protocol/audit.
- Experiment 3: `runs/representation/dev-20261009/` comparison, typed experiment
  report, paired analysis, protocol, audit and raw-response journal.
- Benchmark scope: `evals/manifest.json`; interpretations: `docs/research-findings.md`;
  accepted annotations: `docs/reviews/experiment-3-first-pass-v1.json`.

Comparison files use the existing `ComparisonReport` contract. Experiment 3 uses
`RepresentationReport` and `PairedAnalysis`; Experiment 2's older paired JSON is
handled separately. Missing or malformed evidence is marked unavailable, never
replaced by invented metrics. `runs/` remains ignored, so a fresh checkout may show
methodology and review documentation without local experiment metrics.

Experiments 1 and 2 lack prospective raw-response records. Experiment 3 retains
120 attempts and 119 raw responses; its timeout has no reconstructable response.
The interrupted `runs/representation/dev/` attempt is excluded. Historical pending
review flags remain unchanged; the separate accepted first-pass record documents
later acceptance. Review is assistant-assisted, not independent blind validation.
The dashboard reports recorded audit claims and artifact availability, not a full
independent integrity certification. No held-out ablation results are invented.

## Checks and Boundaries

```powershell
python -m pytest -q tests/test_demo.py tests/test_research_dashboard.py tests/test_overview.py tests/test_presentation.py --basetemp .pytest_cache/demo-tests
python -m pytest -q --basetemp .pytest_cache/all-tests
```

Integration tests run without Streamlit; rendered-app tests run only when the demo
extra is available. Local `.streamlit/` configuration and secrets are ignored,
alongside environment files and caches. There is no authentication or live-provider
selector, and this phase does not deploy the app.

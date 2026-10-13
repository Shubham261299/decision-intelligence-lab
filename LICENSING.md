# DecisionLab Licensing Scope

Copyright (c) 2026 Shubham261299, the DecisionLab repository owner.
The owner confirmed independent personal creation and selected the license split
below. Attribution uses the verified personal GitHub handle, not an inferred legal
name. Repository history establishes 2026; no earlier copyright years are asserted.

## MIT: Original Software

The [MIT License](LICENSE) applies to original project software:

- `src/decisionlab/**/*.py` and `src/decisionlab/prompts/*.txt`.
- `app/**/*.py`, `tests/**/*.py`, and `evals/build_robustness.py`.
- Original software configuration: `pyproject.toml`, `.gitignore`, `.gitattributes`,
  `.env.example` and `software-first-release.json`; original code snippets in project documentation.

This grants rights only to the owner's original software, not imported libraries,
model weights, provider services, vendored code, or material with another notice.

## CC BY 4.0: Eligible Original Research Material

[Creative Commons Attribution 4.0 International](LICENSE-CC-BY-4.0) applies to the
owner's eligible original material, to the extent the owner holds licensable rights:

- Original prose in `README.md`, `AGENTS.md`, and `docs/**/*.md`, excluding code
  snippets covered above, attributed quotations and third-party license text.
- Original synthetic benchmark inputs, expectations, rubrics and manifests in
  `evals/**/*.json` and `evals/**/*.jsonl`; original example data in `examples/`.
- Original protocols and research metadata, including `docs/representation-protocol.json`,
  `docs/research/index.json` and the owner's original review annotations.
- The owner's original selection, organization and explanatory contributions in
  the research archive. Mixed evidence files are not blanket-licensed in full.

Suggested attribution: "DecisionLab — Shubham261299 (2026)", with the repository
URL `https://github.com/Shubham261299/decision-intelligence-lab`, the CC BY 4.0 link
`https://creativecommons.org/licenses/by/4.0/`, and an indication of modifications.
Keep supplied attribution/provenance; licensing does not establish factual accuracy,
independent authorship of model output, blind review, or scientific reliability.
Facts or other material outside copyright are not made exclusive by this notice.

## Excluded or Uncertain Scope

Exclusion from DecisionLab’s license grant does not itself prohibit redistribution. Applicable rights and terms must be assessed separately.

The [existing-repository release plan](docs/software-first-release.md) preserves
the initial commit's findings and accepted annotations. Their presence in history
does not establish redistribution rights. Raw archive copies and private rights-review
records remain local and are not approved for addition. Retained local documentation
does not supply permission or clearance by itself.

- Dependencies and tools (including Pydantic, Streamlit, pytest, Ruff and mypy),
  Ollama/OpenAI software or services, and model weights retain their own terms.
  Their names and recorded identifiers are not relicensed here. They are not
  bundled with the project licenses.
- Model-generated raw text, generated recommendation explanations/assessments,
  provider envelopes and any third-party passages embedded in research records
  are excluded from these grants. This includes the corresponding fields in
  `comparison.json`, `experiment.json`, review packets and assistant-analysis files.
  Model-generated text is not presumed to be owned by the project owner or to
  inherit the model's license. Separate rights review remains unresolved for that
  material; the archive is still local/unpublished and no blanket redistribution
  clearance is claimed.
- Third-party quotations, screenshots, logos, trademarks or separately noticed
  content, if present, retain their own applicable terms. No third-party rights
  are granted by these notices. Privacy, publicity, trademark and patent rights
  are not resolved by the research license.
- Review keys, interrupted runs, credentials, caches and uncurated local outputs
  are not part of the public release inventory and are not newly licensed here.

No historical evidence file is rewritten to apply this split. Use this scope
notice with the archive index; original provenance and assistant-assisted review
limitations remain intact. If an embedded passage has unresolved third-party
rights, stop before publishing it rather than assuming these licenses cover it.

## Official Text Sources

- MIT: https://opensource.org/license/mit (only the holder/year placeholders are filled).
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/legalcode.txt
  (official legal code, without changing its terms).

The CC legal code itself is a Creative Commons public-domain legal tool, not an
original DecisionLab work; see https://creativecommons.org/policies/.

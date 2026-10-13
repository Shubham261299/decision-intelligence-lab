# Existing-Repository Release Plan (Option A)

DecisionLab retains `Shubham261299/decision-intelligence-lab` as its single
portfolio repository, including existing Git history. No second repository,
exported snapshot, replacement root commit, history rewrite or force-push is planned.
This filename is retained so existing documentation links keep working.

## Existing History Versus Local Evidence

The initial commit `0fed538` contains `docs/research-findings.md`,
`docs/reviews/experiment-3-first-pass-v1.json` and `tests/test_human_review.py`.
They remain unchanged in existing history. Cached `origin/main` points to that
commit; current GitHub contents require a separately authorized remote refresh.
Existing publication does not establish redistribution clearance. Accepted review
is user-accepted and assistant-assisted, not independent blind expert validation.

All untracked `docs/research/**` files, including its README, index, rights-review
records and 23 archived evidence copies, remain local and unapproved for addition.
Original `runs/**`, review keys, interrupted runs, secrets and caches also stay local.
Do not delete or rewrite evidence to conceal its history or alter reported results.

## Licensing and Presentation

[MIT](../LICENSE) applies to original software; [CC BY 4.0](../LICENSE-CC-BY-4.0)
applies to eligible authored documentation/data under [the scope notice](../LICENSING.md).
Dependencies and model/third-party content are not relicensed. Unresolved embedded
content remains subject to separate review, including content already in history.

The public app does not load local raw reports or review packets. That restriction
is not removal of already committed findings or annotations. The root
`software-first-release.json` is retained only because the existing presentation
helper and tests require it. Its 138-file legacy export inventory is superseded;
it is not a staging list or an exclusion from Git history. There is no export step.
[Demo](demo.md) and [offline walkthrough](offline-quickstart.md) remain usable
without live providers or saved raw reports.

## Explicit Proposed Staging List ? Not Executed

The portfolio work is already committed locally as `4d22dae`; do not recreate
that commit or stage its unchanged files. Review that commit together with the
following proposed follow-up paths:

```text
README.md
LICENSING.md
LICENSE
LICENSE-CC-BY-4.0
docs/software-first-release.md
docs/PLANS.md
docs/demo.md
docs/offline-quickstart.md
app/overview.py
app/research_dashboard.py
app/release_scope.py
tests/test_overview.py
tests/test_research_dashboard.py
tests/test_software_first.py
software-first-release.json
```

The presentation helper, tests and JSON profile are directly required by current
app changes; they are not a separate-snapshot publishing mechanism. Exclude every
`docs/research/**` and `runs/**` path. Leave the archive-specific `.gitattributes`
change unstaged: it is not needed while the archive is unpublished. Do not use
`git add .`. Existing tracked findings, accepted annotations and their test require
no staging because they are unchanged; they are not silently omitted from history.

## Approval Gates

1. Review the local portfolio commit and the exact follow-up diff and license scope.
2. Run offline tests and `git diff --check`; keep archive bytes and benchmarks intact.
3. Obtain separate authorization to refresh remote information. Do not assume the
   cached ahead/behind status is current, or that the existing remote is empty.
4. Confirm the personal commit identity and Git push authentication separately.
5. Only after approval, stage explicit approved paths and review the complete
   staged diff. Commit and push require separate authorization; no such commands
   are executed by this preparation task. Stop on unexpected remote divergence;
   do not rewrite history, change remotes or force-push.

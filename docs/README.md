# Documentation index

Since the reorganisation of 2026-09-23 the documents are grouped by purpose:

| Folder | Contents |
|---|---|
| [`design/`](design/) | Design notes. The main document is [`design/DESIGN.md`](design/DESIGN.md) (twenty iterations, seventeen lessons); also the course plan [`ROADMAP-course-plan.md`](design/ROADMAP-course-plan.md) (2026-07-11) and the design memo [`DESIGN-gear-and-potion-autonomy.md`](design/DESIGN-gear-and-potion-autonomy.md) |
| [`prereg/`](prereg/) | Per-case pre-registrations (`PREREG-*.md`: v23 to v33, the R1/R2 re-anchoring cases, R7 to R9, B1, G1 and E-fix) |
| [`forensics/`](forensics/) | Forensic, autopsy and archive-audit reports |
| [`protocol/`](protocol/) | Protocol specifications for R19-R21, the operator notes for the case driver [`OPS-launcher.md`](protocol/OPS-launcher.md), and the R7/R8-era protocol notes moved out of the README, [`PROTOCOL-V4-NOTES.md`](protocol/PROTOCOL-V4-NOTES.md) |
| [`rounds/`](rounds/) | Round papers for R9-R19: pre-registrations, verdicts, probe reports and reviews; the R9, R10 and R12 launch records ([R9](rounds/r9-LAUNCH-RECORD-20260823.md), [R10](rounds/r10-LAUNCH-RECORD-20260827.md), [R12](rounds/r12-LAUNCH-RECORD-20260830.md)), the [R11 notes](rounds/r11-NOTES-20260828.md) and the [R18-B launch evidence](rounds/r18-B-LAUNCH-EVIDENCE-20260907.md). Also the report of the option brain's round-9 exam of 2026-09-24, [`round9-exam.md`](rounds/round9-exam.md), with per-game statistics [`round9-exam-results.json`](rounds/round9-exam-results.json) and poster images in `round9-media/`. The option brain's round numbers are unrelated to the R9-R19 campaigns |
| [`assets/`](assets/) | Charts and baseline files; code reads them by path, so do not move them |
| [`milestones/`](milestones/) | Milestone evidence packages, each sealed by its `manifest.json` + `verify.py`; their contents must not change |

Patches 0002 and 0003 were proposed upstream as DevilutionX PR #8606.

## Translation

Documents were translated to English in September 2026; some internal notes were not published.

## Path placeholders

On 2026-09-23 local folder names in paths were replaced with neutral placeholders in model zip metadata, the r2
evaluation archives, the leaderboard provenance markers and 25 text files; SHA-256 values recorded before that
date for those files refer to the earlier bytes. The model cards and drivers give the earlier and current
digests of the models and archives.

## Old paths

Documents written before 2026-09-23 may cite paths in their old form. Map them as follows:

| Old path | Current path |
|---|---|
| `docs/PREREG-*.md` | `docs/prereg/` |
| `docs/DESIGN.md`, `docs/DESIGN-*.md`, `docs/ROADMAP-*.md` | `docs/design/` |
| `docs/FORENSICS-*.md`, `docs/AUTOPSY-*.md`, `docs/AUDIT-*.md` | `docs/forensics/` |
| `docs/*-R19.md`, `docs/*-R20.md`, `docs/*-R21.md`, `docs/RESOURCE-PROTOCOL-L2.md`, `docs/OPS-launcher.md` | `docs/protocol/` |
| `train/runs/r*.md` (round papers) | `docs/rounds/` |

Paths in the code, the README, the model cards and `design/DESIGN.md` already use the current paths.

Files with non-English names were renamed during the English pass.

## Removed drafts

The following drafts were identical to the FROZEN version of the same round and were removed; where a document
mentions one of them, read the FROZEN version: `r10-PREREG-DRAFT-v0.1-20260827`, `r12-PREREG-DRAFT-v0.1-20260830`,
`r13-PREREG-DRAFT-v0.1-20260830`, `r17-PREREG-DRAFT-20260906`, `r18-A-PREREG-20260906`,
`r18-B-PREREG-DRAFT-20260907`, `r18-C-PREREG-20260907`, `r18-DE-PREREG-20260907`, `r18-F-PREREG-20260907`,
`r18-G-PREREG-20260907` (all byte-identical), and `r16-PREREG-DRAFT-20260901` (identical to
[`r16-PREREG-FROZEN-20260902.md`](rounds/r16-PREREG-FROZEN-20260902.md) except two header lines).

## Not in the repository

These categories are not kept in the repository; citations of them in the documents (for example ledger line
numbers) are historical references:

- run logs and raw run outputs;
- raw event ledgers, including the gate ledgers of the recal-g1 and v32 cases;
- internal memos, launch orders not listed above, superseded drafts and implementer probe reports;
- one-off scripts.

## What remains under `train/runs/`

The few files still tracked under `train/runs/` are receipts that code or tests read:

- the 6 evaluation anchors in `eval-assembled/`: the four `r2-*.json`, `v24-G3-leg7.json` and `v26-G3-leg6.json`
  (read by the v25 to v32 drivers and `tests/test_active_reference_contracts.py`);
- [`probe-zeroflip/report.json`](../train/runs/probe-zeroflip/report.json) (read by `run_v32_sovereign.py`);
- [`r10-staging/probe_r17_deployment.py`](../train/runs/r10-staging/probe_r17_deployment.py) (read by
  `tests/test_resource_protocol.py`) and
  [`r10-staging/probe_r15_deployment.py`](../train/runs/r10-staging/probe_r15_deployment.py), which it reads to
  record a source fingerprint;
- [`r10-staging/r16-deploy-arm.json`](../train/runs/r10-staging/r16-deploy-arm.json), the R16 deployment rows that
  the `probe_r17_deployment.py` regression compares against.

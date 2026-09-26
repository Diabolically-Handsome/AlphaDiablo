"""v33 "content case: course 5 dry-window curriculum x 4b positive examples" driver
(sole executor of the rev4 clauses of docs/prereg/PREREG-v33-content-case.md; skeleton = run_b1_infra.py style
(frozen constant block / stage_done idempotence / pre() preflight / separate --smoke entry) +
the exam_or_adopt/biteq/leg loop/winner ruling sections of run_v32_sovereign.py).

Numbering note: v33 was assigned after the panel review passed (numbering note, assigned before the first event); the ledger directory
train/runs/v33-content/ exists under its real name from the first event on.

Stage order (D2/appendix: driver structure; strictly serial to prevent hindsight, enforced by STAGE_SEQUENCE):
  S0  pre() preflight (W1/W7/W-PIN per-file sha/event-line pins/304000 reconciliation/308000 virginity/
      W-H8/W9/W-C inheritance/CASE_RUNTIME capture)
  S1  G0_BASELINE transcription (sha check of the four E0 dual-baseline files; captured outside the driver, before implementation)
  S2  S-bc1 BC-v1 regeneration split assertion (demos bytes ≡ frozen constant / policy torch.equal
      against the _previous archived file) -> BC_REGEN
  S3  S-bc2 BC-v2 collection+training -> N12_GATE (with denominator definition / per-game breakdown / OC use slot;
      P-N12: the single OC threshold 0.65->0.70, one resample)
  S4  G0 six-piece set: G0-1 dual-endpoint bit-level identity -> G0-2a zero-intrusion pair (tensor level, seed 308000)
      -> G0-2b functional smoke test (measured p sequence ≡ registry prefix) -> G0-ghost -> G0-demo pool
      -> G0-6 REF_BITEQ two runs (≡113.0 / ≡140.9 full table, freeze prerequisite)
  S5  FREEZE_SHA / CASE_RUNTIME / NEWLINE_ADOPT (slot for the two conditions of the upgrade review)
  S6  [W-LAUNCH launch-order gate] no LAUNCH_ORDER event -> sys.exit(9), halt and wait
      (AWAITING_LAUNCH is not a failure state; no NEEDS_ATTENTION)
  S7  three legs in series (L-base -> L-cur -> L-full; each leg: launch (driver subprocess launched via
      train/launch_case.sh) -> end-of-leg check dry_curriculum.jsonl ≡ full registry table ->
      offline canary sequence (make-up deadline = before that leg's s16 FIRING_START, asserted in the driver)
      → s16 → full32 → full32-m29)
  S8  course 5 main verdict / 4b main verdict (L-full always judged, win or lose) -> winner ruling (tie-break order D4-5)
      -> release criterion (first use of the G1 new lines, full conjunction recorded)
  S9  H1/H2 held-out bundled pair (on the winner, 1 pair/2 runs) -> R-line scorecard -> VERDICT_PATH

**Launch discipline: freeze ≠ launch; panel review passed ≠ launch; no path in this driver may be read as a launch authorization.**

Usage:
  .venv/bin/python train/run_v33_content.py          # whole case (idempotent resume)
  .venv/bin/python train/run_v33_content.py --smoke  # only the G0-2a/2b smoke tests;
                                                     # nothing goes to the main ledger; uses a separate smoke ledger
Every launch goes through train/launch_case.sh (E8):
  train/launch_case.sh train/runs/v33-content train/run_v33_content.py

Exit codes (D5, centralised in EXIT_CODES):
  0 case closed/idempotent; 2 budget exhausted; 3 preflight (PREFLIGHT_FAIL); 4 not idle/lock conflict;
  5 W5 pre-launch drift; 6 runtime drift during the case; 7 CASE_HALT_G0; 8 REF_DIVERGENCE;
  9 AWAITING_LAUNCH (frozen, waiting for launch; not a failure state); anything else P1.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import traceback
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_contract import (PROTOCOL_VERSION, EvalContractError,
                           OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           strict_json_loads, verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V33 = RUNS / "v33-content"                     # ledger directory under its assigned name (numbering note)
LEDGER = V33 / "gate_ledger.jsonl"
SMOKE_LEDGER = V33 / "smoke_ledger.jsonl"      # separate --smoke ledger (nothing goes to the main ledger)
EVAL = RUNS / "eval-assembled"

# ======================================================================
# exit code table (D5 verbatim, centralised)
# ======================================================================
EXIT_CODES = {
    0: "case closed/idempotent",
    2: "budget exhausted (P2 evaluation runs / P3 leg launches / P5 mandatory runs)",
    3: "preflight failed (P4 PREFLIGHT_FAIL)",
    4: "not idle/lock conflict (W8)",
    5: "W5 pre-launch drift",
    6: "runtime drift during the case",
    7: "CASE_HALT_G0",
    8: "REF_DIVERGENCE",
    9: "AWAITING_LAUNCH (FREEZE_SHA recorded, LAUNCH_ORDER not yet recorded: standby state;"
       " not a failure, no NEEDS_ATTENTION)",
}

# ======================================================================
# W7 artifact pins (full-file sha, frozen driver constants; a mismatch means P4, no launch)
# ======================================================================
KING_ZIP = ROOT / "train" / "models" / "v28-worker-leg1" / "model_final.zip"
# 2026-09-23: re-saved with neutral local paths (content otherwise unchanged); was 2f7bc9dd810956c3
KING_ZIP_SHA = "0c6f014da19c3bf27b208d55adc76be13fcad8bc5f744b95f4f743be97426434"
KING_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
KING_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
KING_STEPS = 3_497_984
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
H_NPZ_SHA = "0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7"   # == DEFAULT_MANAGER_SHA256
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_NPZ_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
KING_SD = RUNS / "v32" / "king_anchor_sd.pt"
KING_SD_SHA = "009aaad29d2653cde3f4e8ed2fafd8861a0f1f572a140c64118df9e3fa3df35d"  # value settled in v32 (in-case parity re-check)

# BC-v1 files (W7 split assertion; regeneration is not optional: after E1 changed worker_env, old receipts must fail preflight)
BC_V1_DIR = RUNS / "bc-worker"
BC_SD = BC_V1_DIR / "policy_sd.pt"
BC_V1_DEMOS = BC_V1_DIR / "demos.npz"
BC_V1_DEMOS_SHA = "3bf892d611e41853eca8fce0cb146753af41ad2c3a21b6c581df1041fb1d9363"
# f052067a… is the **old settled value** of policy_sd.pt (v32 BC_REGEN), used only as the lineage reference
# for the torch.equal comparison -- torch.save bytes are not reproducible, so a byte-equality assertion can never pass (dropped);
# the new sha is settled by this case's BC_REGEN event.
BC_SD_PREV_LINEAGE_SHA = "f052067a589cfcdedaf1754ae6d241d736bb97f6fc798683f395c76cb0ff98e6"

# BC-v2 files (E2 separate output directory; enters the auxiliary loss only via --bc-aux-demos, canonical path unchanged)
BC_V2_DIR = RUNS / "bc-worker-v2"
BC_V2_DEMOS = BC_V2_DIR / "demos.npz"
BC_V2_SD = BC_V2_DIR / "policy_sd.pt"
BC_V2_REPORT = BC_V2_DIR / "bc_report_v2.json"
N12_GATE_MIN = 122            # D7: >=122 (= half of the audit point estimate 244)
RECALL12_GATE_MIN = 0.5       # D7: held-out class-12 recall >=0.5
PREVENTIVE_MAIN = 0.65        # D7 preventive threshold, main plan
PREVENTIVE_OC = 0.70          # the single registered OC (P-N12 resample knob)
N12_DENOMINATOR_DEF = ("denominator = labelled a12 states of the teacher-v2 hysteresis simulation under the current split (window-start trigger states);"
                       " metric = held-out argmax hits; fail-closed"
                       " (pinned by statistics B-1; the 1,975-state definition is dropped)")
N12_CLUSTER_NOTE = ("denominator cluster note (verified correction): the audit showed that under the held-out trigger-state definition "
                    "74.7% is concentrated in two clusters, games 178/187 (880/596/1,975); for the labelled-window (window-start)"
                    " definition the per-game concentration follows this N12_GATE's measured per-game breakdown; recall readings are noisy")

# ======================================================================
# W-PIN input archive pins (G1 form, full-file sha frozen constants; preflight mismatch -> PREFLIGHT_FAIL 3)
# The "all ledgers" set of the 304000 reconciliation is exactly this closed ledger set (D1; no open-ended "all").
# ======================================================================
W_PIN = {
    "v32-ref-launch": (EVAL / "v32-ref-launch.json",
                       "48033577f8f124ae81fc5436eb44e5aa0bf541a4437d7fec99d8f5b1209c71fa"),
    "v32-ref-science": (EVAL / "v32-ref-science.json",
                        "1736185e286c1f2a98f6d4f503c72b40e322dcc31151e0e83068714c1b29f5f0"),
    "b1-ref8k-launch": (EVAL / "b1-ref8k-launch.json",
                        "3b4ef1681134d51d61c3081195fc620ea4ad4d7c7f034597b9f92782cabe6a19"),
    "b1-ref8k-science": (EVAL / "b1-ref8k-science.json",
                         "e23a83383b8e286d9baa85ee9142970103457b0b5a3b249c4f0826b184910ef4"),
    "AUDIT-content-case-archive": (ROOT / "docs" / "forensics" / "AUDIT-content-case-archive.md",
                     "97bbd75f3d3f0455379a451fd785e5d0bf2afece4e43ebcbd0eac181f85b8bb0"),  # English text; pre-translation digest 433f3a97…
    # upgrade review (the pending item E) + g1_results.json: source files of the two NEWLINE_ADOPT
    # conditions; adding the g1_results sha to W-PIN is the explicit option for condition one of that review.
    "AUDIT-content-case-G1-dual-label-review": (ROOT / "docs" / "forensics" / "AUDIT-content-case-G1-dual-label-review.md",
                        "dc5c7c100674b54e127ff24337b40c107cd39a1189c4fba1fb9e4c6bfdf44186"),  # English text; pre-translation digest 8e4be014…
    "g1_results.json": (RUNS / "recal-g1" / "g1_results.json",
                        "aa03addea7003e2f6f5b2e9fb595dd75179d8a5cddb14b478c0073c6a48e74b0"),
    # the four E0 dual-baseline files (king+throne x both endpoints; captured = before the change, captured before implementation)
    "E0-king-p0": (ROOT / "docs" / "assets" / "g0v32_traj_baseline.json",
                   "2f567d7abfc5b4714cac4ff3236b09ab67e22485d2b05b52d9dc956692a26d6f"),
    "E0-king-p1": (ROOT / "docs" / "assets" / "g0v32_traj_baseline_skip_dry_true.json",
                   "bdb43707064d24b5f985c33f211755c8331d215bd41d1630ebdc4019fefff977"),
    "E0-throne-p0": (ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne.json",
                     "e235e96972f8f2eeaaebec2bc8970c3e3f77b715f26c364250c42d8d37131cfc"),
    "E0-throne-p1": (ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne_skip_dry_true.json",
                     "8d541b56d10c18b3421de3dd32be4f17051a03488d531916c24177814a0b28ba"),
}
# W-PIN closed ledger set (full text of each ledger: v29-v32 / reanchor-r2 / infra-b1 / recal-g1)
W_PIN_LEDGERS = {
    "v29": (RUNS / "v29" / "gate_ledger.jsonl",
            "27ad88b295594f2d7131c81be29a51fa246da82f7a932a939a524e056dd06f67"),
    "v30": (RUNS / "v30" / "gate_ledger.jsonl",
            "7187c30c0f8e255fd7f7ab23d01b5164a5557d9c5130633529e85d147e7a463e"),
    "v31": (RUNS / "v31" / "gate_ledger.jsonl",
            "19ed09c664947228a3e07d80feb56edf8bc05124b853bc1ad796b2c6af8d09e5"),
    "v32": (RUNS / "v32" / "gate_ledger.jsonl",
            "8c197617097548f04e82bb3157b679bf6792ce3da937bd4e8ec657d920fe0d22"),
    "reanchor-r2": (RUNS / "reanchor-r2" / "gate_ledger.jsonl",
                    "a3351a9ba525d8d3f400c61716169d991ae16c05b8f1e2f326a74f71fb4828d5"),
    "infra-b1": (RUNS / "infra-b1" / "gate_ledger.jsonl",
                 "cdc24b16b1ac11a86a64ffb031ea5a10b432d61e5581271346926c0f7d53ed15"),
    "recal-g1": (RUNS / "recal-g1" / "gate_ledger.jsonl",
                 "f0b7874edd8a778c7fb3d0be6ff88fd676737ebe8b3fed77b6a498821d547839"),
}
# event-line pins (archive sha + line reference + line-bytes sha; rev3 blocker correction: G1 has
# **five** LINE_DERIVED lines (:10-14) + the WOULDTRIP_RULING line (:20); "six event lines" dropped)
# form: name -> (ledger key, line number (1-based), event name, line_id or None, line-bytes sha256)
W_PIN_EVENT_LINES = {
    "B1-CANARY_SET": ("infra-b1", 19, "CANARY_SET", None,
                      "173ce61ad03db883e82ec40b7e40228a7ab496b0bcf464cfbf850679eb68e612"),
    "G1-RG1.1-width": ("recal-g1", 10, "LINE_DERIVED", "RG1.1-width",
                       "edf605cd7e3735037e89a4dee56a35f24f7695fdee0c0c2a60a77ae5384bb241"),
    "G1-RG1.2a-mean-worker-leg": ("recal-g1", 11, "LINE_DERIVED",
                                  "RG1.2a-mean-worker-leg",
                                  "099e060db3604d73db72e72b341720f96a28c0639ebc3d76849b7d2287f73d94"),
    "G1-RG1.2b-mean-manager-arm": ("recal-g1", 12, "LINE_DERIVED",
                                   "RG1.2b-mean-manager-arm",
                                   "92490e2c81883f02084554b5487a388f2fe4eaeaef61ff236aef2c7161071146"),
    "G1-RG1.3-death": ("recal-g1", 13, "LINE_DERIVED", "RG1.3-death",
                       "323fb1b7904f75d7a7ec1c39de62e85c63281be96fb09b6f173bad46a0f5fee2"),
    "G1-RG1.4-floor": ("recal-g1", 14, "LINE_DERIVED", "RG1.4-floor",
                       "63d819976044228014af33b8b2e1a54545d222af821464a0eedceaaf361ffacc"),
    "G1-WOULDTRIP_RULING": ("recal-g1", 20, "WOULDTRIP_RULING", None,
                            "95443a26551cb477d10b25c8cbc36e544a2415466cbb1256cf7ff31f4b99a2cb"),
}
# K1 provenance (upgrade review limb 2 item 4: infra-b1/gate_ledger.jsonl:14 exam_ok b1-ref8k-launch)
K1_PROVENANCE = {"ledger": "infra-b1/gate_ledger.jsonl", "line": 14,
                 "archive_sha16": "3b4ef1681134d51d"}

# FREEZE_SHA / CASE_RUNTIME structured placeholders (recorded at freeze, no fake values; the text holds no self-referencing sha:
# the freeze sha goes only into the ledger FREEZE_SHA event; the five case-level sha values are settled by the first CASE_RUNTIME event)
FREEZE_SHA_PLACEHOLDER: str | None = None
CASE_RUNTIME_PLACEHOLDER: dict | None = None

# ======================================================================
# leg recipe constants (D3; shared command form = v32-sov verbatim + B1 instrument knobs; frozen)
# ======================================================================
LEG_STEPS = 499_712
NT_TARGET = 3_997_696          # = king 3,497,984 + 499,712 (incremental semantics; nt gate of each leg)
QUANTUM = 2_048                # 512 n-steps × 4 envs
SEED = 304_000                 # same seed for all three legs (P8 reuse; the written exemption is in the 304000 reconciliation assertion)
BETA = 0.015625                # distillation anchor β, frozen original value (design decision 5; anchor formula untouched)
BC_AUX_LAMBDA = 0.015625       # λ_bc (D7 registered discretionary constant, same magnitude as β)
MAIN_TABLE = "linear:1.0:0.5:147,hold:0.5:97"   # main annealing table (fixed by design decision 2 and its addendum)
DRY_CURRICULUM_LEG_START = 3_497_984            # leg-relative rollout anchor (E1 item 2)
CALIB_PROBES = tuple(KING_STEPS + 49_152 * k for k in range(1, 11))
CKPT_EVERY = 98_304
SENTINEL_EVERY = 49_152
DRY_ANCHOR_EVERY = 49_152
# rev4 D3 correction addition (section 12 appendix 2 item 1, to be confirmed): the E5 (1)(2) record-only instrument knobs join the shared
# command form -- the obligation of approved design decision 5 (dry/fresh distill_ce as calibration data for course 2) + E5 (2)
# "baseline built from this case's first launch"/RC.14; zero intrusion is guaranteed by the G0-2a live run with all knobs on.
DISTILL_CE_PROBE_EVERY = 49_152
DRYWIN_METRICS_EVERY = 49_152
CANARY_STEPS = tuple(KING_STEPS + 98_304 * k for k in range(1, 5))   # four mid-leg points
EXTRA_CKPT_STEP = KING_STEPS + 491_520          # extra ckpt produced mechanically: archived, not in the sequence
LEG_TIMEOUT = 21_600           # 6h per leg

# LEGS table = D3 per-leg additions verbatim (authoritative, frozen; the shared form itself does not produce p≡1.0,
# so L-base must carry --skip-dry explicitly; --bc-aux-demos resolves the D3 literal path inside the repo)
LEGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("v33-base", ("--skip-dry",)),
    ("v33-cur", ("--dry-curriculum-schedule", MAIN_TABLE)),
    ("v33-full", ("--dry-curriculum-schedule", MAIN_TABLE,
                  "--bc-aux-lambda", "0.015625",
                  "--bc-aux-demos", str(BC_V2_DEMOS),
                  "--bc-aux-liveness-preflight")),
)
LEG_NAMES = tuple(name for name, _ in LEGS)
LEG_SHORT = {"v33-base": "base", "v33-cur": "cur", "v33-full": "full"}
CURRICULUM_LEGS = ("v33-cur", "v33-full")

POOL_PROBE = "7000-7031"
POOL_S16 = "7000-7015"
POOL_HOLD = "8000-8031"
REF_TAGS = {"launch": "v33-ref-launch", "science": "v33-ref-science"}
REF_MEANS = {"launch": 113.0, "science": 140.9}   # agg anchors for the REF_BITEQ full-table identity
HOLDOUT_TAGS = ("v33-win-hold8k", "v33-win-hold8k-m29")
CANARY_TAGS = tuple(f"v33-{LEG_SHORT[leg]}-canary{k}-{m}"
                    for leg in LEG_NAMES for k in range(1, 5)
                    for m in ("h", "m29"))
LEG_EXAM_TAGS = tuple(f"v33-{LEG_SHORT[leg]}-{suffix}"
                      for leg in LEG_NAMES
                      for suffix in ("s16", "full32", "full32-m29"))
ALL_EXAM_TAGS = (tuple(REF_TAGS.values()) + LEG_EXAM_TAGS + HOLDOUT_TAGS
                 + CANARY_TAGS)

# ======================================================================
# G0 smoke-test constants (D2-2a; the three limbs of the virgin-seed rule are pinned: (1) every event of the W-PIN closed ledger set + the evaluation-pool
# range guard; (2) 307000 is no longer virgin (infra-b1:3 G0_NULLINTRUSION on record), pre-registered =
# 308000; (3) every G0-2b short run uses the same rule and seed)
# ======================================================================
SMOKE_SEED = 308_000
SMOKE_STEPS = 102_400          # 50 rollouts, divisible by 2048
SMOKE_END = KING_STEPS + SMOKE_STEPS    # 3,600,384
SMOKE_CALIB = (KING_STEPS + 49_152, KING_STEPS + 98_304)
SMOKE_HOLD_TABLE = "hold:1.0:50"        # G0-2a schedule pinned at p≡1.0 (covers all 50 rollouts)
# from rev4 the E5 (1)(2) knobs enter every smoke arm with the shared command form (with_b1_knobs=True),
# value 49,152 (DISTILL_CE_PROBE_EVERY/DRYWIN_METRICS_EVERY; each fires >=1 time in the window);
# the bare arm with_b1_knobs=False does not carry them (v32-sov verbatim, must not change).
SMOKE_RUNS = {"bare": "v33-smoke-bare", "knobs": "v33-smoke-knobs",
              "func-p": "v33-smoke-funcp", "func-aux": "v33-smoke-funcaux"}
SMOKE_TIMEOUT = 3_600
V32_SOV_CALIB = (3_747_984, 3_947_984)  # calib step table from the v32-sov command form verbatim (bare arm)

# ======================================================================
# criterion constants (D4/D7; all pre-registered, frozen for the case)
# ======================================================================
# the four G1 new-line constants (cited; LINE_DERIVED line references, see W_PIN_EVENT_LINES;
# preflight re-checks them against the actual ledger values, so a mis-copied constant is caught):
G1_PAIRED_DIFF = 17.32     # RG1.2a (worker-leg family; the "single-vector line" label is mandatory)
G1_PAIRED_WINS = 22        # RG1.1(k* = min{k: P(X≥k|32,.5) ≤ 0.05})
G1_DIED_MAX = 8            # RG1.3 (the "K1 single-archive dependency" label is mandatory)
FLOOR_FRAC = (85.0, 92.0)  # RG1.4 keeps the current 85/92
ABANDON_FRAC = (75.0, 112.4)   # abandon gate, applies to this case (pinned by D4-1; blocks only release and the
                               # title context, not the criterion 2/3 scientific main verdicts)
A12_USE_LINE = 0.1         # v32 registered constant (discrete meaning: >=4 real drinks in 32 games)
SURV_MEAN_BAND = -2.0      # 4b tier mean-difference limb >= -2 (vs L-cur x M29 pairing, re-registered)
R4 = {"descend": 0.0204, "override_sentinel": 0.03,
      "override_void": 0.08, "cap": 0.05}   # sentinel-limb untouched list (G1 L1)
COURSE5_ANCHOR_PRIOR = 5   # prior point (from P8, demoted to a listed prior, not the anchor; anchor = measured L-base)
DEEPWATER_SEED = 7017      # standing control for the deep-water seed (7017 structural slot note)
MULTIPLE_COMPARISON = {    # D4-1 listing obligation (G1 L1a form; a listing, not a decision line)
    "per_candidate_width_alpha": 0.0251,
    "effective_alpha_3arms": 0.073,        # ≈ 1 − 0.9749³
    "joint_alpha_3arms": 0.013,            # derived for 3 arms from the G1 joint 0.0045 basis
    "unit_note": "α unit is the per-candidate single-limb margin (G1 L1a); 3 arms decide",
}
DELEVERAGE_Q95 = 15.99     # line-side Q95 recomputed without 7017 (condition for limb 1 of the upgrade review)
DELEVERAGE_Q95_RAW = 15.987129
OC_ALPHA_COLUMNS = {"worker_leg_old_alpha_mean": 0.357,
                    "worker_leg_new_alpha_mean": 0.050,
                    "note": "OC old_vs_new four columns submitted together (upgrade review limb 1 caveat 6)"}

# G1 verification notes 1 and 2 verbatim (the two conditions of the upgrade review; source of the NEWLINE_ADOPT payload =
# the rev3 verification-amendment chapter of docs/prereg/PREREG-G1-gate-recalibration.md, transcribed verbatim after direct check)
G1_VERIFY_NOTE_1 = ("Absent a registered obligation for a later case to test across pools, cross-pool coverage is forward-looking discretion, not an existing obligation; "
                    "in the current eligibility context (H x 7000) all operational margin of this line (1->8) comes"
                    " solely from the out-of-domain archive (K1)")
G1_VERIFY_NOTE_2 = ("The veto-branch wording becomes \"the only healthy vector in the family is rejected for circularity, so no line can be set for this family\"; "
                    "RG1.2a/2b both carry the \"single-vector line\" label (2b also carries the note \"driven by the high-leverage seed 7017"
                    " (Δ−149.86)\"), which must travel with every D7 citation, with the deleverage sensitivity row"
                    " presented alongside; honest note on the blocking direction: \"setting the line from fresh vectors systematically raises the bar for fresh-type"
                    " (high-variance) candidates\", mandatory whenever the D4-7 revival audit cites this line.")

# R-line band constants (D8; closed interval [lo, hi]; point estimates without a stated source are discretionary, not estimates)
RC2_BAND = {"point": -12.0, "band": (-45.0, 5.0)}
RC3_BAND = {"point": 4.0, "band": (-12.0, 25.0)}
RC4_BAND = {"point": 2.0, "band": (-12.0, 18.0)}
RC6_BAND = {"point": 0.8, "band": (0.0, 8.0)}
RC8_BAND_H = {"point": 2, "band": (0, 8)}
RC8_BAND_M29 = {"point": 5, "band": (0, 10)}
RC9_BAND = {"point": 30.0, "band": (8.0, 55.0)}
RC9_MS = {"healthy": {"max": (20.0, (5.0, 45.0)), "over": (2, (0, 8))},
          "damaged": {"max": (60.0, (45.0, 130.0)), "over": (12, (8, 17))}}
RC9_BRANCH_LINE = -20.0        # the two MS branches are triggered mechanically by the measured RC.2 value (no choosing a band afterwards)
MS_FLAG_LINE = 20.0
RC10_BAND = {"healthy": (0.5, 1.0), "damaged": (0.0, 0.5)}
RC12_BAND = {"point": 4.0, "band": (0.0, 10.0)}   # pp; zero point 0.7515
DRY_REF_THRONE = 0.7515
DRY_REF_LINEAGE = 0.6305
RC13_N12_BAND = {"point": 244, "band": (122, 700)}
RC13_RECALL_BAND = {"point": 0.8, "band": (0.5, 1.0)}
RC15_BAND = {"point": -10.0, "band": (-50.0, 10.0),
             "d2_point": 5, "d2_band": (2, 14)}

CASE_RT: dict | None = None
_LEDGER_PATH = LEDGER          # --smoke switches to SMOKE_LEDGER (nothing goes to the main ledger)

# stage order (D2 strictly serial to prevent hindsight; a later stage refuses to run while an earlier one is not done)
STAGE_SEQUENCE = ("G0_BASELINE", "BC_REGEN", "N12_GATE", "G0_ENDPOINT",
                  "G0_NULLINTRUSION", "G0_SMOKE", "G0_GHOST", "DEMO_LEDGER",
                  "REF_BITEQ", "FREEZE_SHA", "LAUNCH_ORDER", "LEGS")


# ======================================================================
# ledger and shared utilities (inherited verbatim from run_b1_infra / run_v32_sovereign + case-level extensions)
# ======================================================================

def log(event: dict):
    V33.mkdir(parents=True, exist_ok=True)
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(_LEDGER_PATH, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    V33.mkdir(parents=True, exist_ok=True)
    with open(V33 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalFailure(message)


class PreflightFailure(RuntimeError):
    """P4: preflight failed -> no launch, report for review (exit code 3)."""


def pre(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightFailure(message)


def runtime_five(snapshot) -> dict:
    rt = snapshot["runtime"]
    return {"bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
            "game_data": rt["content"]["game_data"]["sha256"],
            "assets": rt["content"]["assets"]["sha256"],
            "protocol": rt["python_protocol"]["sha256"]}


def assert_case_runtime(snapshot, where: str):
    require(CASE_RT is not None, "CASE_RUNTIME not settled")
    current = runtime_five(snapshot)
    if current != CASE_RT:
        log({"event": "CASE_HALT_RUNTIME_DRIFT", "where": where,
             "case": CASE_RT, "current": current})
        attention(f"case-level runtime drift ({where}); halting for review")
        raise SystemExit(6)


def run(cmd, logfile, timeout) -> int:
    V33.mkdir(parents=True, exist_ok=True)
    with open(V33 / logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return 124


def _archive_residue(run_dir: pathlib.Path) -> None:
    """Archive leftovers before launch (B1 _prepare_run_dir form): an existing directory is renamed as a whole to
    <name>-prev<ns>, so the jsonl evidence starts from zero (correction of 2026-07-19;
    root cause of the G0-2a FAIL = leftover smoke output appended on top)."""
    if run_dir.exists():
        run_dir.rename(run_dir.with_name(
            f"{run_dir.name}-prev{time.time_ns()}"))


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def zip_data_field(p: pathlib.Path, key: str):
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read("data")).get(key)


def read_ledger() -> list[dict]:
    if not _LEDGER_PATH.is_file():
        return []
    out = []
    for i, line in enumerate(_LEDGER_PATH.read_text().splitlines(), 1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise OperationalFailure(f"ledger line {i} unparsable: {exc}") from exc
    return out


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def stage_done(events, name) -> bool:
    return any(e.get("event") == name for e in events)


def leg_starts(events, leg) -> int:
    return sum(1 for e in events
               if e.get("event") == "leg_start" and e.get("leg") == leg)


def firing_count(events, tag) -> int:
    return sum(1 for e in events
               if e.get("event") == "FIRING_START" and e.get("tag") == tag)


def s16_fired(events, leg) -> bool:
    """Make-up deadline predicate (P-canary / compliance M-4): that leg's s16 FIRING_START is already recorded."""
    return firing_count(events, f"v33-{LEG_SHORT[leg]}-s16") > 0


def by_seed(rows, lo, hi) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "abnormal seed set (duplicate seed)")
    require(set(m) == set(range(lo, hi + 1)), f"abnormal seed set (must be {lo}-{hi})")
    return m


def depth2_count(rows) -> int:
    return sum(1 for r in rows if r["depth"] >= 2)


def d_windows(row) -> int:
    return str(row["mode_seq"]).count("D")


def dive_per_ep(rows) -> float:
    return sum(str(r["mode_seq"]).count("D") for r in rows) / max(1, len(rows))


def a12_per_ep(agg) -> float:
    hist = agg.get("worker_action_hist", {}) or {}
    return round(int(hist.get("12", 0)) / max(1, int(agg.get("n", 32))), 2)


def a13_per_ep(agg) -> float:
    hist = agg.get("worker_action_hist", {}) or {}
    return round(int(hist.get("13", 0)) / max(1, int(agg.get("n", 32))), 2)


def impl_bundle_sha16() -> str:
    import train_ppo
    return train_ppo._implementation_bundle_sha256()[:16]


def dry_curriculum_table() -> tuple[float, ...]:
    """Full main annealing table (index -> p; single source of truth = train_ppo._parse_dry_curriculum_schedule;
    endpoints use interpolation semantics: item k of a linear segment = p0+(p1−p0)·k/(n−1), the last item of quantum 147 reaches exactly
    0.5 -- strict decrease occupies 299,008 steps and p=0.5 occupies 200,704 steps; this registered semantics is the source of truth)."""
    import train_ppo
    table = train_ppo._parse_dry_curriculum_schedule(MAIN_TABLE)
    require(len(table) * QUANTUM == LEG_STEPS,
            f"main annealing table quantum count unbalanced: {len(table)}x{QUANTUM} != {LEG_STEPS}")
    return table


def assert_stage_prereqs(events, stage: str):
    """Stage order no-hindsight (D2): every stage before `stage` must be recorded (strictly serial)."""
    idx = STAGE_SEQUENCE.index(stage)
    predicates = {
        "N12_GATE": lambda ev: any(e.get("event") == "N12_GATE"
                                   and e.get("gate") == "PASS" for e in ev),
        "G0_NULLINTRUSION": lambda ev: any(
            e.get("event") == "G0_NULLINTRUSION" and e.get("verdict") == "PASS"
            for e in ev),
        "G0_SMOKE": lambda ev: any(e.get("event") == "G0_SMOKE"
                                   and e.get("verdict") == "PASS" for e in ev),
        "REF_BITEQ": lambda ev: {e.get("ref") for e in ev
                                 if e.get("event") == "REF_BITEQ"} >= {"launch",
                                                                       "science"},
    }
    missing = []
    for name in STAGE_SEQUENCE[:idx]:
        done = predicates.get(name, lambda ev, _n=name: stage_done(ev, _n))(events)
        if not done:
            missing.append(name)
    pre(not missing, f"stage order no-hindsight (D2 strictly serial): before {stage} missing {missing}")


# ======================================================================
# seed reconciliation assertions (D1 rewrite + D2-2a virgin-seed rule; pure functions for positive/negative unit tests)
# ======================================================================

def w_pin_ledger_lines() -> dict[str, list[str]]:
    """Full-text lines of the W-PIN closed ledger set (the only enumeration for the 304000/308000 reconciliation)."""
    return {name: path.read_text().splitlines()
            for name, (path, _sha) in W_PIN_LEDGERS.items()}


def assert_seed_304000_provenance(lines_by_ledger: dict[str, list[str]]):
    """304000 reconciliation assertion (rev3 D1 verbatim): scanning every event of the W-PIN closed ledger set, the historical occurrence
    is exactly and only the infra-b1 P8 leg_start; the reuse exemption is written down (same seed, carrying the cross-case note of the
    P8 triangle reconciliation: the L-base vs P8 difference = the autonomy knob); no seed search, no seed change on relaunch."""
    hits = [(name, i, raw)
            for name, lines in lines_by_ledger.items()
            for i, raw in enumerate(lines, 1) if "304000" in raw]
    pre(len(hits) == 1,
        f"304000 reconciliation assertion failed: {len(hits)} occurrences in the closed ledger set (must be exactly 1 "
        f"infra-b1 P8 leg_start): {[(n, i) for n, i, _ in hits]}")
    name, lineno, raw = hits[0]
    pre(name == "infra-b1", f"the single 304000 occurrence is not in infra-b1 (found in {name}:{lineno})")
    try:
        e = json.loads(raw)
    except json.JSONDecodeError:
        pre(False, f"304000 hit line unparsable: {name}:{lineno}")
    pre(e.get("event") == "leg_start" and e.get("leg") == "b1-p8"
        and e.get("seed") == 304_000,
        f"the single 304000 occurrence is not a P8 leg_start event: {name}:{lineno} {e.get('event')}")


def assert_seed_virgin(lines_by_ledger: dict[str, list[str]], seed: int):
    """Virginity assertion (all-event semantics, same as the 304000 clause): zero occurrences in the closed ledger set."""
    token = str(seed)
    hits = [(name, i) for name, lines in lines_by_ledger.items()
            for i, raw in enumerate(lines, 1) if token in raw]
    pre(not hits, f"seed {seed} virginity assertion failed (hit in the all-event scan of the closed ledger set): {hits}")


def assert_pool_guard(seed: int, what: str):
    """B1 evaluation-pool check limb unchanged (range guard for 7000/8000/9000, rank 0-3)."""
    pre(not any(lo <= seed + rank <= hi
                for rank in range(4)
                for lo, hi in ((7000, 7031), (8000, 8031), (9000, 9031))),
        f"{what} seed {seed} collides with an evaluation pool range (7000/8000/9000 guard)")


# ======================================================================
# evaluation machinery (exam_or_adopt skeleton inherited + P2 ledger budgets + bundled channel)
# ======================================================================

def exam(worker, tag, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"archive immutability: {out} already exists, refusing to overwrite")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"exam:{tag}")          # W11 identity restated for every run
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    worker_arg = (worker if snapshot["worker"]["kind"] in {"script", "bc"}
                  else snapshot["worker"]["path"])
    # discipline: --manager-npz explicit every time, no default fallback; no run in this case uses --board
    cmd = [PY, "train/eval_assembled.py", "--worker", str(worker_arg),
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():
            sealed = out.with_suffix(f".{time.time_ns()}.void")
            out.rename(sealed)
            log({"event": "RESIDUE_SEALED", "tag": tag, "void": sealed.name,
                 "why": "leftover archive sealed after a non-zero exit of the evaluation process (P2)"})
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            sealed = out.with_suffix(f".{time.time_ns()}.void")
            out.rename(sealed)
            log({"event": "RESIDUE_SEALED", "tag": tag, "void": sealed.name,
                 "why": "leftover archive sealed after an archive identity/schema check failure (P2)"})
        return None
    d["agg"]["_sha"] = sha16(out)
    return d


def validate_adopted(tag, worker, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    lo, hi = (int(x) for x in seeds.split("-", 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"adopt:{tag}")
    expected = expected_eval_identity(snapshot, tag=tag,
                                      seeds=list(range(lo, hi + 1)))
    d = read_eval_archive(out, **expected)
    verify_eval_identity(snapshot, ROOT)
    d["agg"]["_sha"] = sha16(out)
    return d


def exam_case(events, worker, tag, seeds, manager_npz=None,
              bundled_with=None, extra: dict | None = None):
    """Exam-run finality clause (v32 verbatim, exam_ok/exam_adopted, no .void re-runs) +
    P2 ledger budget (each run's FIRING_START counts, 2 allowed; exhausted -> P5 plan A halt)."""
    out = EVAL / f"{tag}.json"
    prior = [e for e in events
             if e.get("event") == "exam_ok" and e.get("tag") == tag]
    if out.exists():
        require(bool(prior), f"{tag} leftover archive present but no exam_ok in the ledger; halting for review")
        d = validate_adopted(tag, worker, seeds, manager_npz)
        require(d["agg"]["_sha"] == prior[-1]["sha"],
                f"{tag} archive and ledger exam_ok sha mismatch")
        log({"event": "exam_adopted", "tag": tag, "sha": d["agg"]["_sha"]})
        return d
    require(not prior, f"{tag} recorded in the ledger but the archive is missing (REF_INVALID type); halting for review")
    while True:
        fired = firing_count(events, tag)
        require(fired < 2, f"{tag} evaluation run budget exhausted (ledger budget 2) -- P5 plan A halt")
        ev = {"event": "FIRING_START", "tag": tag, "attempt": fired + 1}
        log(ev)
        events.append(ev)
        d = exam(worker, tag, seeds, manager_npz)
        if d is not None:
            a = d["agg"]
            ok = {"event": "exam_ok", "tag": tag, "mean": a["ret_mean"],
                  "died": a["died"], "sha": a["_sha"]}
            if bundled_with:
                ok["bundled_with"] = bundled_with
            if extra:
                ok.update(extra)
            log(ok)
            events.append({"event": "exam_ok", "tag": tag, "sha": a["_sha"]})
            return d
        log({"event": "exam_crash", "tag": tag,
             "note": "evaluation failed; re-exam within the P2 budget (leftover archive already RESIDUE_SEALED)"})


def holdout_account(events, tag):
    """HOLDOUT_EXPOSURE counted per run (W-H8; this case: 1 pair/2 runs, on the winner -- the first case
    under B1's "at most one pair per case" rule)."""
    if any(e.get("event") == "HOLDOUT_EXPOSURE" and e.get("tag") == tag
           for e in events):
        return
    n = sum(1 for e in events if e.get("event") == "HOLDOUT_EXPOSURE") + 1
    ev = {"event": "HOLDOUT_EXPOSURE", "tag": tag, "cumulative_shots": n,
          "note": "counted per run; this case: 1 pair/2 runs on the winner (first case under B1's 'at most one pair per case' rule);"
                  " the registered-side virginity caveat carries over from B1 residual 14"}
    log(ev)
    events.append(ev)


# ======================================================================
# REF_BITEQ (G0-6 freeze prerequisite) and P-refdiv
# ======================================================================

def biteq_diffs(new_doc, ref_key: str) -> dict:
    """REF_BITEQ pure comparison: the new reference vs the same-named v32 reference pinned by W-PIN, bit level per seed per field
    + core agg fields + ret_mean identity (≡113.0 / ≡140.9 full table). Returns the differences (empty = pass)."""
    path, expected = W_PIN[f"v32-ref-{ref_key}"]
    require(path.is_file() and sha256(path) == expected,
            f"W-PIN reference archive sha drift: v32-ref-{ref_key}")
    old = strict_json_loads(path.read_bytes())
    old_rows, new_rows = old["rows"], new_doc["rows"]
    require(len(old_rows) == len(new_rows) == 32, "reference row count abnormal")
    row_diff = [o["seed"] for o, n in zip(sorted(old_rows, key=lambda r: r["seed"]),
                                          sorted(new_rows, key=lambda r: r["seed"]))
                if o != n]
    core = ("ret_mean", "ret_median", "died", "depth_median", "kills_mean",
            "farm_tau_mean", "override_rate", "cap_rate")
    agg_diff = [k for k in core if old["agg"].get(k) != new_doc["agg"].get(k)]
    if new_doc["agg"]["ret_mean"] != REF_MEANS[ref_key]:
        agg_diff.append(f"ret_mean!={REF_MEANS[ref_key]}")
    return {"row_diff_seeds": row_diff, "agg_diff": agg_diff} \
        if (row_diff or agg_diff) else {}


def p_refdiv(events, ref_key: str, diffs: dict):
    """P-refdiv: drift triage (CASE_RUNTIME re-check) -> CASE_HALT_G0 (7) or
    REF_DIVERGENCE (8); the mismatch happened before freeze, so no downgrade and no "the new reading is also plausible"."""
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    if CASE_RT is not None and runtime_five(snapshot) != CASE_RT:
        log({"event": "CASE_HALT_G0", "via": "REF_BITEQ+runtime-drift",
             "ref": ref_key, **diffs, "current": runtime_five(snapshot)})
        attention(f"REF_BITEQ mismatch ({ref_key}) with runtime drift: CASE_HALT_G0")
        raise SystemExit(7)
    log({"event": "REF_DIVERGENCE", "ref": ref_key, **diffs,
         "triage": "runtime_five all green yet the full table mismatches -- halt for review (P-refdiv, no downgrade)"})
    attention(f"REF_BITEQ mismatch ({ref_key}): REF_DIVERGENCE halt for review")
    raise SystemExit(8)


# ======================================================================
# preflight (W lines)
# ======================================================================

def preflight(events, smoke: bool = False):
    """S0: W1/W7/W-PIN/event-line pins/seed reconciliation/W-H8/W9/CASE_RUNTIME.
    With smoke=True (freeze prerequisite, live run on a dirty tree allowed) W1/W9/W-H8 are skipped."""
    global CASE_RT
    pre(PROTOCOL_VERSION == 3, "contract version drift (evaluation protocol bundle != 3)")
    # cross-file identity of the contract revision and the main-table literal (single-source cross pin)
    import train_ppo
    pre(train_ppo._CONTRACT_REVISION == 5, "train_ppo contract revision != 5 (E4)")
    pre(train_ppo._DRY_CURRICULUM_MAIN_TABLE == MAIN_TABLE,
        "main annealing table literal does not match the train_ppo single source of truth")
    pre(train_ppo._BC_V1_DEMOS_SHA256 == BC_V1_DEMOS_SHA,
        "BC-v1 demos frozen constant does not match train_ppo (E6)")
    # ---- W-PIN per-file sha reconciliation (mismatch -> PREFLIGHT_FAIL 3) ----
    for name, (path, expected) in {**W_PIN, **W_PIN_LEDGERS}.items():
        pre(path.is_file() and sha256(path) == expected,
            f"W-PIN archive pin mismatch: {name}")
    # ---- event-line pins (archive sha + line number + line-bytes sha + event name/line_id) ----
    ledger_lines = w_pin_ledger_lines()
    for name, (lkey, lineno, event_name, line_id, line_sha) in \
            W_PIN_EVENT_LINES.items():
        lines = ledger_lines[lkey]
        pre(len(lines) >= lineno, f"W-PIN event line missing: {name} ({lkey}:{lineno})")
        raw = lines[lineno - 1]
        pre(hashlib.sha256(raw.encode()).hexdigest() == line_sha,
            f"W-PIN event line byte drift: {name} ({lkey}:{lineno})")
        e = json.loads(raw)
        pre(e.get("event") == event_name,
            f"W-PIN event line event-name mismatch: {name} != {event_name}")
        if line_id is not None:
            pre(e.get("line_id") == line_id,
                f"W-PIN event line line_id mismatch: {name} != {line_id}")
    # re-check the four G1 new-line constants against the actual ledger values (cited values, guard against mis-copying)
    g1 = {json.loads(ledger_lines["recal-g1"][i - 1]).get("line_id"):
          json.loads(ledger_lines["recal-g1"][i - 1]) for i in range(10, 15)}
    pre(round(float(g1["RG1.2a-mean-worker-leg"]["value"]), 2) == G1_PAIRED_DIFF,
        "G1 mean-difference limb constant != RG1.2a ledger value")
    pre(int(g1["RG1.1-width"]["value"]) == G1_PAIRED_WINS,
        "G1 width limb constant != RG1.1 ledger value")
    pre(int(g1["RG1.3-death"]["value"]) == G1_DIED_MAX,
        "G1 death limb constant != RG1.3 ledger value")
    pre("85/92" in str(g1["RG1.4-floor"]["value"]),
        "G1 floor limb != RG1.4 ledger value (keeps the current 85/92)")
    # ---- W7 artifact pins ----
    pre(KING_ZIP.is_file() and sha256(KING_ZIP) == KING_ZIP_SHA, "king zip drift")
    pre(sha256(KING_NPZ) == KING_NPZ_SHA, "king npz drift")
    pre(sha256(H_NPZ) == H_NPZ_SHA, "H npz drift (!= DEFAULT_MANAGER_SHA256)")
    pre(sha256(M29_NPZ) == M29_NPZ_SHA, "M29 npz drift")
    pre(zip_steps(KING_ZIP) == KING_STEPS, "king zip step count abnormal")
    pre(KING_SD.is_file() and sha256(KING_SD) == KING_SD_SHA,
        "KING_SD drift (the v32 artifact is reused)")
    # ---- 304000 reconciliation + 308000 virginity + evaluation-pool range guard (D1/D2-2a) ----
    assert_seed_304000_provenance(ledger_lines)
    assert_seed_virgin(ledger_lines, SMOKE_SEED)
    assert_pool_guard(SEED, "training")
    assert_pool_guard(SMOKE_SEED, "smoke test")
    if smoke:
        log({"event": "smoke_preflight_ok", "impl_sha16": impl_bundle_sha16()})
        return None
    # ---- W1 freeze notarization (skipped for smoke; the real case needs a clean tree + key files committed) ----
    # correction (2026-07-19): the original "last touched == HEAD" was an overly strict proxy -- a clean tree already guarantees that
    # the on-disk content ≡ the notarized HEAD state, and later commits of unrelated files are not drift; the FREEZE_SHA value
    # always takes the actual HEAD, so the notarization semantics stay complete. Here the key files only need to be committed.
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l != "?? train/leaderboard-assembled-v3.md"]
    pre(not dirty, f"W1: working tree not clean {dirty}")
    head = git("rev-parse", "HEAD")
    for path in ("docs/prereg/PREREG-v33-content-case.md", "train/run_v33_content.py"):
        touch = git("log", "-1", "--format=%H", "--", path)
        pre(bool(touch), f"W1: {path} not committed (no notarizing commit)")
    freezes = [e for e in events if e.get("event") == "FREEZE_SHA"]
    if freezes and freezes[-1]["sha"] != head:
        # P6 chained re-freeze: the chain may continue only if a REFREEZE_REASON file is present
        reason_file = V33 / "REFREEZE_REASON"
        pre(reason_file.is_file(),
            "W1: HEAD != last FREEZE_SHA in the ledger (a P6 chained re-freeze needs REFREEZE_REASON)")
        ev = {"event": "FREEZE_SHA", "sha": head,
              "prev_sha": freezes[-1]["sha"],
              "reason": reason_file.read_text().strip()}
        log(ev)
        events.append(ev)
    # ---- in-case KING_SD parity re-check (W7) ----
    pre(run([PY, "train/check_teacher_parity.py", str(KING_SD),
             str(KING_NPZ)], "parity-king.log", 600) == 0,
        "G-KL-W: king anchor sd and npz attestation failed (in-case 0/1000 re-check)")
    # ---- W-H8 held-out pool discipline (registered side) ----
    holdout_virgin_scan(events)
    # ---- W9 target-archive prerequisite ----
    for t in ALL_EXAM_TAGS:
        has_ledger = any(e.get("event") in ("exam_ok", "CANARY_EVAL")
                         and e.get("tag") == t for e in events)
        if not has_ledger and t not in CANARY_TAGS:
            pre(not (EVAL / f"{t}.json").exists(),
                f"W9: target archive already exists: {t} (restart protocol: .void it first)")
    for leg in LEG_NAMES:
        if leg_starts(events, leg) == 0:
            pre(not (RUNS / leg).exists(), f"run directory left over: {leg}")
    # ---- CASE_RUNTIME capture and W10 resume reconciliation (note: the B1-style W-E0 bit-level assertion cannot be used --
    # this case's E1 changes worker_env.py, so the protocol bundle sha must differ from the v32/B1 settled values; never
    # report it as "sha identical"; the substitute zero-drift proof = the two REF_BITEQ runs + G0-1 dual endpoints) ----
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    CASE_RT = runtime_five(snapshot)
    prior_rt = [e for e in events if e.get("event") == "CASE_RUNTIME"]
    if prior_rt and prior_rt[0]["five"] != CASE_RT:
        log({"event": "CASE_HALT_ENV_DRIFT", "where": "preflight-W10",
             "expected": prior_rt[0]["five"], "current": CASE_RT})
        attention("W5/W10 pre-launch runtime reconciliation mismatch; halting for review")
        raise SystemExit(5)
    # ---- W-C canary scoring set inherited (pinned from the B1 CANARY_SET event line, not re-extracted) ----
    if not stage_done(events, "CANARY_SET"):
        b1_line = json.loads(
            ledger_lines["infra-b1"][W_PIN_EVENT_LINES["B1-CANARY_SET"][1] - 1])
        ev = {"event": "CANARY_SET", "inherited_from": "infra-b1:19 (line sha pinned)",
              "n_D": b1_line["n_D"], "depth2_seeds": b1_line["depth2_seeds"],
              "controls": b1_line["controls"], "C": b1_line["C"],
              "note": "C set and n_D inherited and pinned (W-C, not re-extracted); the scoring seed set only constrains "
                      "R-line readings, not the exam surface (canary exams cover the whole 7000-7031 pool)"}
        log(ev)
        events.append(ev)
    log({"event": "preflight_ok", "king_zip": KING_ZIP_SHA[:16],
         "king_sd": KING_SD_SHA[:16], "impl_sha16": impl_bundle_sha16(),
         "seed": SEED, "smoke_seed": SMOKE_SEED})
    return None


def holdout_virgin_scan(events):
    """W-H8 held-out pool discipline (registered side; inherited verbatim from B1, exemption set extended to the four B1 runs + this case's pair)."""
    sanctioned = {f"{t}.json" for t in
                  ("b1-ref8k-launch", "b1-ref8k-science",
                   "p8-hold8k", "p8-hold8k-m29") + HOLDOUT_TAGS}
    pattern = re.compile(r"\b80(?:[0-2][0-9]|3[01])\b")
    offenders = []
    for arch in sorted(EVAL.glob("*.json")):
        if arch.name in sanctioned:
            continue
        try:
            doc = json.loads(arch.read_text())
            seeds = doc.get("meta", {}).get("seeds", [])
        except (json.JSONDecodeError, UnicodeDecodeError):
            offenders.append(f"{arch.name}: unparsable")
            continue
        if any(8000 <= int(s) <= 8031 for s in seeds):
            offenders.append(arch.name)
    for board in sorted((ROOT / "train").glob("leaderboard*.md")):
        if pattern.search(board.read_text()):
            offenders.append(board.name)
    pre(not offenders, f"W-H8 held-out pool registered-side assertion failed; halting for review: {offenders}")


# ======================================================================
# S1 G0_BASELINE transcription (four E0 files; captured outside the driver, before the freeze commit)
# ======================================================================

def g0_baseline_stage(events):
    for key in ("E0-king-p0", "E0-king-p1", "E0-throne-p0", "E0-throne-p1"):
        path, expected = W_PIN[key]
        pre(path.is_file() and sha256(path) == expected,
            f"E0 dual-baseline archive pin mismatch: {key}")
    if stage_done(events, "G0_BASELINE"):
        return
    ev = {"event": "G0_BASELINE",
          "baselines": {k: W_PIN[k][1] for k in
                        ("E0-king-p0", "E0-king-p1",
                         "E0-throne-p0", "E0-throne-p1")},
          "captured": "before the change (before E0 implementation, captured from clean stashed code by the probe_g0_traj rig "
                      "outside the case; king+throne x both endpoints; executed outside the driver, which only checks the sha "
                      "constants + this transcription)",
          "note": "the 'before the change' burden of proof for G0-1 is carried separately by the E0 stage; new code may not vouch for new code"}
    log(ev)
    events.append(ev)


# ======================================================================
# S2 S-bc1: BC-v1 regeneration split assertion (W7 split form) -> BC_REGEN
# ======================================================================

def _torch_state_equal(a_path: pathlib.Path, b_path: pathlib.Path) -> bool:
    import torch
    a = torch.load(a_path, map_location="cpu", weights_only=True)
    b = torch.load(b_path, map_location="cpu", weights_only=True)
    if set(a) != set(b):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def bc1_stage(events):
    assert_stage_prereqs(events, "BC_REGEN")
    if stage_done(events, "BC_REGEN"):
        prior = [e for e in events if e.get("event") == "BC_REGEN"][-1]
        require(BC_SD.is_file() and sha256(BC_SD) == prior["policy_sha256"],
                "BC-v1 policy cross-launch identity chain broken (!= BC_REGEN settled value)")
        require(sha256(BC_V1_DEMOS) == BC_V1_DEMOS_SHA,
                "BC-v1 demos bytes != frozen constant (cross-launch identity chain broken)")
        return
    require(run([PY, "train/bc_worker.py"],
                f"bc-regen-v1.{time.time_ns()}.log", 3_600) == 0,
            "BC-v1 regeneration failed (R-W FAIL type, whole case halts)")
    # split assertion 1: demos.npz byte sha ≡ frozen constant (np.savez_compressed bytes are deterministic)
    demos_sha = sha256(BC_V1_DEMOS)
    require(demos_sha == BC_V1_DEMOS_SHA,
            f"CASE_HALT: BC-v1 demos bytes {demos_sha[:16]} != frozen constant "
            f"{BC_V1_DEMOS_SHA[:16]} (W7 split assertion 1)")
    # split assertion 2: policy_sd.pt vs the old _previous archived file, tensor-level torch.equal
    # (torch.save bytes are not reproducible -- a byte-equality assertion can never pass, dropped)
    prev_dirs = sorted((BC_V1_DIR / "_previous").iterdir(),
                       key=lambda p: int(p.name)) \
        if (BC_V1_DIR / "_previous").is_dir() else []
    require(bool(prev_dirs), "CASE_HALT: BC-v1 _previous archive missing, no comparison file")
    prev_policy = prev_dirs[-1] / "policy_sd.pt"
    require(prev_policy.is_file(), "CASE_HALT: no policy_sd.pt in the _previous archive")
    require(_torch_state_equal(BC_SD, prev_policy),
            "CASE_HALT: BC-v1 policy torch.equal against the _previous archived file failed"
            " (W7 split assertion 2)")
    import train_ppo
    rec = train_ppo._validate_bc_report(BC_SD, "data_gate", verify_replay=False)
    ev = {"event": "BC_REGEN", "held_out_top1": rec["held_out_top1"],
          "pairs": rec["pairs"], "policy_sha256": sha256(BC_SD),
          "demos_sha256": demos_sha,
          "prev_archive": prev_dirs[-1].name,
          "prev_policy_torch_equal": True,
          "prev_lineage_sha256": BC_SD_PREV_LINEAGE_SHA,
          "note": "the new policy sha becomes the settled value with this event; f052067a… is the old settled value"
                  " kept only as the lineage reference for the torch.equal comparison; refreshing the provenance field is legitimate"
                  " (v32 _previous snapshot precedent)"}
    log(ev)
    events.append(ev)


# ======================================================================
# S3 S-bc2: BC-v2 collection+training -> N12_GATE (P-N12: the single OC 0.65->0.70, one resample)
# ======================================================================

def _n12_event(events, verdict: str, threshold: float, oc_consumed: bool,
               report: dict | None):
    ev = {"event": "N12_GATE", "gate": verdict,
          "preventive_threshold": threshold, "oc_consumed": oc_consumed,
          "n12_gate_min": N12_GATE_MIN, "recall_12_gate_min": RECALL12_GATE_MIN,
          "denominator_definition": N12_DENOMINATOR_DEF,
          "cluster_note": N12_CLUSTER_NOTE}
    if report is not None:
        ev.update({"n12": report.get("n12"),
                   "n12_by_episode": report.get("n12_by_episode"),
                   "recall_12": report.get("recall_12"),
                   "recall_12_denominator": report.get("recall_12_denominator"),
                   "held_out_top1": report.get("held_out_top1"),
                   "pairs": report.get("pairs"),
                   "demos_sha256": report.get("demos_sha256"),
                   "policy_sha256": report.get("policy_sha256")})
    log(ev)
    events.append(ev)
    return ev


def _read_v2_report_if_any() -> dict | None:
    if not BC_V2_REPORT.is_file():
        return None
    try:
        return strict_json_loads(BC_V2_REPORT.read_bytes())
    except Exception:
        return None


def n12_gate_demos_sha(gate_event: dict) -> str:
    """Key lookup for the N12_GATE settled value (rev4 section 12 appendix 2 item 4: the vacuous fallback is dropped in favour of a hard failure --
    an N12_GATE ledger line missing the demos_sha256 key means the identity chain has no anchor, and require fails)."""
    gate_sha = gate_event.get("demos_sha256")
    require(isinstance(gate_sha, str) and len(gate_sha) == 64,
            "N12_GATE ledger line missing demos_sha256 key (identity chain has no anchor, hard failure;"
            " rev4 section 12 appendix 2 item 4)")
    return gate_sha


def assert_lfull_demos_chain(events):
    """Fix 1c (rev4 section 12 appendix 2 item 4): before the L-full leg launch, assert that the measured BC_V2_DEMOS bytes
    sha256 ≡ N12_GATE settled value (cross-launch identity chain; the consumer side of --bc-aux-demos)."""
    passed = [e for e in events
              if e.get("event") == "N12_GATE" and e.get("gate") == "PASS"]
    require(bool(passed), "L-full launch prerequisite: N12_GATE PASS not recorded")
    gate_sha = n12_gate_demos_sha(passed[-1])
    require(BC_V2_DEMOS.is_file(),
            f"L-full launch prerequisite: BC-v2 demos missing: {BC_V2_DEMOS}")
    actual = sha256(BC_V2_DEMOS)
    require(actual == gate_sha,
            f"L-full launch prerequisite: BC-v2 demos bytes {actual[:16]} != N12_GATE "
            f"settled value {gate_sha[:16]} (cross-launch identity chain broken)")


def bc2_stage(events):
    assert_stage_prereqs(events, "N12_GATE")
    passed = [e for e in events
              if e.get("event") == "N12_GATE" and e.get("gate") == "PASS"]
    if passed:
        import bc_worker
        rec = bc_worker._validate_bc_v2_report(
            BC_V2_SD, expected_manager_sha256=sha256(M29_NPZ))
        gate_sha = n12_gate_demos_sha(passed[-1])
        require(rec["demos_sha256"] == gate_sha,
                "BC-v2 demos cross-launch identity chain broken (!= N12_GATE settled value)")
        return
    def _failed_at(threshold: float) -> bool:
        return any(e.get("event") == "N12_GATE" and e.get("gate") == "FAIL"
                   and e.get("preventive_threshold") == threshold
                   for e in events)

    if _failed_at(PREVENTIVE_OC):
        # the single OC is already used and FAIL is recorded: contingency exhausted, no further collection (a restart may not revive it)
        attention("BC-v2 N12 gate: the single OC (0.70) is used and FAIL is recorded -- contingency exhausted;"
                  " no freeze, report for review (P-N12); class weighting needs a separate approval and nothing may stand in for it meanwhile")
        print("P-N12: contingency exhausted (recorded in the ledger); halted pending a decision (not a failure state)", flush=True)
        raise SystemExit(0)
    oc_path = stage_done(events, "OC_CONSUMED") or _failed_at(PREVENTIVE_MAIN)
    attempts = ([(PREVENTIVE_OC, True)] if oc_path
                else [(PREVENTIVE_MAIN, False), (PREVENTIVE_OC, True)])
    for threshold, is_oc in attempts:
        if is_oc and not stage_done(events, "OC_CONSUMED"):
            # clause on using the OC early (added in verification): the mandatory note that the convergence rule is pending confirmation (open item C)
            ev = {"event": "OC_CONSUMED", "knob": "preventive_threshold",
                  "from": PREVENTIVE_MAIN, "to": PREVENTIVE_OC,
                  "note": "convergence rule pending confirmation (open item C): the automatic class-weighting contingency is removed and needs a separate"
                          " approval, and nothing may stand in for it meanwhile; if launch review 2(f) rejects the convergence rule,"
                          " the frozen case moves automatically to a P6 re-freeze pending a decision -- early use is not a fait accompli"}
            log(ev)
            events.append(ev)
        cmd = [PY, "train/bc_worker.py", "--v2",
               "--manager-npz", str(M29_NPZ)]
        if is_oc:
            cmd += ["--preventive-threshold", str(PREVENTIVE_OC)]
        rc = run(cmd, f"bc-v2-{threshold}.{time.time_ns()}.log", 7_200)
        report = _read_v2_report_if_any()
        if rc == 0:
            import bc_worker
            rec = bc_worker._validate_bc_v2_report(
                BC_V2_SD, expected_manager_sha256=sha256(M29_NPZ))
            _n12_event(events, "PASS", threshold, is_oc, rec)
            return
        require(report is not None and report.get("data_gate") == "FAIL",
                f"BC-v2 collection/training operational failure (rc={rc} and no FAIL receipt); halting for review")
        _n12_event(events, "FAIL", threshold, is_oc, report)
    # contingency exhausted: no freeze, report for review (P-N12 is not a failure state; NEEDS_ATTENTION, halted pending a decision)
    attention("BC-v2 N12/recall gate contingency exhausted (main plan 0.65 + the single OC 0.70, once each):"
              " no freeze, report for review; never launch with a gate failure; class weighting needs a separate approval and nothing may stand in for it meanwhile (P-N12)")
    print("P-N12: contingency exhausted, no freeze, report for review -- halted pending a decision (not a failure state)", flush=True)
    raise SystemExit(0)


# ======================================================================
# S4 G0 six-piece set
# ======================================================================

def g0_endpoint_stage(events):
    """G0-1 dual-endpoint bit-level identity: p≡0.0 ≡ skip_dry=False / p≡1.0 ≡ skip_dry=True,
    reconciled bit for bit per seed per beat against the pre-implementation E0 dual baselines (king+throne) (the replay probe
    asserts the wage identity W≡R−bonus per window). Any failure -> CASE_HALT_G0 (7)."""
    assert_stage_prereqs(events, "G0_ENDPOINT")
    if stage_done(events, "G0_ENDPOINT"):
        return
    replays = (
        ("king-p0", [PY, "train/probe_g0_traj.py", "replay"]),
        ("throne-p0", [PY, "train/probe_g0_traj.py", "replay", "throne"]),
        ("king-p1", [PY, "train/probe_g0_traj_skip_dry.py", "replay"]),
        ("throne-p1", [PY, "train/probe_g0_traj_skip_dry.py", "replay", "throne"]),
    )
    for name, cmd in replays:
        rc = run(cmd, f"g0-endpoint-{name}.{time.time_ns()}.log", 1_800)
        if rc != 0:
            log({"event": "CASE_HALT_G0", "via": "G0_ENDPOINT", "which": name,
                 "rc": rc})
            attention(f"G0-1 dual-endpoint bit-level identity failed ({name}); no freeze, no launch")
            raise SystemExit(7)
    ev = {"event": "G0_ENDPOINT", "verdict": "PASS",
          "endpoints": {"p0": "skip_dry=False ≡ p≡0.0", "p1": "skip_dry=True ≡ p≡1.0"},
          "workers": ["king", "throne"], "seeds": "7000-7015",
          "baselines": {k: W_PIN[k][1][:16] for k in
                        ("E0-king-p0", "E0-king-p1",
                         "E0-throne-p0", "E0-throne-p1")},
          "bridge_note": "bridge-attestation downgrade clause (registered in D2-1): if any anchor-bridge attestation fails, gold evaluation only reports numbers,"
                         " without comparing to the board, and the title verdict is suspended and reported; the D4-1 title flow requires this item to pass first;"
                         " old bridge attestations are not inherited automatically"}
    log(ev)
    events.append(ev)


def _leg_shared_cmd(run_name: str, seed: int, total_steps: int,
                    calib_probes: tuple[int, ...],
                    with_b1_knobs: bool = True,
                    calib_record_only: bool = False) -> list[str]:
    """Shared command form (D3 verbatim: v32-sov verbatim + B1 instrument knobs + the last two
    E5 (1)(2) record-only knobs added by the rev4 correction (section 12 appendix 2 item 1); autonomy is on by default,
    no --no-drink-sovereignty; the shared form itself has no --skip-dry;
    with_b1_knobs=False is only for the G0-2a bare arm = v32-sov verbatim, carrying no new knob)."""
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
           "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "512",
           "--num-envs", "4", "--lr", "3e-4", "--ent-coef", "0.005",
           "--seed", str(seed), "--total-steps", str(total_steps),
           "--run-name", run_name, "--distill-beta", str(BETA),
           "--teacher-sd", str(BC_SD), "--teacher-override", str(KING_SD),
           "--manager-npz", str(M29_NPZ),
           "--resume-from", str(KING_ZIP), "--allow-legacy-resume",
           "--calib-probes", ",".join(str(x) for x in calib_probes)]
    if calib_record_only:
        cmd.append("--calib-record-only")
    if with_b1_knobs:
        cmd += ["--ckpt-every-steps", str(CKPT_EVERY),
                "--sentinel-every", str(SENTINEL_EVERY),
                "--dry-anchor-every", str(DRY_ANCHOR_EVERY),
                "--distill-ce-probe-every", str(DISTILL_CE_PROBE_EVERY),
                "--drywin-metrics-every", str(DRYWIN_METRICS_EVERY)]
    return cmd


def leg_cmd(leg: str) -> list[str]:
    """Full CLI of the three legs = shared command form + the D3 per-leg additions verbatim (the LEGS table is authoritative)."""
    extras = dict(LEGS)[leg]
    return _leg_shared_cmd(leg, SEED, LEG_STEPS, CALIB_PROBES) + list(extras)


def _smoke_cmd(variant: str) -> list[str]:
    """G0-2a/2b smoke-test commands (seed 308000, 102,400 steps = 50 rollouts).
    bare = v32-sov command form verbatim incl. --skip-dry (calib step table 3747984,3947984 verbatim,
    never fires in the window, B1 instrument knobs not carried -- the bare arm ≡ HEAD behaviour);
    knobs = all new knobs on: schedule pinned at p≡1.0 (hold:1.0:50) + λ_bc=0 (bc_aux flag
    present, predicate inactive) + E5 (1)(2) probes + B1 instruments (from rev4 both the E5 (1)(2) and B1 knobs are injected with the
    shared form with_b1_knobs=True; the calib step table switches to two in-window points, record-only
    zero intrusion already shown in B1; implementation discretion note);
    func-p / func-aux = G0-2b functional short runs (full main table, leg-relative prefix of the first 50 rollouts)."""
    if variant == "bare":
        return (_leg_shared_cmd(SMOKE_RUNS["bare"], SMOKE_SEED, SMOKE_STEPS,
                                V32_SOV_CALIB, with_b1_knobs=False,
                                calib_record_only=True)
                + ["--skip-dry"])
    if variant == "knobs":
        return (_leg_shared_cmd(SMOKE_RUNS["knobs"], SMOKE_SEED, SMOKE_STEPS,
                                SMOKE_CALIB, calib_record_only=True)
                + ["--dry-curriculum-schedule", SMOKE_HOLD_TABLE,
                   "--bc-aux-lambda", "0.0",
                   "--bc-aux-demos", str(BC_V2_DEMOS)])
    if variant == "func-p":
        return (_leg_shared_cmd(SMOKE_RUNS["func-p"], SMOKE_SEED, SMOKE_STEPS,
                                SMOKE_CALIB)
                + ["--dry-curriculum-schedule", MAIN_TABLE])
    if variant == "func-aux":
        return (_leg_shared_cmd(SMOKE_RUNS["func-aux"], SMOKE_SEED, SMOKE_STEPS,
                                SMOKE_CALIB)
                + ["--dry-curriculum-schedule", MAIN_TABLE,
                   "--bc-aux-lambda", str(BC_AUX_LAMBDA),
                   "--bc-aux-demos", str(BC_V2_DEMOS)])
    raise ValueError(f"unknown smoke arm: {variant}")


def _load_zip_states(path: pathlib.Path):
    import torch
    with zipfile.ZipFile(path) as z:
        policy = torch.load(io.BytesIO(z.read("policy.pth")),
                            map_location="cpu", weights_only=True)
        optim = torch.load(io.BytesIO(z.read("policy.optimizer.pth")),
                           map_location="cpu", weights_only=True)
    return policy, optim


def _tree_equal(a, b, path, diffs):
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a, b)):
            diffs.append(path)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}:keys")
            return
        for k in sorted(a, key=str):
            _tree_equal(a[k], b[k], f"{path}.{k}", diffs)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}:len")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _tree_equal(x, y, f"{path}[{i}]", diffs)
        return
    if a != b:
        diffs.append(path)


def _progress_lines(run_dir: pathlib.Path) -> list[dict]:
    lines = []
    for raw in (run_dir / "progress.jsonl").read_text().splitlines():
        rec = json.loads(raw)
        rec.pop("t", None)                    # wall-clock field is not RNG-related; dropped
        lines.append(rec)
    return lines


def sentinel_line_counts(sentinel_path: pathlib.Path) -> dict[str, int]:
    """Fix 2 (rev4 section 12 appendix 2 item 5): count the two line types of sentinel.jsonl separately -- v23 sentinel lines and
    dry-anchor lines share the file (both train_ppo callbacks write to the same name); the per-limb counts feed
    the four-item G0-2a "the instruments really ran" check (same literal discrimination as the RC.12 consumer at :2147)."""
    rows = (sentinel_path.read_text().splitlines()
            if sentinel_path.is_file() else [])
    return {"sentinel": sum(1 for x in rows if '"sentinel": "v23"' in x),
            "dry_anchor": sum(1 for x in rows if '"dry-anchor"' in x)}


def instrument_jsonl_digest(path: pathlib.Path) -> dict:
    """Fix 5b (rev4 section 12 appendix 2 item 1/D8): end-of-leg transcription payload of an instrument jsonl -- file bytes
    sha256 + line count + aggregated final-section payload (final=True lines; if none, take the last line as is,
    with a fall_back_last_line note); an empty or missing file fails loudly."""
    require(path.is_file(), f"instrument file missing: {path} (E5 knob on but no output)")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    require(bool(rows), f"instrument file empty: {path} (E5 knob on but zero lines)")
    finals = [r for r in rows if r.get("final")]
    digest = {"sha256": sha256(path), "lines": len(rows),
              "final": finals[-1] if finals else rows[-1]}
    if not finals:
        digest["fall_back_last_line"] = True
    return digest


# Fix 4 (rev4 section 12 appendix 2 item 3): lower bound of the visible equivalent of episode seed-sequence identity: the common prefix's
# cumulative len must be >= one env's rollout steps (n-steps 512) -- within the first rollout the two runs' weights are
# bit-identical (λ_bc only enters the weights at the first train()), so their completed-window lines must match exactly; if the initial episode
# seed stream were contaminated, the first line would already diverge.
SEED_EQUIV_MIN_PREFIX_STEPS = 512


def progress_common_prefix(lines_a: list[dict], lines_b: list[dict]) -> dict:
    """Fix 4 (rev4 section 12 appendix 2 item 3): the strongest visible equivalent of the episode seed-sequence identity assertion.

    Reported as is (faithful execution of the no-silent-weakening clause): progress.jsonl worker lines have only
    the four fields ep/t/reward/len -- the episode seed appears only in the WorkerWindowEnv.reset info
    and never in a line, so **no episode seed field sequence exists in the visible run outputs** (no fabrication).
    Equivalent = identity of the common prefix of the two runs' (ep, reward, len) sequences before the first divergence: trajectories
    can only diverge after the first train() (where λ_bc is consumed), so the common prefix must be non-empty and its cumulative len
    must cover the first-rollout lower bound (SEED_EQUIV_MIN_PREFIX_STEPS); if the seed stream were contaminated by the aux rng
    being present, the divergence would show in the first line. The missing direct proof over the full seed sequence goes to limitations."""
    n = 0
    for a, b in zip(lines_a, lines_b):
        if a != b:
            break
        n += 1
    diverged = n < min(len(lines_a), len(lines_b))
    return {"prefix_lines": n,
            "prefix_len_steps": sum(int(x.get("len", 0)) for x in lines_a[:n]),
            "first_divergence_index": (n if diverged else None),
            "lines": [len(lines_a), len(lines_b)],
            "field": "(ep, reward, len) common prefix (the episode seed field is not visible;"
                     " equivalent definition)"}


def g0_nullintrusion_stage(events):
    """G0-2a zero-intrusion pair (tensor level): all new knobs on (p≡1.0, λ_bc=0) vs the bare-recipe arm
    (v32-sov command form verbatim incl. --skip-dry), 2x102,400 steps; policy state_dict +
    optimizer state, torch.equal per tensor + RNG-related telemetry field by field;
    this item is the guarantee that L-base ≡ the v32-sov recipe (E4). FAIL -> CASE_HALT_G0 (7)."""
    impl16 = impl_bundle_sha16()
    for e in events:
        if (e.get("event") == "G0_NULLINTRUSION" and e.get("verdict") == "PASS"
                and e.get("impl_sha16") == impl16):
            print(f"G0-2a already passed (impl {impl16}); idempotent skip", flush=True)
            return
    pre(BC_V2_DEMOS.is_file(), "G0-2a prerequisite: BC-v2 demos missing (pass S-bc2 first)")
    results = {}
    for variant in ("bare", "knobs"):
        run_dir = RUNS / SMOKE_RUNS[variant]
        _archive_residue(run_dir)   # correction: archive leftovers so jsonl output does not pile up (B1 _prepare_run_dir; the 2026-07-19 FAIL is on record)
        t0 = time.time()
        rc = run(_smoke_cmd(variant), f"smoke-{variant}.log", SMOKE_TIMEOUT)
        nt = zip_steps(run_dir / "model_final.zip")
        results[variant] = {"rc": rc, "nt": nt,
                            "dt_min": round((time.time() - t0) / 60, 1)}
        if rc != 0 or nt != SMOKE_END:
            log({"event": "G0_NULLINTRUSION", "verdict": "FAIL",
                 "impl_sha16": impl16, "runs": results,
                 "why": f"smoke {variant} fell short (rc={rc}, nt={nt}, "
                        f"target={SMOKE_END})"})
            attention("G0-2a smoke run failed; no freeze, no launch")
            raise SystemExit(7)
    knobs_dir = RUNS / SMOKE_RUNS["knobs"]
    bare_dir = RUNS / SMOKE_RUNS["bare"]
    # the instruments really ran ("proven zero-intrusion" must not actually mean "never ran"; rev4 section 12 appendix 2 item 5, all four checked:
    # the sentinel/dry-anchor limbs added -- the two line types of sentinel.jsonl counted separately)
    knobs_sentinel = sentinel_line_counts(knobs_dir / "sentinel.jsonl")
    evidence = {
        "knobs_ckpt": sorted(p.name for p in (knobs_dir / "ckpt").glob("*.zip"))
        if (knobs_dir / "ckpt").is_dir() else [],
        "knobs_calib_lines": (len((knobs_dir / "calib.jsonl").read_text()
                                  .splitlines())
                              if (knobs_dir / "calib.jsonl").is_file() else 0),
        "knobs_sentinel_lines": knobs_sentinel["sentinel"],
        "knobs_dry_anchor_lines": knobs_sentinel["dry_anchor"],
        "knobs_distill_probe_lines": (
            len((knobs_dir / "distill_ce_probe.jsonl").read_text().splitlines())
            if (knobs_dir / "distill_ce_probe.jsonl").is_file() else 0),
        "knobs_drywin_lines": (
            len((knobs_dir / "drywin_metrics.jsonl").read_text().splitlines())
            if (knobs_dir / "drywin_metrics.jsonl").is_file() else 0),
        "knobs_curriculum_lines": (
            len((knobs_dir / "dry_curriculum.jsonl").read_text().splitlines())
            if (knobs_dir / "dry_curriculum.jsonl").is_file() else 0),
        "bare_calib_lines": (len((bare_dir / "calib.jsonl").read_text()
                                 .splitlines())
                             if (bare_dir / "calib.jsonl").is_file() else 0),
    }
    instrument_ok = (evidence["knobs_calib_lines"] >= 1
                     and evidence["knobs_sentinel_lines"] >= 1
                     and evidence["knobs_dry_anchor_lines"] >= 1
                     and evidence["knobs_distill_probe_lines"] >= 1
                     and evidence["knobs_drywin_lines"] >= 1
                     and evidence["knobs_curriculum_lines"] == SMOKE_STEPS // QUANTUM
                     and f"model_{KING_STEPS + CKPT_EVERY}_steps.zip"
                     in evidence["knobs_ckpt"])
    # re-check that the knobs arm schedule is pinned at p≡1.0 (all 50 entries of the hold table are 1.0)
    hold_ok = True
    if (knobs_dir / "dry_curriculum.jsonl").is_file():
        pushed = [json.loads(x) for x in (knobs_dir / "dry_curriculum.jsonl")
                  .read_text().splitlines()]
        hold_ok = ([(p["rollout_index"], p["p"]) for p in pushed]
                   == [(i, 1.0) for i in range(SMOKE_STEPS // QUANTUM)])
    # tensor-level criterion
    pol_b, opt_b = _load_zip_states(bare_dir / "model_final.zip")
    pol_k, opt_k = _load_zip_states(knobs_dir / "model_final.zip")
    tensor_diffs: list[str] = []
    _tree_equal(pol_b, pol_k, "policy", tensor_diffs)
    _tree_equal(opt_b, opt_k, "optim", tensor_diffs)
    # RNG-related telemetry field by field (B1 precedent; wall clock dropped)
    telemetry_diffs: list[str] = []
    prog_b, prog_k = _progress_lines(bare_dir), _progress_lines(knobs_dir)
    if len(prog_b) != len(prog_k):
        telemetry_diffs.append(f"progress line count {len(prog_b)} != {len(prog_k)}")
    else:
        for i, (a, b) in enumerate(zip(prog_b, prog_k)):
            if a != b:
                telemetry_diffs.append(f"progress[{i}]: {a} != {b}")
                if len(telemetry_diffs) >= 20:
                    break
    verdict = ("PASS" if not tensor_diffs and not telemetry_diffs
               and instrument_ok and hold_ok else "FAIL")
    ev = {"event": "G0_NULLINTRUSION", "verdict": verdict,
          "impl_sha16": impl16, "seed": SMOKE_SEED, "steps": SMOKE_STEPS,
          "window": [KING_STEPS, SMOKE_END], "runs": results,
          "bare_arm": "v32-sov command form verbatim incl. --skip-dry (dependency pinned, engineering B1)",
          "knobs_arm": "dry_curriculum (hold p≡1.0) + bc_aux (λ=0, present but inactive) + "
                       "E5 (1)(2) probes + B1 instruments (calib switched to two in-window points, record-only)",
          "tensor_verdict": ("all tensors equal under torch.equal" if not tensor_diffs
                             else tensor_diffs[:20]),
          "telemetry_verdict": ("RNG-related fields equal field by field" if not telemetry_diffs
                                else telemetry_diffs[:20]),
          "instruments_actually_fired": instrument_ok,
          "knobs_hold_table_identity": hold_ok,
          "instrument_evidence": evidence,
          "criteria_note": "file byte-level comparison dropped (SB3 zip timestamps/pickle not reproducible);"
                           " smoke telemetry, beyond the bit-level criterion, must not feed any go/no-go input"}
    log(ev)
    events.append(ev)
    if verdict != "PASS":
        attention("G0-2a zero intrusion failed: instruments go back for redesign; no freeze, no launch")
        raise SystemExit(7)


def _read_curriculum_jsonl(run_dir: pathlib.Path) -> list[tuple[int, float]]:
    p = run_dir / "dry_curriculum.jsonl"
    require(p.is_file(), f"dry_curriculum.jsonl missing: {run_dir.name}")
    return [(int(rec["rollout_index"]), float(rec["p"]))
            for rec in (json.loads(x) for x in p.read_text().splitlines())]


def verify_curriculum_prefix(run_dir: pathlib.Path, table, n_rollouts: int
                             ) -> list[str]:
    """Assert measured p sequence ≡ registry prefix (added in verification; closes the gap all three G0 items were blind to).
    Returns the list of differences (empty = identical)."""
    actual = _read_curriculum_jsonl(run_dir)
    expected = [(i, float(table[i])) for i in range(n_rollouts)]
    diffs = []
    if len(actual) != len(expected):
        diffs.append(f"line count {len(actual)} != {len(expected)}")
    for (ai, ap), (ei, ep) in zip(actual, expected):
        if ai != ei or ap != ep:
            diffs.append(f"[{ei}] measured ({ai},{ap}) != registered ({ei},{ep})")
            if len(diffs) >= 10:
                break
    return diffs


def _bc_aux_grad12_report(zip_path: pathlib.Path, demos_npz: pathlib.Path) -> dict:
    """G0-2b: recompute the a12 head gradient against the rev3 training-only positive/negative calibration targets.

    The old smoke test only showed that positive-example CE gives row 12 a gradient, which would pass a 90% collapse. This item
    must consume both the hard negatives and the KING start anchor, with both gradient groups in the same objective.
    """
    import numpy as np
    import torch as th
    import train_ppo

    with zipfile.ZipFile(zip_path) as z:
        sd = th.load(io.BytesIO(z.read("policy.pth")),
                     map_location="cpu", weights_only=True)
    anchor_sd, _ = _load_zip_states(KING_ZIP)
    x, y, episode_id, masks, _ = train_ppo._load_bc_aux_demos_v2(
        demos_npz, expected_manager_sha256=sha256(M29_NPZ))
    bx, by, bm = train_ppo._build_bc_aux_training_bank(
        x, y, episode_id, masks)
    positive = np.flatnonzero(by == 12)
    negative = np.flatnonzero(by != 12)
    require(len(positive) > 0 and len(negative) > 0,
            "G0-2b: the rev3 training-only bank must contain both positives and hard negatives")
    size = min(256, len(by))
    pos_n = min(len(positive), max(1, int(round(
        size * train_ppo._BC_AUX_POSITIVE_FRACTION))))
    neg_n = min(len(negative), max(1, size - pos_n))
    index = np.concatenate([positive[:pos_n], negative[:neg_n]])
    obs = th.tensor(bx[index], dtype=th.float32)
    mask_t = th.tensor(bm[index])
    h = th.tanh(obs @ sd["mlp_extractor.policy_net.0.weight"].T
                + sd["mlp_extractor.policy_net.0.bias"])
    h = th.tanh(h @ sd["mlp_extractor.policy_net.2.weight"].T
                + sd["mlp_extractor.policy_net.2.bias"])
    wa = sd["action_net.weight"].clone().requires_grad_(True)
    logits = h @ wa.T + sd["action_net.bias"]
    logp = th.log_softmax(
        th.where(mask_t, logits, th.full_like(logits, -1e8)), dim=-1)
    p12 = logp[:, 12].exp().clamp(1e-7, 1.0 - 1e-7)
    pos_loss = th.nn.functional.binary_cross_entropy(
        p12[:pos_n], th.full_like(
            p12[:pos_n], train_ppo._BC_AUX_POSITIVE_TARGET))
    neg_loss = th.nn.functional.binary_cross_entropy(
        p12[pos_n:], th.full_like(
            p12[pos_n:], train_ppo._BC_AUX_NEGATIVE_TARGET))
    anchor_logits = train_ppo._policy_logits_from_sb3_state_dict(
        anchor_sd, bx[index])
    anchor_logp = th.log_softmax(
        th.where(mask_t, anchor_logits,
                 th.full_like(anchor_logits, -1e8)), dim=-1)
    anchor_probs = anchor_logp.exp()[pos_n:]
    anchor_kl = (anchor_probs * (
        anchor_logp[pos_n:] - logp[pos_n:]
    )).sum(dim=-1).mean()
    objective = (
        train_ppo._BC_AUX_POSITIVE_FRACTION * pos_loss
        + (1.0 - train_ppo._BC_AUX_POSITIVE_FRACTION) * neg_loss
        + train_ppo._BC_AUX_ANCHOR_KL_COEF * anchor_kl)
    (g,) = th.autograd.grad(objective, [wa])
    gtotal = float(th.linalg.vector_norm(g))
    g12 = float(g[12].abs().sum())
    require(gtotal > 0.0 and g12 > 0.0,
            "G0-2b: rev2 total/a12 head gradient is zero")
    return {
        "objective_revision": train_ppo._BC_AUX_OBJECTIVE_REVISION,
        "n_pairs_12": int(len(positive)),
        "n_hard_negatives": int(len(negative)),
        "batch_positive": int(pos_n),
        "batch_negative": int(neg_n),
        "objective": float(objective.detach()),
        "positive_bce": float(pos_loss.detach()),
        "negative_bce": float(neg_loss.detach()),
        "anchor_kl": float(anchor_kl.detach()),
        "grad_total_l2": gtotal,
        "grad12_abs_sum": g12,
        "all_m12_true": bool(bm[:, 12].all()),
    }


def _bc_aux_behavior_report(zip_path: pathlib.Path,
                            demos_npz: pathlib.Path) -> dict:
    """E5 held-out behaviour hard gate shared by existing and new smoke tests (purely offline, real masks)."""
    import train_ppo

    sd, _ = _load_zip_states(zip_path)
    anchor_sd, _ = _load_zip_states(KING_ZIP)
    x, y, episode_id, masks, _ = train_ppo._load_bc_aux_demos_v2(
        demos_npz, expected_manager_sha256=sha256(M29_NPZ))
    metrics = train_ppo.bc_aux_behavior_metrics(
        sd, x, y, episode_id, masks, anchor_sd=anchor_sd,
        heldout_only=True)
    return {"metrics": metrics,
            "gate": train_ppo.bc_aux_behavior_gate(
                metrics, require_root_anchor=True)}


def g0_funcsmoke_stage(events):
    """G0-2b functional smoke test (no equality assertion): p<1 short run (prefix of the first 50 rollouts of the main table) +
    λ_bc>0 short run; measured p sequence ≡ registry prefix, dry windows really enter the learning distribution, the auxiliary CE is really consumed
    + non-zero head-12 gradient, class-12 demos cover m[12]=True in full. FAIL -> CASE_HALT_G0 (7).
    Implementation discretion reported: a full-trajectory proof that "the training RNG state trajectory is equal point by point" is beyond the visible run outputs;
    the gap is approximately closed by the identity of the two short runs' dry_curriculum.jsonl (the p draw stream is not disturbed by λ being present) + prefix identity,
    and recorded as is under limitations."""
    impl16 = impl_bundle_sha16()
    for e in events:
        if (e.get("event") == "G0_SMOKE" and e.get("verdict") == "PASS"
                and e.get("impl_sha16") == impl16):
            print(f"G0-2b already passed (impl {impl16}); idempotent skip", flush=True)
            return
    pre(BC_V2_DEMOS.is_file(), "G0-2b prerequisite: BC-v2 demos missing (pass S-bc2 first)")
    table = dry_curriculum_table()
    n_roll = SMOKE_STEPS // QUANTUM
    results = {}
    for variant in ("func-p", "func-aux"):
        run_dir = RUNS / SMOKE_RUNS[variant]
        _archive_residue(run_dir)   # correction: same as G0-2a, archive leftovers
        rc = run(_smoke_cmd(variant), f"smoke-{variant}.log", SMOKE_TIMEOUT)
        nt = zip_steps(run_dir / "model_final.zip")
        results[variant] = {"rc": rc, "nt": nt}
        if rc != 0 or nt != SMOKE_END:
            log({"event": "G0_SMOKE", "verdict": "FAIL", "impl_sha16": impl16,
                 "runs": results, "why": f"{variant} fell short (rc={rc}, nt={nt})"})
            attention("G0-2b functional smoke run failed; no freeze, no launch")
            raise SystemExit(7)
    funcp_dir = RUNS / SMOKE_RUNS["func-p"]
    funcaux_dir = RUNS / SMOKE_RUNS["func-aux"]
    prefix_diffs = {v: verify_curriculum_prefix(RUNS / SMOKE_RUNS[v], table,
                                                n_roll)
                    for v in ("func-p", "func-aux")}
    p_stream_equal = (_read_curriculum_jsonl(funcp_dir)
                      == _read_curriculum_jsonl(funcaux_dir))
    # Fix 4 (rev4 section 12 appendix 2 item 3): episode seed-sequence identity -- assertion of the strongest visible equivalent
    # (progress.jsonl has no seed field; definition and rationale in progress_common_prefix)
    seed_seq = progress_common_prefix(_progress_lines(funcp_dir),
                                      _progress_lines(funcaux_dir))
    seed_seq_ok = (seed_seq["prefix_lines"] >= 1
                   and seed_seq["prefix_len_steps"] >= SEED_EQUIV_MIN_PREFIX_STEPS)
    # the dry-window branch really enters the learning distribution (E5 (2) drywin dry group n>0; fresh windows n>0 as usual)
    dry_seen = fresh_seen = False
    drywin_path = funcp_dir / "drywin_metrics.jsonl"
    if drywin_path.is_file():
        for raw in drywin_path.read_text().splitlines():
            rec = json.loads(raw)
            windows = rec.get("windows", {})
            dry_n = (windows.get("dry") or {}).get("n", 0)
            fresh_n = (windows.get("fresh") or {}).get("n", 0)
            dry_seen = dry_seen or dry_n > 0
            fresh_seen = fresh_seen or fresh_n > 0
    # evidence that λ_bc is consumed: mount printout + scalar persisted in the zip + offline autograd head-12 gradient
    aux_log = sorted(V33.glob("smoke-func-aux.log"))
    mounted = bool(aux_log) and "[4b]" in aux_log[-1].read_text()
    aux_lambda_in_zip = zip_data_field(funcaux_dir / "model_final.zip",
                                       "bc_aux_lambda")
    grad_report = _bc_aux_grad12_report(funcaux_dir / "model_final.zip",
                                        BC_V2_DEMOS)
    behavior_report = _bc_aux_behavior_report(
        funcaux_dir / "model_final.zip", BC_V2_DEMOS)
    verdict = ("PASS" if not prefix_diffs["func-p"]
               and not prefix_diffs["func-aux"] and p_stream_equal
               and seed_seq_ok
               and dry_seen and fresh_seen and mounted
               and aux_lambda_in_zip == BC_AUX_LAMBDA
               and behavior_report["gate"]["verdict"] == "PASS"
               else "FAIL")
    ev = {"event": "G0_SMOKE", "verdict": verdict, "impl_sha16": impl16,
          "seed": SMOKE_SEED, "steps": SMOKE_STEPS, "runs": results,
          "p_prefix_identity": {v: (d or "≡ registry prefix (identical)")
                                for v, d in prefix_diffs.items()},
          "p_stream_equal_across_aux_onoff": p_stream_equal,
          "episode_seed_seq_identity": {**seed_seq, "ok": seed_seq_ok,
                                        "min_prefix_steps":
                                            SEED_EQUIV_MIN_PREFIX_STEPS},
          "dry_branch_entered_learning": dry_seen,
          "fresh_branch_present": fresh_seen,
          "bc_aux_mount_marker": mounted,
          "bc_aux_lambda_in_zip": aux_lambda_in_zip,
          "grad12": grad_report,
          "a12_behavior": behavior_report,
          "limitations": "the episode seed-sequence identity limb is done (rev4 section 12 appendix 2 item 3), but under the"
                         " strongest-visible-equivalent definition: progress.jsonl has no episode seed"
                         " field (worker lines only have ep/t/reward/len), so the gap is approximately closed by the identity of the two runs' (ep, "
                         "reward, len) common prefix (>= the first-rollout coverage bound) + "
                         "p draw stream identity + the unit-test family of the derivation functions + the G0-2a torch.equal "
                         "fallback -- the missing direct proof over the full seed sequence and of 'the training RNG state trajectory is equal"
                         " point by point' is recorded as is (listed separately in the hand-over note);"
                         " direct output evidence of the dry-window skip branch (scripted runs) is not in the SB3 infos"
                         " stream; it is approximated by the conjunction of p<1 and the dry group really firing"}
    log(ev)
    events.append(ev)
    if verdict != "PASS":
        attention("G0-2b functional smoke test failed; no freeze, no launch")
        raise SystemExit(7)


def g0_ghost_stage(events):
    """G0-ghost (v12 lesion regression, reusing the v32 probe): with belt=0, 12 is still illegal; the _drain fallback
    is not disabled by the curriculum/auxiliary loss; the death-beat termination ladder is unchanged; no drinking after death."""
    assert_stage_prereqs(events, "G0_GHOST")
    if stage_done(events, "G0_GHOST"):
        return
    rc = run([PY, "train/probe_g0_ghost.py"],
             f"g0-ghost.{time.time_ns()}.log", 1_800)
    if rc != 0:
        log({"event": "CASE_HALT_G0", "via": "G0_GHOST", "rc": rc})
        attention("G0-ghost failed; no freeze, no launch")
        raise SystemExit(7)
    ev = {"event": "G0_GHOST", "verdict": "PASS"}
    log(ev)
    events.append(ev)


def g0_demo_ledger_stage(events):
    """G0-demo pool and per-generation accounting (DEMO_LEDGER): v1 bytes pinned / v1 path forbids (11,12)
    unchanged, v2 path forbids 11 and allows 12 / v2 masks on-manifold / schema isolation mutually exclusive."""
    assert_stage_prereqs(events, "DEMO_LEDGER")
    if stage_done(events, "DEMO_LEDGER"):
        return
    import bc_worker
    require(bc_worker.forbidden_actions_for_generation(1) == (11, 12),
            "per-generation collection ban broken: v1 must forbid (11,12) unchanged")
    require(bc_worker.forbidden_actions_for_generation(2) == (11,),
            "per-generation collection ban broken: v2 must forbid 11 and allow 12")
    require(sha256(BC_V1_DEMOS) == BC_V1_DEMOS_SHA,
            "DEMO_LEDGER: BC-v1 demos bytes != frozen constant")
    v2_rec = bc_worker._validate_bc_v2_report(
        BC_V2_SD, expected_manager_sha256=sha256(M29_NPZ))
    require(v2_rec["teacher_generation"] == 2, "v2 receipt generation record abnormal")
    # the v1 validator fails loudly on v2 files (schema isolation, mutually exclusive, E2)
    import train_ppo
    v1_rejects_v2 = False
    try:
        train_ppo._validate_bc_report(BC_V2_SD, "data_gate", verify_replay=False)
    except Exception:
        v1_rejects_v2 = True
    require(v1_rejects_v2, "schema isolation broken: the v1 validator did not reject the v2 file")
    import numpy as np
    d = np.load(BC_V2_DEMOS)
    require({"X", "Y", "episode_id", "masks"} <= set(d.files),
            "v2 demos schema lacks per-sample masks (E2)")
    sel = d["Y"] == 12
    require(bool(sel.any()) and bool(d["masks"][sel][:, 12].all()),
            "v2 demos class-12 examples m[12] assertion failed")
    require(not d["masks"][:, 11].any(), "v2 demos m[11] always-False assertion failed")
    ev = {"event": "DEMO_LEDGER",
          "v1": {"demos_sha256": BC_V1_DEMOS_SHA,
                 "forbidden": [11, 12], "path": "canonical (the four train_ppo sites unchanged, not a character touched)"},
          "v2": {"demos_sha256": v2_rec["demos_sha256"],
                 "policy_sha256": v2_rec["policy_sha256"],
                 "forbidden": [11], "n12": v2_rec["n12"],
                 "preventive_threshold": v2_rec["preventive_threshold"],
                 "masks": "per-sample env.action_masks() captured live (the only on-manifold "
                          "source of truth; back-derived definitions would be a second source and are not allowed)"},
          "schema_isolation": "v1/v2 validators mutually exclusive, measured (v1 rejects the v2 file)",
          "note": "all BC-v2 class-12 samples are real executed beats in the autonomy-on environment (no counterfactual labels;"
                  " the assertion that removes overridden beats entirely is unchanged and lives on the bc_worker collection side)"}
    log(ev)
    events.append(ev)


def refs_stage(events):
    """G0-6 in-case two refs runs (REF_BITEQ): R1 ≡113.0 / R2 ≡140.9 full table, per seed
    per field; passing is a FREEZE_SHA prerequisite -- the frozen, waiting-for-launch state carries no unproven drift risk; a mismatch goes to
    P-refdiv, no downgrade. refs are reference re-checks, not exams of this case's candidates, so running them before the approved launch is legitimate
    (recorded as is under launch clause 2(g))."""
    assert_stage_prereqs(events, "REF_BITEQ")
    refs = {}
    for ref_key, manager in (("launch", None), ("science", str(M29_NPZ))):
        tag = REF_TAGS[ref_key]
        d = exam_case(events, str(KING_NPZ), tag, POOL_PROBE,
                      manager_npz=manager,
                      extra={"note": "G0-6 REF_BITEQ reference re-check run (freeze prerequisite,"
                                     " before FREEZE_SHA and the approved launch; section 9 of the design memo (not published))"})
        diffs = biteq_diffs(d, ref_key)
        if diffs:
            p_refdiv(events, ref_key, diffs)
        if not any(e.get("event") == "REF_BITEQ" and e.get("ref") == ref_key
                   for e in events):
            ev = {"event": "REF_BITEQ", "ref": ref_key, "tag": tag, "rows": 32,
                  "agg_core": "equal", "ret_mean": d["agg"]["ret_mean"]}
            log(ev)
            events.append(ev)
        refs[ref_key] = d
    return refs


# ======================================================================
# S5 FREEZE_SHA / CASE_RUNTIME / NEWLINE_ADOPT
# ======================================================================

def freeze_stage(events):
    assert_stage_prereqs(events, "FREEZE_SHA")
    head = git("rev-parse", "HEAD")
    if not stage_done(events, "FREEZE_SHA"):
        ev = {"event": "FREEZE_SHA", "sha": head,
              "note": "the commit is the notarization (freeze discipline as in R1/R2/B1/G1; the text holds no self-referencing sha);"
                      " freeze ≠ launch"}
        log(ev)
        events.append(ev)
    if not stage_done(events, "CASE_RUNTIME"):
        require(CASE_RT is not None, "CASE_RUNTIME capture missing")
        ev = {"event": "CASE_RUNTIME", "five": CASE_RT,
              "note": "five case-level sha values settled (honest statement on record that the W-E0 bit-level assertion cannot be used:"
                      " the protocol bundle must change because of E1; substitute zero-drift proof = the two REF_BITEQ runs + "
                      "G0-1 dual endpoints, both on record)"}
        log(ev)
        events.append(ev)
    if not stage_done(events, "NEWLINE_ADOPT"):
        ev = {"event": "NEWLINE_ADOPT",
              "case": "v33-content (first case citing the G1 new lines; ledger counterpart of the separately listed 'new-line adoption'"
                      " item of the freeze report, where the G1 D7-6 first-citation clause is fulfilled)",
              "lines": {"paired_diff": G1_PAIRED_DIFF,
                        "paired_diff_raw": 17.318781,
                        "wins": G1_PAIRED_WINS, "died_max": G1_DIED_MAX,
                        "floor": "85/92 keeps the current value (RG1.4)"},
              "labels": {"RG1.2a": "single-vector line (n_vectors=1; engineering immunity, not a distributional proof)",
                         "RG1.3": "K1 single-archive dependency line (on b1-ref8k-launch)"},
              # slot for the two conditions of the upgrade review (AUDIT-content-case-G1-dual-label-review dc5c7c10…; pre-translation digest 8e4be014…)
              "upgrade_review": {
                  "doc_sha256": W_PIN["AUDIT-content-case-G1-dual-label-review"][1],
                  "verdict": "both limbs hold, with conditions",
                  "g1_verify_note_1": G1_VERIFY_NOTE_1,
                  "g1_verify_note_2": G1_VERIFY_NOTE_2,
                  "deleverage_q95": DELEVERAGE_Q95,
                  "deleverage_q95_raw": DELEVERAGE_Q95_RAW,
                  "g1_results_sha256": W_PIN["g1_results.json"][1],
                  "oc_alpha_columns": OC_ALPHA_COLUMNS,
                  "k1_provenance": K1_PROVENANCE,
                  "loo": "drop K1 -> line 6, the counterfactual flip set becomes empty (recal-g1:16)"},
              "alpha_listing": MULTIPLE_COMPARISON,
              "note": "the two label limbs received a separate upgrade-review opinion (open item E fulfilled); its 8+7"
                      " mandatory caveats and the obligation to back-fill first-exam operating characteristics take effect with the verdict"}
        log(ev)
        events.append(ev)


# ======================================================================
# S6 W-LAUNCH launch-order gate
# ======================================================================

def launch_gate(events):
    """After FREEZE_SHA and before the first leg_start of the three legs: the ledger must hold a LAUNCH_ORDER event
    (the operator's transcription of the approved launch instruction with its timestamp). Without it -> halt and wait, exit code 9
    (AWAITING_LAUNCH is not a failure state; no NEEDS_ATTENTION).
    **Panel review passed ≠ launch; no path in this function constitutes a launch authorization.**"""
    if not stage_done(events, "LAUNCH_ORDER"):
        print("AWAITING_LAUNCH: FREEZE_SHA recorded, LAUNCH_ORDER not recorded -- "
              "the driver halts and waits before the first leg_start of the three legs (exit code 9, not a failure state); "
              "the only valid launch order = an explicit approval, transcribed verbatim into the ledger by the operator",
              flush=True)
        sys.exit(9)


# ======================================================================
# S7 three legs in series + offline canary sequence + leg exams
# ======================================================================

def dry_curriculum_table_stage(events):
    """The DRY_CURRICULUM_TABLE event records the full table before the (curriculum) leg launch (E1 item 2)."""
    if stage_done(events, "DRY_CURRICULUM_TABLE"):
        return
    table = dry_curriculum_table()
    ev = {"event": "DRY_CURRICULUM_TABLE", "schedule": MAIN_TABLE,
          "leg_start": DRY_CURRICULUM_LEG_START, "quantum": QUANTUM,
          "n_rollouts": len(table), "legs": list(CURRICULUM_LEGS),
          # rev4 section 12 appendix 2 item 6: full-precision float recorded (json.dumps(float) round-trips losslessly),
          # identical to the source of truth used by verify_curriculum_prefix; the round(p,10) definition is dropped.
          "table": [float(p) for p in table],
          "endpoint_note": "endpoints use interpolation semantics (registered during implementation): item k of a linear segment = "
                           "p0+(p1−p0)·k/(n−1), the last item of quantum 147 reaches exactly 0.5 -- "
                           "strict decrease occupies 299,008 steps and p=0.5 occupies 200,704 steps; "
                           "this registered semantics is the source of truth; recording the full table seals it"}
    log(ev)
    events.append(ev)


def assert_leg_lock_free(leg: str) -> None:
    """Fix B (launch-time audit, major): before launch, probe RUNS/<leg>/.run.lock with a non-blocking flock
    -- if the lock is held, an orphan leg is present (the previous run's train_ppo process is still alive); OperationalFailure
    halts for review and writes no leg_start (uses no budget). Probe reliability follows the train_ppo._RunLock
    semantics (flock is a kernel lock, released automatically on a crash/SIGKILL; a leftover lock file does not hold the lock);
    a lock obtained by the probe is released at once; the lock file is neither written nor deleted."""
    lock_path = RUNS / leg / ".run.lock"
    if not lock_path.exists():
        return
    with open(lock_path, "r", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = f.read().strip() or "unknown"
            raise OperationalFailure(
                f"{leg} pre-launch lock probe: .run.lock is held ({owner}) -- the previous run's "
                "train_ppo process is still present (orphan leg); collect evidence and kill the leg process group, then rerun; "
                "this run writes no leg_start (uses no budget)") from exc
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def leg_train_log_name(leg: str, attempt: int) -> str:
    """Fix D (launch-time audit, minor): the leg training log file name carries the run number (train-<leg>-a<N>.log,
    N = sequence number of this run's leg_start) -- run() truncates with "w", so a fixed name would destroy the first run's
    crash log evidence on the second run."""
    return f"train-{leg}-a{attempt}.log"


def leg_stage(events, leg: str) -> str:
    """Single-leg launch (P3 ledger budget 2 per leg; nt gate 3,997,696; always resumes from the king zip;
    no seed change on relaunch) + end-of-leg check dry_curriculum.jsonl ≡ full registry table (curriculum legs)."""
    assert_stage_prereqs(events, "LEGS")
    if leg == "v33-full":
        # Fix 1c (rev4 section 12 appendix 2 item 4): before launch, demos bytes ≡ N12_GATE settled value
        assert_lfull_demos_chain(events)
    model_path = RUNS / leg / "model_final.zip"
    out = RUNS / leg / "policy.npz"
    if not (model_path.exists() and zip_steps(model_path) == NT_TARGET):
        require(leg_starts(events, leg) < 2,
                f"{leg} leg launch budget exhausted (ledger budget 2 per leg, 6 for the three legs) -- "
                "OPERATIONAL_FAILURE, whole case halts; never hand-edit the ledger to keep it alive")
        # Fix B: lock probe before every launch (an orphan leg halts the case, before leg_start is recorded)
        assert_leg_lock_free(leg)
        # Fix D: the run sequence number is computed before leg_start is recorded (N = recorded leg_starts + 1)
        attempt = leg_starts(events, leg) + 1
        ev = {"event": "leg_start", "leg": leg, "seed": SEED,
              "extras": list(dict(LEGS)[leg]),
              "recipe": "shared command form = v32-sov verbatim + B1 instrument knobs + the rev4 correction"
                        " addition of the E5 (1)(2) record-only knobs (D3 / section 12 appendix 2 item 1);"
                        " per-leg additions = LEGS table verbatim",
              "launcher": "train/launch_case.sh (E8; the driver itself is launched through it,"
                          " legs are driver subprocesses; PID bookkeeping as described in the launcher header)",
              "resume_from": "king zip (P3 relaunches always resume from the king zip; seed always 304000)"}
        log(ev)
        events.append(ev)
        t0 = time.time()
        rc = run(leg_cmd(leg), leg_train_log_name(leg, attempt),
                 timeout=LEG_TIMEOUT)
        nt = zip_steps(model_path)
        log({"event": "leg_done", "leg": leg, "rc": rc, "nt_zip": nt,
             "dt_min": round((time.time() - t0) / 60, 1)})
        require(rc == 0 and nt == NT_TARGET,
                f"{leg} fell short (rc={rc}, nt={nt}, target={NT_TARGET}) -- "
                "operational failure, halting for review; a restart may use the second run (P3)")
    else:
        log({"event": "leg_skip_complete", "leg": leg})
    # end-of-leg check: curriculum leg dry_curriculum.jsonl ≡ full DRY_CURRICULUM_TABLE
    # (this closes the gap of "the table was recorded but a different one ran"; mismatch -> CASE_HALT_G0 7)
    if leg in CURRICULUM_LEGS:
        diffs = verify_curriculum_prefix(RUNS / leg, dry_curriculum_table(),
                                         LEG_STEPS // QUANTUM)
        if diffs:
            log({"event": "CASE_HALT_G0", "via": "DRY_CURRICULUM_VERIFY",
                 "leg": leg, "diffs": diffs[:10]})
            attention(f"{leg} end-of-leg dry_curriculum.jsonl does not match the full registry table")
            raise SystemExit(7)
        if not any(e.get("event") == "DRY_CURRICULUM_VERIFIED"
                   and e.get("leg") == leg for e in events):
            ev = {"event": "DRY_CURRICULUM_VERIFIED", "leg": leg,
                  "n_rollouts": LEG_STEPS // QUANTUM, "verdict": "≡ full registry table"}
            log(ev)
            events.append(ev)
    # Fix 5b (rev4 section 12 appendix 2 item 1/D8): end-of-leg DRYWIN_METRICS event transcription -- the E5 (1)(2)
    # knobs are on with the shared command form; the drywin_metrics.jsonl file sha256 + line count + final
    # section aggregate payload are recorded; distill_ce_probe.jsonl of the same type goes into this event (one event, two files,
    # following the DEMO_LEDGER v1/v2 two-payload form); where the DRYWIN_METRICS entry of the D8 vocabulary is written.
    if not any(e.get("event") == "DRYWIN_METRICS" and e.get("leg") == leg
               for e in events):
        ev = {"event": "DRYWIN_METRICS", "leg": leg,
              "every": {"drywin_metrics": DRYWIN_METRICS_EVERY,
                        "distill_ce_probe": DISTILL_CE_PROBE_EVERY},
              "drywin_metrics": instrument_jsonl_digest(
                  RUNS / leg / "drywin_metrics.jsonl"),
              "distill_ce_probe": instrument_jsonl_digest(
                  RUNS / leg / "distill_ce_probe.jsonl"),
              "note": "record only (E5 (1)(2); the RC.14 baseline is built first in this case, no prior band;"
                      " the dry/fresh distill_ce split is leg-side material for the obligation to supply course 2 calibration data (design decision 12);"
                      " record of the knobs added by the rev4 section 12 appendix 2 item 1 correction)"}
        log(ev)
        events.append(ev)
    if not out.exists():
        require(run([PY, "train/export_worker_npz.py", str(model_path),
                     str(out)], f"export-{leg}.log", 600) == 0 and out.exists(),
                f"{leg} npz export failed (the make-up export channel uses no budget; after restart the skip branch runs)")
        log({"event": "npz_exported", "leg": leg, "sha256": sha256(out)})
    return str(out)


def canary_eval_done(events, tag) -> dict | None:
    hits = [e for e in events
            if e.get("event") == "CANARY_EVAL" and e.get("tag") == tag]
    return hits[-1] if hits else None


def canary_exam(events, worker_npz, tag, manager_npz, l1_fired: bool):
    """P-canary (B1 verbatim): does not use the evaluation run budget; one archive per ckpt x manager is final;
    retries only for operational failures without an archive; make-up deadline = before that leg's s16 FIRING_START (asserted in the
    driver); a failure is recorded only and the leg continues."""
    out = EVAL / f"{tag}.json"
    prior = canary_eval_done(events, tag)
    if prior is not None:
        require(out.exists(), f"canary {tag} recorded in the ledger but the archive is missing; halting for review")
        d = validate_adopted(tag, worker_npz, POOL_PROBE, manager_npz)
        require(d["agg"]["_sha"] == prior["sha"],
                f"canary {tag} archive and ledger sha mismatch")
        return d, False
    if out.exists():
        # Fix E (launch-time audit, minor): recovery path for adopting a leftover archive -- when validate_adopted raises
        # EvalContractError it must not fall through to P1; handled per P-canary "record only, the leg continues":
        # the leftover archive is sealed as .void + an event is recorded + this function continues its normal order (re-exam within the deadline).
        try:
            d = validate_adopted(tag, worker_npz, POOL_PROBE, manager_npz)
            return d, True
        except EvalContractError as exc:
            sealed = out.with_suffix(f".{time.time_ns()}.void")
            out.rename(sealed)
            log({"event": "OPERATIONAL-canary", "tag": tag,
                 "void": sealed.name,
                 "why": f"leftover archive adoption check failed ({exc}); per P-canary record only, the leg continues:"
                        " leftover archive already .void; re-exam if within the deadline, otherwise undecidable"})
    if l1_fired:
        log({"event": "OPERATIONAL-canary", "tag": tag,
             "why": "make-up deadline passed (before that leg's s16 FIRING_START, compliance M-4);"
                    " no make-up -- the R line for this point reads 'undecidable, recorded as is'"})
        return None, False
    retries = 0
    while retries < 2:
        d = exam(worker_npz, tag, POOL_PROBE, manager_npz)
        if d is not None:
            return d, True
        retries += 1
        log({"event": "OPERATIONAL-canary", "tag": tag, "retry": retries,
             "why": "operational failure without an archive; retry (counted in the ledger)"})
    log({"event": "OPERATIONAL-canary", "tag": tag,
         "why": "still failing after retry -- record only, the leg continues; the R line for this point reads 'undecidable'"})
    return None, False


def canary_stage_leg(events, leg: str) -> dict:
    """Per-leg offline canary sequence (copied from B1; four mid-leg points x {H, M29}; record only;
    L-full adds the A12_CANARY mid-leg instrument event)."""
    short = LEG_SHORT[leg]
    l1_fired = s16_fired(events, leg)
    canary_docs = {}
    for k, step in enumerate(CANARY_STEPS, 1):
        ckpt = RUNS / leg / "ckpt" / f"model_{step}_steps.zip"
        npz = V33 / "canary" / leg / f"policy_{step}.npz"
        if not npz.exists():
            require(ckpt.is_file(), f"canary ckpt missing: {ckpt}")
            npz.parent.mkdir(parents=True, exist_ok=True)
            rc = run([PY, "train/export_worker_npz.py", str(ckpt), str(npz)],
                     f"canary-export-{leg}-{step}.log", 600)
            if rc != 0 or not npz.exists():
                log({"event": "OPERATIONAL-canary", "leg": leg,
                     "ckpt_step": step,
                     "why": f"npz export failed (rc={rc}); every reading at this point is 'undecidable'"})
                continue
        for mtag, manager in (("h", None), ("m29", str(M29_NPZ))):
            tag = f"v33-{short}-canary{k}-{mtag}"
            d, fresh = canary_exam(events, str(npz), tag, manager, l1_fired)
            if d is None:
                continue
            canary_docs[tag] = d
            if fresh or canary_eval_done(events, tag) is None:
                rows = sorted(d["rows"], key=lambda r: r["seed"])
                ev = {"event": "CANARY_EVAL", "tag": tag, "leg": leg, "k": k,
                      "ckpt_step": step,
                      "manager": "H" if mtag == "h" else "M29",
                      "sha": d["agg"]["_sha"],
                      "mean": d["agg"]["ret_mean"], "died": d["agg"]["died"],
                      "seeds": [r["seed"] for r in rows],
                      "ret": [r["ret"] for r in rows],
                      "died_vec": [int(r["died"]) for r in rows],
                      "depth": [r["depth"] for r in rows],
                      "d_windows": [d_windows(r) for r in rows],
                      "farm_tau_mean": [r["farm_tau_mean"] for r in rows],
                      "a12_per_ep": a12_per_ep(d["agg"]),
                      "discipline": "record only; never a basis for any in-leg intervention; the only end checkpoint"
                                    " = nt 3,997,696; every canary reading is"
                                    " descriptive; the decision surface = the final exam archives and the release line (panel M3)"}
                log(ev)
                events.append({"event": "CANARY_EVAL", "tag": tag,
                               "sha": d["agg"]["_sha"]})
            if leg == "v33-full" and not any(
                    e.get("event") == "A12_CANARY" and e.get("tag") == tag
                    for e in events):
                # E5 (3) mid-leg instrument (RC.11 re-extinction dynamics sequence; record only). Implementation discretion
                # reported: evaluation archives have no per-game action histogram, so the E5 (3) per-game definition (episodes_with_
                # a12/a12_max) is not available and record_a12_canary is not called (no fake values).
                hist = d["agg"].get("worker_action_hist", {}) or {}
                ev = {"event": "A12_CANARY", "tag": tag, "leg": leg,
                      "checkpoint_step": step,
                      "manager": "H" if mtag == "h" else "M29",
                      "a12_total": int(hist.get("12", 0)),
                      "episodes": int(d["agg"].get("n", 32)),
                      "a12_per_episode": a12_per_ep(d["agg"]),
                      "per_episode_note": "evaluation archives have no per-game action histogram -- the per-game breakdown"
                                          " (episodes_with_a12/a12_max) is not available;"
                                          " reported as is (no fake values; the E5 (3) per-game definition waits for"
                                          " upstream data)"}
                log(ev)
                events.append(ev)
    return canary_docs


def leg_exams(events, leg: str, npz_path: str) -> dict:
    """Three leg exams: s16 (H only, compliant under E1 exemption 1, half pool) -> full32 (H) -> full32-m29."""
    short = LEG_SHORT[leg]
    docs = {}
    docs["s16"] = exam_case(events, npz_path, f"v33-{short}-s16", POOL_S16,
                            extra={"note": "quick screen (B1 E1 exemption 1, half pool, H only, "
                                           "compliant)"})
    for suffix, manager, bundle in (
            ("full32", None, f"v33-{short}-full32-m29"),
            ("full32-m29", str(M29_NPZ), f"v33-{short}-full32")):
        tag = f"v33-{short}-{suffix}"
        d = exam_case(events, npz_path, tag,
                      POOL_PROBE, manager_npz=manager, bundled_with=bundle)
        a = d["agg"]
        if not any(e.get("event") == "exam_detail" and e.get("tag") == tag
                   for e in events):
            ev = {"event": "exam_detail", "tag": tag, "leg": leg,
                  "mean": a["ret_mean"], "died": a["died"],
                  "a12_per_ep": a12_per_ep(a), "a13_per_ep": a13_per_ep(a),
                  "rwin_per_ep": round(sum(str(r["mode_seq"]).count("R")
                                           for r in d["rows"]) / 32, 2),
                  "depth2": depth2_count(d["rows"]),
                  "dive": round(dive_per_ep(d["rows"]), 2),
                  "override": a["override_rate"], "cap": a["cap_rate"],
                  "sha": a["_sha"]}
            log(ev)
            events.append(ev)
        docs[suffix] = d
    return docs


# ======================================================================
# S8 criteria (course 5 main verdict / 4b main verdict / eligibility / winner / release) -- pure functions for unit tests
# ======================================================================

def qual_of(doc) -> dict:
    """Per-leg eligibility written out (D4-1 / compliance m-4): died<=8 (same-value replacement of G1 L2-0 (a)) ∧
    sentinel limbs untouched (level change/cap/override, G1 L1 untouched list) ∧ not voided."""
    a = doc["agg"]
    dive = dive_per_ep(doc["rows"])
    void = (a["override_rate"] >= R4["override_void"]
            or (dive > 1 and a["died"] > 6))
    sentinel_ok = (a["farm_descend_rate"] <= R4["descend"]
                   and a["cap_rate"] < R4["cap"]
                   and a["override_rate"] < R4["override_sentinel"])
    out = {"died_ok": a["died"] <= G1_DIED_MAX,
           "sentinel_zero_touch": sentinel_ok, "void": void}
    out["qual_ok"] = out["died_ok"] and sentinel_ok and not void
    if (not out["qual_ok"] and out["died_ok"] and not void
            and a["cap_rate"] >= R4["cap"]
            and a["farm_descend_rate"] <= R4["descend"]
            and a["override_rate"] < R4["override_sentinel"]):
        out["cap_drift_note"] = ("disqualified only by cap_rate -- note on a candidate gate-semantics drift + "
                                 "τ̄ difference vs ref and per-window distribution (R5) go with the verdict")
    return out


def decide_winner(means: dict, died: dict, quals: dict) -> dict:
    """Winner clause (D4-5): among eligible legs the highest full32 (H) mean; difference <=0.05 -> fewer deaths;
    still tied -> L-full (the prescribed leg first); if the highest mean is blocked by eligibility -> next in line (substitution
    recorded)."""
    pool = [n for n in means if quals[n]["qual_ok"]]
    if not pool:
        return {"winner": None, "pool": [], "substituted": False}
    prelim = max(means, key=lambda n: means[n])
    band = [n for n in pool if max(means[n2] for n2 in pool) - means[n] <= 0.05]
    if len(band) > 1:
        dmin = min(died[n] for n in band)
        band = [n for n in band if died[n] == dmin]
        winner = "v33-full" if "v33-full" in band else band[0]
    else:
        winner = band[0]
    return {"winner": winner, "pool": pool, "prelim": prelim,
            "substituted": winner != prelim and prelim not in pool}


def course5_ruling(anchor_hits: int, cur_hits: int) -> dict:
    """Course 5 main verdict (D4-2 re-established): the only decision quantity = F-lock v1 hit count; decision leg = L-cur x H;
    anchor = measured L-base in the case (prior point 5 from P8, listed alongside, not the anchor). All decision lines are pre-registered."""
    out = {"anchor_hits_l_base": anchor_hits, "cur_hits": cur_hits,
           "anchor_prior_note": f"prior point {COURSE5_ANCHOR_PRIOR} from P8"
                                " (two variable levels away; demoted to a listed prior, not the anchor)"}
    if anchor_hits <= 3:
        out["verdict"] = ("floor effect, undecidable -- the course 5 main verdict gives no pass/fail; the dry-window behaviour"
                          " instrument (E5 (2)), D-window retention and τ distribution are recorded as is for a follow-up case")
        out["branch"] = "floor"
        return out
    if cur_hits <= anchor_hits - 2:
        out["verdict"] = f"direction success (L-cur {cur_hits} ≤ anchor−2 = {anchor_hits - 2})"
        out["branch"] = "success"
    elif cur_hits == anchor_hits - 1:
        out["verdict"] = "decrease within the noise band (descriptive, = anchor−1)"
        out["branch"] = "noise"
    else:
        out["verdict"] = f"no improvement (L-cur {cur_hits} ≥ anchor {anchor_hits})"
        out["branch"] = "no_improvement"
    if anchor_hits == 4:
        out["note_low_anchor"] = "low baseline reduces discriminating power (anchor=4, success line ≤2)"
    return out


def a12_tier_ruling(a12_full: float, died_full: int, paired_mean: float,
                    died_cur: int, a12_cur: float) -> dict:
    """4b main-verdict tier table (D4-3 re-registered; decision order tier 1 -> tier 2 -> tier 3, exhaustiveness guaranteed by the complement tier;
    ctrl = L-cur x M29; constants re-registered for this case's context and submitted for review (open item B))."""
    out = {"a12_full": a12_full, "died_full": died_full,
           "paired_mean_vs_cur_m29": round(paired_mean, 2),
           "died_cur_m29": died_cur, "a12_cur": a12_cur,
           "main_line": a12_full >= A12_USE_LINE,
           "binom_note": "the died tier is a directional reading, not evidence of significance (binomial note attached)"}
    if (a12_full >= A12_USE_LINE and died_full <= died_cur
            and paired_mean >= SURV_MEAN_BAND):
        out["tier"] = 1
        out["verdict"] = (f"tier 1 (injection converted into survival): a12/game {a12_full}≥{A12_USE_LINE}"
                          f" ∧ died {died_full}≤ctrl {died_cur} ∧ paired mean difference "
                          f"{paired_mean:.2f}≥{SURV_MEAN_BAND}")
    elif (a12_full < A12_USE_LINE and paired_mean >= SURV_MEAN_BAND
          and died_full <= died_cur + 1):
        out["tier"] = 2
        out["verdict"] = (f"tier 2 (examples given but not used): a12/game {a12_full}<{A12_USE_LINE}"
                          f" ∧ mean difference {paired_mean:.2f}≥{SURV_MEAN_BAND} ∧ died "
                          f"{died_full}≤ctrl+1 (the v32 F1 explicit expectation and the primary risk case of R11/residual 4"
                          ")")
        if 0 < a12_full < A12_USE_LINE and died_full <= died_cur:
            out["low_use_note"] = "low usage and improved survival appear together; attribution unproven (v32 verbatim)"
    else:
        out["tier"] = 3
        if a12_full >= A12_USE_LINE:
            sub = "(a) used but not converted (a12≥0.1 but the survival/mean limbs are not met)"
        else:
            sub = "(b) not used and worse"
        out["verdict"] = f"tier 3 (complement tier, guarantees exhaustiveness): {sub}"
        out["tier3_subcase"] = sub
    if a12_cur >= A12_USE_LINE:
        out["control_crossline"] = ("control crosses the line (pre-registered, statistics M-1): L-cur "
                                    f"a12/game {a12_cur}≥{A12_USE_LINE} -> the verdict must carry"
                                    " 'curriculum-only revival candidate (compound survival path, same family as v32 residual 2"
                                    "), 4b package attribution downgraded to \"gain unproven\"'; the tier"
                                    " is still decided, but the attribution sentence is reworded accordingly")
    if out["tier"] in (2, 3):
        out["circle12_exit"] = ("exit syntax of the design decision 12 ladder clause (D4-4): referred to a new design review"
                                " (retry with a different mechanism, or honestly declare extinction and release); no prejudgment,"
                                " no automatic transfer to another case")
    out["conjunction_note"] = "4b package = the conjunction λ_bc>0 + v2 files; this caveat is mandatory (with the R7 teacher-generation mixing note)"
    return out


def death_seed_sets(full_m29_rows: list, ctrl_m29_rows: list) -> dict:
    """Fix G (launch-time audit, minor, context-correction note): for the 4b verdict "difference in the set of dead seeds"
    the control now uses the D4-3 in-case context ctrl = L-cur×M29 (full32-m29) rows -- replacing the original
    v32-ref-science (a cross-case reference, not this case's ctrl); rescued/new_deaths semantics follow
    (rescued/newly dead relative to ctrl)."""
    ctrl_dead = {r["seed"] for r in ctrl_m29_rows if r["died"]}
    full_dead = {r["seed"] for r in full_m29_rows if r["died"]}
    return {"ctrl": "L-cur×M29 (full32-m29; D4-3 in-case context correction,"
                    " replaces v32-ref-science)",
            "rescued": sorted(ctrl_dead - full_dead),
            "new_deaths": sorted(full_dead - ctrl_dead)}


# ======================================================================
# statistical discipline tools (inherited from B1; for the R lines)
# ======================================================================

def median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    require(n > 0, "median input is empty")
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def binom_tail_ge(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def sign_test(diffs) -> dict:
    neg = sum(1 for d in diffs if d < 0)
    pos = sum(1 for d in diffs if d > 0)
    ties = len(diffs) - neg - pos
    n = neg + pos
    p = binom_tail_ge(max(neg, pos), n) if n else 1.0
    return {"neg": neg, "pos": pos, "ties": ties, "p_one_sided": round(p, 4)}


def deleveraged_mean(diff_by_seed: dict) -> dict:
    require(len(diff_by_seed) >= 2, "the deleveraged mean needs >=2 seeds")
    lever = max(diff_by_seed, key=lambda s: abs(diff_by_seed[s]))
    rest = [v for s, v in diff_by_seed.items() if s != lever]
    return {"dropped_seed": lever,
            "dropped_value": round(diff_by_seed[lever], 2),
            "mean": round(sum(rest) / len(rest), 2)}


def loo_7017(diff_by_seed: dict) -> dict:
    """7017 leave-one-out reading (mandatory, listed alongside; standing control for the deep-water seed, −150-level leverage)."""
    rest = [v for s, v in diff_by_seed.items() if s != DEEPWATER_SEED]
    return {"dropped_seed": DEEPWATER_SEED,
            "dropped_value": round(diff_by_seed.get(DEEPWATER_SEED, float("nan")), 2)
            if DEEPWATER_SEED in diff_by_seed else None,
            "mean": round(sum(rest) / len(rest), 2) if rest else None}


def band_judge(x: float, lo: float, hi: float, integer: bool = False,
               upper_open: bool = False) -> dict:
    """Fix H (launch-time audit, minor): with upper_open=True the band is half-open [lo, hi) --
    only the RC.10 damage-branch band [0.0, 0.5) uses it (retention exactly 0.5 must fall into the healthy branch);
    the default closed-interval semantics stay unchanged globally (no global semantic change)."""
    in_band = (lo <= x < hi) if upper_open else (lo <= x <= hi)
    if integer:
        borderline = min(abs(x - lo), abs(x - hi)) <= 1
    else:
        borderline = min(abs(x - lo), abs(x - hi)) <= 0.05 * (hi - lo)
    out = {"x": round(float(x), 4), "band": [lo, hi], "in_band": in_band}
    if upper_open:
        out["upper_open"] = True
    if borderline:
        out["borderline_note"] = "borderline + line not recalibrated (mandatory note per the borderline clause)"
    return out


def paired_diff_stats(leg_rows: dict, ref_rows: dict) -> dict:
    """Paired mean-difference statistics block (D4 statistical discipline: the mean always comes with median + sign test + deleveraged mean + 7017 leave-one-out).
    Fix C (launch-time audit, major): adds mean_raw, the full-precision mean -- the release conjunction's mean-difference limb and
    the RC.9 MS branch trigger always decide on raw (the v32 precedent compares raw; the value rounded to 2dp is only for
    display in the ledger, since rounding within a 0.005 window could silently flip a verdict)."""
    diffs = {s: leg_rows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)}
    vals = list(diffs.values())
    mean_raw = sum(vals) / len(vals)
    return {"mean": round(mean_raw, 2),
            "mean_raw": mean_raw,
            "median": round(median(vals), 2),
            "sign": sign_test(vals),
            "deleveraged": deleveraged_mean(diffs),
            "loo_7017": loo_7017(diffs),
            "wins": sum(1 for v in vals if v > 0),
            "by_seed": {str(s): round(v, 2) for s, v in diffs.items()}}


def ms_vectors(leg_h: dict, leg_m29: dict, ref_h: dict, ref_m29: dict) -> dict:
    seeds = sorted(leg_h)
    require(set(seeds) == set(leg_m29) == set(ref_h) == set(ref_m29),
            "MS: the four archives have different seed sets")
    signed = {}
    for s in seeds:
        dh = leg_h[s]["ret"] - ref_h[s]["ret"]
        dm = leg_m29[s]["ret"] - ref_m29[s]["ret"]
        signed[s] = round(dh - dm, 2)
    ms = {s: abs(v) for s, v in signed.items()}
    over = sorted(s for s, v in ms.items() if v > MS_FLAG_LINE)
    return {"signed_dh_minus_dm": signed,
            "ms_max": round(max(ms.values()), 2),
            "ms_median": round(median(list(ms.values())), 2),
            "over_line_seeds": over, "n_over_line": len(over),
            "flag_line": MS_FLAG_LINE}


# ======================================================================
# replay channel (supplies F-lock τ; inherited from B1 E4)
# ======================================================================

def run_obsdrift(worker_npz, archive: pathlib.Path, out: pathlib.Path,
                 manager=None) -> dict:
    if out.exists():
        report = strict_json_loads(out.read_bytes())
        if (report.get("archive_sha256") == sha256(archive)
                and report.get("fidelity_ok")):
            return report
        out.rename(out.with_suffix(f".{time.time_ns()}.stale"))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "train/probe_b1_obsdrift.py", "--worker", str(worker_npz),
           "--archive", str(archive), "--out", str(out)]
    if manager:
        cmd += ["--manager", str(manager)]
    rc = run(cmd, f"obsdrift-{out.stem}.{time.time_ns()}.log", 3_600)
    require(rc == 0 and out.exists(),
            f"replay failed / fidelity mismatch: {archive.name} (fidelity anchor clause; halting for review)")
    return strict_json_loads(out.read_bytes())


def flock_hits(leg_full32_doc, ref_launch_rows: dict, leg_npz: str,
               leg: str) -> dict:
    """F-lock composite signature v1 hits (firewall: v1 decides with its current construction constants; v2 goes back as an instrument
    and never enters this case's decision lines; the uncalibrated-specificity and horizon caveats are mandatory)."""
    from probe_composite_signature import composite_signature
    replay = run_obsdrift(leg_npz, EVAL / f"v33-{LEG_SHORT[leg]}-full32.json",
                          V33 / "replay" / f"v33-{LEG_SHORT[leg]}-full32.json")
    tau = {int(s): v.get("farm_tau_median")
           for s, v in replay.get("per_seed", {}).items()}
    sig = composite_signature(ref_launch_rows,
                              by_seed(leg_full32_doc["rows"], 7000, 7031), tau)
    return {"n_hits": sig["n_hits"], "hits": sig["hits"],
            "hit_7017": DEEPWATER_SEED in sig["hits"],
            "caveats": ["signature specificity uncalibrated (mandatory with the verdict)",
                        "horizon caveat (max-steps 3000, frozen)"]}


# ======================================================================
# S9 R-line scorecard + VERDICT_PATH
# ======================================================================

def scorecard_stage(events, refs, leg_docs, canary_docs, verdicts,
                    hold_docs):
    if stage_done(events, "VERDICT_PATH"):
        return
    ref_launch = by_seed(refs["launch"]["rows"], 7000, 7031)
    ref_science = by_seed(refs["science"]["rows"], 7000, 7031)
    rows = {leg: {"h": by_seed(leg_docs[leg]["full32"]["rows"], 7000, 7031),
                  "m29": by_seed(leg_docs[leg]["full32-m29"]["rows"],
                                 7000, 7031)}
            for leg in LEG_NAMES}
    card = {}
    band_outcomes = []

    def add(name, judged):
        card[name] = judged
        if isinstance(judged, dict) and "in_band" in judged:
            band_outcomes.append((name, judged["in_band"]))

    card["RC1_ref_biteq"] = "two REF_BITEQ runs recorded (≡113.0/≡140.9 full table; freeze prerequisite fulfilled)"
    # RC.2 L-base full32 (H) paired mean difference
    st2 = paired_diff_stats(rows["v33-base"]["h"], ref_launch)
    rc2 = band_judge(st2["mean"], *RC2_BAND["band"])
    rc2.update({"point": RC2_BAND["point"], **{k: st2[k] for k in
                ("median", "sign", "deleveraged", "loo_7017")}})
    add("RC2_base_paired_mean", rc2)
    # Fix C: the two MS branches are triggered mechanically on the raw mean (a rounded reading near the line could silently flip the band choice)
    branch = "healthy" if st2["mean_raw"] >= RC9_BRANCH_LINE else "damaged"
    # RC.3 course 5 effect (L-cur − L-base, same-seed pairing)
    d3 = {s: rows["v33-cur"]["h"][s]["ret"] - rows["v33-base"]["h"][s]["ret"]
          for s in sorted(ref_launch)}
    rc3 = band_judge(sum(d3.values()) / 32, *RC3_BAND["band"])
    rc3.update({"point": RC3_BAND["point"],
                "note": "below the band -> course 5 harm candidate; the verdict is worded per cell, never lumped as 'pass/fail'"})
    add("RC3_course5_effect", rc3)
    # RC.4 4b effect (L-full − L-cur)
    d4 = {s: rows["v33-full"]["h"][s]["ret"] - rows["v33-cur"]["h"][s]["ret"]
          for s in sorted(ref_launch)}
    rc4 = band_judge(sum(d4.values()) / 32, *RC4_BAND["band"])
    rc4.update({"point": RC4_BAND["point"],
                "note": "RC.3 and RC.4 share L-cur, so their errors are negatively correlated by construction (their sum ≡ "
                        "L-full−L-base); they must not be stacked as independent evidence (statistics minor-4);"
                        " the conjunction caveat is mandatory"})
    add("RC4_yi_effect", rc4)
    card["RC5_course5_main"] = verdicts["course5"]
    card["RC6_a12_main"] = {**verdicts["a12"],
                            "point": RC6_BAND["point"],
                            "band": list(RC6_BAND["band"]),
                            "prior_note": "v32 R32.3 pre-registered point 1.5, measured 0.0"
                                          " (F1: autonomy never exercised) -- this case is the first non-zero"
                                          " expectation after injection, with no measured precedent (rev3 correction)"}
    card["RC7_launch_limbs"] = verdicts["launch_limbs"]
    # RC.8 died per leg
    for leg in LEG_NAMES:
        rc8h = band_judge(leg_docs[leg]["full32"]["agg"]["died"],
                          *RC8_BAND_H["band"], integer=True)
        rc8h.update({"point": RC8_BAND_H["point"],
                     "died_vec": [int(r["died"]) for r in sorted(
                         leg_docs[leg]["full32"]["rows"],
                         key=lambda r: r["seed"])]})
        add(f"RC8_died_h_{LEG_SHORT[leg]}", rc8h)
        rc8m = band_judge(leg_docs[leg]["full32-m29"]["agg"]["died"],
                          *RC8_BAND_M29["band"], integer=True)
        rc8m["point"] = RC8_BAND_M29["point"]
        add(f"RC8_died_m29_{LEG_SHORT[leg]}", rc8m)
    # RC.9 dual-manager reading difference + two MS branches (triggered mechanically by the measured RC.2 value, no choosing a band afterwards)
    for leg in LEG_NAMES:
        diffs9 = [rows[leg]["m29"][s]["ret"] - rows[leg]["h"][s]["ret"]
                  for s in sorted(ref_launch)]
        rc9 = band_judge(sum(diffs9) / 32, *RC9_BAND["band"])
        rc9["point"] = RC9_BAND["point"]
        add(f"RC9_dual_manager_{LEG_SHORT[leg]}", rc9)
        ms = ms_vectors(rows[leg]["h"], rows[leg]["m29"],
                        ref_launch, ref_science)
        msb = RC9_MS[branch]
        rc9max = band_judge(ms["ms_max"], *msb["max"][1])
        rc9max.update({"point": msb["max"][0], "branch": branch,
                       "ms_limitation": "MS≈0 does not show the worker is undamaged (a detector of manager-sensitive damage,"
                                        " not of all damage; mandatory caveat with the verdict)"})
        add(f"RC9_MS_max_{LEG_SHORT[leg]}_{branch}", rc9max)
        rc9over = band_judge(ms["n_over_line"], *msb["over"][1], integer=True)
        rc9over["point"] = msb["over"][0]
        add(f"RC9_MS_overline_{LEG_SHORT[leg]}_{branch}", rc9over)
    # RC.10 canary D-window retention (canary4 x H, C set depth2; branches as in RC.9)
    cs = [e for e in events if e.get("event") == "CANARY_SET"][0]
    d_c, n_d = cs["depth2_seeds"], cs["n_D"]
    for leg in LEG_NAMES:
        c4 = canary_docs.get(leg, {}).get(f"v33-{LEG_SHORT[leg]}-canary4-h")
        if c4 is None:
            card[f"RC10_retention_{LEG_SHORT[leg]}"] = \
                "undecidable (canary data missing, P-canary); recorded as is"
            continue
        c4_rows = by_seed(c4["rows"], 7000, 7031)
        kept = [s for s in d_c if d_windows(c4_rows[s]) >= 1]
        # Fix H: the damage band [0.0, 0.5) is half-open (retention exactly 0.5 falls into the healthy branch); only this band
        rc10 = band_judge(len(kept) / n_d, *RC10_BAND[branch],
                          upper_open=(branch == "damaged"))
        rc10.update({"kept_seeds": sorted(kept), "n_D": n_d, "branch": branch,
                     "formula": "retention := |{s in D_C: canary4 x H has >=1 D window for the seed}|"
                                " / n_D (RB.8 formula verbatim; always descriptive)"})
        add(f"RC10_retention_{LEG_SHORT[leg]}", rc10)
    # RC.11 canary a12/game sequence (L-full; no prior band, recorded point by point)
    a12_seq = [{"tag": e["tag"], "step": e["checkpoint_step"],
                "manager": e["manager"], "a12_per_episode": e["a12_per_episode"]}
               for e in events if e.get("event") == "A12_CANARY"]
    card["RC11_a12_canary_seq"] = {
        "seq": a12_seq,
        "note": "expected shape = non-zero after injection, risk of decay in the middle/late part (mirror of R11 v14/v15);"
                " no prior band, recorded point by point as is; canaries only observe and never rescue, the final-exam criteria are the backstop"}
    # RC.12 dry-anchor increment (end of leg; zero point 0.7515; never had normative meaning)
    for leg in LEG_NAMES:
        sent = RUNS / leg / "sentinel.jsonl"
        dry_lines = ([json.loads(x) for x in sent.read_text().splitlines()
                      if '"dry-anchor"' in x] if sent.is_file() else [])
        finals = [r for r in dry_lines if r.get("final")] or dry_lines[-1:]
        if not finals:
            card[f"RC12_dry_anchor_{LEG_SHORT[leg]}"] = "undecidable (sentinel data missing)"
            continue
        inc = (finals[-1]["mismatch"] - DRY_REF_THRONE) * 100
        rc12 = band_judge(inc, *RC12_BAND["band"])
        rc12.update({"point": RC12_BAND["point"],
                     "refs": {"throne": DRY_REF_THRONE,
                              "lineage": DRY_REF_LINEAGE},
                     "note": "descriptive, not a criterion limb (the G1 D6 direction reversal is on record; the +8.35pp "
                             "healthy-leg counterexample note is mandatory); the probe set uses the same BC-v1 demos "
                             "bytes for all three legs (E6 guarantee chain)"})
        add(f"RC12_dry_anchor_{LEG_SHORT[leg]}", rc12)
    # RC.13 BC-v2 receipt (freeze prerequisite gate already passed)
    n12_ev = [e for e in events
              if e.get("event") == "N12_GATE" and e.get("gate") == "PASS"][-1]
    rc13 = band_judge(n12_ev["n12"], *RC13_N12_BAND["band"], integer=True)
    rc13.update({"point": RC13_N12_BAND["point"],
                 "recall_12": band_judge(n12_ev["recall_12"],
                                         *RC13_RECALL_BAND["band"]),
                 "oc_consumed": n12_ev.get("oc_consumed"),
                 "cluster_note": N12_CLUSTER_NOTE})
    add("RC13_n12", rc13)
    # RC.14 dry-window behaviour instrument (baseline built first in this case, no prior band; starting point for closing audit gap i)
    rc14 = {}
    for leg in LEG_NAMES:
        rc14[leg] = {
            "drywin_metrics": (str(RUNS / leg / "drywin_metrics.jsonl")
                               if (RUNS / leg / "drywin_metrics.jsonl").is_file()
                               else "not available (the E5 (2) knob is on with the rev4 D3 shared command form"
                                    " but produced no output -- missing operational artifact, recorded as is;"
                                    " the end-of-leg DRYWIN_METRICS event should have stopped this earlier)"),
            "distill_ce_probe": (str(RUNS / leg / "distill_ce_probe.jsonl")
                                 if (RUNS / leg
                                     / "distill_ce_probe.jsonl").is_file()
                                 else "not available (as above; the leg-side gap in the course 2 calibration-data obligation"
                                      " is recorded as is)")}
    card["RC14_drywin_instruments"] = {
        **rc14, "note": "no prior band, recorded as is; within this case it can only show internal consistency, not a norm"}
    # RC.15 8000-pool winner pair
    if hold_docs:
        k1 = by_seed(strict_json_loads(
            W_PIN["b1-ref8k-launch"][0].read_bytes())["rows"], 8000, 8031)
        h1_rows = by_seed(hold_docs["h"]["rows"], 8000, 8031)
        st15 = paired_diff_stats(h1_rows, k1)
        rc15 = band_judge(st15["mean"], *RC15_BAND["band"])
        rc15.update({"point": RC15_BAND["point"], "median": st15["median"],
                     "sign": st15["sign"]})
        add("RC15_holdout_paired", rc15)
        rc15d = band_judge(depth2_count(hold_docs["h"]["rows"]),
                           RC15_BAND["d2_band"][0], RC15_BAND["d2_band"][1],
                           integer=True)
        rc15d["point"] = RC15_BAND["d2_point"]
        add("RC15_holdout_depth2", rc15d)
    else:
        card["RC15_holdout"] = "undecidable (no winner or the held-out pair was not run); recorded as is"
    # family-level reading clause (this card reads 14 bands; <=3 scattered out-of-band lines is normal, no cherry-picking)
    n_out = sum(1 for _, ok in band_outcomes if not ok)
    family = {"n_band_readings": len(band_outcomes), "n_out_of_band": n_out,
              "out_names": [n for n, ok in band_outcomes if not ok],
              "clause": "<=3 scattered out-of-band lines is normal (similar to B1's ratio of <=3 out of about 17 bands); no"
                        " cherry-picking; only clustered out-of-band lines escalate to NEEDS_ATTENTION"}
    if n_out >= 4:
        attention(f"R lines clustered out of band ({n_out}): {family['out_names']}")
    ev = {"event": "VERDICT_PATH", "case": "v33-content",
          "golden_authorized": verdicts.get("golden_authorized", False),
          "scorecard": card, "family_ledger": family,
          "winner": verdicts.get("winner"),
          "course5": verdicts["course5"], "a12_tier": verdicts["a12"],
          "attribution_grammar": {
              "course5": "course 5 effect = L-cur − L-base",
              "yi": "4b effect = L-full − L-cur (4b package = the conjunction λ_bc>0 + v2 files)",
              "triangle": "the same seed only guarantees initial equivalence; 'the only variable' is a configuration-level statement, not trajectory pairing"
                          " (v32 R32 split definition, verbatim)"},
          "mandatory_notes": [
              "horizon statement: max-steps 3000, frozen, mandatory with the verdict",
              "training reward is blind to this failure (−6% vs −34), never a criterion",
              "H-track default and non-retroactivity clauses carried over from B1 D3",
              "M29-side evaluation determinism is presumed (B1 residual 15 not settled); the presumption note goes with every M29 verdict",
              "note on point estimates: every R-line point estimate without a stated source is discretionary, not an estimate",
              "residuals 1-17: see section 12 of the pre-registration; carried into the verdict appendix (written by the operator)"]}
    log(ev)
    events.append(ev)
    attention(f"v33 case-closing scorecard recorded: course5={verdicts['course5'].get('branch')}; "
              f"4b tier={verdicts['a12'].get('tier')}; "
              f"winner={verdicts.get('winner')}; the verdict appendix is written by the operator")


def log_launch_check_no_winner(events, limbs: dict, quals: dict) -> None:
    """Fix F (launch-time audit, minor): the no-winner branch's launch_check record gets the stage_done
    idempotence guard (symmetric with the winner branch) -- re-entry (resume) does not record twice."""
    if stage_done(events, "launch_check"):
        return
    log({"event": "launch_check", **limbs, "quals": quals})
    events.append({"event": "launch_check"})


# ======================================================================
# main flow
# ======================================================================

def _main():
    events = read_ledger()
    if stage_done(events, "VERDICT_PATH"):
        print("case already closed: idempotent exit", flush=True)
        return
    preflight(events)                                   # S0
    g0_baseline_stage(events)                           # S1 E0 transcription
    bc1_stage(events)                                   # S2 S-bc1
    bc2_stage(events)                                   # S3 S-bc2 + N12
    g0_endpoint_stage(events)                           # S4 G0-1
    g0_nullintrusion_stage(events)                      # S4 G0-2a
    g0_funcsmoke_stage(events)                          # S4 G0-2b
    g0_ghost_stage(events)                              # S4 G0-ghost
    g0_demo_ledger_stage(events)                        # S4 G0-demo pool
    refs = refs_stage(events)                           # S4 G0-6 REF_BITEQ
    freeze_stage(events)                                # S5 FREEZE/RT/NEWLINE
    launch_gate(events)                                 # S6 W-LAUNCH(exit 9)

    # ---- S7 three legs in series: launch -> canaries -> s16 -> full32 -> full32-m29 ----
    dry_curriculum_table_stage(events)
    npz = {}
    leg_docs = {}
    canary_docs = {}
    for leg in LEG_NAMES:
        npz[leg] = leg_stage(events, leg)
        canary_docs[leg] = canary_stage_leg(events, leg)
        leg_docs[leg] = leg_exams(events, leg, npz[leg])

    # ---- S8 criteria (scientific main verdicts first, decoupled from release) ----
    ref_launch = by_seed(refs["launch"]["rows"], 7000, 7031)
    # the dead sci_rows assignment was removed (audit report item 1: after the D4-3 context correction the science control is gone; Fix G is in death_seed_sets)
    R = refs["launch"]["agg"]["ret_mean"]
    abandon = round(R * ABANDON_FRAC[0] / ABANDON_FRAC[1], 1)
    floor_line = round(R * FLOOR_FRAC[0] / FLOOR_FRAC[1], 1)
    # course 5 main verdict (RC.5): F-lock v1 hits; anchor = measured L-base in the case
    sig = {leg: flock_hits(leg_docs[leg]["full32"], ref_launch,
                           npz[leg], leg) for leg in LEG_NAMES}
    course5 = course5_ruling(sig["v33-base"]["n_hits"],
                             sig["v33-cur"]["n_hits"])
    course5.update({
        "hits_by_leg": {leg: sig[leg]["n_hits"] for leg in LEG_NAMES},
        "l_full_note": "the L-full hit count must be listed alongside but is a combined course 5 + 4b reading and never enters the course 5"
                       " decision; if it disagrees with L-cur, list both as is, never aggregate",
        "seat_7017_note": {leg: sig[leg]["hit_7017"] for leg in LEG_NAMES},
        "firewall": "F-lock decides with the current v1 construction constants; v2 goes back as an auxiliary delivered instrument"
                    " and never enters this case's decision lines",
        "caveats": sig["v33-base"]["caveats"]})
    if not stage_done(events, "COURSE5_MAIN"):
        log({"event": "COURSE5_MAIN", **course5})
        events.append({"event": "COURSE5_MAIN"})
    # 4b main verdict (D4-3): L-full is always judged, win or lose; ctrl = L-cur x M29
    full_m29 = by_seed(leg_docs["v33-full"]["full32-m29"]["rows"], 7000, 7031)
    cur_m29 = by_seed(leg_docs["v33-cur"]["full32-m29"]["rows"], 7000, 7031)
    paired_fc = sum(full_m29[s]["ret"] - cur_m29[s]["ret"]
                    for s in sorted(cur_m29)) / 32
    a12_verdict = a12_tier_ruling(
        a12_per_ep(leg_docs["v33-full"]["full32-m29"]["agg"]),
        leg_docs["v33-full"]["full32-m29"]["agg"]["died"], paired_fc,
        leg_docs["v33-cur"]["full32-m29"]["agg"]["died"],
        a12_per_ep(leg_docs["v33-cur"]["full32-m29"]["agg"]))
    base_m29_agg = leg_docs["v33-base"]["full32-m29"]["agg"]
    a12_verdict["base_m29_parallel"] = {
        "mean": base_m29_agg["ret_mean"], "died": base_m29_agg["died"],
        "a12": a12_per_ep(base_m29_agg),
        "note": "counterpart of the retraining-regression safeguard: if L-cur and L-base survive equally, the causal 'conversion'"
                " caveat is tightened"}
    # Fix G: the set-difference control = L-cur x M29 (D4-3 in-case context; correction note in death_seed_sets)
    a12_verdict["death_seed_sets"] = death_seed_sets(
        leg_docs["v33-full"]["full32-m29"]["rows"],
        leg_docs["v33-cur"]["full32-m29"]["rows"])
    a12_verdict["h_side_parallel"] = {
        "a12_full_h": a12_per_ep(leg_docs["v33-full"]["full32"]["agg"]),
        "note": "readings under H listed alongside, with the 'structurally tight' caveat (ref depth2 7 vs 15);"
                " not a separate line; decoupled from the release criterion"}
    if not stage_done(events, "A12_MAIN"):
        log({"event": "A12_MAIN", **a12_verdict})
        events.append({"event": "A12_MAIN"})

    # ---- eligibility / winner / release (first use of the G1 new lines, full conjunction recorded) ----
    means = {leg: leg_docs[leg]["full32"]["agg"]["ret_mean"]
             for leg in LEG_NAMES}
    died = {leg: leg_docs[leg]["full32"]["agg"]["died"] for leg in LEG_NAMES}
    quals = {leg: qual_of(leg_docs[leg]["full32"]) for leg in LEG_NAMES}
    if not stage_done(events, "quals"):
        log({"event": "quals", "quals": quals, "means": means, "died": died,
             "qualification": "eligibility = died≤8 (K1 single-archive dependency label) ∧ sentinel limbs untouched"
                              " (level change/cap/override) ∧ not voided (written out in D4-1)"})
        events.append({"event": "quals"})
    win = decide_winner(means, died, quals)
    winner = win["winner"]
    verdicts = {"course5": course5, "a12": a12_verdict, "winner": winner,
                "golden_authorized": False}
    hold_docs = None
    if winner is None:
        verdicts["launch_limbs"] = {"verdict": "all three legs lost eligibility -- no winner, release not testable"
                                               " (scientific main verdicts already recorded, decoupled from release)"}
        log_launch_check_no_winner(events, verdicts["launch_limbs"], quals)
    else:
        if win.get("substituted") and not stage_done(events, "substitution"):
            log({"event": "substitution", "blocked": win["prelim"],
                 "why": quals[win["prelim"]]})
            events.append({"event": "substitution"})
        wrows = by_seed(leg_docs[winner]["full32"]["rows"], 7000, 7031)
        st = paired_diff_stats(wrows, ref_launch)
        w_died = died[winner]
        abandon_pass = means[winner] >= abandon
        limbs = {
            # Fix C: the release mean-difference limb uses mean_raw (raw semantics, v32 precedent; a near-line reading like
            # 17.315 would falsely pass as 17.32 after rounding); the recorded paired_mean still shows 2dp.
            "paired_mean_ge_17.32": st["mean_raw"] >= G1_PAIRED_DIFF,
            "wins_ge_22of32": st["wins"] >= G1_PAIRED_WINS,
            "died_le_8": w_died <= G1_DIED_MAX,
            "floor_85_92": means[winner] >= floor_line,
            "sentinel_zero_touch": quals[winner]["sentinel_zero_touch"],
            "non_void": not quals[winner]["void"],
        }
        launch = all(limbs.values()) and abandon_pass
        deleverage_note = None
        # the note uses the same definition as the decision limb (raw; audit report item 2 -- verdict and note must not use two definitions)
        if DELEVERAGE_Q95 < st["mean_raw"] < G1_PAIRED_DIFF:
            deleverage_note = ("leverage-sensitive band: the block is decided by the single-seed noise contribution of 7017"
                               " (upgrade review caveat 3)")
        elif st["mean_raw"] <= DELEVERAGE_Q95:
            deleverage_note = "also blocked below the deleveraged line; the block is robust to the 7017 leverage"
        verdicts["launch_limbs"] = {
            "winner": winner, "paired_mean": st["mean"], "wins": st["wins"],
            "died": w_died, "mean": means[winner],
            "lines": {"paired_diff": G1_PAIRED_DIFF,
                      "wins": G1_PAIRED_WINS, "died_max": G1_DIED_MAX,
                      "floor": floor_line, "abandon": abandon},
            "limbs": limbs, "abandon_pass": abandon_pass, "launch": launch,
            "labels": ["single-vector line (RG1.2a)", "K1 single-archive dependency (RG1.3)"],
            "alpha_listing": MULTIPLE_COMPARISON,
            "deleverage_note": deleverage_note,
            "asymmetry_note": "a winner with died∈{7,8} passing the release line while gold evaluation structurally falls back a tier"
                              " ('failed succession') is an expected case announced in advance by design decision 9;"
                              " gold evaluation keeps deaths ≤6 (G1 L2-0; the release line does not extend to the gold-evaluation P lines)"}
        if not stage_done(events, "launch_check"):
            log({"event": "launch_check", **verdicts["launch_limbs"]})
            events.append({"event": "launch_check"})
        if launch:
            verdicts["golden_authorized"] = True
        if launch and not stage_done(events, "GOLDEN_AUTHORIZED"):
            golden_cmd = (f"{PY} train/eval_assembled.py --worker {npz[winner]} "
                          f"--manager-npz {H_NPZ} --seeds 9000-9031 "
                          f"--tag v33-golden --board")
            log({"event": "GOLDEN_AUTHORIZED", "leg": winner,
                 "mean": means[winner], "died": w_died,
                 "wins": st["wins"], "mean_diff": st["mean"],
                 "worker_npz_sha": sha16(npz[winner]),
                 "golden_cmd": golden_cmd,
                 "handover_expectation": ("died∈{7,8} failed-succession expectation slot: "
                                          + ("present (died="
                                             f"{w_died})" if w_died >= 7
                                             else "absent")),
                 "notice": "burning the gold pool once is a campaign-level notice rule (design decision 9 acknowledged); the gold run is"
                           " started manually by the operator, one arm, once; the title flow requires the G0-1 dual baselines to pass first"
                           " (D2-1 downgrade clause)"})
            attention(f"gold run waiting for manual start: {winner} (release line passed; gold-evaluation death line ≤6 asymmetry on record)")
        # H1/H2 held-out bundled pair (on the winner)
        d_h = exam_case(events, npz[winner], HOLDOUT_TAGS[0], POOL_HOLD,
                        bundled_with=HOLDOUT_TAGS[1])
        holdout_account(events, HOLDOUT_TAGS[0])
        d_m = exam_case(events, npz[winner], HOLDOUT_TAGS[1], POOL_HOLD,
                        manager_npz=str(M29_NPZ), bundled_with=HOLDOUT_TAGS[0])
        holdout_account(events, HOLDOUT_TAGS[1])
        hold_docs = {"h": d_h, "m29": d_m}

    # ---- S9 R-line scorecard ----
    scorecard_stage(events, refs, leg_docs, canary_docs, verdicts, hold_docs)


def _smoke_main():
    """--smoke: run only the G0-2a/2b live smoke tests; nothing goes to the main ledger; uses a separate smoke ledger
    (run_b1_infra --smoke form; freeze prerequisite, dirty tree allowed)."""
    global _LEDGER_PATH
    _LEDGER_PATH = SMOKE_LEDGER
    events = read_ledger()
    preflight(events, smoke=True)
    g0_nullintrusion_stage(events)
    g0_funcsmoke_stage(events)
    print("smoke stage complete (verdicts in smoke-ledger events G0_NULLINTRUSION/G0_SMOKE;"
          " main ledger untouched)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run only the G0-2a/2b smoke tests (separate smoke ledger, nothing goes to the main ledger)")
    args = ap.parse_args()
    try:
        with exclusive_lock(V33 / ".driver.lock", "v33 driver"):
            if args.smoke:
                _smoke_main()
            else:
                _main()
    except OutputReservationError as e:
        log({"event": "OPERATIONAL_FAILURE", "why": f"W8 lock conflict: {e}"})
        attention("W8 not idle / lock conflict:\n" + str(e))
        raise SystemExit(4) from e
    except PreflightFailure as e:
        log({"event": "PREFLIGHT_FAIL", "why": str(e)})
        attention("P4 preflight failed, no launch; report for review:\n" + str(e))
        raise SystemExit(3) from e
    except OperationalFailure as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("operational failure:\n" + str(e))
        raise SystemExit(2) from e
    except SystemExit:
        raise
    except Exception as e:
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("driver died abnormally (P1):\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

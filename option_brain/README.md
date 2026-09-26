# option_brain: the code behind "version 9"

This folder holds the code that played the round-9 exam games (2026-09-24) and the tools that ran, checked and
scored the exam. Results: [`docs/rounds/round9-exam.md`](../docs/rounds/round9-exam.md).

**Model weights are not included.** Neither the LoRA adapters nor the executor networks (model158 and the goal
head) nor any training data or teacher labels are published; only their sha256 values are (below and in
[`PROVENANCE.md`](PROVENANCE.md)). Our own game logs are not published either. Without the weights the games
cannot be re-run, and without the logs the replay tool cannot be pointed at our games. The code is here to show
what ran: the files are the frozen round-9 copies, except that the comments and messages of seven files (and six
comment lines of `runtime-deps/runtime-deps.patch`) were translated to English or neutralised; their original
sha256 values are in [`PROVENANCE.md`](PROVENANCE.md) where they are published.

## How one decision works

```
live_runner.py ─ reads the paused game state through the bridge (the game does not advance while the model thinks)
  ├─ options.py        builds the menu of options that can be executed now (A, B, C, ...), minus hidden ones
  │   └─ undo_guard.py hides options that would just undo the previous action
  ├─ facts.py          computes the fact lines from engine data and formulas (hit chance, readiness, ...)
  ├─ render.py         writes the prompt: mission, location, hero, enemies, recent results, fact lines, menu
  ├─ hf_broker.py      one forward pass of Mistral-Small-3.2-24B (4-bit NF4) + LoRA adapter;
  │                    the answer is the most probable legal letter (option_sft.load_model)
  ├─ options.compile_goal  turns the letter into a goal for the hands
  └─ game_executor_r3.py   the "hands": walking, attack chains, chase, auto-belt (code), on top of
        strategy-brain-sft-20260922/game_executor_r2.py + game_support_r3.py
        strategist-rl-king-continuation-20260921/  (config, navigation, revision1-3 runtimes)
        strategist-rl-hands-20260921/  (runtime: bridge scope and gym adapter; model.py: frozen model158)
        skeleton-king-dual-brain-20260921-r16/session.py  (native journal with a sha256 hash chain;
                                                             death is final, no reload)
  → native bridge (native/bridge-r3) → DevilutionX engine (patches/ + native/engine-r11-loadmonster.patch)
```

The model never issues engine commands itself. It picks one letter per decision; code turns that into a goal and
executes it until the goal ends (completed, interrupted by a new enemy or by damage, and so on); then the game
pauses again for the next decision. In the old arm of the exam this happened on average every 46.7 game ticks
(about 2.3 s of game time at 20 ticks per second).

The folder names mirror the original workspace because the code finds its dependencies by relative path
(for example `live_runner.py` adds `../strategy-brain-sft-20260922` to `sys.path`). The import closure of a
game process was traced on 2026-09-24; every module it loads from the workspace is here, plus two frozen
versions of this repository's own files (see `runtime-deps/`). `strategist-rl-hands-20260921/runtime.py`
imports a module called `config`; in a game process that name is already bound to
`strategist-rl-king-continuation-20260921/config.py`, so the hands folder's own `config.py` is never loaded
and is not included.

The files are working copies, so their comments keep development timestamps and pointers to unpublished working
notes. Internal planning documents cited in comments (`PREREG-ROUND9.md`, `ROADMAP-20260923.md`, `REPORT.md` and
others) are not published.

## What is where

| Path | What |
|---|---|
| `option-brain-20260922/live_runner.py` | Plays one game: menu, prompt, broker request, goal, stop rules, logs |
| `option-brain-20260922/options.py`, `undo_guard.py` | The option menu and its hiding rules (round-9 menu `r9`) |
| `option-brain-20260922/facts.py`, `engine_numbers.py` | Fact lines (v3 for the released model, v5 for the candidate) |
| `option-brain-20260922/render.py` | System prompt and state prompt |
| `option-brain-20260922/hf_broker.py`, `option_sft.py` | Inference (NF4 + LoRA, argmax over legal letters, top-8 probabilities logged) and training/loading code |
| `option-brain-20260922/rollin_broker.py` | Hashed by `live_runner.py` at start; not used in exam games |
| `option-brain-20260922/game_executor_r3.py` | The r3 hands (attack chains, target rule, hold floor, chase to the last-seen tile) |
| `option-brain-20260922/quest_lottery.py`, `bench/butcher_probe.py` | Which seeds carry the Skeleton King / Butcher quest (copy of the engine's quest draw; tick-0 check) |
| `option-brain-20260922/menu_r9.py` | Offline re-implementation of the round-9 menu rules for existing logs (not run in games; its lazy import of a training script is not included) |
| `option-brain-20260922/run_mixed_v4.sh`, `run_exam9_wave.sh`, `run_exam9_chain.sh`, `broker_watchdog.sh` | Launching a wave of 16 games with one broker; the four-wave chain; the broker watchdog |
| `option-brain-20260922/check_exam9_wave.py`, `score_exam9.py`, `exam9_lib.py`, `exam9_config.json` | Per-wave integrity check, scoring, shared rules (failure classes, loop signatures), the single config |
| `option-brain-20260922/make_exam9_freeze.sh`, `seed_audit9.py` | Freezing (sha256 list of every file that runs) and the seed-leak audit |
| `option-brain-20260922/launch_smoke9.sh`, `launch_gate9.sh`, `fidelity9.py` | Smoke test, training-world gate, adapter re-scoring (fidelity) |
| `option-brain-20260922/TEACHER.md` | Written guidance given to the teacher models for the round-9 labels (new arm); the student never sees it |
| `teacher/` | The two guidance versions closest in time to the round-8 labels of the released adapter (the exact text used cannot be confirmed) |
| `replay/replay_verify.py` | CPU replay of a native journal through the same engine and bridge, row by row (no model, no brain) |
| `runtime-deps/runtime-deps.patch` | Turns `python/diablogym/env.py`, `options_env.py` and `train/leashed_ppo.py` of the version-9 release (tag `round9-stable-kills-20260924`) into the frozen versions the game process loads |
| `native/` | Bridge-r3 sources, the LoadMonster engine fix, sanitized build records ([README](native/README.md)) |
| `localize.py` | Makes a copy with your own paths (the published files use placeholders) |
| `PROVENANCE.md`, `provenance.json` | Published sha256 of every file, and the original sha256 where it is published |

## The released model ("version 9") and the candidate

| | `d9facts3` (released, "version 9") | `d13facts5` (candidate, not adopted) |
|---|---|---|
| Base | `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, revision `95a6d26c4bfb886c58daf9d3f7332c857cb27b43` (Apache-2.0), 4-bit NF4 for training and inference | same |
| Adapter | LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o attention projections (19,660,800 trainable parameters) | same shape |
| Adapter sha256 (`adapter_model.safetensors`) | `1d3b24942f1a2a5825f5aa8ab7a000b1d5cb27639c33ee889e4ce13dd0a6dd72` | `3380ec0f5aeb5622fc67e4edf77836439cdcdd93cadf0ff48d2c2f02d26cab37` |
| `adapter_config.json` sha256 | `3f9445229c861be44b12227dd8f07c9bbfecca43f2e38e046f0d3ebfad9419dd` | `2520ec010298e1eef68178f487ecdfe756df0781de111e371cb9ad10e926bea1` |
| Fact lines | version 3 | version 5 |
| Trained | 2026-09-23 (round 8) | 2026-09-24 (round 9d) |
| Data | 3,260 rows: 909 decisions of an earlier OpenAI Codex agent ("Astra"), 2,037 states labelled by Claude teacher workflows over 8 DAgger rounds (two independent teachers, a third to adjudicate), 74 label copies, 240 formula-derived attribute-point labels | 4,455 rows: the above lineage plus round-9 teacher labels and formula rules |
| Training | one epoch, lr 1e-4, 20 warm-up steps then cosine to 0.1x, one answer letter per row plus a copy with shuffled options; best validation checkpoint (step 780 of 783, 73.1% agreement with the teachers) | 1,070 steps, 84.8% validation agreement; failed one pre-set offline criterion (N2) |

Training is imitation learning (supervised learning on teacher labels, with DAgger), not reinforcement learning.
The executor's small networks were trained earlier and are frozen: model158 (PPO, sha256
`bd85ce98491f8f1c0056d3924f4c0eb3bdb6f9cc4f04cf5b734f78815f3dc1db`; in the round-8 games it chose "attack" on
every one of 82,252 beats, so it acts as a fixed rule) and the goal head (sha256
`a2cb7d3b909f93b61148eea2a4878d2eef9eef922e16fca4676e2f974fc5251c`; its walking step must equal the code's
shortest-path step or the game raises an error).

## Running it (outline)

You need, besides this folder: Linux or WSL2; a GPU with enough memory for a 24B model in NF4 (the exam used
one RTX 5090); the base model at the revision above; an adapter (not published); model158 and the goal head
(not published); the engine and bridge built from `patches/`, `native/` and DevilutionX `34c4cfc`; a checkout of
this repository at tag `round9-stable-kills-20260924` with `runtime-deps.patch` applied where `strategist-rl-king-continuation-20260921/config.py`
(`SOURCE`, `NETWORK_SOURCE`) expects it; and your own `DIABDAT.MPQ`. The exam used Python 3.12.3 with
torch 2.12.1 (CPU) for the game process, and torch 2.12.1+cu130, transformers 5.17.0, peft 0.21.0,
bitsandbytes 0.50.2, accelerate 1.15.0 and mistral_common 1.11.7 for the broker.

```bash
python option_brain/localize.py /work/experiments --workspace /work --root /work/run --home /work/home --gpu GPU-...
cd /work/experiments/option-brain-20260922
# one wave: 8 King + 8 Butcher games against one broker (see run_exam9_wave.sh for the checks it makes first)
FACTS_VERSION=3 EXTRA_HANDS=--allow-heldout bash run_mixed_v4.sh <batch> <adapter dir> <seed>:skeleton_king:150000:720:<tag> ...
```

Every script checks its inputs (file hashes, adapter fact-line version, free GPU, no other broker) and refuses to
run otherwise; expect to adapt paths and checks to your machine.

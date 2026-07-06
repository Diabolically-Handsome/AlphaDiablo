# AlphaDiablo / DiabloGym

[![CI](https://github.com/Diabolically-Handsome/AlphaDiablo/actions/workflows/ci.yml/badge.svg)](https://github.com/Diabolically-Handsome/AlphaDiablo/actions/workflows/ci.yml)

**A fast, deterministic Diablo I reinforcement-learning environment** built on
[DevilutionX](https://github.com/diasurgical/devilutionX), plus the training
pipeline that took a PPO agent from *hiding in a corner* to *opening doors,
smashing barrels, looting potions and fighting its way down through the
dungeon* — thirteen documented runs, one diagnosed failure mode eliminated
(or one hypothesis falsified) per run.

- 🚀 **~13,000× realtime**: full game logic, headless — 254k engine ticks/s raw,
  ~7,500 `env.step()`/s with full observations (M-series MacBook, measured)
- 🎲 **Deterministic**: `reset(seed)` owns the dungeon seeds *and* the global RNG
  stream; evaluations are bit-reproducible across processes (verified per-seed,
  see protocol notes in [train/evaluate.py](train/evaluate.py)); engine source
  pinned to an exact upstream commit by [bootstrap.sh](bootstrap.sh)
- 🧩 **Gymnasium API**: structured observations (entity features + 11×11 local
  map), macro-actions (engage / explore / descend / drink / pick-up-potion)
- 📊 **Zero-dependency live dashboard** for training runs
- 🩹 Ships **upstream fixes** for six DevilutionX headless-mode bugs — asset
  fallbacks, monster-missile anims (a bat swoop was the first crash), the
  unloaded SFX table (the Butcher's greeting was the second) — in `patches/`

![learning curves](docs/assets/learning-curves.png)

*Left: training-time kills (sampled policy, rolling 100) across the six
iterations that built the champion. Right: the gold standard — deterministic
(argmax) evaluation on 32 fixed seeds. Full run-by-run post-mortems in
[docs/DESIGN.md](docs/DESIGN.md) (Chinese; lesson summaries below).*

## Results (32-seed deterministic gold standard)

| model | params | mean kills | median | max | zero-kill | reached L2 |
|---|---|---|---|---|---|---|
| v5 vision, no explore macro¹ | 45,771 | 7.6 | 0 | 45 | 19/32 | 0/32 |
| v6 macro-MLP | 45,836 | 8.8 | 3.5 | 36 | 15/32 | 0/32 |
| v8 LSTM-128 | 451,596 | 8.4 | 3.0 | 43 | 13/32 | 0/32 |
| v9c entity-attention | 701,980 | 3.8 | 0 | 38 | 21/32 | 0/32 |
| v10 = v6 recipe, 3000-step episodes | 45,836 | 5.5 | 0 | 49 | 18/32 | 0/32 |
| v11 = v6 + descend option | 45,901 | 19.4 | 14.5 | **70** | 2/32 | **27/32** |
| v12 = v11 + belt-potion action | 45,966 | 12.3 | 10.0 | 46 | 9/32 | 26/32 |
| **v13 = potion system made learnable (champion)** | 46,543 | **35.2** | **29.0** | 65 | **1/32** | 25/32 |

¹ *Evaluated post-hoc on the current env (same observation; it never selects
the explore macro). Protocol: seeds 9000-9031, 1500 steps, argmax, idle
machine, pinned engine — [train/leaderboard.md](train/leaderboard.md).*

Honesty notes: each row is a **single training run** (v1-v10 unseeded; v11
onward uses `--seed`), and a 32-seed mean has an SEM of ≈2 kills — so the
v5/v6/v8 means are statistically indistinguishable and ordering claims rest
on the distribution shape (median, zero-kill), not the means. The v11 jump,
by contrast, moves every column at once and is far outside that noise band.
Leaderboard checkpoints are not distributed yet (a tagged release is
planned); rows come from the author's runs and are deterministically
re-evaluable given the checkpoint. Champion honesty numbers: v13's
pre-registered predictions went **2/4** — real-drink share >50% and mean
≥16 hit; deaths ≤10 **missed** (12/32), reached-L2 ≥26 **missed** (25/32).
Deaths did fall 17/32 → 12/32 while the kill rate nearly doubled, but one
seed (9001) migrated the v12 idle-spam attractor onto the new pickup key —
1,448 no-op presses (lesson 12). Observation changes (286→290 in v13) end
direct re-evaluation of older checkpoints on the current env; each row
stands on the env version it was scored under (same policy as v1-v4/v7).

Four findings we did not expect:

1. **At this scale, task design beats architecture** (directional evidence,
   one run per architecture). With a 3M-step budget, a 46k-parameter MLP
   equipped with two hand-built macro-actions matches a 10×-larger LSTM, and
   a 15×-larger entity-attention model never trained stably — even with
   double the budget (6M steps) it ended at 3.8. The single-episode max is
   too noisy to rank architectures (the memoryless v5 hit 45; the LSTM 43).
   The wins came from reward attribution, action granularity and an
   exploration option — not from bigger brains.
2. **The remaining failures are dead zeros, not slow episodes.** Doubling the
   evaluation horizon to 3,000 steps changes *nothing*: per-seed kill counts
   are bit-identical at both horizons for both v6 and v10, all 32 seeds. When
   the spawn pocket has no reachable prey, the agent never recovers — a
   planning/exploration failure, not a time budget one.
3. **Capability lives in the action space, not the parameter count.** The
   dead zeros turned out to be a *sensor* problem: closed doors are
   indistinguishable from walls in the walkability channel, so part of every
   level is invisible-by-construction. A static "sealed spawn" analysis
   predicts zero-kill episodes for the MLP, the LSTM and the attention model
   with zero false positives (15/15 cells) — information destroyed at the
   sensor is unrecoverable by any downstream architecture. v11 added **one
   action** (a descend option that plans through doors/barrels with a
   full-map BFS and operates them en route), left observation, rewards and
   architecture untouched, and doubled the gold standard — where a 15×
   parameter increase had previously *lost* points. Emergent bonus: on the
   deepest sealed seed the policy uses the descend macro as a *door-opening
   key* and farms the unsealed rooms without ever taking the stairs.
4. **One observation bit bought a ~190× improvement in button discipline.**
   v12 and v13 share the same drink button. v12 could not see the belt and
   spent 99.5% of its presses on an empty one; v13 can, and spends 93.4% of
   its presses on a stocked one — 25 of its 57 argmax drinks fire below
   half HP, the deepest at 1% HP. What makes a skill learnable is not the
   action but the observability of the action's precondition (lessons 5,
   11, 12) — and the mean nearly doubled (19.4 → 35.2) once the potion
   economy closed.

### Twelve lessons from thirteen runs (short version)

1. Don't tax the intermediate costs of the behaviour you want, and don't leave
   zero-cost sanctuaries in the reward landscape (v1's wall-hugger).
2. Shaping must be attributed to the agent's own actions — monsters walking
   toward you is not progress (v2's fishing exploit).
3. When atomic actions are finer-grained than the task's causal structure,
   package them as temporally-extended options (v3's engage macro).
4. Densify rewards on conserved task progress (damage fractions), never on
   countable events (swing counts) — anything countable gets farmed (v4).
5. Rewards can only cash in information that exists in the observation; when
   failures cluster spatially, fix perception first (v5's 11×11 map).
6. Don't force a reactive policy to learn planning — wrap planning as an
   option and let the policy choose (v6's explore macro: the median episode
   went from 0 kills to 3.5 and zero-kill episodes from 19/32 to 15/32; the
   mean gap, +1.2, is within eval noise).
7. Macro engineering has degenerate attractors: each patch bred a new exploit;
   after three patch rounds (v7-v7d) we froze the interface instead.
8. Eight evaluation seeds lied to us in *both* directions (champion inflated
   77%, v5 deflated 15%); 32 fixed seeds, argmax, frozen protocol — and treat
   machine load as part of the protocol.
9. Architecture upgrades pay off only when the bottleneck is the brain: the
   LSTM matched but didn't beat the macro-MLP; attention never trained
   stably; doubling episode length changed nothing. The bottleneck is the
   spawn-pocket deadlock — task structure again.
10. Perception bounds what can be known, the action set bounds what can be
    done, architecture only tunes the efficiency in between (v11: one new
    option, +120% mean kills; v9c: 15× parameters, −57%). Audit those three
    layers in that order — the cheapest miracles live in the action space.
11. A new action is also a new hiding place. v12's drink action did its
    designed job on a few seeds (one argmax clutch heal from 8.6% HP) and
    deaths fell 17/32 → 10/32 — but mean kills regressed 19.4 → 12.3, and
    4,715 of 4,740 presses hit an empty belt. The belt count was
    deliberately kept out of the observation (protocol comparability), so
    the policy could never learn when *not* to press: lesson 5 applies to
    action preconditions too. Door-blindness, then bottle-blindness —
    self-inflicted this time. v11 keeps the crown.
12. Discipline is a function of observability, and hiding places are
    conserved. Giving the policy eyes on the belt (v13) turned 99.5% waste
    into 93.4% discipline and doubled the champion — but the idle-spam
    attractor from lesson 11 did not die, it migrated: one seed presses the
    new pickup key 1,448 times as its no-op corner. Remove a hiding place
    and risk-averse probability mass flows to the next zero-risk action;
    budget for attractor migration whenever you add one.

## Quickstart (macOS, Apple Silicon)

```bash
# 0. Requirements: Homebrew, Xcode CLT, Python ≥3.11
python3 -m venv .venv && .venv/bin/pip install -e ".[train,build]"

# 1. Game data (pick one):
#    - Free shareware (dungeon levels 1-2, no quest monsters):
mkdir -p "$HOME/Library/Application Support/diasurgical/devilution"
curl -L -o "$HOME/Library/Application Support/diasurgical/devilution/spawn.mpq" \
  https://github.com/diasurgical/devilutionx-assets/releases/download/v5/spawn.mpq
#    - Full game: buy Diablo on GOG, extract DIABDAT.MPQ with `brew install innoextract`,
#      drop it in the same folder (see docs/DESIGN.md notes).

# 2. Engine + bridge (clones DevilutionX at the pinned commit, applies patches, builds)
./bootstrap.sh && ./build.sh

# 3. Verify: random agent + determinism + descend/seed-differentiation
.venv/bin/python tests/smoke_random_agent.py
.venv/bin/python tests/descend_seed_test.py

# 4. Train + watch
.venv/bin/python train/train_ppo.py --total-steps 3000000 --num-envs 4
.venv/bin/python train/dashboard.py        # → http://127.0.0.1:8787

# 5. Evaluate against the leaderboard protocol (idle machine!)
.venv/bin/python train/evaluate.py train/runs/<run>/model_final
```

## How it works

| Layer | Where | What |
|---|---|---|
| C++ bridge | `src/diablogym.cpp` | Embeds the whole engine as a shared library (`HeadlessMode`), drives the game loop tick-by-tick from Python, injects actions at the **network command layer** (same path as multiplayer — a trained agent can later join a TCP co-op game as a headless client) |
| Env | `python/diablogym/env.py` | Gymnasium env: 290-dim obs (player/monster entities + stairs direction + 11×11 walkability & monster-occupancy map + belt/floor-potion fields), `Discrete(14)` with engage/explore/descend/drink/pickup macro-actions, per-hit damage rewards |
| Training | `train/train_ppo.py` | SB3 PPO, subprocess vec-envs, per-episode JSONL metrics |
| Evaluation | `train/evaluate.py` | Frozen 32-seed deterministic protocol; appends to the leaderboard |
| Monitoring | `train/dashboard.py` | stdlib-only live dashboard (SVG charts, 2s polling) |
| Engine fixes | `patches/` | Headless asset-fallback fixes (town cel/til/sol/min), applied idempotently by `build.sh` |

Determinism notes: the engine reseeds its global RNG from the wall clock when
creating a hero (`CreatePlayer`) and paces turns against real time
(`nthread_has_500ms_passed`). The bridge re-seeds the global RNG from the
episode seed on every `reset()`, and the evaluation protocol requires an idle
machine — under heavy load a trajectory can slip by one logic turn. Both
quirks are documented in [train/evaluate.py](train/evaluate.py).

## Roadmap

- [x] v0 walking skeleton: embed, reset(seed), step, obs, actions
- [x] Phase 1 — autonomous fighter on dungeon level 1 (v6: 8.8 mean kills)
- [x] **Crack the spawn-pocket deadlock** — root cause was door-blindness in
  the walkability channel; the v11 descend option (door/barrel-aware BFS)
  cut zero-kill episodes 15/32 → 2/32
- [x] Descend to L2 — 27/32 episodes reach it now (deepest runs chain to L4)
- [x] Survive down there: v12's blind drink action cut deaths at a kill-rate
  cost (lesson 11); v13 made the potion system *learnable* (belt count +
  nearest floor heal into the observation, door-aware pickup macro) —
  deaths 17/32 → 12/32 while mean kills doubled to 35.2
- [ ] Gear up: equip armor and weapons from the floor (v14)
- [ ] Clear-rate objective
- [ ] The Butcher 🥩 (his greeting already crashed our headless engine once —
  see patches/0003; killing him is next)
- [ ] Cross-class generalization (Rogue / Sorcerer — `hero_class` already exposed)
- [ ] Multiplayer co-op deployment (carry your creator through the game)

## 中文速览

基于 DevilutionX 的暗黑破坏神 I 强化学习环境:无头引擎裸跑 ~13,000 倍实时
(含观测的 env.step 约 7,500 步/秒,~1,500 倍实时)、种子级确定性(评估跨进程
位级可复现)、Gymnasium 接口、宏动作(交战/探索/下楼/喝药/捡药)、零依赖训练
监控面板。十三轮迭代把 PPO 从"面壁思过"练到"开门、砸桶、捡药续命、一路下杀"
(32 种子金标准均击杀 **35.2**,较上代冠军近乎翻倍;实喝纪律 0.5%→93.4%),
并留下十二课教训:奖励税、塑形归因、动作时序、防磨刀、感知天花板、探索
option、宏退化吸引子、评估运气税、任务设计>架构、能力住在动作空间、新动作
也是新藏身处、**纪律是观测的函数而藏身处守恒**——每一课都有数据实锤,完整
踩坑史见 [docs/DESIGN.md](docs/DESIGN.md)。

## Legal

MIT for the code in this repository. `patches/` contains derivative snippets of
DevilutionX (Sustainable Use License — non-commercial); the build fetches
DevilutionX from upstream rather than vendoring it. **No copyrighted game assets
are included**: bring your own `DIABDAT.MPQ` (GOG) or use Blizzard's freely
available shareware `spawn.mpq`. Diablo® is a trademark of Blizzard
Entertainment. This is an unofficial research project, unaffiliated with
Blizzard Entertainment or DeepMind.

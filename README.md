# AlphaDiablo / DiabloGym

[![CI](https://github.com/Diabolically-Handsome/AlphaDiablo/actions/workflows/ci.yml/badge.svg)](https://github.com/Diabolically-Handsome/AlphaDiablo/actions/workflows/ci.yml)

**A fast, deterministic Diablo I reinforcement-learning environment** built on
[DevilutionX](https://github.com/diasurgical/devilutionX), plus the training
pipeline that took a PPO agent from *hiding in a corner* to *autonomously
hunting Fallen packs on dungeon level 1* — ten documented runs, one diagnosed
failure mode eliminated (or one hypothesis falsified) per run.

- 🚀 **~13,000× realtime**: full game logic, headless — 254k engine ticks/s raw,
  ~7,500 `env.step()`/s with full observations (M-series MacBook, measured)
- 🎲 **Deterministic**: `reset(seed)` owns the dungeon seeds *and* the global RNG
  stream; evaluations are bit-reproducible across processes (verified per-seed,
  see protocol notes in [train/evaluate.py](train/evaluate.py)); engine source
  pinned to an exact upstream commit by [bootstrap.sh](bootstrap.sh)
- 🧩 **Gymnasium API**: structured observations (entity features + 11×11 local
  map), macro-actions (engage / explore)
- 📊 **Zero-dependency live dashboard** for training runs
- 🩹 Ships **upstream fixes** for four DevilutionX headless-mode asset bugs
  (`patches/`)

![learning curves](docs/assets/learning-curves.png)

*Left: training-time kills (sampled policy, rolling 100) across the six
iterations that built the champion. Right: the gold standard — deterministic
(argmax) evaluation on 32 fixed seeds. Full run-by-run post-mortems in
[docs/DESIGN.md](docs/DESIGN.md) (Chinese; lesson summaries below).*

## Results (32-seed deterministic gold standard)

| model | params | mean kills | median | max | zero-kill | reached L2 |
|---|---|---|---|---|---|---|
| v5 vision, no explore macro¹ | 45,771 | 7.6 | 0 | 45 | 19/32 | 0/32 |
| **v6 macro-MLP (champion)** | 45,836 | **8.8** | **3.5** | 36 | 15/32 | 0/32 |
| v8 LSTM-128 | 451,596 | 8.4 | 3.0 | 43 | 13/32 | 0/32 |
| v9c entity-attention | 701,980 | 3.8 | 0 | 38 | 21/32 | 0/32 |
| v10 = v6 recipe, 3000-step episodes | 45,836 | 5.5 | 0 | 49 | 18/32 | 0/32 |

¹ *Evaluated post-hoc on the current env (same observation; it never selects
the explore macro). Protocol: seeds 9000-9031, 1500 steps, argmax, idle
machine, pinned engine — [train/leaderboard.md](train/leaderboard.md).*

Honesty notes: each row is a **single training run** (training was unseeded;
`--seed` exists now), and a 32-seed mean has an SEM of ≈2 kills — so the
v5/v6/v8 means are statistically indistinguishable and ordering claims below
rest on the distribution shape (median, zero-kill), not the means. Leaderboard
checkpoints are not distributed yet (a tagged release is planned); rows come
from the author's runs and are deterministically re-evaluable given the
checkpoint. And the least flattering number in the table: despite a +8.0
reward per level descended and stairs-direction features in the observation,
**no agent of any generation has ever taken the stairs to level 2**.

Two findings we did not expect:

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
   planning/exploration failure (**the** open problem), not a time budget one.

### Nine lessons from ten runs (short version)

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
| Env | `python/diablogym/env.py` | Gymnasium env: 286-dim obs (player/monster entities + stairs direction + 11×11 walkability & monster-occupancy map), `Discrete(11)` with engage/explore macro-actions, per-hit damage rewards |
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
- [x] Phase 1 — autonomous fighter on dungeon level 1 (champion: 8.8 mean
  kills over 32 seeds; single-episode best 49 across generations)
- [ ] **Crack the spawn-pocket deadlock** (15/32 episodes end at 0 kills when
  no prey is reachable early — candidates: spawn curriculum, cross-room
  frontier exploration, bigger LSTM budget)
- [ ] Clear-rate objective & descend-to-L2 curriculum
- [ ] The Butcher 🥩 (requires full game data)
- [ ] Cross-class generalization (Rogue / Sorcerer — `hero_class` already exposed)
- [ ] Multiplayer co-op deployment (carry your creator through the game)

## 中文速览

基于 DevilutionX 的暗黑破坏神 I 强化学习环境:无头引擎裸跑 ~13,000 倍实时
(含观测的 env.step 约 7,500 步/秒,~1,500 倍实时)、种子级确定性(评估跨进程
位级可复现)、Gymnasium 接口、宏动作(交战/探索)、零依赖训练监控面板。十轮
迭代把 PPO 从"面壁思过"练到"自主猎杀堕落者"(从早期 8 种子旧口径的 0,到现役
冠军在 32 种子金标准下的 8.8),并留下九课教训:奖励税、塑形归因、动作时序、
防磨刀、感知天花板、探索 option、宏退化吸引子、评估运气税、任务设计>架构——
每一课都有数据实锤,完整踩坑史见 [docs/DESIGN.md](docs/DESIGN.md)。

## Legal

MIT for the code in this repository. `patches/` contains derivative snippets of
DevilutionX (Sustainable Use License — non-commercial); the build fetches
DevilutionX from upstream rather than vendoring it. **No copyrighted game assets
are included**: bring your own `DIABDAT.MPQ` (GOG) or use Blizzard's freely
available shareware `spawn.mpq`. Diablo® is a trademark of Blizzard
Entertainment. This is an unofficial research project, unaffiliated with
Blizzard Entertainment or DeepMind.

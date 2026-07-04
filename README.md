# AlphaDiablo / DiabloGym

**A fast, deterministic Diablo I reinforcement-learning environment** built on
[DevilutionX](https://github.com/diasurgical/devilutionX), plus the training
pipeline that took a PPO agent from *hiding in a corner* to *reliably clearing
Fallen packs on dungeon level 1* over six documented iterations.

- 🚀 **~20,000× realtime**: full game logic, headless, ~400k ticks/s on an M-series MacBook
- 🎲 **Deterministic**: `reset(seed)` → identical dungeon, monsters and RNG stream
- 🧩 **Gymnasium API**: structured observations (entity features + 11×11 local map), macro-actions (engage / explore)
- 📊 **Zero-dependency live dashboard** for training runs
- 🩹 Ships **upstream fixes** for four DevilutionX headless-mode asset bugs (`patches/`)

![learning curves](docs/assets/learning-curves.png)

*Six iterations, one diagnosed failure mode eliminated per run. Gold standard =
deterministic (argmax) evaluation, mean kills over 8 fixed seeds: 0 → 0.2 → 2.8
→ 3.0 → 6.5 → **15.6**, zero-kill episodes 5/8 → **0/8**. Full post-mortems in
[docs/DESIGN.md](docs/DESIGN.md).*

## Quickstart (macOS, Apple Silicon)

```bash
# 0. Requirements: Homebrew, Xcode CLT, Python ≥3.11
python3 -m venv .venv && .venv/bin/pip install numpy gymnasium pybind11 stable-baselines3 torch

# 1. Game data (pick one):
#    - Free shareware (dungeon levels 1-2, no quest monsters):
mkdir -p "$HOME/Library/Application Support/diasurgical/devilution"
curl -L -o "$HOME/Library/Application Support/diasurgical/devilution/spawn.mpq" \
  https://github.com/diasurgical/devilutionx-assets/releases/download/v5/spawn.mpq
#    - Full game: buy Diablo on GOG, extract DIABDAT.MPQ with `brew install innoextract`,
#      drop it in the same folder (see docs/DESIGN.md notes).

# 2. Engine + bridge (clones DevilutionX to a temp dir, applies patches, builds)
./bootstrap.sh && ./build.sh

# 3. Verify: random agent + determinism + descend/seed-differentiation
.venv/bin/python tests/smoke_random_agent.py
.venv/bin/python tests/descend_seed_test.py

# 4. Train + watch
.venv/bin/python train/train_ppo.py --total-steps 3000000 --num-envs 4
.venv/bin/python train/dashboard.py        # → http://127.0.0.1:8787
```

## How it works

| Layer | Where | What |
|---|---|---|
| C++ bridge | `src/diablogym.cpp` | Embeds the whole engine as a shared library (`HeadlessMode`), drives the game loop tick-by-tick from Python, injects actions at the **network command layer** (same path as multiplayer — a trained agent can later join a TCP co-op game as a headless client) |
| Env | `python/diablogym/env.py` | Gymnasium env: 286-dim obs (player/monster entities + stairs direction + 11×11 walkability & monster-occupancy map), `Discrete(11)` with engage/explore macro-actions, per-hit damage rewards |
| Training | `train/train_ppo.py` | SB3 PPO, subprocess vec-envs, per-episode JSONL metrics |
| Monitoring | `train/dashboard.py` | stdlib-only live dashboard (SVG charts, 2s polling) |
| Engine fixes | `patches/` | Headless asset-fallback fixes (town cel/til/sol/min), applied idempotently by `build.sh` |

## Roadmap

- [x] v0 walking skeleton: embed, reset(seed), step, obs, actions
- [x] v1 phase 1: reliable fighter on dungeon level 1 (this release)
- [ ] Clear-rate objective & descend-to-L2 curriculum
- [ ] The Butcher 🥩 (requires full game data)
- [ ] Cross-class generalization (Rogue / Sorcerer — `hero_class` already exposed)
- [ ] Multiplayer co-op deployment (carry your creator through the game)

## 中文速览

基于 DevilutionX 的暗黑破坏神 I 强化学习环境:无头 2 万倍实时、种子级确定性、
Gymnasium 接口、宏动作(交战/探索)、零依赖训练监控面板。六轮迭代把 PPO 从
"面壁思过"练到"确定性策略稳定清剿堕落者"(金标准 0 → 15.6),完整踩坑史见
[docs/DESIGN.md](docs/DESIGN.md)——奖励税、塑形白嫖、动作时序、感知天花板、
探索 option,每一课都有数据实锤。

## Legal

MIT for the code in this repository. `patches/` contains derivative snippets of
DevilutionX (Sustainable Use License — non-commercial); the build fetches
DevilutionX from upstream rather than vendoring it. **No copyrighted game assets
are included**: bring your own `DIABDAT.MPQ` (GOG) or use Blizzard's freely
available shareware `spawn.mpq`. Diablo® is a trademark of Blizzard
Entertainment. This is an unofficial research project, unaffiliated with
Blizzard Entertainment or DeepMind.

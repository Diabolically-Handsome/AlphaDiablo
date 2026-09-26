# From a normal level-1 start to a living Skeleton King kill

**AlphaDiablo defeated Diablo I's Skeleton King and ended alive at level 7,
96/96 HP, with 13 healing potions remaining.** Native King kills = 1, quest
completed = true, hero dead = false. The run stopped immediately at that point.

Recorded September 22, 2026, 00:11 UTC.
[中文](README.zh-CN.md) · [Machine-readable result](result.json)

## Watch

[![Final encounter, recorded tick 55000](media/poster.png)](https://github.com/Diabolically-Handsome/AlphaDiablo/releases/download/skeleton-king-first-kill-20260921/skeleton-king-final-fight-60fps.mp4)

- **[Final encounter: continuous 85.3-second MP4](https://github.com/Diabolically-Handsome/AlphaDiablo/releases/download/skeleton-king-first-kill-20260921/skeleton-king-final-fight-60fps.mp4)** — all 1,706 game ticks and 5,118 native display subframes, including every drink, interruption and pause in actual movement.
- [Opening: continuous 12:15.2 MP4](https://github.com/Diabolically-Handsome/AlphaDiablo/releases/download/skeleton-king-first-kill-20260921/normal-start-opening-12m15s.mp4) — the preserved beginning of the same world, **not** a second win or a video of the kill.
- [Optional final-fight status subtitles](https://github.com/Diabolically-Handsome/AlphaDiablo/releases/download/skeleton-king-first-kill-20260921/skeleton-king-final-fight-60fps.en.srt) — HP is rounded down; the potion counter is **belt only**, not the total inventory.
- [Release: videos, evidence ZIP and checksums](https://github.com/Diabolically-Handsome/AlphaDiablo/releases/tag/skeleton-king-first-kill-20260921).

Both videos are silent 640×352 viewport recordings at 60 fps. Actual game logic
runs at 20 Hz; each tick supplies three native animation/camera display
subframes, not AI-generated interpolation. The MP4 uses lossy H.264; retaining
every time step does **not** mean pixel-lossless encoding. Thinking time while
the game was paused is excluded, rather than filled with repeated stills.

| Released video | Native ticks (inclusive) | Position in the full game timeline |
| --- | --- | --- |
| Opening | 1–14,704 | 00:00–12:15.20 |
| Final encounter | 54,682–56,387 | 45:34.05–46:59.35 |

**The intervening video is not published yet.** A full-run export stopped on a
display-only unique-monster corpse binding fault. It was not a gameplay death.
For this release, all 63,153 recorded commands/tick entries were replayed and
matched exactly; rendering was enabled only for the final encounter. That
excerpt passed, but does not fix or conceal the full-film fault. Original failed
exports and frames remain preserved locally. The evidence ZIP covers the
**entire command history**, including the omitted middle, losses and retreats.

## What happened

One normal-difficulty Warrior world, seed **20260927**, began at level 1 in town.
The fixed seed order was 20260920–20260951; only Skeleton King quest availability
was checked. The first seven lacked that quest; 20260927 was the first match.
The selection record is included. No selection by loot or map difficulty, no
resource/level injection, and no death rollback are claimed or used.

The first two King encounters ended in living retreats. The same character then
earned level 7, spent five earned stat points on dexterity, identified normal
loot for the native fee, repaired equipment and bought supplies. On the third
encounter it carried 20 healing potions. The final weapon was a normally found,
paid-identified **Morning Star of zest** (base damage 1–10, +7 vitality), with a
small shield, leather armor and skull cap. The strategist selected the fighting
position and ordered seven drinks; 13 potions remained.

Final totals: **level 7, XP 28,179, armor 32, 736 gold, 186 kills including the
King, 56,387 ticks / 46:59.35 of game time**. This world did **not** kill the
Butcher; the earlier Butcher milestone is a separate run. There were no extra
ticks to finish the death animation, collect the crown or continue afterward.

## Who controlled what

The high-level strategist was an OpenAI Codex agent, not a
fine-tuned local manager. It selected goals, routes, preparation, equipment,
stat allocation, retreats and explicit healing interventions using visible
state and explored-map memory. The historical opening used **1,624 Ministral 3
8B requests**. Subsequent execution has **8,850** separately attributed records:

| Component | Decisions |
| --- | ---: |
| Frozen model158 combat worker | 1,803 |
| Previously trained navigation/goal head | 3,861 |
| Strategist-directed services | 293 |
| Strategist-directed holds | 2,893 |

No optimizer ran and neither worker nor navigation-head weights changed. Native
hit reactions, blocking, misses and attack interruption remained active. An
offline forward-pass audit reproduced 1,803 worker and 3,861 navigation choices.
Earlier engineering versions repaired interfaces and native loading; this is
**not** a claim that the whole project has an unmodified vanilla engine. The
successful continuation's frozen engine identity is in [provenance.json](provenance.json).

The worker still has a clear limitation: all 1,803 choices were attacks, even
though drinking was legal in 273 decisions. In the final fight it chose attack
412 times, with drinking available 55 times. **All seven drinks were ordered by
the strategist**, not learned during this run.

## Check the evidence

Download `skeleton-king-evidence.zip` from the release, then run:

```text
python verify.py skeleton-king-evidence.zip
python verify.py skeleton-king-evidence.zip path/to/downloaded-media
```

[The checker](verify.py) requires Python 3.11+ and only its standard library.
It validates file hashes, the native journal chain and every tick, earlier
segments as exact prefixes, final state/quest/survival, execution attribution,
seven drinks, goal-to-before-state bindings and the final clip's frame index.
The optional second argument checks the downloaded media hashes as well.
This is **record consistency checking**, not an independent engine replay or
an independent success. Running the actual game additionally requires the
specific experimental runtime, checkpoints and legally owned game assets,
which are not bundled here.

The ZIP includes the complete native command journal, public final checkpoint,
prior command prefixes, attributed execution records, 448 candidate teacher
goals, combat/forward-pass audits, seed selection and frame metadata. Private
absolute paths were replaced with neutral source labels; native command and
checkpoint bytes were left unchanged. The original/public hashes are recorded.
No MPQ/game asset archive, native save archive, model weights or credentials
are included. The 448 goals are **unreviewed demonstrations, not ground truth**.

## What this establishes — and what it does not

This is a concrete successful **assistant-led hybrid agent** demonstration from
normal starting conditions. It is one world with surviving retries and allowed
thinking pauses, not multiple independent wins, human-speed real-time play,
pure-RL autonomy, a stable win rate, a world-first claim, or a full Diablo clear.
Preparation, equipment, position and combat randomness all changed; this does
not isolate the manager as the sole cause or prove the old worker is sufficient.

This publication preserves evidence. It does not deploy a model, resume
training, replace the active certified checkpoint, or initiate another game.

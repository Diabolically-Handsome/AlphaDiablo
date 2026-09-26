# Native side of the round-9 games (engine and bridge sources)

Every game of the round-9 exam, and of the earlier round-8 checks, ran on the same two native binaries:

| Binary | sha256 | How it was built |
|---|---|---|
| Engine `liblibdevilutionx_so.so` ("engine-repair-r11") | `41eb66efd17492ec2d2c0a54795b5e0ddf3f11efc9c7c544cd82299dfa435da2` | DevilutionX + patches below; see `engine-r11-build.json` |
| Bridge `_diablogym.cpython-312-x86_64-linux-gnu.so` ("bridge-r3") | `05fc300940ac491d9e05fdd37dacecf7e9e139811de102d0c886caee8f0952b1` | `bridge-r3/` compiled against the patched engine; see `bridge-r3-build.json` |

**The compiled `.so` files are not included**, and no game data is included (bring your own `DIABDAT.MPQ`).
A rebuild from these sources is not byte-identical (the translated messages alone change the binary), and an
exact journal replay needs the exact binaries.

## Engine source = upstream + this repository's patches + one fix

- Upstream: [DevilutionX](https://github.com/diasurgical/devilutionX) commit `34c4cfc` (the commit that
  `bootstrap.sh` pins).
- Patches: `patches/0001` to `patches/0014` of this repository; `0014` differs from the version that ran only in
  three comment lines translated to English. The development copies used for the games differ from the committed
  ones only in line endings of context lines (the engine sources use CRLF); `build.sh` applies them with
  `--ignore-space-change`.
- One more fix, `engine-r11-loadmonster.patch` (not in `patches/`, so the environment build in this repository
  is unchanged): when a monster is loaded from a level save, the six `reducePlayer*` fields (attribute drains)
  are restored from the monster's type data. Upstream does not store them in the save record, so a reused
  monster slot could keep the previous occupant's values. The engine's own level-change path saves and reloads
  a level's monsters, so this matters whenever the hero returns to a level.
- Checked on 2026-09-24, before the three comment lines of `0014` were translated: `git archive 34c4cfc Source test`, then
  `git apply --ignore-space-change` of `patches/0001`–`0014` gives a tree whose `Source/` and `test/` equal the
  development engine tree (`diff -r --strip-trailing-cr`); applying `engine-r11-loadmonster.patch` on top gives
  the r11 `Source/` tree exactly. With the current `0014` the tree differs from it only in those three comment
  lines.
- What the patches do (four kinds): crash guards for headless running; default-off hooks that can only refuse a
  level change; a shop/unequip transaction refactor that keeps prices and random-number use; read-only
  interfaces. Neither they nor the LoadMonster fix above change a combat, drop, price, experience, quest or
  map-generation formula or data table.

To build the engine the way the games used it, apply `patches/0001`–`0014` and then
`option_brain/native/engine-r11-loadmonster.patch` to DevilutionX `34c4cfc` (the three translated comment lines
of `0014` do not change the compiled code):

```bash
git -C devilutionX apply --ignore-space-change "$PWD/option_brain/native/engine-r11-loadmonster.patch"
```

## Bridge source (`bridge-r3/`, 17 files)

The 17 files are identical to the `source_files` digests in `bridge-r3-build.json` except for comment and message
translation (in `diablogym.cpp`, `resource_identify.hpp`, `resource_protocol.hpp` and `resource_sweep.hpp`; original
and published digests in [`../PROVENANCE.md`](../PROVENANCE.md)). They differ from
this repository's `src/` (the bridge of the Gymnasium environment in `python/diablogym`): three headers are the
same (`resource_combinations.hpp`, `resource_identify.hpp`, `resource_sweep.hpp`); `diablogym.cpp`,
`resource_loot.hpp` and `resource_protocol.hpp` are newer; eleven headers are new. The most relevant one is
`manual_control.hpp`: the manual, lockstep game interface the option brain plays through. It lets the hero use
only main dungeon levels 0 to 3, the Skeleton King's tomb and a town portal to levels 1–3
(`ManualTransitionGuard`); that is the depth limit of this bridge.

`src/` is left as it is, because `build.sh` and the tests of this repository build and check that version.

**Game options.** At game start the bridge pins the engine options that would change world or action results
(`diablogym.cpp`, around line 1966): 20 ticks per second, randomized quests, no Lua mods, and the automatic
conveniences (auto-pickup, belt auto-refill, auto-equip) switched off, so that only issued commands change the
inventory.

**Test probes.** The bridge binary also contains test probes that would be cheats if called, for example
`probe_invincible`, `probe_add_experience`, `probe_resource_add_gold` and `probe_warp_main_level`. The
published code has no call site for any of them. The only `probe_` calls in the code are read-only:

- in a game process, `probe_tile` (walkability of one tile), called by `strategist-rl-hands-20260921/runtime.py`
  and in five places of the runtime `diablogym/env.py` (see `runtime-deps/`);
- outside the games, `probe_tile` and `probe_is_spawn` in `bench/butcher_probe.py`, the tick-0 quest check made
  before the exam, which never enters the dungeon.

The native journals record game commands, not `probe_` calls. The journals of all 48 completed exam games contain
only 18 ordinary command kinds (`tick`, `walk`, `attack_stand`, `attack`, `pickup`, `operate`, `stat`, `drink`,
`buy`, `equip`, `talk`, `dismiss`, `sell`, `belt`, `unbelt_exact`, `repair`, `reset`, `identify`), but that
census alone cannot show that no cheat probe was called. The stronger evidence is the replay: in the 38 exam
games replayed from their journals, every row's hash chain, which includes the engine's checkpoint hash, was
equal; a state change that was not in the journal would have broken it. Nothing structurally blocks the probes:
the Python wrapper refuses names that start with `debug`, `test_`, `force`, `inject` or `set_`, but not
`probe_`.

## Build records

`bridge-r3-build.json` and `engine-r11-build.json` are sanitized copies of the two build records: absolute paths
are replaced by placeholders (`$ENGINE_R11`, `$ENGINE_BUILD`, `$DEVX`, `$VENV`, `$BRIDGE_R3`, `$REPO`,
`$AD_HOME`); hashes are unchanged. `original_record_sha256` is the sha256 of the unsanitized record. The engine
record summarizes the 250 retained object files of the patched build by count and digest. Its `bridge_sha256`
belongs to a test bridge built in the same step, not to bridge-r3.

## License

The bridge sources are original code of this project (MIT, see `LICENSE`); they include DevilutionX headers
when compiled. `engine-r11-loadmonster.patch`, like the files in `patches/`, is a derivative snippet of
DevilutionX and carries DevilutionX's
[Sustainable Use License](https://github.com/diasurgical/devilutionX/blob/master/LICENSE.md) (non-commercial).
Using this project as a whole is therefore limited to non-commercial purposes (see `NOTICE`).

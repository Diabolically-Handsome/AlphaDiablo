# Provenance of the published option_brain files

Every file below is a copy of the file that ran in the round-9 exam (or, where noted, of the file used to verify it). The published copy differs from the original in the following ways:

1. **Machine-specific values are replaced by placeholders**: the local workspace folder by `$AD_WORKSPACE`, the run root by `$AD_ROOT`, the home folder by `$AD_HOME`, and two GPU UUIDs by `$AD_GPU_UUID` and `$AD_GPU2_UUID`. The replacement is literal. `localize.py` substitutes your own values. The round trip was checked locally against the freeze list.
2. **Two docstring lines** were rewritten: one in `option-brain-20260922/game_executor_r3.py` (column "edits") and line 3 of the runtime `options_env.py` (see "Runtime dependencies"). These do not round-trip; they are documentation and do not change behaviour.
3. **English pass**: after the version-9 release, the comments and messages of seven files were translated to English or neutralised (the four translated bridge sources under "Native sources", `score_exam9.py`, `bench/butcher_probe.py` and `facts.py`; column "edits"), and six comment lines of `runtime-deps.patch` were neutralised (see "Runtime dependencies"). These do not round-trip; they do not change behaviour.

Nothing else was changed: no code was reformatted, and line endings are kept as they were on disk (three files use CRLF). `.gitattributes` marks `option_brain/**` as `-text` so Git does not convert them.

"Freeze list" is the round-9 exam freeze list (361 files, sha256 `5b732f5eb0d1ba68bd5c7915f52210ed89d8724b814fb625d54399a59dc7a9ef`); it was re-checked with `sha256sum -c` after the exam (361/361 OK). The list itself is not published because it names local paths. Published sha256 values are given in full below; some original sha256 values are withheld. The same data is in `provenance.json`.

## Code and tools

| File | Original sha256 | Published sha256 | Placeholders | Edits | Round trip |
|---|---|---|---|---|---|
| `option-brain-20260922/bench/butcher_probe.py` | withheld (in the round-9 freeze list, not published) | `e18fcb2d280de0860fdab3e03f6fac1ec9c51624ac18d5035bb173da5e42de86` | `$AD_WORKSPACE`×2, `$AD_ROOT`×1 | lines 1-2: comment translated to English and a local-directory pointer reworded | no |
| `option-brain-20260922/broker_watchdog.sh` | withheld (in the round-9 freeze list, not published) | `68ce7aa9847c957bd9b430ffed333ce681af3f8cf67c679ed3223c49fc692303` | `$AD_WORKSPACE`×1 | — | yes |
| `option-brain-20260922/check_exam9_wave.py` | `61e69dd02b61cd3bd0ae147320a1e6c2c8518afa4031eab746f7c063b19e4287` | (same) | — | — | yes |
| `option-brain-20260922/engine_numbers.py` | withheld (in the round-9 freeze list, not published) | `0b9c21446db9131178dc0a153523f9bee6320500bdc263f1fe4971266baf1863` | `$AD_HOME`×1 | — | yes |
| `option-brain-20260922/exam9_config.json` | withheld (in the round-9 freeze list, not published) | `da9c58a54259f65087bb160bfcb3564909ca6479cca26fb28f96d27c8f232a6e` | `$AD_WORKSPACE`×1, `$AD_ROOT`×9, `$AD_HOME`×4, `$AD_GPU_UUID`×1 | — | yes |
| `option-brain-20260922/exam9_lib.py` | `7cc8f810a17e0c4bef9fbd08d43e4dbdad5c4db2adc2531b2c85740ae8669af2` | (same) | — | — | yes |
| `option-brain-20260922/facts.py` | `485e9fab50da76bf8cd73c7d10f19c59c587c7ba078608149bf51a92fbde9dbe` | `42371b35f26b40ebd8dcdb142c341af940523bcad3a2be687f666c690c0f6bf6` | — | lines 1539-1540: a workflow id and a local note path removed | no |
| `option-brain-20260922/fidelity9.py` | `6802ab3aba20478ca0c97b47ef40ffc6017c186547f2f107364449080073c729` | (same) | — | — | yes |
| `option-brain-20260922/game_executor_r3.py` | withheld | `27bb52ada6f9fff5efdb70721f14fd9e28550c8c093aac2f86323dac148d1de0` | — | line 21 docstring reworded | no |
| `option-brain-20260922/hf_broker.py` | `735ed3abcd384a0f8f7af7b1690dc75eda3e244fa97368c8d000a24c9ba57245` | (same) | — | — | yes |
| `option-brain-20260922/launch_gate9.sh` | withheld (in the round-9 freeze list, not published) | `9a6a4bdaec5d7ec2b049cac38fdf0c0a43e5a0b41156cabd86cd4c6bc58a08de` | `$AD_WORKSPACE`×2, `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/launch_smoke9.sh` | withheld (in the round-9 freeze list, not published) | `501e54fb6ad427e38c971961611ec08bc6a84323743b8cb39a00664f250e3576` | `$AD_WORKSPACE`×2, `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/live_runner.py` | withheld (in the round-9 freeze list, not published) | `53500f98c4609fefae007317ae5e5311153d0b6e7538a61a8c51d1793ada8893` | `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/make_exam9_freeze.sh` | withheld (in the round-9 freeze list, not published) | `b1be016230fbc3ba3ecab26aead7671f5fe3dc02a83e9577c399a014fb94e676` | `$AD_WORKSPACE`×2, `$AD_HOME`×3 | — | yes |
| `option-brain-20260922/menu_r9.py` | `4c14eb63aca47ad8a37000f82d5b4d942db2071cccf1146d75619ba6e2b42872` | (same) | — | — | yes |
| `option-brain-20260922/option_sft.py` | withheld (in the round-9 freeze list, not published) | `ff65f1b8381a88fbe128f40c7cada2b66ed01453188e8a298a9860cb83d43491` | `$AD_ROOT`×3, `$AD_HOME`×1, `$AD_GPU_UUID`×1 | — | yes |
| `option-brain-20260922/options.py` | withheld (in the round-9 freeze list, not published) | `8682c1fcc00fa4ae8d39b5dfd5329b4b0f28a8ea9d39a1c4b6595d19c93dbd3f` | `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/quest_lottery.py` | `1dd16eae0f33bd2d6503d205eef31af61d3659e59c74a4e5ef22673358170966` | (same) | — | — | yes |
| `option-brain-20260922/render.py` | `201e917b20c28fc4b711da3d8284edd676f8df6955248bc2bdb6bc4e5e4400e0` | (same) | — | — | yes |
| `option-brain-20260922/rollin_broker.py` | `971625c09751e542a1baa7b54d451aa0545373cf70e6b87fbf92c6d2c6a0359d` | (same) | — | — | yes |
| `option-brain-20260922/run_exam9_chain.sh` | withheld (in the round-9 freeze list, not published) | `2931ae4294eb7061fba3d7b45e2e8b8beae5e3c257635e41280647dd08a620a3` | `$AD_WORKSPACE`×2, `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/run_exam9_wave.sh` | withheld (in the round-9 freeze list, not published) | `9fedd4da759bb9aec7950d06b888916c93464b474041cc5e5bdbb973c6a9f4df` | `$AD_WORKSPACE`×1 | — | yes |
| `option-brain-20260922/run_mixed_v4.sh` | withheld (in the round-9 freeze list, not published) | `9c9f01a76b8ffb48d46c4af8b5db79d58a3388dced65388ff943cc78322dc7bf` | `$AD_WORKSPACE`×1, `$AD_ROOT`×5, `$AD_HOME`×2 | — | yes |
| `option-brain-20260922/score_exam9.py` | `fa276de768e01c3ebd7bab99b8f8add1c8d35a76209fc4c20b8dbdb2720368f7` | `5d5e80d1dee3b8543ef575b4730817b8789a13e89bd705193bdfb1cfa8547ac5` | — | comments and messages translated to English | no |
| `option-brain-20260922/seed_audit9.py` | `8ee8a1953881bb1718c8d5715c4c7f5039e1423f1d855c7e2c87a1f39a3a9582` | (same) | — | — | yes |
| `option-brain-20260922/TEACHER.md` | `1866da9673e484ecb426c2fa08382bc92e8db05c715a960ea7a7bd5ed50c737e` | (same) | — | — | yes |
| `option-brain-20260922/undo_guard.py` | `abf4a10cd491cf77507142073b36c282dff04b6118b272d5ff2012ea8ca7957e` | (same) | — | — | yes |
| `replay/replay_verify.py` | withheld (not published; this file is not in the freeze list) | `7810c3ff5ac756b1a132aa33642d485c1c8d8b97a59f67039d11fff5287de5e2` | `$AD_WORKSPACE`×1, `$AD_ROOT`×1, `$AD_HOME`×1 | — | yes |
| `skeleton-king-dual-brain-20260921-r16/session.py` | withheld (in the round-9 freeze list, not published) | `2ec43612c85940028c003b6be696b9e81ffc656cedc94825911d3e40da3dde31` | `$AD_HOME`×3 | — | yes |
| `strategist-rl-hands-20260921/goal.py` | `b7c20144857cc57e1237040949988a916aa5deddd5852f2071837b050a418d8c` | (same) | — | — | yes |
| `strategist-rl-hands-20260921/model.py` | `0fbcfc9c9078c1ba1ce0f20d187ca46f19df99568d46b1aa717c7ebb210727e0` | (same) | — | — | yes |
| `strategist-rl-hands-20260921/public_view.py` | `6d9c792f27c22154cf20b640697b8deb38ad4ea4fb43077105247d27ac278197` | (same) | — | — | yes |
| `strategist-rl-hands-20260921/runtime.py` | withheld (in the round-9 freeze list, not published) | `4ebb139cfd76a2a49428cc0b0b5d9bb8239d111efc35e97f70de68a3c94183d2` | `$AD_HOME`×2 | — | yes |
| `strategist-rl-king-continuation-20260921/config.py` | withheld (in the round-9 freeze list, not published) | `b20e56b09d1a1fc49d89a10fb5ac6a3502977d1bae7825219b41b90db75ab823` | `$AD_HOME`×8, `$AD_GPU_UUID`×1 | — | yes |
| `strategist-rl-king-continuation-20260921/navigation.py` | `365eeb1f74cd1950abda7e8544e4ebd203191c48de43ce60b987f49ceff941eb` | (same) | — | — | yes |
| `strategist-rl-king-continuation-20260921/revision1/execution_r1.py` | `9d2b04f56e7f789d4c429413a54edb86a072f6b5ffeb0a63b87be1f932754306` | (same) | — | — | yes |
| `strategist-rl-king-continuation-20260921/revision2/trade_runtime.py` | `9701777d807183018124cabbe7803f7ff8b9d3b24d1eb10acfe0e05638d7f5a8` | (same) | — | — | yes |
| `strategist-rl-king-continuation-20260921/revision3/telemetry_runtime.py` | `f93b25996ebe70cec803a61d54088dd4ae816d38adc667c6233a00ef721c7157` | (same) | — | — | yes |
| `strategy-brain-sft-20260922/common.py` | withheld (in the round-9 freeze list, not published) | `3c721956686aae7a1c1559717378ca6aa50de9699a8b6ebd8774411f1512e288` | `$AD_HOME`×1, `$AD_GPU_UUID`×1, `$AD_GPU2_UUID`×1 | — | yes |
| `strategy-brain-sft-20260922/game_executor_r2.py` | `0ecce987eed084754fd01177cbc694f1805254b0e54853558c50c0fe54987eca` | (same) | — | — | yes |
| `strategy-brain-sft-20260922/game_support_r3.py` | `dcc1fa9f64761b363e606d00e3fb58fccfda89de8b44dd6bbce3558c6990c018` | (same) | — | — | yes |
| `strategy-brain-sft-20260922/protocol.py` | `b8edfb14fc202b33dcf013a4095e29ba2fe35edd91ca8601636aca8b45a0a96a` | (same) | — | — | yes |
| `strategy-brain-sft-20260922/purchase_audit.py` | `2d5b69f7a760984247f54d5b94b021b34ca3d1df8c6c4f6fcbb77ba34e4ede6e` | (same) | — | — | yes |
| `teacher/TEACHER.before-r9-132005.md` | `dd75b2e679c599aa042ca42333e93e84436fbc3df7bfd396d822e8439c91e54b` | (same) | — | — | yes |
| `teacher/TEACHER.before-v3-054308.md` | `540f763c272ff310ef781ac83c4594618f6bde65045b6abb91bf549adc142619` | (same) | — | — | yes |

Freeze status: all files above are in the freeze list; their freeze-list digests are the original digests, including the withheld ones, except:

- `option-brain-20260922/TEACHER.md`: not in the freeze list; its sha256 is recorded in PREREG-ROUND9 revision 2 (teacher guidance for the round-9b-9d labels of the new arm).
- `replay/replay_verify.py`: not in the freeze list; written after the exam freeze for the round-8 replays and used unchanged for the round-9 replays (sha256 recorded in both replay summaries).
- `teacher/TEACHER.before-r9-132005.md`: not in the freeze list; one of the two guidance versions closest in time to the round-8 labels of d9facts3 (the exact text used cannot be confirmed).
- `teacher/TEACHER.before-v3-054308.md`: as above.

## Runtime dependencies that are versions of this repository's own files

The game process imports four modules of a frozen copy of this repository's `python/diablogym` package (dated 2026-07-25 to 2026-09-09) and `leashed_ppo.py` of a frozen copy of `train/`. Three of the five differ from the files of the version-9 release (tag `round9-stable-kills-20260924`); the current repository files also differ from the runtime versions in comments and messages. `runtime-deps/runtime-deps.patch` turns the files of that release into the runtime versions:

```
git apply option_brain/runtime-deps/runtime-deps.patch   # in a separate checkout of tag round9-stable-kills-20260924
```

| Runtime file | Repository file (at the release tag) | Original sha256 | After the patch | Note |
|---|---|---|---|---|
| `diablogym/controller_wire.py` | `python/diablogym/controller_wire.py` | `7ff86ea8ff7547e5a7e219bbd7d1ca716918904ca3b1d2b515c1446dae166095` | equal | identical in the release, no hunk |
| `diablogym/env.py` | `python/diablogym/env.py` | `edac8479396648fb7c21b182407fffa3a643fe57d0909ed363a4f52779fd3bfa` | equal | changed by the patch |
| `diablogym/nav.py` | `python/diablogym/nav.py` | `15d79605a958c5239a171c1d06e3a0c8967dc73603d75d3c61c24db2c0648b86` | equal | identical in the release, no hunk |
| `diablogym/options_env.py` | `python/diablogym/options_env.py` | withheld | `08aef1b01c54fd269a04c6ac0245e30d62d5d8c7725374fbd5ee80a6b52019fb` | changed by the patch, except line 3 (kept as in the repository file) and six comment lines whose internal wording was neutralised |
| `train/leashed_ppo.py` | `train/leashed_ppo.py` | `4a9c3dcaf8a84550c0397b7c80327d438e434982ef6d7a44923ba0b484857cfe` | equal | changed by the patch |

`runtime-deps.patch` sha256: `1245b32ab4648d550ea23b73d7b91ddf3f2e63bf16399bd7e88656e0539ac037` (original `82932103e835d2168088643687b240ea4f69bc9efb6da973db27a12f77701591`). Apply it with `core.autocrlf=false`; on Windows the default `true` rewrites line endings and the sha256 values then differ.

**Freeze gap.** The exam freeze list covers `leashed_ppo.py` (same file, same sha256) and, by content, `controller_wire.py` and `nav.py`, but **not the runtime copies of `env.py` and `options_env.py`**. Its import trace imported `leashed_ppo`, which imported the development checkout's `diablogym` (the editable install of the game environment) before the runtime's own `install_package` had run, so the list recorded that checkout's `diablogym` files instead. The game process itself loads the frozen copy above: `install_package` runs before the executor loads `leashed_ppo` and refuses to run if `diablogym` is already imported. The sha256 values above were taken on 2026-09-24, after the exam; the files' modification times are 2026-07-25 to 2026-09-09, before the exam.

## Native sources

See `native/README.md`. The bridge sources are identical to the build record of the bridge used in every exam game, except for comment and message translation in four files (original and published sha256 below); the two build records are sanitized summaries (paths replaced by placeholders).

| File | Original sha256 | Published sha256 |
|---|---|---|
| `native/bridge-r3/butcher_event.hpp` | `757810978cfeaa8c1d01b30cdac0f780243043bb2fa9b08e61f335ad61f2d4c3` | (same) |
| `native/bridge-r3/combat_telemetry.hpp` | `48996fe56f75f45e1e30110b1b6707c2a183c233af15454d9143a3744f3e537e` | (same) |
| `native/bridge-r3/diablogym.cpp` | `68491ca15749f9df5e66e0095eb68bebb44ece8604b9aa2d8c0671b4a4468ad7` | `7dfaaacb599155507491f02766fcf37857cff3bbda24affa4bd88dd895508ecd` |
| `native/bridge-r3/field_equipment.hpp` | `08e881b539317cd8f3132159c00c7d7f73f8c474893bd1a05456a9d9c5229cf9` | (same) |
| `native/bridge-r3/gear_wear_fixture.hpp` | `3186820049ff21192e03175dcbc871d2cd67d1e88375af863c1fb82c612aa0f7` | (same) |
| `native/bridge-r3/manual_control.hpp` | `7b35b4d12515efb98b9db46557818a2a7593af062390bd83a903b8a2e63d0fe6` | (same) |
| `native/bridge-r3/preparation_goal.hpp` | `ff7582480116a1d73e8e6e823505281fe51f8ba37fecac409c6d136191252fbc` | (same) |
| `native/bridge-r3/quest_entrance_visibility.hpp` | `f4bf85166ae04a24a2a5e10e22369653f220a3e9320acb29b3d57a47f6ead377` | (same) |
| `native/bridge-r3/resource_combinations.hpp` | `f7a4646e5dfd8d8c3bce290f788fa2694aa7d5e4fc2aea7d108a817ece27db85` | (same) |
| `native/bridge-r3/resource_identify.hpp` | `c4b8a145ee0de86f8e949c97d801e2d0324dc8156ca76bb73f6bfa491c833cad` | `68f5b1e5b9b18c69e82367f7f6add48460d28eacbb1519024b16c9775e26a2f3` |
| `native/bridge-r3/resource_loot.hpp` | `6c5f2ade962b1157bcb346e403965f1209a40256c615e11ec1c92a08c97420ca` | (same) |
| `native/bridge-r3/resource_protocol.hpp` | `b29d617209628b85b8f760a498a387f5e2b97ba92a334187c08399fd34d02be2` | `78ae637498a0dda869589e7aed34b44f29751a49aa7b5d9bf68d40f71c6e264a` |
| `native/bridge-r3/resource_sweep.hpp` | `3446248045655b8fdd6c10a14050adb5c25904cd1f758d158c67a218e5f5b404` | `fb337594003d314fea691f1ac7097abbebaacfdfe154adef419c1e36e73d5979` |
| `native/bridge-r3/resource_trip_quota.hpp` | `f0520366849be3e83eb76318c9cbec682ca66c66392004f539e0003533474ce1` | (same) |
| `native/bridge-r3/shop_catalog_scope.hpp` | `395fd1d466a22a2fd8873a2d886a892a46d37125cad905fc469d6b2add4e5ebd` | (same) |
| `native/bridge-r3/unbelt_exact.hpp` | `1d6e4a4fc4d985c0a2ac8cad4796ea2e41e6aa7370d55c7224a6e138283120ac` | (same) |
| `native/bridge-r3/witch_trade.hpp` | `83a577fddf02b3a904e169fdc2755ca50b75ee95e1cd2456df71f21fc872df94` | (same) |
| `native/bridge-r3-build.json` | `ca3bd57deb112ef0d8701e8c97ea921dcd14b9dfbb972d4489296f4ee8fcce52` | (same) |
| `native/engine-r11-build.json` | `26bb95f99ae562fa7678258b3bd4d6869c6f48371faca227dffa6464457c5ad3` | (same) |
| `native/engine-r11-loadmonster.patch` | `f515612aae8306bf39d02236d728f8f75fa9baebc9bb1445ff8bdb72a588ee41` | (same) |

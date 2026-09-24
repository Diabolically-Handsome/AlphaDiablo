# Provenance of the published option_brain files

Every file below is a copy of the file that ran in the round-9 exam (or, where noted, of the file used to verify it). The published copy differs from the original in exactly two ways:

1. **Machine-specific values are replaced by placeholders**: the local workspace folder by `$AD_WORKSPACE`, the run root by `$AD_ROOT`, the home folder by `$AD_HOME`, and two GPU UUIDs by `$AD_GPU_UUID` and `$AD_GPU2_UUID`. The replacement is literal. `localize.py` substitutes your own values; substituting the original values reproduces the original bytes. This round trip was checked locally for every file (column "round trip").
2. **Two docstring lines that named the project owner** were rewritten: one in `option-brain-20260922/game_executor_r3.py` (column "edits") and line 3 of the runtime `options_env.py` (see "Runtime dependencies"). These do not round-trip; they are documentation and do not change behaviour.

Nothing else was changed: no code was reformatted, and line endings are kept as they were on disk (three files use CRLF). `.gitattributes` marks `option_brain/**` as `-text` so Git does not convert them.

"Freeze list" is the round-9 exam freeze list (361 files, sha256 `5b732f5eb0d1ba68bd5c7915f52210ed89d8724b814fb625d54399a59dc7a9ef`); it was re-checked with `sha256sum -c` after the exam (361/361 OK). The list itself is not published because it names local paths; the sha256 values of the files published here are given in full below. The same data is in `provenance.json`.

## Code and tools

| File | Original sha256 | Published sha256 | Placeholders | Edits | Round trip |
|---|---|---|---|---|---|
| `option-brain-20260922/bench/butcher_probe.py` | `c68c442f9c0f041f428e0ab2e39be333c4bcbc930ab09c0788588d8e8b3d5b58` | `b47246aa03288eeb741bbe5737a99cfc8b5b0535e474c94ef06f30e904ebb36b` | `$AD_WORKSPACE`×2, `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/broker_watchdog.sh` | `fd235f1c60405bafa511c60e4eaa41efef3a868b8c04c70b9e7b8f553480c91a` | `68ce7aa9847c957bd9b430ffed333ce681af3f8cf67c679ed3223c49fc692303` | `$AD_WORKSPACE`×1 | — | yes |
| `option-brain-20260922/check_exam9_wave.py` | `61e69dd02b61cd3bd0ae147320a1e6c2c8518afa4031eab746f7c063b19e4287` | (same) | — | — | yes |
| `option-brain-20260922/engine_numbers.py` | `02630e17ebc5d98e97d04f86477f37da8650d7c54874dd81d410520d90af14d2` | `0b9c21446db9131178dc0a153523f9bee6320500bdc263f1fe4971266baf1863` | `$AD_HOME`×1 | — | yes |
| `option-brain-20260922/exam9_config.json` | `28223fffe5836fd65fdc50319f323edeb0115371c3361850a5fb90d6f9eef8fd` | `da9c58a54259f65087bb160bfcb3564909ca6479cca26fb28f96d27c8f232a6e` | `$AD_WORKSPACE`×1, `$AD_ROOT`×9, `$AD_HOME`×4, `$AD_GPU_UUID`×1 | — | yes |
| `option-brain-20260922/exam9_lib.py` | `7cc8f810a17e0c4bef9fbd08d43e4dbdad5c4db2adc2531b2c85740ae8669af2` | (same) | — | — | yes |
| `option-brain-20260922/facts.py` | `485e9fab50da76bf8cd73c7d10f19c59c587c7ba078608149bf51a92fbde9dbe` | (same) | — | — | yes |
| `option-brain-20260922/fidelity9.py` | `6802ab3aba20478ca0c97b47ef40ffc6017c186547f2f107364449080073c729` | (same) | — | — | yes |
| `option-brain-20260922/game_executor_r3.py` | `cb998caa1b309126ce113bfcad4534317ce74d1721118ac9e72c9fafbc0b717d` | `27bb52ada6f9fff5efdb70721f14fd9e28550c8c093aac2f86323dac148d1de0` | — | needs the project owner's approval | no |
| `option-brain-20260922/hf_broker.py` | `735ed3abcd384a0f8f7af7b1690dc75eda3e244fa97368c8d000a24c9ba57245` | (same) | — | — | yes |
| `option-brain-20260922/launch_gate9.sh` | `13982dd564e172308f47135308a7961de7c2321a26b630e64c4375fcb90e64ca` | `9a6a4bdaec5d7ec2b049cac38fdf0c0a43e5a0b41156cabd86cd4c6bc58a08de` | `$AD_WORKSPACE`×2, `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/launch_smoke9.sh` | `0e9dd3401e9d30b11e6c83e85ba2b1b6d8f127a01584bfb86386387ace7a32a3` | `501e54fb6ad427e38c971961611ec08bc6a84323743b8cb39a00664f250e3576` | `$AD_WORKSPACE`×2, `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/live_runner.py` | `d6d4a6ba4ad451e11830f8b8893d9875e628e6168e84ff6b313070c855d7f623` | `53500f98c4609fefae007317ae5e5311153d0b6e7538a61a8c51d1793ada8893` | `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/make_exam9_freeze.sh` | `c9b63a81a14d76d3fa24898305abf1ff9318f1a546cd39726a896befb4fbca39` | `b1be016230fbc3ba3ecab26aead7671f5fe3dc02a83e9577c399a014fb94e676` | `$AD_WORKSPACE`×2, `$AD_HOME`×3 | — | yes |
| `option-brain-20260922/menu_r9.py` | `4c14eb63aca47ad8a37000f82d5b4d942db2071cccf1146d75619ba6e2b42872` | (same) | — | — | yes |
| `option-brain-20260922/option_sft.py` | `6f1ed7bdd156acb431dd708a76c78b70f6400a2a9d597b0123885fbb7862cbfe` | `ff65f1b8381a88fbe128f40c7cada2b66ed01453188e8a298a9860cb83d43491` | `$AD_ROOT`×3, `$AD_HOME`×1, `$AD_GPU_UUID`×1 | — | yes |
| `option-brain-20260922/options.py` | `6e343eedd9ff4db115a196fdaf5176c90bc40f234380fac6b664bbb3d21e72a1` | `8682c1fcc00fa4ae8d39b5dfd5329b4b0f28a8ea9d39a1c4b6595d19c93dbd3f` | `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/quest_lottery.py` | `1dd16eae0f33bd2d6503d205eef31af61d3659e59c74a4e5ef22673358170966` | (same) | — | — | yes |
| `option-brain-20260922/render.py` | `201e917b20c28fc4b711da3d8284edd676f8df6955248bc2bdb6bc4e5e4400e0` | (same) | — | — | yes |
| `option-brain-20260922/rollin_broker.py` | `971625c09751e542a1baa7b54d451aa0545373cf70e6b87fbf92c6d2c6a0359d` | (same) | — | — | yes |
| `option-brain-20260922/run_exam9_chain.sh` | `4a390e61060fd9b456c05bfd811fefedd545e2268992aadc95b03303bdc8b78a` | `2931ae4294eb7061fba3d7b45e2e8b8beae5e3c257635e41280647dd08a620a3` | `$AD_WORKSPACE`×2, `$AD_ROOT`×1 | — | yes |
| `option-brain-20260922/run_exam9_wave.sh` | `14ef7c9e30bfa72e1baaa9ff73b96f94ea339deea08de17dc471f521f59678b1` | `9fedd4da759bb9aec7950d06b888916c93464b474041cc5e5bdbb973c6a9f4df` | `$AD_WORKSPACE`×1 | — | yes |
| `option-brain-20260922/run_mixed_v4.sh` | `67045a757a1ffbfbb87cdd99fe764390b13c94e1fb0aeb34e5c92566b802ab4a` | `9c9f01a76b8ffb48d46c4af8b5db79d58a3388dced65388ff943cc78322dc7bf` | `$AD_WORKSPACE`×1, `$AD_ROOT`×5, `$AD_HOME`×2 | — | yes |
| `option-brain-20260922/score_exam9.py` | `fa276de768e01c3ebd7bab99b8f8add1c8d35a76209fc4c20b8dbdb2720368f7` | (same) | — | — | yes |
| `option-brain-20260922/seed_audit9.py` | `8ee8a1953881bb1718c8d5715c4c7f5039e1423f1d855c7e2c87a1f39a3a9582` | (same) | — | — | yes |
| `option-brain-20260922/TEACHER.md` | `1866da9673e484ecb426c2fa08382bc92e8db05c715a960ea7a7bd5ed50c737e` | (same) | — | — | yes |
| `option-brain-20260922/undo_guard.py` | `abf4a10cd491cf77507142073b36c282dff04b6118b272d5ff2012ea8ca7957e` | (same) | — | — | yes |
| `replay/replay_verify.py` | `452b14b8bf22826a9db4ba1ebf4d8d932624247455835ec3df522eded750b417` | `7810c3ff5ac756b1a132aa33642d485c1c8d8b97a59f67039d11fff5287de5e2` | `$AD_WORKSPACE`×1, `$AD_ROOT`×1, `$AD_HOME`×1 | — | yes |
| `skeleton-king-dual-brain-20260921-r16/session.py` | `af06fb2623b3aac127daa4e6a43ae1d7c364ac41b5b971b3e88f2db0fcba68c6` | `2ec43612c85940028c003b6be696b9e81ffc656cedc94825911d3e40da3dde31` | `$AD_HOME`×3 | — | yes |
| `strategist-rl-hands-20260921/goal.py` | `b7c20144857cc57e1237040949988a916aa5deddd5852f2071837b050a418d8c` | (same) | — | — | yes |
| `strategist-rl-hands-20260921/model.py` | `0fbcfc9c9078c1ba1ce0f20d187ca46f19df99568d46b1aa717c7ebb210727e0` | (same) | — | — | yes |
| `strategist-rl-hands-20260921/public_view.py` | `6d9c792f27c22154cf20b640697b8deb38ad4ea4fb43077105247d27ac278197` | (same) | — | — | yes |
| `strategist-rl-hands-20260921/runtime.py` | `7cf1cfedbff150eed8f8fcae507866daa44e12e2bc6745a0c241f0b0059da5f9` | `4ebb139cfd76a2a49428cc0b0b5d9bb8239d111efc35e97f70de68a3c94183d2` | `$AD_HOME`×2 | — | yes |
| `strategist-rl-king-continuation-20260921/config.py` | `eaf9434cecbcf525f6a126f444132985875bfbfc655fc0fcb032bec3a21aca1d` | `b20e56b09d1a1fc49d89a10fb5ac6a3502977d1bae7825219b41b90db75ab823` | `$AD_HOME`×8, `$AD_GPU_UUID`×1 | — | yes |
| `strategist-rl-king-continuation-20260921/navigation.py` | `365eeb1f74cd1950abda7e8544e4ebd203191c48de43ce60b987f49ceff941eb` | (same) | — | — | yes |
| `strategist-rl-king-continuation-20260921/revision1/execution_r1.py` | `9d2b04f56e7f789d4c429413a54edb86a072f6b5ffeb0a63b87be1f932754306` | (same) | — | — | yes |
| `strategist-rl-king-continuation-20260921/revision2/trade_runtime.py` | `9701777d807183018124cabbe7803f7ff8b9d3b24d1eb10acfe0e05638d7f5a8` | (same) | — | — | yes |
| `strategist-rl-king-continuation-20260921/revision3/telemetry_runtime.py` | `f93b25996ebe70cec803a61d54088dd4ae816d38adc667c6233a00ef721c7157` | (same) | — | — | yes |
| `strategy-brain-sft-20260922/common.py` | `ea71f9e7dd5617b03bc7747351400f94beb9d04b55699268dc29eb65f1d950b6` | `3c721956686aae7a1c1559717378ca6aa50de9699a8b6ebd8774411f1512e288` | `$AD_HOME`×1, `$AD_GPU_UUID`×1, `$AD_GPU2_UUID`×1 | — | yes |
| `strategy-brain-sft-20260922/game_executor_r2.py` | `0ecce987eed084754fd01177cbc694f1805254b0e54853558c50c0fe54987eca` | (same) | — | — | yes |
| `strategy-brain-sft-20260922/game_support_r3.py` | `dcc1fa9f64761b363e606d00e3fb58fccfda89de8b44dd6bbce3558c6990c018` | (same) | — | — | yes |
| `strategy-brain-sft-20260922/protocol.py` | `b8edfb14fc202b33dcf013a4095e29ba2fe35edd91ca8601636aca8b45a0a96a` | (same) | — | — | yes |
| `strategy-brain-sft-20260922/purchase_audit.py` | `2d5b69f7a760984247f54d5b94b021b34ca3d1df8c6c4f6fcbb77ba34e4ede6e` | (same) | — | — | yes |
| `teacher/TEACHER.before-r9-132005.md` | `dd75b2e679c599aa042ca42333e93e84436fbc3df7bfd396d822e8439c91e54b` | (same) | — | — | yes |
| `teacher/TEACHER.before-v3-054308.md` | `540f763c272ff310ef781ac83c4594618f6bde65045b6abb91bf549adc142619` | (same) | — | — | yes |

Freeze status: all files above are in the freeze list with the same original sha256, except:

- `option-brain-20260922/TEACHER.md`: not in the freeze list; its sha256 is recorded in PREREG-ROUND9 revision 2 (teacher guidance for the round-9b-9d labels of the new arm).
- `replay/replay_verify.py`: not in the freeze list; written after the exam freeze for the round-8 replays and used unchanged for the round-9 replays (sha256 recorded in both replay summaries).
- `teacher/TEACHER.before-r9-132005.md`: not in the freeze list; one of the two guidance versions closest in time to the round-8 labels of d9facts3 (the exact text used cannot be confirmed).
- `teacher/TEACHER.before-v3-054308.md`: as above.

## Runtime dependencies that are versions of this repository's own files

The game process imports four modules of a frozen copy of this repository's `python/diablogym` package (dated 2026-07-25 to 2026-09-09) and `leashed_ppo.py` of a frozen copy of `train/`. Three of the five differ from the files at this commit. `runtime-deps/runtime-deps.patch` turns the files of this commit into the runtime versions:

```
git apply option_brain/runtime-deps/runtime-deps.patch   # in a separate copy of this repository
```

| Runtime file | Repository file | Original sha256 | After the patch | Note |
|---|---|---|---|---|
| `diablogym/controller_wire.py` | `python/diablogym/controller_wire.py` | `7ff86ea8ff7547e5a7e219bbd7d1ca716918904ca3b1d2b515c1446dae166095` | equal | identical to this commit, no hunk |
| `diablogym/env.py` | `python/diablogym/env.py` | `edac8479396648fb7c21b182407fffa3a643fe57d0909ed363a4f52779fd3bfa` | equal | changed by the patch |
| `diablogym/nav.py` | `python/diablogym/nav.py` | `15d79605a958c5239a171c1d06e3a0c8967dc73603d75d3c61c24db2c0648b86` | equal | identical to this commit, no hunk |
| `diablogym/options_env.py` | `python/diablogym/options_env.py` | `6b2cdc2b10de8ca92ef341bbdde3eaa81e3ddd66bcd9c3ca9229430d25cb32bd` | `0f9366f9a2fa671f8ab24365ad03a5a77e46c2ce2cf754936e138ad2b0c180cf` | changed by the patch, except line 3 (a docstring naming who approved a design), which is kept as in this repository |
| `train/leashed_ppo.py` | `train/leashed_ppo.py` | `4a9c3dcaf8a84550c0397b7c80327d438e434982ef6d7a44923ba0b484857cfe` | equal | changed by the patch |

`runtime-deps.patch` sha256: `82932103e835d2168088643687b240ea4f69bc9efb6da973db27a12f77701591`. Apply it with `core.autocrlf=false`; on Windows the default `true` rewrites line endings and the sha256 values then differ.

**Freeze gap.** The exam freeze list covers `leashed_ppo.py` (same file, same sha256) and, by content, `controller_wire.py` and `nav.py`, but **not the runtime copies of `env.py` and `options_env.py`**. Its import trace imported `leashed_ppo`, which imported the development checkout's `diablogym` (the editable install of the game environment) before the runtime's own `install_package` had run, so the list recorded that checkout's `diablogym` files instead. The game process itself loads the frozen copy above: `install_package` runs before the executor loads `leashed_ppo` and refuses to run if `diablogym` is already imported. The sha256 values above were taken on 2026-09-24, after the exam; the files' modification times are 2026-07-25 to 2026-09-09, before the exam.

## Native sources

See `native/README.md`. The bridge sources are byte-identical to the build record of the bridge used in every exam game; the two build records are sanitized summaries (paths replaced by placeholders).

| File | sha256 |
|---|---|
| `native/bridge-r3/butcher_event.hpp` | `757810978cfeaa8c1d01b30cdac0f780243043bb2fa9b08e61f335ad61f2d4c3` |
| `native/bridge-r3/combat_telemetry.hpp` | `48996fe56f75f45e1e30110b1b6707c2a183c233af15454d9143a3744f3e537e` |
| `native/bridge-r3/diablogym.cpp` | `68491ca15749f9df5e66e0095eb68bebb44ece8604b9aa2d8c0671b4a4468ad7` |
| `native/bridge-r3/field_equipment.hpp` | `08e881b539317cd8f3132159c00c7d7f73f8c474893bd1a05456a9d9c5229cf9` |
| `native/bridge-r3/gear_wear_fixture.hpp` | `3186820049ff21192e03175dcbc871d2cd67d1e88375af863c1fb82c612aa0f7` |
| `native/bridge-r3/manual_control.hpp` | `7b35b4d12515efb98b9db46557818a2a7593af062390bd83a903b8a2e63d0fe6` |
| `native/bridge-r3/preparation_goal.hpp` | `ff7582480116a1d73e8e6e823505281fe51f8ba37fecac409c6d136191252fbc` |
| `native/bridge-r3/quest_entrance_visibility.hpp` | `f4bf85166ae04a24a2a5e10e22369653f220a3e9320acb29b3d57a47f6ead377` |
| `native/bridge-r3/resource_combinations.hpp` | `f7a4646e5dfd8d8c3bce290f788fa2694aa7d5e4fc2aea7d108a817ece27db85` |
| `native/bridge-r3/resource_identify.hpp` | `c4b8a145ee0de86f8e949c97d801e2d0324dc8156ca76bb73f6bfa491c833cad` |
| `native/bridge-r3/resource_loot.hpp` | `6c5f2ade962b1157bcb346e403965f1209a40256c615e11ec1c92a08c97420ca` |
| `native/bridge-r3/resource_protocol.hpp` | `b29d617209628b85b8f760a498a387f5e2b97ba92a334187c08399fd34d02be2` |
| `native/bridge-r3/resource_sweep.hpp` | `3446248045655b8fdd6c10a14050adb5c25904cd1f758d158c67a218e5f5b404` |
| `native/bridge-r3/resource_trip_quota.hpp` | `f0520366849be3e83eb76318c9cbec682ca66c66392004f539e0003533474ce1` |
| `native/bridge-r3/shop_catalog_scope.hpp` | `395fd1d466a22a2fd8873a2d886a892a46d37125cad905fc469d6b2add4e5ebd` |
| `native/bridge-r3/unbelt_exact.hpp` | `1d6e4a4fc4d985c0a2ac8cad4796ea2e41e6aa7370d55c7224a6e138283120ac` |
| `native/bridge-r3/witch_trade.hpp` | `83a577fddf02b3a904e169fdc2755ca50b75ee95e1cd2456df71f21fc872df94` |
| `native/bridge-r3-build.json` | `ca3bd57deb112ef0d8701e8c97ea921dcd14b9dfbb972d4489296f4ee8fcce52` |
| `native/engine-r11-build.json` | `26bb95f99ae562fa7678258b3bd4d6869c6f48371faca227dffa6464457c5ad3` |
| `native/engine-r11-loadmonster.patch` | `f515612aae8306bf39d02236d728f8f75fa9baebc9bb1445ff8bdb72a588ee41` |

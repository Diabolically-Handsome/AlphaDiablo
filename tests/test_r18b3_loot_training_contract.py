"""R18-B3 (2026-09-07): sustain-loot-v1 进入工人训练的完成时钟。

主席今晨裁定「进城卖装备是滚雪球计划的核心」——训练世界必须等于被测世界,
因此 L1 两趟战利品经济(sustain-loot-v1)与 sustain-v6 并列为一等训练服务法。

本卷盯住的法条:

* 窗口(WorkerWindowEnv)在 earned-dive-suffix-v1 下同时接受 sustain-v6 与
  sustain-loot-v1;loot 只在完成时钟(completion-l2-v1 / completion-l2-r18c)
  下成立,配旧 legacy 时钟一律 fail closed;
* 训练器(train/train_ppo.py)的时钟校验、earned 前缀钉死值与 CLI 选项都admit
  loot,而 sustain-v5 之类照旧被拒;
* 配方(train/eval_contract.py)缺省输出与 R18-B3 之前逐字节相同——旧调用者
  拿到的仍是同一个 dict(键序在内);带 completion-l2-r18c 时才换新版本号;
* 迁移操作名(train/migrate_resource_candidate.py)给 loot 一个全新版本串,
  sustain-v6 的旧串一字未动。

窗口一侧沿用 tests/test_completion_r18c_training.py 的私有包别名 + bridge 桩,
不 build/reset/step 任何原生引擎。
"""
import argparse
import ast
import contextlib
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "python"), str(ROOT / "tests")]

PACKAGE = "_r18b3_loot_pure"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "python/diablogym")]
    sys.modules[PACKAGE] = package
if PACKAGE + ".bridge" not in sys.modules:
    bridge_stub = ModuleType(PACKAGE + ".bridge")
    bridge_stub.WM_DIABPREVLVL = 1027  # a fake engine trigger id; never used here
    sys.modules[PACKAGE + ".bridge"] = bridge_stub

worker_env = importlib.import_module(PACKAGE + ".worker_env")
clock = importlib.import_module(PACKAGE + ".completion_clock")

import eval_contract
import migrate_resource_candidate as migration
import train_ppo as training

V1 = "completion-l2-v1"
R18C = "completion-l2-r18c"
PROTOCOLS = (V1, R18C)
SCOPE = "earned-dive-suffix-v1"
LOOT = "sustain-loot-v1"
ARMOR = "sustain-v6"
ARRIVAL = clock.COMPLETION_L2_V1.arrival_microsteps  # 12000, shared by both recipes
R16 = "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0"


# R18-B3 之前 resource_service_recipe("l2-town-v1", "full", "sustain-loot-v1")
# 的逐字节输出(含键序),抄成字面量钉在此处:缺省路径永不许漂移。
OLD_LOOT_RECIPE = {
    'version': 'l1-two-trip-loot-economy-v1',
    'service_policy': 'sustain-loot-v1',
    'mode': 'full',
    'stages': ['collect_observed_gold_and_loot',
               'town_trip',
               'sell_idle_smith_equipment',
               'observe_smith',
               'observe_healer_and_native_heal',
               'joint_minimum_readiness_budget',
               'normal_unequip_repair_or_replace',
               'potions_to_readiness',
               'return_original_l1'],
    'planning_scope': 'observed_single_ordinary_armor_retained_repairs_minimum_medicine',
    'service_microstep_cap': 3000,
    'potion_target': 4,
    'potion_target_source': 'native_readiness.required_belt_heals',
    'navigation_recovery': 'bounded-combat-heal-v1',
    'gold_memory': {'version': 'observed-l1-gold-memory-v1',
                    'source': 'existing-native-main-l1-observations',
                    'identity_fields': ['seed_hi', 'seed_lo', 'create_info', 'base_id'],
                    'pickup': 'normal-revisit-current-active-id-revalidation',
                    'remembered_value_is_wallet': False},
    'collect_budget_scope': 'new-collect-commands-within-actual-microstep-window',
    'collect_budget_exhausted': 'continue_outbound',
    'collect_command_window_microsteps': 900,
    'collect_tail': 'finish-existing-animation-in-outbound-charged-to-service-budget',
    'collect_cutoff': 'record-current-native-state-at-command-window-cutoff',
    'native_ordinary_armor_scope': True,
    'ordinary_armor_catalog': {
        'version': 'ordinary-armor-v1',
        'items': ['normal_helmet', 'normal_shield', 'normal_chest'],
        'projection': 'native_full_readiness_actual_replaced_slots',
        'projection_refresh': 'known_smith_identities_live_player_before_each_joint_plan',
        'projection_refresh_native_microsteps': 0,
        'selection': 'effective_blocking_if_complete_affordable_then_lowest_total',
        'blocking_is_native_readiness_gate': False,
        'missing_block_projection': 'unknown',
        'partial_free_equip_fallback': False,
        'missing_quote_or_projection': 'unresolved_not_physical_impossibility'},
    'native_preserve_equipment_readiness': True,
    'equipment_readiness_preservation': {
        'version': 'equipment-readiness-preservation-v1',
        'scope': 'native-a14-eligibility-plan-and-commit',
        'equipment_failures': ['armor', 'damage', 'weapon', 'durability'],
        'trigger': 'current-equipment-subset-ready',
        'requirement': 'next-equipment-subset-ready',
        'independent_failures': ['level', 'health', 'potions'],
        'unready_equipment': 'legacy-upgrade-rule'},
    'time_protocol': 'completion-l2-v1',
    'max_town_trips': 2,
    'native_loot_economy': True,
    'second_trip_trigger': 'native-deficit-and-real-resource-or-growth-change-after-return',
    'loot_collection': 'observed-non-upgrade-smith-equipment-normal-inventory',
    'sale': 'on-site-identity-and-price-bound-native-smith-inventory-sale',
    'sale_income': 'actual-personal-inventory-gold-receipt-only',
    'gear_replacement': 'retain-replaced-items-or-reject-no-room'}


# ---------------------------------------------------------------------------
# (a)(b) 训练窗口:WorkerWindowEnv
# ---------------------------------------------------------------------------

class StubOptions:
    """Enough of a completion-protocol OptionsEnv for the wrapper; no engine."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_steps = kwargs.get("max_steps", 6000)
        self.env = SimpleNamespace(
            max_steps=ARRIVAL, _steps=0, _resource_actual_microsteps=0, _raw=None,
            _completion_prefix_active=False, _completion_prefix_deadline=None,
            observation_space=SimpleNamespace(shape=(295,)))
        self._completion_clock = SimpleNamespace(
            state=SimpleNamespace(physical_deadline=ARRIVAL))


def prefix_callback():
    """The verified R16 parent's public surface; the wrapper only inspects it."""
    def callback(observation, action_mask):
        return 9
    callback.source_sha256 = worker_env._PREFIX_WORKER_SHA256
    callback.diablogym_worker_observation_view = "dual-v4-asymmetric-v3"
    callback.diablogym_worker_action12_mode = "environment-mask"
    callback.episode_reseed = Mock()
    callback.on_beat = None
    return callback


@contextlib.contextmanager
def patched_options(stub=True):
    """R18-B3 复审 (2026-09-07):OptionsEnv 的桩改由调用方持有。

    从前 build() 自己开一层 patch.object,把调用方那只 Mock 遮住了,于是
    `options.assert_not_called()` 盯的是一只**永远不会被调用**的 Mock ——
    「loot + legacy 在构造 OptionsEnv 之前就死」这条断言当时是空的:把守卫
    挪到 `self.oe = OptionsEnv(...)` 之后,整卷照样全绿。现在正负两路观察的
    是同一只 Mock,顺序断言才真的有牙。
    """
    side_effect = (lambda **kw: StubOptions(**kw)) if stub else None
    with patch.object(worker_env, "OptionsEnv", side_effect=side_effect) as options:
        yield options


def build(protocol, policy, **overrides):
    """构造 WorkerWindowEnv;必须在调用方的 patched_options() 之内调用。"""
    assert isinstance(worker_env.OptionsEnv, Mock), (
        "OptionsEnv 必须由调用方打桩,否则顺序断言又会变空")
    arguments = dict(
        manager_npz=None, manager_heuristic="readiness-v1", max_steps=6000,
        learning_window_scope=SCOPE, policy_observation_view="dual-v4-asymmetric-v3",
        resource_protocol="l2-town-v1", resource_purchase_mode="full",
        resource_service_policy=policy, prefix_worker=prefix_callback(),
        prefix_worker_sha256=worker_env._PREFIX_WORKER_SHA256,
        prefix_max_attempts=2, prefix_max_microsteps=100,
        worker_time_protocol=protocol)
    arguments.update(overrides)
    return worker_env.WorkerWindowEnv(**arguments)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_window_accepts_loot_under_either_completion_clock(protocol):
    with patched_options() as options:
        env = build(protocol, LOOT)
    assert env.resource_service_policy == LOOT
    assert env.worker_time_protocol == protocol
    options.assert_called_once()
    assert env.oe.kwargs["resource_service_policy"] == LOOT
    assert env.oe.kwargs["worker_time_protocol"] == protocol


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_window_still_accepts_sustain_v6_unchanged(protocol):
    with patched_options() as options:
        env = build(protocol, ARMOR)
    assert env.resource_service_policy == ARMOR
    options.assert_called_once()
    assert env.oe.kwargs["resource_service_policy"] == ARMOR
    assert env.oe.kwargs["worker_time_protocol"] == protocol


def test_window_refuses_loot_under_the_legacy_clock():
    """战利品经济没有旧时钟版本:loot + legacy 在构造 OptionsEnv 之前就死。"""
    with patched_options(stub=False) as options:
        with pytest.raises(ValueError,
                           match="sustain-loot-v1 requires an explicit completion-l2"):
            build("legacy", LOOT)
        options.assert_not_called()


@pytest.mark.parametrize("policy", ["legacy-v1", "sustain-v5", "sustain-v4"])
@pytest.mark.parametrize("protocol", [*PROTOCOLS, "legacy"])
def test_window_still_refuses_every_other_service_policy(policy, protocol):
    with patched_options(stub=False) as options:
        with pytest.raises(ValueError, match="earned-dive-suffix-v1 requires"):
            build(protocol, policy)
        options.assert_not_called()


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("key,value", [("resource_protocol", "off"),
                                       ("resource_purchase_mode", "armor")])
def test_loot_still_requires_l2_town_v1_and_full(protocol, key, value):
    with patched_options(stub=False) as options:
        with pytest.raises(ValueError, match="requires l2-town-v1"):
            build(protocol, LOOT, **{key: value})
        options.assert_not_called()


# ---------------------------------------------------------------------------
# (c) 训练器:时钟校验 + earned 前缀钉死值
# ---------------------------------------------------------------------------

def time_config_args(policy, protocol=R18C):
    return dict(worker=True, worker_learning_window_scope=SCOPE,
                resource_protocol="l2-town-v1", resource_purchase_mode="full",
                resource_service_policy=policy, max_steps=6000, farm_scene_cap=3600)


@pytest.mark.parametrize("policy", [ARMOR, LOOT])
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_validate_worker_time_config_admits_loot_and_armor(policy, protocol):
    recipe = training._validate_worker_time_config(protocol, **time_config_args(policy))
    assert recipe == clock.COMPLETION_RECIPES[protocol].as_dict()


@pytest.mark.parametrize("policy", ["sustain-v5", "sustain-v4", "legacy-v1"])
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_validate_worker_time_config_still_rejects_the_other_policies(policy, protocol):
    with pytest.raises(ValueError, match="sustain-v6 or sustain-loot-v1"):
        training._validate_worker_time_config(protocol, **time_config_args(policy))


@pytest.mark.parametrize("scope", ["farm-only", "farm-dive-v1"])
def test_loot_needs_a_completion_clock_in_every_learning_window_scope(scope):
    """R18-B3 复审 (2026-09-07):这条法与作用域无关。

    从前 farm-only / farm-dive-v1 下的 loot + legacy 走完整个参数校验层,直到
    OptionsEnv 构造(原生引擎已启动、run 目录已落盘)才死。
    """
    args = SimpleNamespace(worker=True, worker_learning_window_scope=scope,
                           resource_protocol="l2-town-v1", resource_purchase_mode="full",
                           resource_service_policy=LOOT, max_steps=6000,
                           farm_scene_cap=3600, worker_time_protocol="legacy")
    with pytest.raises(ValueError,
                       match="sustain-loot-v1 requires an explicit completion-l2"):
        training._validate_worker_time_args(args)


@pytest.mark.parametrize("scope", ["farm-only", "farm-dive-v1"])
@pytest.mark.parametrize("policy", ["legacy-v1", "sustain-v5", ARMOR])
def test_the_non_earned_scopes_are_otherwise_untouched(policy, scope):
    args = SimpleNamespace(worker=True, worker_learning_window_scope=scope,
                           resource_protocol="l2-town-v1", resource_purchase_mode="full",
                           resource_service_policy=policy, max_steps=6000,
                           farm_scene_cap=3600, worker_time_protocol="legacy")
    assert training._validate_worker_time_args(args) is None


def prefix_cli_args(policy=ARMOR, protocol="legacy", **overrides):
    """R18-B2 的 earned 前缀 CLI 参数,逐字沿用注册父代的钉死值。"""
    args = dict(
        worker_policy_observation_view="dual-v4-asymmetric-v3", reward_economy="v4",
        farm_scene_cap=3600, worker_action14_logit_bonus=2.5,
        worker_dive_action11_logit_bonus=2.0, worker_potion_action13_logit_bonus=2.0,
        worker_hp_loss_price=0.1, worker_potion_pickup_bonus=2.0,
        worker_no_progress_timeout_credit="zero",
        worker_descend_escrow_fraction=0.5, worker_descend_escrow_power=1.6,
        worker_descend_escrow_readiness_gate=True, reset_layer_clock_on_window=True,
        worker=True, algo="mppo", device="cpu", seed=2164000, max_steps=6000,
        worker_learning_window_scope=SCOPE, worker_prefix_model="registered-parent.zip",
        worker_prefix_max_attempts=3, worker_prefix_max_microsteps=18000,
        resource_protocol="l2-town-v1", resource_purchase_mode="full",
        resource_service_policy=policy, dive_blocker_recovery="adjacent-v1",
        drink_sovereignty=True, distill_beta=0.0, bc_aux_lambda=0.0,
        bc_aux_demos=None, bc_aux_liveness_preflight=False,
        worker_time_protocol=protocol)
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_fixed_prefix_dict_accepts_loot_under_a_completion_clock(protocol):
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        training._validate_worker_prefix_args(prefix_cli_args(LOOT, protocol))


@pytest.mark.parametrize("protocol", ["legacy", *PROTOCOLS])
def test_fixed_prefix_dict_still_accepts_sustain_v6_on_any_clock(protocol):
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        training._validate_worker_prefix_args(prefix_cli_args(ARMOR, protocol))


def test_fixed_prefix_dict_refuses_loot_on_the_legacy_clock():
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        with pytest.raises(ValueError,
                           match="sustain-loot-v1 requires an explicit completion-l2"):
            training._validate_worker_prefix_args(prefix_cli_args(LOOT, "legacy"))


@pytest.mark.parametrize("policy", ["sustain-v5", "sustain-v4", "legacy-v1"])
def test_fixed_prefix_dict_still_refuses_sustain_v5_and_friends(policy):
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        with pytest.raises(ValueError, match="sustain-v6 or sustain-loot-v1"):
            training._validate_worker_prefix_args(prefix_cli_args(policy, R18C))


@pytest.mark.parametrize("key,value", [
    ("device", "cuda"), ("worker", False), ("seed", None), ("max_steps", 10000),
    ("distill_beta", 1.0)])
def test_the_prefix_requires_around_the_fixed_dict_are_untouched_for_loot(key, value):
    """`fixed` 之外的那几条 _require(设备/工人/种子/步长/蒸馏)对 loot 同样成立。"""
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        with pytest.raises(ValueError):
            training._validate_worker_prefix_args(prefix_cli_args(LOOT, R18C, **{key: value}))


def prefix_fixed_dict():
    """把 train_ppo._validate_worker_prefix_args 里的 `fixed` 钉死表原样取出来。

    R18-B3 复审 (2026-09-07):从前这里是一张手抄的 12 条清单,其中只有 7 条
    真是 `fixed` 的键,余下 9 个钉死值从没在 loot 路径上被试过。改成由字典本身
    驱动:将来新增的钉死键自动进卷,漂移立刻红。
    """
    tree = ast.parse((ROOT / "train/train_ppo.py").read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_validate_worker_prefix_args")
    assign = next(node for node in function.body if isinstance(node, ast.Assign)
                  and getattr(node.targets[0], "id", None) == "fixed")
    return ast.literal_eval(assign.value)


def drifted(value):
    """一个必定不合法的取值(元组 = 集合成员,取一个非成员)。"""
    if isinstance(value, tuple):
        return "-or-".join(value) + "-not-a-member"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return str(value) + "-drifted"


FIXED = prefix_fixed_dict()


@pytest.mark.parametrize("key", sorted(FIXED))
def test_every_pinned_value_in_the_fixed_dict_is_untouched_for_loot(key):
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        with pytest.raises(ValueError, match=key):
            training._validate_worker_prefix_args(
                prefix_cli_args(LOOT, R18C, **{key: drifted(FIXED[key])}))


def test_the_fixed_dict_covers_the_whole_registered_pin_list():
    """钉死表既没缩水也没换位:R16 注册父代的 17 个键,顺序与主树一致。"""
    assert len(FIXED) == 17
    assert list(FIXED)[:4] == ["resource_protocol", "resource_purchase_mode",
                               "resource_service_policy", "dive_blocker_recovery"]
    assert FIXED["resource_service_policy"] == (ARMOR, LOOT)


def test_an_already_invalid_caller_still_gets_the_same_first_error_as_before_b3():
    """R18-B3 复审 (2026-09-07):服务法留在 `fixed` 原位 ⇒ 首个报错不变。

    主树对 (resource_protocol='off', resource_service_policy='sustain-v5') 先喊
    resource_protocol;把成员检查提到循环之前会让它先喊服务法。
    """
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        with pytest.raises(ValueError, match=r"requires resource_protocol='l2-town-v1'"):
            training._validate_worker_prefix_args(prefix_cli_args(
                "sustain-v5", R18C, resource_protocol="off",
                resource_purchase_mode="armor"))


# ---------------------------------------------------------------------------
# (d) CLI 选项
# ---------------------------------------------------------------------------

def cli_node(flag):
    tree = ast.parse((ROOT / "train/train_ppo.py").read_text(encoding="utf-8"))
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and node.value.args
             and isinstance(node.value.args[0], ast.Constant)
             and node.value.args[0].value == flag]
    assert len(nodes) == 1
    ap = argparse.ArgumentParser()
    exec(compile(ast.Module(nodes, type_ignores=[]), "<actual-cli-node>", "exec"), {"ap": ap})
    return ap


def test_cli_admits_loot_and_keeps_every_old_choice_and_default():
    ap = cli_node("--resource-service-policy")
    assert ap.parse_args([]).resource_service_policy == "legacy-v1"
    for policy in ("legacy-v1", "sustain-v2", "sustain-v3", "sustain-v4",
                   "sustain-v5", "sustain-v6", LOOT):
        assert ap.parse_args(
            ["--resource-service-policy", policy]).resource_service_policy == policy
    for rejected in ("sustain-loot", "sustain-loot-v2", "loot", "sustain-v7"):
        with pytest.raises(SystemExit):
            ap.parse_args(["--resource-service-policy", rejected])


# ---------------------------------------------------------------------------
# (e)(f) 配方版本号
# ---------------------------------------------------------------------------

def test_the_default_loot_recipe_is_byte_identical_to_the_pre_b3_dict():
    recipe = eval_contract.resource_service_recipe("l2-town-v1", "full", LOOT)
    assert recipe == OLD_LOOT_RECIPE
    assert list(recipe) == list(OLD_LOOT_RECIPE)          # 键序也不许动
    assert eval_contract.resource_service_recipe("l2-town-v1", "full", LOOT,
                                                 time_protocol=V1) == OLD_LOOT_RECIPE


def test_the_r18c_variant_moves_only_the_version_and_the_clock():
    old = eval_contract.resource_service_recipe("l2-town-v1", "full", LOOT)
    new = eval_contract.resource_service_recipe("l2-town-v1", "full", LOOT,
                                                time_protocol=R18C)
    assert new["time_protocol"] == R18C
    assert new["version"] == "l1-two-trip-loot-economy-v1-r18c"
    assert old["version"] == "l1-two-trip-loot-economy-v1"
    assert {key for key in set(old) | set(new)
            if old.get(key) != new.get(key)} == {"version", "time_protocol"}
    assert list(new) == list(old)


def test_the_registered_clock_versions_are_exactly_the_completion_protocols():
    assert (tuple(eval_contract.LOOT_SERVICE_RECIPE_VERSIONS)
            == tuple(clock.COMPLETION_PROTOCOLS))
    assert (eval_contract.LOOT_SERVICE_DEFAULT_TIME_PROTOCOL == V1
            and len(set(eval_contract.LOOT_SERVICE_RECIPE_VERSIONS.values()))
            == len(eval_contract.LOOT_SERVICE_RECIPE_VERSIONS))


@pytest.mark.parametrize("value", ["legacy", "completion-l2", "completion-l2-v2",
                                   "completion-l2-r18d", "", 1, True])
def test_an_unregistered_clock_fails_closed_for_loot(value):
    with pytest.raises(ValueError, match="registered completion-l2 time protocol"):
        eval_contract.resource_service_recipe("l2-town-v1", "full", LOOT,
                                              time_protocol=value)


def test_the_not_supplied_sentinel_is_the_registered_default_for_loot():
    """R18-B3 复审:哨兵是 None(未传),不是「缺省时钟值」——见下一条。"""
    assert eval_contract.resource_service_recipe(
        "l2-town-v1", "full", LOOT, time_protocol=None) == OLD_LOOT_RECIPE


@pytest.mark.parametrize("value", [V1, R18C, "legacy", "", 1, True])
@pytest.mark.parametrize("policy", ["legacy-v1", "sustain-v5", ARMOR])
def test_any_explicit_clock_on_a_non_loot_policy_fails_closed(policy, value):
    """R18-B3 复审 (2026-09-07):连缺省时钟值也不许静默落在非 loot 服务法上。

    从前 time_protocol 的哨兵就是 completion-l2-v1 本身,于是
    resource_service_recipe(...,"sustain-v6",time_protocol="completion-l2-v1")
    被静默接受并忽略,而同一写法配 completion-l2-r18c 却报错。
    """
    assert eval_contract.resource_service_recipe("l2-town-v1", "full", policy) is not None
    with pytest.raises(ValueError, match="only applies to sustain-loot-v1"):
        eval_contract.resource_service_recipe("l2-town-v1", "full", policy,
                                              time_protocol=value)


# ---------------------------------------------------------------------------
# R18-B3 复审:评测档案身份仍然没有时钟键 —— loot 档案一律 fail closed
# ---------------------------------------------------------------------------

LOOT_ARCHIVE_ENVIRONMENT = {
    "resource_protocol": "l2-town-v1", "resource_service_policy": LOOT,
    "resource_readiness_law": "coach-v03", "dive_blocker_recovery": "adjacent-v1",
    "farm_scene_cap": 3600, "reset_layer_clock_on_window": True}


def test_a_loot_eval_archive_cannot_be_minted_under_the_default_clock():
    """训练侧 loot 配方随时钟分版,档案身份却没有时钟键——不许用缺省值作证。"""
    with pytest.raises(eval_contract.EvalContractError,
                       match="sustain-loot-v1 archives require the completion-l2 clock"):
        eval_contract.make_protocol([1, 2, 3],
                                    r16_environment=dict(LOOT_ARCHIVE_ENVIRONMENT))


def test_the_archive_validator_refuses_a_loot_r16_environment():
    with pytest.raises(eval_contract.EvalContractError,
                       match="does not carry yet"):
        eval_contract.validate_r16_environment(dict(LOOT_ARCHIVE_ENVIRONMENT))


def test_the_sustain_v6_archive_identity_is_untouched_by_that_refusal():
    environment = dict(LOOT_ARCHIVE_ENVIRONMENT, resource_service_policy=ARMOR)
    protocol = eval_contract.make_protocol([1, 2, 3], r16_environment=environment)
    assert protocol["resource_service_recipe"] == eval_contract.resource_service_recipe(
        "l2-town-v1", "full", ARMOR)
    assert "time_protocol" not in protocol["resource_service_recipe"]
    eval_contract.validate_r16_environment(environment)


def test_the_shipped_eval_cli_never_offered_loot_in_the_first_place():
    """档案生产端(train/eval_assembled.py)的 choices 里本就没有 loot。"""
    source = (ROOT / "train/eval_assembled.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and node.value.args
             and isinstance(node.value.args[0], ast.Constant)
             and node.value.args[0].value == "--resource-service-policy"]
    assert len(nodes) == 1
    choices = next(keyword.value for keyword in nodes[0].value.keywords
                   if keyword.arg == "choices")
    assert LOOT not in ast.literal_eval(choices)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_the_training_side_helper_states_the_real_clock(protocol):
    assert training._resource_service_recipe_for("l2-town-v1", "full", LOOT, protocol) == (
        eval_contract.resource_service_recipe("l2-town-v1", "full", LOOT,
                                              time_protocol=protocol))
    # 非 loot 的服务法走原调用:时钟一个字都不进配方。
    for policy in ("legacy-v1", "sustain-v5", ARMOR):
        assert training._resource_service_recipe_for("l2-town-v1", "full", policy, protocol) == (
            eval_contract.resource_service_recipe("l2-town-v1", "full", policy))


def test_the_training_side_helper_refuses_loot_on_the_legacy_clock():
    with pytest.raises(ValueError, match="registered completion-l2 time protocol"):
        training._resource_service_recipe_for("l2-town-v1", "full", LOOT, "legacy")


# ---------------------------------------------------------------------------
# 训练契约与续训身份
# ---------------------------------------------------------------------------

def loot_args(protocol=R18C, policy=LOOT):
    return SimpleNamespace(
        worker=True, options=False, flat_clock=False, arch="mlp", max_steps=6000,
        num_envs=4, n_steps=512, gamma=1.0, lr=1e-4, ent_coef=.005,
        skip_dry=False, no_drink_sovereignty=False, dry_curriculum_schedule=None,
        bc_aux_lambda=0., bc_aux_demos=None, bc_aux_liveness_preflight=False,
        distill_beta=0., calib_record_only=False, worker_learning_window_scope=SCOPE,
        resource_protocol="l2-town-v1", resource_purchase_mode="full",
        resource_service_policy=policy, farm_scene_cap=3600,
        worker_time_protocol=protocol, worker_prefix_model="unused-parent.zip",
        worker_prefix_max_attempts=64, worker_prefix_max_microsteps=384000,
        worker_depth_shaping_unit=24., reward_economy="v4")


def loot_contract(protocol=R18C, policy=LOOT):
    import gymnasium as gym
    model = SimpleNamespace(max_grad_norm=.5, action_space=gym.spaces.Discrete(15),
                            observation_space=gym.spaces.Box(-1, 1, (13012,)), device="cpu")
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        return training._training_contract(loot_args(protocol, policy), model, batch_size=256)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_the_training_contract_records_the_clock_the_loot_service_actually_ran(protocol):
    contract = loot_contract(protocol)
    assert contract["resource_service_policy"] == LOOT
    assert contract["worker_time_protocol"] == protocol
    assert contract["resource_service_recipe"]["time_protocol"] == protocol
    assert contract["resource_service_recipe"] == eval_contract.resource_service_recipe(
        "l2-town-v1", "full", LOOT, time_protocol=protocol)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_the_sustain_v6_contract_is_untouched_by_this_reform(protocol):
    contract = loot_contract(protocol, ARMOR)
    assert contract["resource_service_recipe"] == eval_contract.resource_service_recipe(
        "l2-town-v1", "full", ARMOR)
    assert "time_protocol" not in contract["resource_service_recipe"]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_same_clock_loot_resume_validates(protocol):
    current = loot_contract(protocol)
    training._validate_resume_contract(copy.deepcopy(current), current)


@pytest.mark.parametrize("saved,current", [(V1, R18C), (R18C, V1)])
def test_a_loot_run_cannot_change_its_clock_by_ordinary_resume(saved, current):
    with pytest.raises(ValueError, match="separately identified initialization"):
        training._validate_resume_contract(
            loot_contract(saved), loot_contract(current), allow_manager_change=True,
            allow_legacy_resume=True, allow_optimizer_reset=True,
            allow_target_kl_change=True, allow_environment_restart=True)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_loot_and_armor_are_separate_identities_under_the_same_clock(protocol):
    with pytest.raises(ValueError, match="separately identified initialization"):
        training._validate_resume_contract(
            loot_contract(protocol, ARMOR), loot_contract(protocol, LOOT),
            allow_environment_restart=True)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_loot_sits_inside_the_protected_resume_policies(protocol):
    """手改的 loot 配方(哪怕只动版本号)不许靠 legacy 旁路续训。"""
    current = loot_contract(protocol)
    saved = copy.deepcopy(current)
    saved["resource_service_recipe"]["version"] = "l1-two-trip-loot-economy-v9"
    with pytest.raises(ValueError, match="mismatched native armor/preservation scope/version"):
        training._validate_resource_resume_identity(saved, current)
    with pytest.raises(ValueError):
        training._validate_resource_resume_identity(None, current)


# ---------------------------------------------------------------------------
# (g) 迁移操作名
# ---------------------------------------------------------------------------

FROZEN_OPERATIONS = {
    "sustain-v2": "r16-to-sustain-v2-weights-only-v1",
    "sustain-v3": "r16-to-sustain-v3-weights-only-v1",
    "sustain-v4": "r16-to-sustain-v4-weights-only-v1",
    "sustain-v5": "r16-to-sustain-v5-weights-only-v1",
    "sustain-v6": "r16-to-sustain-v6-weights-only-v1"}


def test_the_frozen_operation_names_are_never_renamed():
    for policy, name in FROZEN_OPERATIONS.items():
        assert migration.OPERATIONS[policy] == name
    assert migration.OPERATION == FROZEN_OPERATIONS["sustain-v2"]
    assert migration.DEPTH_SIGNAL_OPERATION == (
        "r16-to-sustain-v6-earned-dive-suffix-v1-depth24-dive-adjacent-v1-weights-only-v1")


def test_the_loot_operation_name_is_registered_but_not_yet_mintable():
    """新版本串已登记(永不改名),但迁移 schema 还没有时钟键——先 fail closed。"""
    assert migration.OPERATIONS[LOOT] == "r16-to-sustain-loot-v1-weights-only-v1"
    assert migration.OPERATIONS[LOOT] not in FROZEN_OPERATIONS.values()
    assert len(set(migration.OPERATIONS.values())) == len(migration.OPERATIONS)


@pytest.mark.parametrize("recovery,scope", [("off", None), ("adjacent-v1", None),
                                            ("adjacent-v1", SCOPE)])
def test_every_loot_migration_fails_closed_until_the_schema_carries_the_clock(recovery, scope):
    """R18-B3 复审 (2026-09-07):迁移收据是冻结产物,不许冻结没人核实过的时钟。

    ALLOWED_CONTRACT_KEYS / target_contract 里没有 worker_time_protocol,而
    train_ppo._validate_resource_warm_start_args 又无条件拒绝一切 loot
    warm-start——真放行只会冻出一份永远没人能消费、却写着
    time_protocol=completion-l2-v1 的收据。
    """
    with pytest.raises(ValueError,
                       match="requires a completion-l2 clock in the migration schema"):
        migration.operation_for(LOOT, recovery, scope)


def test_the_loot_refusal_comes_before_anything_is_written():
    """target_contract 的第一步之一就是 operation_for,早于 mkdir/落盘。"""
    source = (ROOT / "train/migrate_resource_candidate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "migrate")
    body = ast.dump(ast.Module(function.body, type_ignores=[]))
    assert body.index("target_contract") < body.index("mkdir")


def test_the_sustain_v6_earned_operation_string_is_bit_for_bit_the_old_one():
    assert migration.operation_for(ARMOR, "adjacent-v1", SCOPE) == (
        "r16-to-sustain-v6-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v1")
    assert migration.operation_for(ARMOR) == FROZEN_OPERATIONS["sustain-v6"]
    assert migration.operation_for(ARMOR, "adjacent-v1") == (
        "r16-to-sustain-v6-dive-adjacent-v1-weights-only-v1")


def test_the_depth_signal_experiment_stays_a_sustain_v6_only_registration():
    with pytest.raises(ValueError, match="Depth signal requires"):
        migration.operation_for("sustain-v5", "adjacent-v1", SCOPE, 24.0)
    with pytest.raises(ValueError, match="Depth signal requires"):
        migration.operation_for(ARMOR, "off", SCOPE, 24.0)
    # loot + depth24 仍然被拒(现在死在更早的 loot 时钟法上)。
    with pytest.raises(ValueError, match="completion-l2 clock in the migration schema"):
        migration.operation_for(LOOT, "adjacent-v1", SCOPE, 24.0)


@pytest.mark.parametrize("policy", ["legacy-v1", "sustain-loot-v2", "sustain-v7", None])
def test_unregistered_targets_are_still_refused(policy):
    with pytest.raises(ValueError, match="Only explicit"):
        migration.operation_for(policy)


def test_the_migration_cli_offers_loot_and_every_frozen_target():
    assert set(migration.OPERATIONS) == set(FROZEN_OPERATIONS) | {LOOT}

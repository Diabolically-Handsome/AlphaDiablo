"""二分定位:多次 reset 后走路命令是否还有效。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge

DiabloGymEnv()


def probe_walk(tag):
    obs = bridge.observe()
    sx, sy = obs["player_x"], obs["player_y"]
    bridge.act_walk(sx + 4, sy + 4)
    obs = bridge.step(ticks=60)  # 4 格斜走绰绰有余
    moved = max(abs(obs["player_x"] - sx), abs(obs["player_y"] - sy))
    print(f"{tag}: 从 ({sx},{sy}) 走了 {moved} 格 → ({obs['player_x']},{obs['player_y']})  mode={obs['player_mode']}")
    return moved


print("== 实验 A:纯城镇连续 reset ==")
bridge.reset(seed=1001)
a1 = probe_walk("A-reset#1")
bridge.reset(seed=1001)
a2 = probe_walk("A-reset#2")
bridge.reset(seed=1001)
a3 = probe_walk("A-reset#3")

print(f"\n结论 A: reset#1 走 {a1} 格, #2 走 {a2} 格, #3 走 {a3} 格")
if a2 >= 3 and a3 >= 3:
    print("→ 纯城镇多次 reset 正常;脏状态来自『下地牢后再 reset』的路径")
else:
    print("→ 第二次 reset 本身就坏:NetInit/teardown 重入问题")

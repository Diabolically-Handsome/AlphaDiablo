"""Which quests a game seed gets, without starting a game: a Python copy of the quest lottery.

Same steps as the bridge's ManualSeedHasKing (strategist_rl_king_20260921/bridge-r3/bridge-src/manual_control.hpp:27-37):
questSeed = 16th output of std::mt19937(seed); then DevilutionX InitialiseQuestPools(questSeed)
(alphadiablo-dev/devilutionX/Source/quests.cpp:254-272) with DiabloGenerator (Source/engine/random.hpp:
LCG a=0x015A4E35 c=1 mod 2^32, advanceRndSeed = abs(int32(state)), generateRnd(v<=0x7FFF) = (x>>16) % v,
pickRandomlyAmong = values[max(generateRnd(n), 0)]). Checked against the bridge's answer for 64 seeds (seeds.json).
usage: python3 quest_lottery.py <first-seed> <last-seed>
"""
import sys


def mt19937_outputs(seed, n):
    mt = [0] * 624
    mt[0] = seed & 0xFFFFFFFF
    for i in range(1, 624):
        mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
    idx, out = 624, []
    for _ in range(n):
        if idx >= 624:
            for i in range(624):
                y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7FFFFFFF)
                mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            idx = 0
        y = mt[idx]
        idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        out.append(y & 0xFFFFFFFF)
    return out


class DiabloGenerator:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def advance(self):
        self.state = (0x015A4E35 * self.state + 1) & 0xFFFFFFFF
        s = self.state - (1 << 32) if self.state & 0x80000000 else self.state
        return s if s == -(1 << 31) else abs(s)

    def generate(self, v):
        if v <= 0:
            return 0
        x = self.advance()
        return (x >> 16) % v if v <= 0x7FFF else x % v   # Python >> and % on non-negative x match C++

    def pick(self, values):
        return values[max(self.generate(len(values)), 0)]


def missing_quests(seed):
    quest_seed = mt19937_outputs(seed, 16)[15]
    rng = DiabloGenerator(quest_seed)
    gone = [rng.pick(['skeleton_king', 'poisoned_water'])]
    if quest_seed == 988045466:
        rng.advance()
    else:
        gone.append(rng.pick(['butcher', 'ogden_banner', 'gharbad']))
    gone.append(rng.pick(['blind', 'rock', 'blood']))
    gone.append(rng.pick(['mushroom', 'zhar', 'anvil']))
    return gone


if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for s in range(a, b + 1):
        g = missing_quests(s)
        print(s, 'king' if 'skeleton_king' not in g else '-', 'butcher' if 'butcher' not in g else '-')

import numpy as np
import random
import matplotlib.pyplot as plt
import os

# ---------- 编码与解码 ----------
def encode(x, x_min=0, x_max=2):
    """将浮点数x编码为9位二进制字符串（整数1位+十分位4位+百分位4位）"""
    if x < x_min or x >= x_max:
        raise ValueError(f"x must be in [{x_min}, {x_max})")
    int_part = int(x)                     # 0 或 1
    frac = int(round((x - int_part) * 100))  # 两位小数对应的整数
    tenth = frac // 10
    hundredth = frac % 10
    # 编码
    int_bits = bin(int_part)[2:].zfill(1)
    tenth_bits = bin(tenth)[2:].zfill(4)
    hundredth_bits = bin(hundredth)[2:].zfill(4)
    return int_bits + tenth_bits + hundredth_bits

def decode(bits, x_min=0, x_max=2):
    """将9位二进制串解码为浮点数"""
    if len(bits) != 9:
        raise ValueError("bits must be 9 characters")
    int_part = int(bits[0], 2)
    tenth = int(bits[1:5], 2)
    hundredth = int(bits[5:9], 2)
    # 限制范围（防止非法值）
    tenth = min(tenth, 9)
    hundredth = min(hundredth, 9)
    return int_part + tenth/10 + hundredth/100

# 测试编码解码（验证题2要求）
def test_encoding():
    test_vals = [0.0, 0.5, 1.0, 1.51, 1.99]
    print("编码解码测试：")
    for v in test_vals:
        bits = encode(v)
        decoded = decode(bits)
        print(f"{v:.2f} -> {bits} -> {decoded:.2f} (误差 {abs(v-decoded):.4f})")


def crossover(parent1, parent2, pc=0.6):
    """单点交叉，以概率pc交换，否则原样返回"""
    if random.random() > pc:
        return parent1, parent2
    point = random.randint(1, 8)   # 交叉点位置（1~8）
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, pm=0.2):
    """位翻转变异，每一位以概率pm翻转"""
    bits = list(individual)
    for i in range(len(bits)):
        if random.random() < pm:
            bits[i] = '1' if bits[i] == '0' else '0'
    return ''.join(bits)

def objective(x):
    return (x - 3) ** 2   # 目标函数，最小化

def fitness(x):
    return -objective(x)   # 适应度取负，越大越好

class EvolutionaryAlgorithm:
    def __init__(self, Gen, pc, pm, P0, P, x_min=1, x_max=4):
        self.Gen = Gen
        self.pc = pc
        self.pm = pm
        self.P0 = P0
        self.P = P
        self.x_min = x_min
        self.x_max = x_max
        self.best_x_history = []
        self.best_obj_history = []

    def _real_to_enc(self, real_x):
        """将 [1,4] 映射到 [0,2)"""
        return (real_x - self.x_min) / (self.x_max - self.x_min) * 2.0

    def _enc_to_real(self, enc_x):
        """将 [0,2) 映射回 [1,4]"""
        return self.x_min + (enc_x / 2.0) * (self.x_max - self.x_min)

    def initialize_population(self):
        pop = []
        for _ in range(self.P0):
            real_x = random.uniform(self.x_min, self.x_max)  # 在 [1,4] 随机
            enc_x = self._real_to_enc(real_x)  # 映射到 [0,2)
            pop.append(encode(enc_x, 0, 2))  # 用 [0,2) 编码
        return pop

    def evaluate(self, individual):
        enc_x = decode(individual, 0, 2)  # 解码得到 [0,2) 的值
        real_x = self._enc_to_real(enc_x)  # 映射回 [1,4]
        return fitness(real_x)  # 计算适应度

    def select(self, population, fitnesses):
        """轮盘赌选择（选择P0个个体）"""
        # 防止负适应度，平移
        min_f = min(fitnesses)
        if min_f < 0:
            fitnesses = [f - min_f + 1e-10 for f in fitnesses]
        total = sum(fitnesses)
        if total == 0:
            probs = [1/len(population)] * len(population)
        else:
            probs = [f/total for f in fitnesses]
        selected = []
        for _ in range(self.P0):
            r = random.random()
            cum = 0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    selected.append(population[i])
                    break
        return selected

    def run(self, verbose=True):
        # 初始化
        population = self.initialize_population()
        best_individual = None
        best_fitness_val = -float('inf')

        for gen in range(self.Gen):
            # 评估
            fitnesses = [self.evaluate(ind) for ind in population]
            # 更新全局最优
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness_val:
                best_fitness_val = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx]
            enc_best = decode(best_individual, 0, 2)
            best_x = self._enc_to_real(enc_best)
            best_obj = objective(best_x)
            self.best_x_history.append(best_x)
            self.best_obj_history.append(best_obj)

            if verbose:
                print(f"Gen {gen+1}/{self.Gen}: best x = {best_x:.4f}, objective = {best_obj:.4f}")

            # 交叉：生成新个体直到种群规模达到 P
            offspring = []
            while len(offspring) < self.P - self.P0:
                p1, p2 = random.sample(population, 2)
                c1, c2 = crossover(p1, p2, self.pc)
                offspring.extend([c1, c2])
            offspring = offspring[:self.P - self.P0]
            population = population + offspring

            # 变异
            for i in range(len(population)):
                population[i] = mutate(population[i], self.pm)

            # 选择
            fitnesses = [self.evaluate(ind) for ind in population]
            population = self.select(population, fitnesses)

        # 最终结果
        final_enc = decode(best_individual, 0, 2)
        final_x = self._enc_to_real(final_enc)
        final_obj = objective(final_x)
        return {
            'best_x': final_x,
            'best_obj': final_obj,
            'best_individual': best_individual,
            'history_x': self.best_x_history,
            'history_obj': self.best_obj_history
        }

def run_experiments():
    param_sets = [
        {'Gen': 10, 'pc': 0.6, 'pm': 0.2, 'P0': 10, 'P': 15},
        {'Gen': 20, 'pc': 0.6, 'pm': 0.2, 'P0': 10, 'P': 15},
        {'Gen': 20, 'pc': 0.6, 'pm': 0.2, 'P0': 10, 'P': 15},  # 重复，用于稳定性观察
        {'Gen': 20, 'pc': 0.2, 'pm': 0.2, 'P0': 10, 'P': 15},
        {'Gen': 20, 'pc': 0.6, 'pm': 0.2, 'P0': 20, 'P': 30},
        {'Gen': 20, 'pc': 0.6, 'pm': 0.2, 'P0': 5,  'P': 7},
    ]
    results = []
    for idx, params in enumerate(param_sets):
        print(f"\n===== 实验 {idx+1}: {params} =====")
        runs = []
        for run_id in range(3):   # 每个参数运行3次
            print(f"--- 第 {run_id+1} 次运行 ---")
            ea = EvolutionaryAlgorithm(**params, x_min=1, x_max=4)
            res = ea.run(verbose=True)
            runs.append(res)
        # 统计
        best_xs = [r['best_x'] for r in runs]
        best_objs = [r['best_obj'] for r in runs]
        results.append({
            'params': params,
            'best_x_mean': np.mean(best_xs),
            'best_x_std': np.std(best_xs),
            'best_obj_mean': np.mean(best_objs),
            'best_obj_std': np.std(best_objs),
            'history_objs': [r['history_obj'] for r in runs]
        })
        print(f"统计: x = {np.mean(best_xs):.4f} ± {np.std(best_xs):.4f}, "
              f"obj = {np.mean(best_objs):.4f} ± {np.std(best_objs):.4f}")
    return results

def plot_convergence_curves(results, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    labels = [
        "Gen=10, pc=0.6, pm=0.2, P0=10, P=15",
        "Gen=20, pc=0.6, pm=0.2, P0=10, P=15",
        "Gen=20, pc=0.6, pm=0.2, P0=10, P=15 (重复)",
        "Gen=20, pc=0.2, pm=0.2, P0=10, P=15",
        "Gen=20, pc=0.6, pm=0.2, P0=20, P=30",
        "Gen=20, pc=0.6, pm=0.2, P0=5, P=7"
    ]
    for i, (res, label) in enumerate(zip(results, labels)):
        ax = axes[i]
        for hist in res['history_objs']:
            ax.plot(range(1, len(hist)+1), hist, alpha=0.4, linewidth=1)
        # 平均曲线
        all_hists = res['history_objs']
        min_len = min(len(h) for h in all_hists)
        avg = np.mean([h[:min_len] for h in all_hists], axis=0)
        ax.plot(range(1, min_len+1), avg, 'r-', linewidth=2, label='平均')
        ax.axhline(y=0, color='g', linestyle='--', alpha=0.6, label='理论最优')
        ax.set_xlabel('进化代数')
        ax.set_ylabel('目标值 (x-3)²')
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_curves.png", dpi=300)
    plt.close()
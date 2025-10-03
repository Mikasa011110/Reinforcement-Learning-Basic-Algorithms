import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """ 伯努利多臂老虎机， 输入K表示拉杆个数"""
    def __init__(self, K):
        self.probs = np.random.uniform(size=K) # 随机生成K个 0-1 的数，作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs) # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k):
        # 当玩家选择k号拉杆后，根据该拉杆的获奖概率返回1或0
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

np.random.seed(1)
K=10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个10臂伯努利老虎机,获奖概率最大的拉杆为: "f"{bandit_10_arm.best_idx}")
print("获奖概率为 "f"{bandit_10_arm.best_prob:.4f}")


class Solver:
    """多臂老虎机算法基本框架"""
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 记录每根拉杆被拉动的次数
        self.regret = 0. # 当前步的累计懊悔
        self.regrets = [] # 维护一个列表，记录每一步的累计懊悔
        self.actions = [] # 维护一个列表，记录每一步的动作
    
    def update_regret(self, k):
        # 计算累积懊悔并保存，k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 运行一步，选择一个拉杆并更新懊悔(这里的raise NotImplementedError表示该方法需要在子类中实现)
        raise NotImplementedError
    
    def run(self, n):
        # 运行n步
        for _ in range(n):
            k = self.run_one_step() # 选择一个拉杆
            self.counts[k] += 1 # 更新该拉杆被拉动的次数
            self.actions.append(k) # 记录本次动作
            self.update_regret(k) # 更新懊悔


class EpsilonGreedy(Solver):
    """ epsilon贪婪算法，继承自Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)  # 调用父类的初始化方法
        self.epsilon = epsilon  # 探索的概率
        self.estimates = np.array([init_prob] * self.bandit.K) # 初始化每根拉杆的期望奖励估值， 输出是【1.0, 1.0, ..., 1.0】
    
    def run_one_step(self):
        if np.random.random() < self.epsilon: # 以epsilon的概率随机选择一个拉杆
            k = np.random.choice(self.bandit.K)
        else: # 否则选择当前期望奖励估值最大的拉杆
            k = np.argmax(self.estimates)
        r = self.bandit.step(k) # 拉动k号拉杆，获得本次动作的奖励r
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) # 根据增量式均值更新公式，更新k号拉杆的期望奖励估值
        return k

def plot_results(solvers, solver_names):
    """ 生成累积懊悔随时间变化的图像，输入solvers是一个列表，列表中的每个元素都是一种特定的策略。
        solver_names也是一个列表，存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative Regret")
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01, init_prob=1.0)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累计懊悔为: ', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ['Epsilon-Greedy'])


"""尝试不同的epsilon值，观察10arm 老虎机的累积懊悔值"""
np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=eps, init_prob=1.0) for eps in epsilons]
epsilon_greedy_solver_names = ["epsilon={}".format(eps) for eps in epsilons]

for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
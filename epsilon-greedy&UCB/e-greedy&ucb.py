import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit:
    def __init__(self):
        self.epsilon = 0.1  # 设定epsilon值
        self.num_arm = 10  # 设置arm的数量
        self.arms = np.random.uniform(0, 1, self.num_arm)  # 设置每一个arm的均值，为0-1之间的随机数
        self.best = np.argmax(self.arms)  # 找到最优arm的index
        self.T = 100000  # 设置进行行动的次数
        self.hit_epsilon = np.zeros(self.T)  # 用来记录每次行动是否找到最优arm
        self.hit_ucb = np.zeros(self.T)
        self.reward_epsilon = np.zeros(self.num_arm)  # 用来记录每次行动后各个arm的平均收益
        self.reward_ucb = np.zeros(self.num_arm)
        self.num_epsilon = np.zeros(self.num_arm)  # 用来记录每次行动后各个arm被拉动的总次数
        self.num_ucb = np.ones(self.num_arm)*0.000000001
        self.V_epsilon = 0
        self.V_ucb = 0
        self.up_bound = np.zeros(self.num_arm)

    def get_reward(self, i):  # i为arm的index
        return self.arms[i] + np.random.normal(0, 1)  # 生成的收益为arm的均值加上一个波动

    def update_epsilon(self, i):
        self.num_epsilon[i] += 1
        self.reward_epsilon[i] = (self.reward_epsilon[i]*(self.num_epsilon[i]-1)+self.get_reward(i))/self.num_epsilon[i]
        self.V_epsilon += self.get_reward(i)

    def update_ucb(self, i):
        self.num_ucb[i] += 1
        self.reward_ucb[i] = (self.reward_ucb[i]*(self.num_ucb[i]-1)+self.get_reward(i))/self.num_ucb[i]
        self.V_ucb += self.get_reward(i)

    def calculate_epsilon(self):
        for i in range(self.T):
            if np.random.random() > self.epsilon:
                index = np.argmax(self.reward_epsilon)
            else:
                a = np.argmax(self.reward_epsilon)
                index = a
                while index == a:
                    index = np.random.randint(0, self.num_arm)
            if index == self.best:
                self.hit_epsilon[i] = 1  # 如果拿到的arm是最优的arm，则将其记为1
            self.update_epsilon(index)

    def calculate_ucb(self):
        for i in range(self.T):
            for j in range(self.num_arm):
                self.up_bound[j] = self.reward_ucb[j] + np.sqrt(2*np.log(i+1)/self.num_ucb[j])
            index = np.argmax(self.up_bound)
            if index == self.best:
                self.hit_ucb[i] = 1
            self.update_ucb(index)

    def plot_epsilon(self):  # 画图查看收敛性
        x = np.array(range(self.T))
        y1 = np.zeros(self.T)
        y3 = np.zeros(self.T)
        t = 0
        for i in range(self.T):
            t += self.hit_epsilon[i]
            y1[i] = t/(i+1)
        t = 0
        for i in range(self.T):
            t += self.hit_ucb[i]
            y3[i] = t/(i+1)
        y2 = np.ones(self.T)*(1-self.epsilon)
        plt.plot(x, y1, label='epsilon-greedy')
        plt.plot(x, y2, label='1-epsilon')
        plt.plot(x, y3, label='UCB')
        plt.legend(loc='right', prop={'size': 18})
        plt.show()


E = MultiArmedBandit()
E.calculate_epsilon()
E.calculate_ucb()
print(E.V_ucb)
print(E.V_epsilon)
E.plot_epsilon()

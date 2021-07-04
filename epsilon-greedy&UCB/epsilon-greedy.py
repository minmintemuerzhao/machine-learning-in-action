import numpy as np
import matplotlib.pyplot as plt


class EpsilonGreedy:
    def __init__(self):
        self.epsilon = 0.1  # 设定epsilon值
        self.num_arm = 10  # 设置arm的数量
        self.arms = np.random.uniform(0, 1, self.num_arm)  # 设置每一个arm的均值，为0-1之间的随机数
        self.best = np.argmax(self.arms)  # 找到最优arm的index
        self.T = 50000  # 设置进行行动的次数
        self.hit = np.zeros(self.T)  # 用来记录每次行动是否找到最优arm
        self.reward = np.zeros(self.num_arm)  # 用来记录每次行动后各个arm的平均收益
        self.num = np.zeros(self.num_arm)  # 用来记录每次行动后各个arm被拉动的总次数

    def get_reward(self, i):  # i为arm的index
        return self.arms[i] + np.random.normal(0, 1)  # 生成的收益为arm的均值加上一个波动

    def update(self, i):
        self.num[i] += 1
        self.reward[i] = (self.reward[i]*(self.num[i]-1)+self.get_reward(i))/self.num[i]

    def calculate(self):
        for i in range(self.T):
            if np.random.random() > self.epsilon:
                index = np.argmax(self.reward)
            else:
                a = np.argmax(self.reward)
                index = a
                while index == a:
                    index = np.random.randint(0, self.num_arm)
            if index == self.best:
                self.hit[i] = 1  # 如果拿到的arm是最优的arm，则将其记为1
            self.update(index)

    def plot(self):  # 画图查看收敛性
        x = np.array(range(self.T))
        y1 = np.zeros(self.T)
        t = 0
        for i in range(self.T):
            t += self.hit[i]
            y1[i] = t/(i+1)
        y2 = np.ones(self.T)*(1-self.epsilon)
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.show()


E = EpsilonGreedy()
E.calculate()
E.plot()

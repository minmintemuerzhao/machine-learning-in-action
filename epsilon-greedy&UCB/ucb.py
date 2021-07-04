
import numpy as np
import matplotlib.pyplot as plt


# Set δ = 1/n**4 in Hoeffding's bound
# Choose a with highest Heoffding bound

class UCB:
    def __init__(self):
        self.num_arm = 10  # 设置arm的数量
        self.arms = np.random.uniform(0, 1, self.num_arm)  # 设置每一个arm的均值，为0-1之间的随机数
        self.best = np.argmax(self.arms)  # 找到最优arm的index
        self.T = 100000  # 设置进行行动的次数
        self.hit = np.zeros(self.T)  # 用来记录每次行动是否找到最优arm
        self.reward = np.zeros(self.num_arm)  # 用来记录每次行动后各个arm的平均收益
        self.num = np.ones(self.num_arm)*0.00001  # 用来记录每次行动后各个arm被拉动的总次数
        self.V = 0
        self.up_bound = np.zeros(self.num_arm)

    def get_reward(self, i):  # i为arm的index
        return self.arms[i] + np.random.normal(0, 1)  # 生成的收益为arm的均值加上一个波动

    def update(self, i):
        self.num[i] += 1
        self.reward[i] = (self.reward[i]*(self.num[i]-1)+self.get_reward(i))/self.num[i]
        self.V += self.get_reward(i)

    def calculate(self):
        for i in range(self.T):
            for j in range(self.num_arm):
                self.up_bound[j] = self.reward[j] + np.sqrt(2*np.log(i+1)/self.num[j])
            index = np.argmax(self.up_bound)
            if index == self.best:
                self.hit[i] = 1
            self.update(index)

    def plot(self):  # 画图查看收敛性
        x = np.array(range(self.T))
        y1 = np.zeros(self.T)
        t = 0
        for i in range(self.T):
            t += self.hit[i]
            y1[i] = t/(i+1)
        plt.plot(x, y1)
        plt.show()


U = UCB()
U.calculate()
U.plot()

# 版权声明：本文为CSDN博主「NLP饶了我，NN再爱我一次」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_42419611/article/details/115015754

class TreeNode:
    def __init__(self, s, d):
        self.vec = s  # 特征向量
        self.Dimension = d  # 即划分空间时的特征维度
        self.left = None  # 左子节点
        self.right = None  # 右子节点
        self.father = None  # 父节点（搜索时需要往回退）

    def __str__(self):
        return str(self.vec)  # print 一个 Node 类时会打印其特征向量


def dis(arr1, arr2):
    res = 0
    for a, b in zip(arr1, arr2):
        res += (a - b) ** 2
    return res ** 0.5


def build(arr, l, father):
    if len(arr) == 0:  # 样本空间为空则返回
        return

    # 找x^l的中位数和对应特征向量,即arr[:][l]的中位数及arr[x][:]
    size = len(arr)
    # 直接对arr进行排序，因为要得到特征向量和划分子空间，由此直接对arr排序最便捷
    # 对二维数组排序，排序值为l列（升序）：
    arr.sort(key=(lambda x: x[l]))

    mid = int((size - 1) / 2)

    root = TreeNode(arr[mid], l)
    root.father = father
    root.left = build(arr[0:mid], (l + 1) % n, root)  # 0:mid不包括mid，即[0,mid)
    root.right = build(arr[mid + 1:], (l + 1) % n, root)
    # print(root,root.father)
    return root


def dfs(depth, root, father, stack):
    if root == None:
        return father;

    stack.append(root)

    if target[depth % n] < root.vec[depth % n]:  # depth%n=l,l为特征的上标
        return dfs(depth + 1, root.left, root, stack)
    else:
        return dfs(depth + 1, root.right, root, stack)
    return root


Featureset = [[1, 6, 2],
              [2, 9, 3],
              [5, 1, 4],
              [9, 4, 7],
              [4, 2, 6],
              [6, 3, 5],
              [7, 2, 5],
              [9, 1, 4]]
n = len(Featureset[0][:])  # 特征的维度(个数)
Features = [1, 3, 4, 2, 5, 5, 2, 7]
temp = Featureset

# 递归构造kd树
root = build(temp, 0, None)

# 深度搜索kd树，找到一个当前最近点
target = [1, 4, 5]
stack = []
nearest = dfs(0, root, root.father, stack)  # 找到最近邻
nearest_dis = dis(nearest.vec, target)  # nearest_dis为当前最近邻离target的距离，即最小距离，也是超球体的半径

visited = {}  # 用来判断兄弟节点是否已经讨论过

# 利用stack进行回溯，而非递归
while stack[-1] != root:
    print('STACK:', end=' ')
    for i in stack: print(i, end=' ')
    print()

    # 先定义好当前节点cur、父亲节点father、兄弟节点bro
    cur = stack[-1]
    stack.pop()
    father = cur.father
    bro = father.left
    if father.left == cur:
        bro = father.right

    # 如果当前节点与target的距离小于最近距离，则更新最近结点和最近距离
    if dis(cur.vec, target) < nearest_dis:
        nearest = cur
        nearest_dis = dis(cur.vec, target)

    # print(visited.get(hash(bro)))
    # 当前节点没有递归过
    if visited.get(hash(cur)) == None:
        # 若超球体和父节点的超平面相交，相交则父节点的另一侧，即兄弟节点所在划分域可能存在更近的节点
        if father.vec[father.Dimension] - target[father.Dimension] < nearest_dis:
            visited.update({hash(bro): 'yes'})
            dfs(father.Dimension, bro, father, stack)

print("最近距离：", nearest_dis)
print("最近邻：", nearest)

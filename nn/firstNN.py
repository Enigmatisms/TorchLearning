#-*-coding:utf-8-*-
"""
    MNIST 手写数字数据集分类的CNN实现
    第一个CNN网络，所以会有比较完全的注释说明整个代码每处地方都干了什么
"""
import torch
from torch import nn                                # nn模块，必须使用到的
from torch.autograd import Variable as Var          # 需要将训练使用的数据 / label都变成Variable才能放入网络中
from torchvision import datasets                    # 跑数据集需要
from torch.utils.data.dataloader import DataLoader  # 数据集加载工具
from torch import optim                             # 优化器，传入网络参数进行优化
from torchvision import transforms                  # 进来的数据集为PILImage 需要变成Tensor

"""
    初学者的架构思考：
        首先是卷积层，不同的卷积核（对应Channel）会生成不同的卷积后的特征层
        每一层卷积之后都要立刻进行Normalization，其意义是：变换输出结果到一个特定空间范围使其符合特定的分布
            类同于数据白化，归一化去除中心以及方差的不一致性
            当然这里存在一个再平移以及再缩放
        此后进行激活，得到一个多Channel的中间输出
        此后重复如下结构：
        Conv(Ch_i, kw, kw) -> (Ch_o, wo, ho) -> Normalization -> Actibation (ReLU) -> Conv(Ch_o(上一层的channels) ... ) -> ...
        当然其中可以穿插Pooling层，Pooling层是必要的吗？对于较大的输入是必要的，MNIST小图的话好像没什么意义
        直到最后：Flatten -> FC -> Activation -> FC -> Output.

        注意其中有些线性连接层
"""
# 简单的CNN网络架构
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层
        self.conv1 = nn.Sequential(     
            nn.Conv2d(1, 16, 5, padding = 2),       # 注意，pytorch默认没有padding，所以卷积需要padding
            nn.ReLU(),                              # ReLU
            nn.MaxPool2d(2)                         # 注意MaxPool （pooling操作）是采样，相当于resize（有损失的）
        )
        # 第二层，注意输入输出层的size对应
        self.conv2 = nn.Sequential(     
            nn.Conv2d(16, 32, 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(2)             # 变为7 * 7大小
        )
        self.conv3 = nn.Sequential(     # 第三层
            nn.Conv2d(32, 16, 5, padding = 2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(       # 输出全连接 + 正则化 + 激活 + 全连接
            nn.Linear(16 * 7 * 7, 20),
            nn.BatchNorm1d(20),
            nn.Softmax(),
            nn.Linear(20, 10)
        )
        # 如果是简单分类判别的话

    def forward(self, x):               # 前向传递
        x = self.conv1(x)               # 反正就是无休止地过中间层了
        x = self.conv2(x)
        x = self.conv3(x)

        # 此处由于是个batch，batch输出的是一个3维矩阵
        # 比如本处是：16 * 7 * 7 的三维数组，需要进入全连接层就需要进行flatten操作
        x = x.view(x.size(0), -1)       # 变成向量（每个channel内部）
        # 进入全连接层，返回最后的输出
        return self.out(x)

## TODO：需要定义反向传播的方式以及优化器
## TODO：需要能够加载数据
# 使用transform以及内置的datasets完成数据加载
if __name__ == "__main__":
    use_load = False
    batch_size = 50
    epoches = 3
    lrate = 1e-3
    inspect_point = 50

    tf = transforms.ToTensor()

    # 对于MNIST 数据集，加载时需要进行transform转换成一个Tensor
    tests = datasets.MNIST("..\\data\\", False, transform = tf, download = False)

    # 使用mini batch会更好，test_data也可以batch操作
    test_data = DataLoader(tests, batch_size = batch_size, shuffle = False)     # 不打乱，便于观察输出

    if use_load == False:
        cnn = CNN()
        cnn.cuda()                                          # 直接使用GPU训练
        opt = optim.Adam(cnn.parameters(), lr = lrate)      # 优化器设定
        loss_func = nn.CrossEntropyLoss()                   # 误差函数设定
        train_set = datasets.MNIST("..\\data\\", True, transform = tf, download = False)
        tdata = DataLoader(train_set, batch_size = batch_size, shuffle = True)      # 测试集，需要打乱

        batch_cnt = 1
        for i in range(epoches):                            # 可以对训练集进行多次训练
            for k, (bx, by) in enumerate(tdata):            # 取数据
                if torch.cuda.is_available():
                    _bx = bx.cuda()
                    _by = by.cuda()
                else:
                    _bx = Var(bx)
                    _by = Var(by)
                out = cnn(_bx)                              # 过网络了，注意，其中require_grad的变量包含在CNN网络参数里
                loss = loss_func(out, _by)                  # 误差evaluation，使用前向out与标准_by比较
                opt.zero_grad()                             # 标准优化器操作
                loss.backward()
                opt.step()
                # 每batch个值会进行一次输出
                if k % inspect_point == 0:          
                    print("Epoch: %d, loss: %f"%(k, loss.data.item()))        # 一维tensor转化为python标量
                    print("Batch Counter:", batch_cnt)
                    print("===========================================")
                # 这里并没有加入验证集（validation）
                batch_cnt += 1
        print("Training completed.")
        # 模型保存
        torch.save(cnn, "..\\models\\first_cnn.pkl")
    else:
        cnn = torch.load("..\\models\\first_cnn.pkl")

    eval_cnt = 0
    for i, (bx, by) in enumerate(test_data):
        if torch.cuda.is_available():
            bx = bx.cuda()
            by = by.cuda()
        out = cnn(bx)

        # 对于一个2D矩阵输出，torch.max会返回每一个行上的最大值位置以及值
        _, pred = torch.max(out, 1)     # 返回最大值（_）与对应索引（pred）
        rights = (pred == by).sum()
        eval_cnt += rights.item()
        print("Batch(%d) prediction shape:"%(i), pred.shape, ", rights out of 50: ", rights.item())
    
    print("Total %d test samples."%(len(test_data) * batch_size))
    print("Correction ratio: %f."%(eval_cnt / (len(test_data) * batch_size)))

    # 三轮训练最后的loss已经在波动了，但是会比98.31%(一轮训练)好一些，到99.31%
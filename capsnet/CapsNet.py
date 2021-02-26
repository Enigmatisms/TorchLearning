"""
对CapsuleNN 别人实现的注解 学习一下别人写的代码
    实现胶囊网络失败了，torch太不熟练了，并且之前也没思考过高维矩阵操作的问题
    果然，使用这样的神经网络框架而不去深入了解设计思想就和调库侠没什么区别

本实现来自Github，本代码是对此实现中的细节进行注解。并借此机会学习一下别人是怎么写的。希望以后自己不要那么菜了。
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829
PyTorch implementation by Kenta Iwasaki @ Gram.AI.
From https://github.com/gram-ai/capsule-networks
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3

def softmax(input, dim=1):
    """
        transpose是为了更方便进行view + softmax组织 但是我不知道这样写的意义何在
        如果直接对input进行 F.softmax(input, dim = dim) 不可以吗 为什么一定要transpose
    """
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    # contiguous操作的意义是什么
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        """
            nn.Parameter 说明vector映射需要经过学习才能得出
            上一层没有输出的时候（非胶囊层），只需要定义本层为胶囊层即可，至少
            在本MNIST分类网络中，只有两层胶囊层，第二层（digitCaps）本身并不含卷积胶囊结构
        """
        if num_route_nodes != -1:
            """
                至于为什么要写成(, ,  in_channels, out_channels) in_channels是8的原因（而不是32）
                是因为，在上一层输出时，进行了ravel操作（但是这样实现是为了什么？）
                个人的理解是：由于动态路由的性质就是：高一层的所有node需要接收来自低一层的所有输入
                那么实际类似一个FC，只不过FC是向量的全连接，并且全连接关系是动态路由的方式
                低层每一个node实际上没有地位上的差异
            """
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            # 每个卷积保存在List中（自动加入parameters()中）
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        # 常用操作了，需要平方累加的部分求和就是模平方 此处keepdim的理由很明确，只是对原来的矩阵进行归一化
        # 原来的矩阵由于是【高维矩阵】，norm过程需要进行维度保持
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        # 如果需要进行动态路由（也就是上一层也是胶囊层）
        if self.num_route_nodes != -1:
            # 维度一致化 (10, n, 1152, 1, 8) @ (10, n, 1152, 8, 16) weights是[8x16]
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            # TODO: 直接生成logit值，之后softmax变为概率，但为什么一定要Variable？
            logits = Variable(torch.zeros(*priors.size())).cuda()   # 可以处理为zeros_like
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)                      # dim = 2对应的是 1152(PrimaryCaps输出数)
                # 由于这里是加权操作，所以只需要进行对应相乘再相加即可 1152 保持输出shape进行归一化
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    # 由于最后一维是16 (最后一维也就是计算内积结果的维度，需要sum得到内积的求和结果)
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            # 此处 -1 自动推导了结果，将(n, 32, 6, 6)直接压缩成(n, 1152, 1)，以进行叠加
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        # in_channels 是8是因为PrimaryCaps输出 "capsule(x).view(x.size(0), -1, 1)" 相当于压扁
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)
        """
            个人认为，实际上此处160 -> 512 貌似没有什么意义？毕竟实际只有一个16维的向量将重构原来的数字
            因为进行了mask操作 详见维度扩增操作 y[:, :, None] 直接将比如[0, 1, 0] 变成了[[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        """
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),                  # 激活函数多用inplace 节省显存
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)

        # 输出(n, 10, 16)，原始输出(10, n, 1, 1, 16) squeeze后(10, n, 16) 
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        # 进行模长的计算（由于x仍然是胶囊的输出，输出进行了squash，最后根据长度来计算softmax）
        # 输出为(n, 10)
        classes = (x ** 2).sum(dim=-1) ** 0.5

        # 输出为(n, 10)softmax非降维手段，只是求指数化概率
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            # 由于此处 classes 为(n, 10) 求最大值所在位置
            _, max_length_indices = classes.max(dim=1)
            # 可能参与运算的操作数都需要是Variable 此处表示的意思是 在一个size为10 * 10的单位矩阵中
            # 按照行抽取，之后y将会被送入y[:, :, None] 扩增，得到论文中需要的mask
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        # 此处 y原来是一个个的行向量组成的2D矩阵，行向量中只含有一个1，其他全是0
        # 加了一维（此处None最后变为16，相当于复制了16份结果（最后形成某行全是1，其他行全是0的mask结构））
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return classes, reconstructions

# LOSS也是可以进行自定义的
class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        # 重构误差是均方误差
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        # ReLU相当于是一个max截断函数 此处应该是可以多维计算
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        # 论文这样处理貌似是保留了部分 label不对应的重构误差
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()     # 维度叠加

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)     # 变为(n, 28 * 28)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm

    # torchnet工具可以继续了解一下
    import torchnet as tnt

    model = CapsuleNet()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss()

    # 此后就完全是 tnt自动训练的内容
    def get_iterator(mode):
        dataset = MNIST(root='../data', download=False, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    # 定义获取一个sample之后的loss计算操作
    def processor(sample):
        data, labels, training = sample

        data = augmentation(data.unsqueeze(1).float() / 255.0)
        labels = torch.LongTensor(labels)

        # label 就是用只含一个1的行向量组合成的
        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.

        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data

        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        reconstruction_logger.log(
            make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
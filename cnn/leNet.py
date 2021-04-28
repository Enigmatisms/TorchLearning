#-*-coding:utf-8-*-
"""
    MNIST 手写数字数据集分类的CNN实现
    基于我二月份某个代码以及另一篇论文复现时的代码
    - LeNet
"""
import os
import shutil
import argparse
from datetime import datetime
import torch
from torch import nn                                
from torchvision import datasets                    
from torch.utils.data.dataloader import DataLoader 
from torch import optim                             
from torchvision import transforms     
from torch.utils.tensorboard import SummaryWriter

# LeNet
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(     
            nn.Conv2d(1, 6, 5, padding = 2),      
            nn.ReLU(True),                             
            nn.MaxPool2d(2)                       
        )
        self.conv2 = nn.Sequential(     
            nn.Conv2d(6, 16, 5, padding = 0),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, 5, padding = 0),
            nn.ReLU(True),
        )
        self.out = nn.Sequential(
            nn.Linear(120, 84),
            # nn.Dropout(0.1),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):               
        x = self.conv1(x)              
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        return self.out(x)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 10, help = "Training lasts for . epochs")
    parser.add_argument("--batch_sz", type = int, default = 50, help = "Batch size for miniBatch")
    parser.add_argument("--lr", type = float, default = 5e-3, help = "Learning rate")
    parser.add_argument("--inspect", type = int, default = 100, help = "Print loss information every <inspect> batches")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    args = parser.parse_args()

    batch_size = args.batch_sz
    epochs = args.epochs
    lrate = args.lr
    inspect_point = args.inspect
    del_dir = args.del_dir

    tf = transforms.ToTensor()

    tests = datasets.MNIST("..\\data\\", False, transform = tf, download = False)

    test_data = DataLoader(tests, batch_size = batch_size, shuffle = False)     # 不打乱，便于观察输出
    train_set = datasets.MNIST("..\\data\\", True, transform = tf, download = False)
    tdata = DataLoader(train_set, batch_size = batch_size, shuffle = True)      # 测试集，需要打乱

    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    writer = SummaryWriter(log_dir = logdir+time_stamp)

    cnn = CNN()
    cnn.cuda()                                          
    opt = optim.Adam(cnn.parameters(), lr = lrate)      
    sch = optim.lr_scheduler.ExponentialLR(opt, 0.985, -1)
    loss_func = nn.CrossEntropyLoss()                   
    batch_cnt = 0
    for i in range(epochs):                            
        for k, (bx, by) in enumerate(tdata):           
            _bx = bx.cuda()
            _by = by.cuda()
            out = cnn(_bx)                              
            loss = loss_func(out, _by)                  
            opt.zero_grad()                             
            loss.backward()
            opt.step()
            batch_cnt += 1
            if k % inspect_point != 0: continue
            sch.step()          
            cnn.eval()
            eval_cnt = 0
            for i, (bx, by) in enumerate(test_data):
                bx = bx.cuda()
                by = by.cuda()
                out = cnn(bx)
                _, pred = torch.max(out, 1) 
                rights = (pred == by).sum()
                eval_cnt += rights.item()
            acc = eval_cnt / (len(test_data) * batch_size)
            print("Batch Counter: %d\tloss: %f\tacc: %f"%(batch_cnt, loss.data.item(), acc))
            writer.add_scalar('Eval/Total Loss', loss.data.item(), batch_cnt)
            writer.add_scalar('Eval/Acc', acc, batch_cnt)
            cnn.train()
    print("Training completed.")
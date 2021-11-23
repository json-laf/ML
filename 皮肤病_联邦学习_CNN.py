import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import syft as sy
import torchvision

# pysyft的hook
hook = sy.TorchHook(torch)
# 创建虚拟节点
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
mike = sy.VirtualWorker(hook, id="mike")

class Arguments():
    def __init__(self):
        self.batch_size = 2
        self.test_batch_size = 1
        self.epochs = 30
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False

# 配置参数
args = Arguments()
# 使用cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
# 设置worker
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# 联邦数据，数据分布在不同工作节点上
federated_train_loader = sy.FederatedDataLoader(
    torchvision.datasets.ImageFolder('JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format-master/training-images',
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize([28, 28]),  # 裁剪图片
                                         torchvision.transforms.Grayscale(1),  # 单通道
                                         torchvision.transforms.ToTensor(),  # 将图片数据转成tensor格式
                                         torchvision.transforms.Normalize(  # 归一化
                                             (0.1307,), (0.3081,))
                                     ])).federate((bob, alice)), batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder('JPG-PNG-to-MNIST-NN-Format/JPG-PNG-to-MNIST-NN-Format-master/test-images',
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize([28, 28]),  # 裁剪图片
                                         torchvision.transforms.Grayscale(1),  # 单通道
                                         torchvision.transforms.ToTensor(),  # 将图片数据转成tensor格式
                                         torchvision.transforms.Normalize(  # 归一化
                                             (0.1307,), (0.3081,))
                                     ])), batch_size=args.test_batch_size, shuffle=True, **kwargs)
# 深度网络结构

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
# 训练

def train(args, model, device, federated_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        # 把模型发给联邦学习节点
        model.send(data.location)
        data, target = data.to(device), target.to(device)

        # 把grad清零
        optimizer.zero_grad()
        output = model(data)
        # 计算损失
        loss = F.nll_loss(output, target)
        # 计算梯度
        loss.backward()
        optimizer.step()
        # 从远程节点更新模型

        model.get()
        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                       100. * batch_idx / len(federated_train_loader), loss.item()))


# 测试
acc_ls = []
loss_ls = []

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc_ls.append(round(100. * correct / len(test_loader.dataset)) / 100)
    loss_ls.append(test_loss)


if __name__ == '__main__':
    # 模型初始化
    model = Net().to(device)
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # 求解
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    # 训练结果存盘
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")
    print("acc列表为：{}，平均acc为{}".format(acc_ls, sum(acc_ls) / len(acc_ls)))
    print("loss列表为：{}，平均loss为{}".format(loss_ls, sum(loss_ls) / len(loss_ls)))

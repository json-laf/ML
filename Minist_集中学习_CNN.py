import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.utils.data as data


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 30
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False  # 是否使用GPU相关的
        self.seed = 1
        self.log_interval = 30  # 打印进度有关的
        self.save_model = False


args = Arguments()
# 使用cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./Mnist/data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 这是ImageNet数据集的方差和均值
    ])), batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./Mnist/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True)


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
        # model.send(data.location)
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
        # model.get()
        if batch_idx % args.log_interval == 0:
            # loss = loss.get()
            # print(batch_idx)
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))
    acc_ls.append(100. * correct / len(test_loader.dataset) / 100)
    loss_ls.append(test_loss)


if __name__ == '__main__':
    # 模型初始化
    model = Net().to(device)
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # 求解
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    # 训练结果存盘
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")
    print("acc列表为：{}，平均acc为{}".format(acc_ls, sum(acc_ls) / len(acc_ls)))
    print("loss列表为：{}，平均loss为{}".format(loss_ls, sum(loss_ls) / len(loss_ls)))

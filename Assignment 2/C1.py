import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse


# Define the ResNet-18 model
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out

# class ResNet18(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet18, self).__init__()
#         self.in_channels = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)

#         self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

#     def make_layer(self, block, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))

#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)

#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)

#         return out

# # Define the hyperparameters
# learning_rate = 0.1
# momentum = 0.9
# weight_decay = 5e-4
# num_epochs = 10

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def getargs():

    # Define command line arguments
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to the dataset directory')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers. Default is set to 2')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Optimizer for training (sgd, adam, etc.). Default is set to SGD')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Whether to use CUDA for training, By Default it is set True, so it will try to use GPU if availavle else it will default to CPU')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size. Default is set to 128')
    parser.add_argument('--test_batch_size', type=int, default=100,
                        help='Test batch size. Default is set to 100')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for optimizer. Default is set to 0.1')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer. Default is set to 0.9')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for optimizer. Default is set to 5e-4')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train for. Default is set to 5')
    args = parser.parse_args()

    return(args)

def main():

    args = getargs()
    
    data_path = args.data_path
    num_workers = args.num_workers
    optimizer = args.optimizer
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    learning_rate = args.learning_rate
    momentum = args.momentum
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    
    #Make Code Device agnostic
    if args.use_cuda:
        device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    net = ResNet(BasicBlock, [2, 2, 2, 2])
    net = net.to(device)

    if(str(optimizer).lower() == 'sgd'):
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif(str(optimizer).lower() == 'adam'):
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    print("Loading Data")
    
    Transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010))
    ])

    TrainDataset = torchvision.datasets.CIFAR10(root = data_path, train = True, download = True, transform = Transform)
    TrainLoader = torch.utils.data.DataLoader(TrainDataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

    TestDataset = torchvision.datasets.CIFAR10(root = data_path, train = False, download = True, transform = Transform)
    TestLoader = torch.utils.data.DataLoader(TestDataset, batch_size = test_batch_size, shuffle = False, num_workers = num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        print('\nTrain Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(TrainLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100 * correct/total
            return train_loss, accuracy
        
    def test(epoch):
        print('\nTest Epoch: %d' % epoch)
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(TestLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    for epoch in range(0, num_epochs):
        value, accuracy = train(epoch)
        test(epoch)
        print("Train Loss is {}, Accuracy is {} for Epoch: {}".format(value, accuracy, epoch))


#main()
if __name__ == "__main__":
    main()
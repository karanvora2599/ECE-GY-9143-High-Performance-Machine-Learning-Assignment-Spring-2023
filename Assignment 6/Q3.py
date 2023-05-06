import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

# Define BasicBlock and ResNet classes here
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
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

def total_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).view(-1).float().sum(0, keepdim=True)
        return correct.mul_(100.0 / batch_size).item()

def train(rank, world_size, model, batch_size, epochs, total_data_transferred):
    torch.manual_seed(0)
    device = torch.device(f'cuda:{rank}')
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    model.to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler
    )

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_accuracy = 0.0        
        comm_time = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            start_comm = time.time()
            outputs = model(inputs)
            comm_time += time.time() - start_comm
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy(outputs, labels)

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch + 1}: Loss {epoch_loss:.4f}, Accuracy {epoch_accuracy:.2f}%")
        print(f"Batch size: {batch_size}, Training time for epoch: {epoch_time:.2f} seconds")
        print(f"Batch size: {batch_size}, Communication time for epoch: {comm_time:.4f} seconds")
        bandwidth_utilization = total_data_transferred / comm_time
        print(f"Bandwidth utilization for {world_size} GPUs and batch size {batch_size}: {bandwidth_utilization/1e6:.2f} MB/sec")


def main(gpu_count, model, batch_size=100, epochs=10):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    total_parameters = total_model_parameters(model)
    bytes_per_parameter = 4 # float32
    total_data_transferred = total_parameters * bytes_per_parameter
    
    for gpu_count in gpu_counts:        
        print(f"\nRunning with {gpu_count} GPUs")
        world_size = gpu_count
        mp.spawn(train, args=(world_size, model, batch_size, epochs, total_data_transferred), nprocs=world_size, join=True)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

if __name__ == "__main__":
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    gpu_count = 1
    epochs = 2

    batch_sizes = [32, 128, 512]
    gpu_counts = [1, 2, 4]
    for batch_size in batch_sizes:
        try:
            main(gpu_count, model, batch_size, epochs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} is too large for the available GPU memory.")
                break
            else:
                raise e
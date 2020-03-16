'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from os.path import join, isdir
import time
import argparse
from models import *
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--mom', default=0.9, type=float, help='momentum')
parser.add_argument('--optim', default='adam', choices=['sgd', 'adam'], help='momentum')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--sparsity', default=0.0, type=float, help='convolution backward weight sparsity')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--normal', action='store_true', default=False, help='use pytorch\'s conv layer')
parser.add_argument('--plain', action='store_true', default=False, help='use plain VGG')
parser.add_argument('--model', choices=['VGG_tiny', 'VGG_mini', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'sp1'], default='VGG_tiny', help='pick a VGG')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Device: ', device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
print('Timestamp: ', timestamp)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.plain:
    net = VGG(args.model)
else:
    if args.model == 'sp1':
        net = SpatialModel1(normal=args.normal)
    else:
        net = SpatialVGG(args.model, normal=args.normal)

print('Num of parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    checkpoint_path = join('./checkpoint/', args.resume)
    print('==> Resuming from checkpoint: {}'.format(checkpoint_path))
    assert os.path.isfile(checkpoint_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    timestamp = args.resume[5:-4]

log_dir = join('./logs', timestamp)
if not isdir(log_dir):
    os.makedirs(log_dir)

wandb.init(project='node hijacking')
wandb.config.update(args)

criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.decay)
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)

h, w = 32, 32
shape_map_center = torch.stack(torch.meshgrid(torch.arange(0.5, h, step=1), torch.arange(0.5, w, step=1))).unsqueeze(
    0).repeat(args.batch, 1, 1, 1)
shape_map_var = torch.ones((args.batch, 2, h, w)) / 4  # 4 is a hyper parameter determining the diameter of pixels.
shape_map_cov = torch.zeros((args.batch, 1, h, w))
shape_map = torch.cat([shape_map_center, shape_map_var, shape_map_cov], dim=1).to(device)

def shapes_kernel_loss(model):
    """Added a term that prevents shape kernels to be zero."""
    loss = 0.
    for n, p in model.module.features.named_parameters():
        if 'shapes_kernel' in n:
            loss += 0.0 / (p.mean() * p.std())
    return loss

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs = inputs if args.plain else torch.cat([inputs, shape_map[:inputs.shape[0], ...]], dim=1)
        outputs = net(inputs)
        loss = criterion(outputs, targets)  # + shapes_kernel_loss(net)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # if batch_idx % 10 == 0:
        #     for n, p in net.module.features.named_parameters():
        #         # logging histogram of parameters in the "shape pathway"
        #         if ('shape' in n or 'conv2' in n) and p.grad is not None:
        #             writer.add_histogram(n + '_grad', p.grad, epoch*len(trainloader) + batch_idx)

    wandb.log({'Train Loss': train_loss/(batch_idx+1)}, step=epoch)
    wandb.log({'Train Acc.': 100.*correct/total}, step=epoch)
    # for n, p in net.module.features.named_parameters():
    #     # logging histogram of parameters in the "shape pathway"
    #     if 'shape' in n or 'conv2' in n:
    #         writer.add_histogram(n, p, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(testloader), total=len(testloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs if args.plain else torch.cat([inputs, shape_map[:inputs.shape[0], ...]], dim=1)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        wandb.log({'Test Loss': test_loss/(batch_idx+1)}, step=epoch)
        wandb.log({'Test Acc.': 100.*correct/total}, step=epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'timestamp': timestamp,
            **vars(args)
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + timestamp + '.pth')
        best_acc = acc


for epoch in tqdm(range(start_epoch, start_epoch+200)):
    train(epoch)
    test(epoch)

wandb.log({'hparam/accuracy': best_acc})

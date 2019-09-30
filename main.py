'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from os.path import join
import time
import argparse
from models import *
from tqdm import tqdm
from tensorboardX import SummaryWriter

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
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Device: ', device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
    net = VGG('VGG11')
else:
    net = SpatialVGG('VGG11', normal=args.normal)

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

time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
log_dir = join('./logs', time_stamp)
os.makedirs(log_dir)
writer = SummaryWriter(logdir=log_dir, comment=str(args)+'_'+time_stamp, flush_secs=5)
writer.add_text('args', str(args))
# writer.add_graph(net, torch.zeros(4, 8, 32, 32).to(device))  # doesn't work reliably
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
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % 10 == 0:
            for name, module in net.named_modules():
                if name == 'features':
                    for n, p in module.named_parameters():
                        # logging histogram of parameters in the "shape pathway"
                        if 'shape' in n or 'conv2' in n:
                            print(n + '_grad', p.grad, epoch)

    writer.add_scalar('Train Loss', train_loss/(batch_idx+1), epoch)
    writer.add_scalar('Train Acc.', 100.*correct/total, epoch)
    for name, module in net.named_modules():
        if name == 'features':
            for n, p in module.named_parameters():
                # logging histogram of parameters in the "shape pathway"
                if 'shape' in n or 'conv2' in n:
                    print(n , p, epoch)


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
        writer.add_scalar('Test Loss', test_loss/(batch_idx+1), epoch)
        writer.add_scalar('Test Acc.', 100.*correct/total, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'timestamp': time_stamp,
            **vars(args)
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_'+time_stamp+'.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

writer.add_hparams(vars(args), {'hparam/accuracy': best_acc})

from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Net
from torchvision import transforms
from torchvision.datasets import CIFAR10
from getdata import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')

train_set_img = CIFAR10('dataset', train=True, download=True, 
                                transform=transforms.Compose([
                                                transforms.Grayscale(1),
                                                transforms.Resize(8),
                                                transforms.ToTensor()
                                        ]))

train_set_target = CIFAR10('dataset', train=True, download=True, 
                                transform=transforms.Compose([
                                                transforms.Grayscale(1),
                                                transforms.Resize(32),
                                                transforms.ToTensor()
                                        ]))


test_set_img = CIFAR10('dataset', train=False, download=True, 
                                transform=transforms.Compose([
                                                transforms.Grayscale(1),
                                                transforms.Resize(8),
                                                transforms.ToTensor()
                                        ]))

test_set_target = CIFAR10('dataset', train=False, download=True, 
                                transform=transforms.Compose([
                                                transforms.Grayscale(1),
                                                transforms.Resize(32),
                                                transforms.ToTensor()
                                        ]))

training_image_data_loader = DataLoader(dataset=test_set_img, num_workers=opt.threads, batch_size=opt.batchSize)
testing_image_data_loader = DataLoader(dataset=test_set_img, num_workers=opt.threads, batch_size=opt.testBatchSize)
training_target_data_loader = DataLoader(dataset=test_set_target, num_workers=opt.threads, batch_size=opt.batchSize)
testing_target_data_loader = DataLoader(dataset=test_set_target, num_workers=opt.threads, batch_size=opt.testBatchSize)

print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    for iteration, (batch_img, batch_target) in enumerate(zip(training_image_data_loader, training_target_data_loader), 1):

        input = batch_img[0].to(device)
        target = batch_target[0].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_image_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_image_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for (batch_img, batch_target) in zip(testing_image_data_loader, testing_target_data_loader):
            input, target = batch_img[0].to(device), batch_target[0].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_image_data_loader)))


def checkpoint(epoch):
    model_out_path = "checkpoints/blur_model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)
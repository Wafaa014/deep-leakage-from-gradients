import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot
from model import LeNet, weights_init

#read the arguments
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')

args = parser.parse_args()

#loading CIFAR100 dataset and applying preprocessing transforms
dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0])


gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long()
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(tt(gt_data[0]))
plt.show()

# loading the model
net = LeNet()

# setting the random seed value
torch.manual_seed(1234)

# initialize the weights and setting the criterion
net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient 
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))
#pprint(original_dy_dx)

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).requires_grad_(True)

plt.imshow(tt(dummy_data[0]))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label],0.1) #slower but gets better results 
#optimizer = torch.optim.SGD([dummy_data, dummy_label],0.1)

history = []
for iters in range(300):
    def closure():
        optimizer.zero_grad()
        # passing the dummy data to the model and computing the loss and its gradient
        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        # computing the distance between the 2 gradients and optimizing it
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if iters % 10 == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0]))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()

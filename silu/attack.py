import os
import time
from pathlib import Path
from functools import reduce
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data.dataset import random_split
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torchvision.models as models
import torchvision.transforms

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                      help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                     choices=model_names,
                     help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--content-size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')

parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')

parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory to save the output images')

class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
        x_adv = x.cuda().requires_grad_(True)
        h_adv = self.net(x_adv)
        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        if targeted:
            cost = self.criterion(h_adv, y.cuda())
        else:
            cost = -self.criterion(h_adv, y.cuda())

        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)


        h = self.net(x.cuda())
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h

    def i_fgsm(self, x, y, targeted=False, eps=0.03, alpha=1, iteration=100, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)
            if targeted:
                cost = self.criterion(h_adv, y)
            else:
                cost = -self.criterion(h_adv, y)

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            def clipped(x, _min, _max):
                return torch.max(torch.min(x, _max), _min)
            x_adv.grad.sign_()
            x_adv = x_adv - alpha/iteration*x_adv.grad
            x_adv = clipped(x_adv, x-eps, x+eps)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]#return image path

def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop:
        transform_list.append(torchvision.transforms.CenterCrop(size))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

if __name__ == "__main__":
	args = parser.parse_args()
	traindir = os.path.join(args.data, 'train')
	valdir = os.path.join(args.data, 'val')
	model = models.__dict__[args.arch](pretrained=True)
	torch.cuda.set_device(args.gpu)
	model = model.cuda(args.gpu)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

	p_content_dir = Path(args.data).resolve()
	content_tf = input_transform(args.content_size, args.crop)
	'''
	dataset = []
	content_dir = []
	for p in p_content_dir.glob("*"):
		content_dir.append(p.stem)
	    #print(content_dir)
	    ds = []
	    #print(p_content_dir/p.stem)
	    for ext in ['JPEG', 'jpg']:
	        ds += [x.stem + x.suffix for x in list((p_content_dir/p.stem).glob('*.'+ext))]
	    dataset.append(ds)

	a_content_dir = reduce((lambda x, y: x+y), dataset)
	for ii in range(len(content_dir)):
		cd = content_dir[ii]
		ds = dataset[ii]
		for nnn in ds:
			content_path = p_content_dir / cd / nnn
			print(content_path)
			content_img = Image.open(content_path).convert("RGB")
			image = content_tf(content_img)
			image = image.cuda()
	'''
	val_loader = torch.utils.data.DataLoader(
        MyImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
	att = Attack(model, F.nll_loss)
	output_dir = Path(args.output_dir)
	
	_mean = [0.485, 0.456, 0.406]
	_std = [0.229, 0.224, 0.225]
	for i, data in enumerate(val_loader):
		(image, target), (path, _) = data
		#print(path[0])
		cd = path[0].split("/")[-2]
		name = path[0].split("/")[-1]
		output_dir_cd = output_dir.joinpath(cd)
		if not output_dir_cd.is_dir():
			output_dir_cd.mkdir(parents=True)
		output_path = output_dir_cd.joinpath(name)
		
		x_adv, h_adv, h = att.fgsm(image, target, eps=0.1)
		for j in range(0,3):
			x_adv[0][j]=x_adv[0][j]*_std[j]+_mean[j] 
		save_image(x_adv.cuda(), output_path, padding=0)
		#x_adv, h_adv, h = att.fgsm(image, target, eps=0.0)
		
		

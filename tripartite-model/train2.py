import math 
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim, Tensor
from tripartite_model import TripartiteModel

import numpy as np
import matplotlib.pyplot as plt

import argparse

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR10
from torchvision import transforms

parser = argparse.ArgumentParser() 
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--neg_k', type=int, default=5)
parser.add_argument('--n_hidden', type=int, default=32)
parser.add_argument('--n_couplings', type=int, default=16)
parser.add_argument('--radius', type=float, default = 1.0)
parser.add_argument('--frozen', action='store_true')
parser.add_argument('--normal_init', action='store_true')
parser.add_argument('--lr', type=float, default = 1e-3)
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--n_epochs', type=int, default = 5)
parser.add_argument('--neg_sampling', type=int, default = 3)
parser.add_argument('--pdf_loss', action='store_true')

args = parser.parse_args()

model = TripartiteModel(dim=2, n_hidden=args.n_hidden, n_couplings=args.n_couplings, clip=1.0, radius=args.radius,
                        k=args.k, neg_k=args.neg_k
                        )

data_transforms =  transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

N_CLASSES = 10
NEG_SAMPLING = args.neg_sampling

def target_transform(target: int) -> Tuple[int, np.array]:
    rand_dist = np.arange(N_CLASSES)[np.arange(N_CLASSES) != target]
    return (target, np.random.choice(rand_dist, NEG_SAMPLING))

dataset = CIFAR10('./caltech/', download=True, transform=data_transforms, target_transform=target_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=2)


if torch.cuda.is_available():
    print("using CUDA")
    device = 'cuda:0'
else:
    device = 'cpu'

model.to(device)

vision = resnet18(weights=ResNet18_Weights.DEFAULT)
if args.frozen:
    for param in vision.parameters():
        param.requires_grad = False
num_ftrs = vision.fc.in_features
vision.fc = nn.Linear(num_ftrs, 2)
vision.to(device)

if args.normal_init:
    concepts = torch.normal(torch.zeros((N_CLASSES, model.feature_size), device=device)) * 0.05
else:
    identity = model.couplings[0].generate_identity_feature().repeat(len(model.couplings))
    concepts = identity.repeat(N_CLASSES, 1)
    concepts = concepts.to(device)

concepts.requires_grad=True

params = [concepts] + [x for x in vision.parameters()]
optimizer = optim.Adam(params, lr=args.lr)
losses = []

for epoch in range(args.n_epochs):
    for i, (img, (pos_target, neg_target)) in enumerate(dataloader):
        img = img.to(device)
        neg_target = neg_target.to(device)
        pos_target = pos_target.to(device)
        optimizer.zero_grad()
        features = vision(img)
        if args.pdf_loss:
            _, ladj, log_probs = model.transform(features, concepts[pos_target], with_ladj=True, with_log_probs=True)
            pos_loss = (ladj + log_probs).mean()
            neg_loss = model.transform(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True).mean()
            neg_loss = (ladj + log_probs).mean()
        else:
            pos_loss = model(features, concepts[pos_target]).mean()
            neg_loss = model(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True).mean()
        loss = pos_loss + neg_loss 
        loss.backward()
        losses.append(loss)
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch+1}/{args.n_epochs}, step {i}, loss={sum([l.item() for l in losses])/len(losses)}")
            losses = [] 

import math 
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim, Tensor
from tripartite_model import TripartiteModel

import numpy as np
import matplotlib.pyplot as plt


from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR10
from torchvision import transforms

from tqdm import tqdm 

parser = argparse.ArgumentParser() 
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--k', type=float, default=2)
parser.add_argument('--n_hidden', type=int, default=32)
parser.add_argument('--n_couplings', type=int, default=16)
parser.add_argument('--radius', type=float, default = 1.0)
parser.add_argument('--frozen', action='store_true')
parser.add_argument('--lr', type=float, default = 1e-3)
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--n_epochs', type=int, default = 5)
parser.add_argument('--clip', type=float, default = 1.0)
parser.add_argument('--neg_sampling', type=int, default = 3)
parser.add_argument('--pdf_loss', action='store_true')
parser.add_argument('--no_neg_sampling', action='store_true')

args = parser.parse_args()

model = TripartiteModel(dim=args.dim,
                        n_hidden=args.n_hidden,
                        n_couplings=args.n_couplings,
                        clip=args.clip, 
                        radius=args.radius,
                        k=args.k 
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
train, test = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))], generator=torch.Generator().manual_seed(42))

train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True, num_workers=2)


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
vision.fc = nn.Linear(num_ftrs, args.dim)
vision.to(device)

identity = model.couplings[0].generate_identity_feature().repeat(len(model.couplings))
concepts = identity.repeat(N_CLASSES, 1)
concepts += torch.normal(torch.zeros((N_CLASSES, model.feature_size))) * 0.01
concepts = concepts.to(device)

concepts.requires_grad=True

params = [concepts] + [x for x in vision.parameters()]
optimizer = optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
losses = []
pos_losses = []
neg_losses = []

def validate(model, vision, concepts):
    with torch.no_grad():
        n = 0
        pos_correct = 0
        neg_correct = 0
        pos_mean = 0
        neg_mean = 0
        for img, (pos_target, neg_target) in tqdm(test_dataloader):
            n += len(pos_target)
            img = img.to(device)
            neg_target = neg_target.to(device)
            pos_target = pos_target.to(device)
            optimizer.zero_grad()
            features = vision(img)
            pos_position, pos_outputs = model(features, concepts[pos_target], return_position=True)
            pos_correct += (pos_outputs < -math.log(0.5)).sum().item()
            neg_position, neg_outputs = (model(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True, return_position=True))
            neg_mean += torch.exp(-neg_outputs).sum()
            pos_mean += torch.exp(-pos_outputs).sum()
            neg_correct += (neg_outputs < -math.log(0.5)).sum().item()
        print(f'Positive examples: {pos_correct/n:.3f}\tNegative examples{neg_correct/(NEG_SAMPLING*n):.3f}')
        print(f'Positive mean: {pos_mean/n:.3f}\tNegative mean{neg_mean/(NEG_SAMPLING*n):.3f}')


for epoch in range(args.n_epochs):
    for i, (img, (pos_target, neg_target)) in enumerate(train_dataloader):
        img = img.to(device)
        neg_target = neg_target.to(device)
        pos_target = pos_target.to(device)
        optimizer.zero_grad()
        features = vision(img)
        if args.pdf_loss:
            _, ladj, log_probs = model.transform(features, concepts[pos_target], with_ladj=True, with_log_probs=True)
            pos_loss = (-ladj - log_probs).mean()
            _, ladj, log_probs = model.transform(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True, with_ladj=True, with_log_probs=True)
            neg_loss = (-ladj - log_probs).mean()
        else:
            pos_loss = model(features, concepts[pos_target]).mean()
            neg_loss = model(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True).mean()
            neg_loss = neg_loss.mean()

        if args.no_neg_sampling:
            loss = pos_loss
        else:
            loss = args.alpha*pos_loss + (1-args.alpha)*neg_loss 

        loss.backward()
        losses.append(loss)
        pos_losses.append(pos_loss)
        neg_losses.append(neg_loss)
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch+1}/{args.n_epochs}, step {i}")
            print(f"loss={sum([l.item() for l in losses])/len(losses)} ")
            print(f"pos_loss={sum([l.item() for l in pos_losses])/len(pos_losses)} ")
            print(f"neg_loss={sum([l.item() for l in neg_losses])/len(neg_losses)} ")
            validate(model, vision, concepts)
            losses = [] 
            neg_losses = [] 
            pos_losses = [] 
    scheduler.step()

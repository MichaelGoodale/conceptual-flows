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




def train_model(alpha=0.9, dim=2, k=2, n_hidden=32, n_couplings=16, 
                radius=1.0, frozen=False, lr=1e-3, batch_size=128, n_epochs=5,
                clip=1.0, neg_sampling=3, pdf_loss=False, no_neg_sampling=False):

    model = TripartiteModel(dim=dim,
                            n_hidden=n_hidden,
                            n_couplings=n_couplings,
                            clip=clip, 
                            radius=radius,
                            k=k 
                            )

    data_transforms =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    CLASSES =  ['airplane',
      'automobile',
      'bird',
      'cat',
      'deer',
      'dog',
      'frog',
      'horse',
      'ship',
      'truck']

    N_CLASSES = len(CLASSES)
    NEG_SAMPLING = neg_sampling

    def target_transform(target: int) -> Tuple[int, np.array]:
        rand_dist = np.arange(N_CLASSES)[np.arange(N_CLASSES) != target]
        return (target, np.random.choice(rand_dist, NEG_SAMPLING))

    dataset = CIFAR10('./caltech/', download=True, transform=data_transforms, target_transform=target_transform)
    train, test = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))], generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=2)


    if torch.cuda.is_available():
        print("using CUDA")
        device = 'cuda:0'
    else:
        device = 'cpu'

    model.to(device)

    vision = resnet18(weights=ResNet18_Weights.DEFAULT)
    if frozen:
        for param in vision.parameters():
            param.requires_grad = False
    num_ftrs = vision.fc.in_features
    vision.fc = nn.Linear(num_ftrs, dim)
    vision.to(device)

    identity = model.couplings[0].generate_identity_feature().repeat(len(model.couplings))
    #concepts = identity.repeat(N_CLASSES, 1)
    concepts = torch.normal(torch.zeros((N_CLASSES, model.feature_size))) 
    concepts = concepts.to(device)

    concepts.requires_grad=True

    params = [concepts] + [x for x in vision.parameters()]
    optimizer = optim.Adam(params, lr=lr)
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
            cutoff = math.log(0.5)
            for img, (pos_target, neg_target) in tqdm(test_dataloader):
                n += len(pos_target)
                img = img.to(device)
                neg_target = neg_target.to(device)
                pos_target = pos_target.to(device)
                optimizer.zero_grad()
                features = vision(img)
                pos_outputs = model(features, concepts[pos_target])
                pos_correct += (pos_outputs > cutoff).sum().item()
                pos_mean += torch.exp(pos_outputs).sum()

                neg_outputs = (model(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True))
                neg_correct += (neg_outputs > cutoff).sum().item()
                neg_mean += torch.exp(neg_outputs).sum()
            print(f'Positive examples: {pos_correct/n:.3f}\tNegative examples{neg_correct/(NEG_SAMPLING*n):.3f}')
            print(f'Positive mean: {pos_mean/n:.3f}\tNegative mean{neg_mean/(NEG_SAMPLING*n):.3f}')
            boundary = model.distribution.generate_boundary(1000).to(device)
            for concept_idx, name in enumerate(CLASSES):
                plt.scatter(*model.inverse_transform(boundary, concepts[concept_idx])[0].T.cpu().detach(), label=name)
            plt.legend()
            plt.show()


    for epoch in range(n_epochs):
        for i, (img, (pos_target, neg_target)) in enumerate(train_dataloader):
            img = img.to(device)
            neg_target = neg_target.to(device)
            pos_target = pos_target.to(device)
            optimizer.zero_grad()
            features = vision(img)

            _, log_probs = model.transform(features, concepts[pos_target], with_log_probs=True)
            pos_loss = (-log_probs).mean()
            _, log_probs = model.transform(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True, with_log_probs=True)
            neg_loss = (-log_probs).mean()
            loss = alpha*pos_loss + (1-alpha)*neg_loss 

            loss.backward()
            losses.append(loss)
            pos_losses.append(pos_loss)
            neg_losses.append(neg_loss)
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, step {i}")
                print(f"loss={sum([l.item() for l in losses])/len(losses)} ")
                print(f"pos_loss={sum([l.item() for l in pos_losses])/len(pos_losses)} ")
                print(f"neg_loss={sum([l.item() for l in neg_losses])/len(neg_losses)} ")
                validate(model, vision, concepts)
                losses = [] 
                neg_losses = [] 
                pos_losses = [] 
        scheduler.step()


if __name__ == '__main__':
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

    args = parser.parse_args()

    train_model(alpha=args.alpha, dim=args.dim, k=args.k, n_hidden=args.n_hidden,
                n_couplings=args.n_couplings, radius=args.radius, frozen=args.frozen,
                lr=args.lr, batch_size=args.batch_size, n_epochs=args.n_epochs,
                clip=args.clip, neg_sampling=args.neg_sampling)

import math 
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim, Tensor
import torch.nn.functional as F
from tripartite_model import TripartiteModel

import numpy as np
import matplotlib.pyplot as plt


from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR10
from torchvision import transforms

from tqdm import tqdm 




def train_model(alpha: float = 0.9, dim: int = 2, k: float = 2, n_hidden: int = 32, n_couplings: int = 16, 
                frozen: bool = False, lr:float = 1e-3, batch_size: int = 128, n_epochs: int = 5,
                clip: float = 1.0, c:float = 2/3., neg_sampling: int = 3, pdf_loss: bool = False, no_neg_sampling: bool = False, scale:float = 0.15,
                sample_batch_size: int = 32):

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

    def get_alternatives(target: int, n=NEG_SAMPLING) -> int:
        rand_dist = np.arange(N_CLASSES)[np.arange(N_CLASSES) != target]
        return np.random.choice(rand_dist, n)

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

    model = TripartiteModel(dim=dim,
                            n_hidden=n_hidden,
                            n_couplings=n_couplings,
                            clip=clip, 
                            k=k,
                            c=c
                            )

    model.to(device)

    generated_model = TripartiteModel(dim=dim,
                            n_hidden=8,
                            n_couplings=2,
                            clip=clip, 
                            k=k,
                            c=c
                            )

    generated_model.to(device)

    vision = resnet18(weights=ResNet18_Weights.DEFAULT)
    if frozen:
        for param in vision.parameters():
            param.requires_grad = False
    num_ftrs = vision.fc.in_features
    vision.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                              nn.ReLU(),
                              nn.Linear(num_ftrs, generated_model.feature_size))
    vision.to(device)

    identity = model.couplings[0].generate_identity_feature().repeat(len(model.couplings))
    concepts = torch.zeros((N_CLASSES, model.feature_size))
    concepts = torch.normal(concepts) * scale
    concepts = concepts.to(device)

    concepts.requires_grad=True

    params = [concepts] + [x for x in vision.parameters()]
    optimizer = optim.AdamW(params, lr=lr)
    losses = []

    def validate(model, vision, concepts):
        with torch.no_grad():
            n = 0
            pos_correct = 0
            neg_correct = 0
            pos_mean = 0
            neg_mean = 0
            cutoff = 0.5
            for img, (pos_target, neg_target) in tqdm(test_dataloader):
                n += len(pos_target)
                img = img.to(device)
                neg_target = neg_target.to(device)
                pos_target = pos_target.to(device)
                optimizer.zero_grad()
                features = vision(img)

                features = generated_model.sample(features, sample_batch_size).view(-1, dim)
                pos_target = pos_target.repeat_interleave(sample_batch_size, 0)
                neg_target = neg_target.repeat_interleave(sample_batch_size, 0)

                pos_outputs = model(features, concepts[pos_target])
                pos_outputs = torch.exp(pos_outputs).reshape(-1, sample_batch_size).mean(dim=-1)
                pos_correct += (pos_outputs > cutoff).sum().item()
                pos_mean += pos_outputs.sum()

                neg_outputs = (model(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True))
                neg_outputs = torch.exp(neg_outputs).reshape(-1, sample_batch_size).mean(dim=-1)
                neg_correct += (neg_outputs > cutoff).sum().item()
                neg_mean += neg_outputs.sum()
            print(f'Positive examples: {pos_correct/(n):.3f}\tNegative examples{neg_correct/(NEG_SAMPLING*n):.3f}')
            print(f'Positive mean: {pos_mean/(n):.3f}\tNegative mean{neg_mean/(NEG_SAMPLING*n):.3f}')
            boundary = model.distribution.generate_boundary(1000).to(device)
            for concept_idx, name in enumerate(CLASSES):
                plt.scatter(*model.inverse_transform(boundary, concepts[concept_idx])[0].T.cpu().detach(), color=plt.cm.tab20(concept_idx), label=name)
                plt.scatter(*model.sample(concepts[concept_idx], n=100)[0].T.cpu().detach(), alpha=0.15, color=plt.cm.tab20(concept_idx),)
            plt.legend()
            plt.show()


    validate(model, vision, concepts)
    for epoch in range(n_epochs):
        for i, (img, (pos_target, neg_target)) in enumerate(tqdm(train_dataloader)):
            uniq_concepts = pos_target.unique()

            neg_target = neg_target.to(device)
            img = img.to(device)
            pos_target = pos_target.to(device)
            optimizer.zero_grad()
            features = vision(img)

            features = generated_model.sample(features, sample_batch_size).view(-1, dim)
            pos_target = pos_target.repeat_interleave(sample_batch_size, 0)
            neg_target = neg_target.repeat_interleave(sample_batch_size, 0)

            _, log_probs = model.transform(features, concepts[pos_target], with_log_probs=True)
            pos_loss = (-log_probs).mean()
            pos_loss_class = -model(features, concepts[pos_target]).mean()
            neg_loss = -model(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_target.view(-1)], negative_example=True).mean()*NEG_SAMPLING
            loss = alpha*pos_loss + (1-alpha)*(pos_loss_class+neg_loss)

            loss.backward()
            losses.append(loss)
            optimizer.step()
        print(f"Epoch {epoch+1}/{n_epochs}, step {i}")
        print(f"loss={sum([l.item() for l in losses])/len(losses)} ")
        validate(model, vision, concepts)
        losses = [] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--k', type=float, default=2)
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--n_couplings', type=int, default=16)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--n_epochs', type=int, default = 5)
    parser.add_argument('--clip', type=float, default = 1.0)
    parser.add_argument('--neg_sampling', type=int, default = 3)
    parser.add_argument('--center', type=float, default = 2/3.)
    parser.add_argument('--scale', type=float, default = 0.15)
    parser.add_argument('--sample_batch_size', type=int, default = 32)

    args = parser.parse_args()

    train_model(alpha=args.alpha, dim=args.dim, k=args.k, n_hidden=args.n_hidden,
                n_couplings=args.n_couplings, frozen=args.frozen,
                lr=args.lr, batch_size=args.batch_size, n_epochs=args.n_epochs, scale=args.scale
                clip=args.clip, neg_sampling=args.neg_sampling, c=args.center)

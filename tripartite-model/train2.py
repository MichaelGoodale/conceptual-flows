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


from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import CIFAR10
from torchvision import transforms

from tqdm import tqdm 




def train_model(alpha=0.9, dim=2, k=2, n_hidden=32, n_couplings=16, 
                radius=1.0, frozen=False, lr=1e-3, batch_size=128, n_epochs=5,
                clip=1.0, c=2/3., neg_sampling=3, pdf_loss=False, no_neg_sampling=False):

    model = TripartiteModel(dim=dim,
                            n_hidden=n_hidden,
                            n_couplings=n_couplings,
                            clip=clip, 
                            radius=radius,
                            k=k,
                            c=c
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

    model.to(device)

    vision = resnet50(weights=ResNet50_Weights.DEFAULT)
    if frozen:
        for param in vision.parameters():
            param.requires_grad = False
    num_ftrs = vision.fc.in_features
    vision.fc = nn.Linear(num_ftrs, dim)
    vision.to(device)

    identity = model.couplings[0].generate_identity_feature().repeat(len(model.couplings))
    concepts = torch.zeros((N_CLASSES, model.feature_size))
    torch.nn.init.xavier_uniform_(concepts, gain=torch.nn.init.calculate_gain('selu'))
    concepts = concepts.to(device)

    concepts.requires_grad=True

    params = [concepts] + [x for x in vision.parameters()]
    optimizer = optim.AdamW(params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    losses = []
    real_losses = []
    sample_losses = []

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
        for i, (img, (pos_target, neg_samples)) in enumerate(tqdm(train_dataloader)):
            uniq_concepts = pos_target.unique()
            neg_targets = torch.tensor(np.vstack([get_alternatives(x.item()) for x in uniq_concepts]))

            neg_samples = neg_samples.to(device)
            neg_targets = neg_targets.to(device)
            img = img.to(device)
            pos_target = pos_target.to(device)
            optimizer.zero_grad()
            features = vision(img)

            _, log_probs = model.transform(features, concepts[pos_target], with_log_probs=True)
            pos_loss = (-log_probs).mean()
            log_probs = model(features.repeat_interleave(NEG_SAMPLING, 0), concepts[neg_samples.view(-1)], negative_example=True)
            neg_loss = (-log_probs).mean()
            real_loss = pos_loss + neg_loss

            # Sample from each distribution and pass to negative of different.
            batch = model.sample(concepts[neg_targets.view(-1)], batch_size)
            neg_weights = concepts[uniq_concepts].repeat_interleave(NEG_SAMPLING, 0).unsqueeze(1).expand(-1, batch_size, -1)
            log_probs = model(batch, neg_weights, negative_example=True)
            sample_loss = (-log_probs).mean()
            loss = alpha*real_loss + (1-alpha)*sample_loss

            loss.backward()
            losses.append(loss)
            real_losses.append(real_loss)
            sample_losses.append(sample_loss)
            optimizer.step()
        print(f"Epoch {epoch+1}/{n_epochs}, step {i}")
        print(f"loss={sum([l.item() for l in losses])/len(losses)} ")
        print(f"real_loss={sum([l.item() for l in real_losses])/len(real_losses)} ")
        print(f"sample_loss={sum([l.item() for l in sample_losses])/len(sample_losses)} ")
        validate(model, vision, concepts)
        losses = [] 
        real_losses = [] 
        sample_losses = [] 
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
    parser.add_argument('--center', type=float, default = 2/3.)

    args = parser.parse_args()

    train_model(alpha=args.alpha, dim=args.dim, k=args.k, n_hidden=args.n_hidden,
                n_couplings=args.n_couplings, radius=args.radius, frozen=args.frozen,
                lr=args.lr, batch_size=args.batch_size, n_epochs=args.n_epochs,
                clip=args.clip, neg_sampling=args.neg_sampling, c=args.center)

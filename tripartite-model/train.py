import torch
from tqdm import tqdm
from torch import optim 
from tripartite_model import TripartiteModel
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
import numpy as np

import math 

import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--n_hidden', type=int, default=16)
parser.add_argument('--n_couplings', type=int, default=64)
parser.add_argument('--radius', type=float, default = 2.0)
parser.add_argument('--lr', type=float, default = 1e-3)
parser.add_argument('--batch_size', type=int, default = 128)

args = parser.parse_args()

model = TripartiteModel(dim=2, n_hidden=args.n_hidden, n_couplings=args.n_couplings, clip=1.0, radius=args.radius)

if torch.cuda.is_available():
    print("USING CUDA")
    device = 'cuda:0'
else:
    device = 'cpu'

model.to(device)

W = torch.normal(torch.zeros(model.feature_size, device=device)) * 0.05
W.requires_grad=True

optimizer = optim.Adam([W], lr=args.lr)

X, y = make_moons(n_samples = 1000)
A = X[y==0] 
B = X[y==1] 

for i in range(2000):
    optimizer.zero_grad() 
    data = A[np.random.choice(np.arange(len(A)), size=args.batch_size)]
    data = torch.tensor(data, device=device, dtype=torch.float)
    _, log_probs = model.transform(data, W, with_log_probs=True)
    loss = (-log_probs).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Step {i}, loss={loss.item()}")

samples = model.sample(W, n=2000)
#neg_samples = model.sample(W, n=2000, negative_example=True)

plt.scatter(*A.T, label='A')
#plt.scatter(*B.T, label='B')
##plt.scatter(*neg_samples.T.cpu().detach(), label='Negative examples')
#plt.scatter(*samples.T.cpu().detach(), label='Positive examples')
plt.legend()
plt.show()

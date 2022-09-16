import torch
from tqdm import tqdm
from torch import optim 
from tripartite_model import TripartiteModel
import matplotlib.pyplot as plt
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


w1 = lambda z: torch.sin(2 * math.pi * z[:, 0] / 4)
w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
w3 = lambda z: 3 * (1.0 / (1 + torch.exp(-(z[:, 0] - 1) / 0.3)))

def density_wave(z):
    z = torch.reshape(z, [z.shape[0], 2])
    z1, z2 = z[:, 0], z[:, 1]
    u = 0.5 * ((z2 - w1(z))/0.4) ** 2
    u[torch.abs(z1) > 4] = 1e8
    return torch.exp(-u)

optimizer = optim.Adam([W], lr=args.lr)

for i in range(3_000):
    optimizer.zero_grad() 
    x, det_jac, log_probs = model.sample(W, args.batch_size, with_ladj=True, with_log_probs=True)
    loss = torch.abs(log_probs + det_jac - torch.log(density_wave(x) + 1e-9)).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Step {i}, loss={loss.item()}")

samples = model.sample(W)
plt.scatter(*samples.T.detach())
plt.show()

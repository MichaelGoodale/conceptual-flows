import torch
from tqdm import tqdm
from torch import optim 
from tripartite_model import TripartiteModel
import matplotlib.pyplot as plt
import math 


model = TripartiteModel(dim=2, n_hidden=16, n_couplings=64, clip=1.0, radius=5.0)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

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

optimizer = optim.Adam([W], lr=1e-3)

torch.autograd.set_detect_anomaly(True)

for i in range(3_000):
    optimizer.zero_grad() 
    x, det_jac, log_probs = model.sample(W, 128, with_ladj=True, with_log_probs=True)
    loss = torch.abs(log_probs+det_jac - torch.log(density_wave(x) + 1e-9)).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Step {i}, loss={loss.item()}")

samples = model.sample(W)
plt.scatter(*samples.T.detach())
plt.show()

import math
from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.distributions as D
import torch.nn.functional as F


class BallHomeomorphism():
    ''' Takes points in Rn and transforms them to the unit ball, or vice versa'''

    def __init__(self, dim, radius=1., eps=1e-9):
        self.dim = dim
        self.radius = radius
        self.eps = eps

    def to_ball(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        ladj = (self.radius ** self.dim) / ((norm+1) ** (self.dim+1))
        ladj = torch.log(torch.abs(ladj) + self.eps).squeeze(dim=-1)
        return self.radius*x / (1 + norm), ladj

    def from_ball(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        ladj = self.radius / ((self.radius - norm) ** (self.dim+1))
        ladj = torch.log(torch.abs(ladj)+self.eps).squeeze(dim=-1)
        return x / (self.radius - norm), ladj

class MaskedCouplingFlow(nn.Module):
    def __init__(self, dim, mask=None, n_hidden=64, n_layers=2, activation=F.selu, clip=1.0, eps=1e-9):
        super().__init__()
        self.dim = dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        if mask == None:
            mask = torch.ones(dim)
            mask[::2] = 0
        self.register_buffer('mask', mask)
        self.clip = clip
        self.eps = eps

    def net_forward(self, z: Tensor, W: List[Tensor], B: List[Tensor]) -> Tensor:
        ''' Run through a neural network of self.n_layer dimensions 
        given the weights, biases and the input.'''
        z = z.unsqueeze(1)
        for w, b in zip(W, B):
            z = self.activation(torch.matmul(z, w) + b)
        return z.squeeze(1)

    @property
    def feature_size(self) -> int:
        '''Determine how big a feature vector should be to represent this coupling'''

        weight_size = (self.dim * 2 + (self.n_layers - 2) * self.n_hidden) * self.n_hidden
        bias_size = self.dim + (self.n_layers - 1) * self.n_hidden
        return 2*(weight_size + bias_size)

    def generate_identity_feature(self) -> Tensor:
        w = []
        for i in range(self.n_layers):
            if i == 0:
                w.append(torch.eye(self.dim, self.n_hidden))
                w.append(torch.eye(self.dim, self.n_hidden))
                w.append(torch.zeros(self.n_hidden))
                w.append(torch.zeros(self.n_hidden))
            elif i == self.n_layers - 1:
                w.append(torch.eye(self.n_hidden, self.dim))
                w.append(torch.eye(self.n_hidden, self.dim))
                w.append(torch.zeros(self.dim))
                w.append(torch.zeros(self.dim))
            else:
                w.append(torch.eye(self.n_hidden, self.n_hidden))
                w.append(torch.eye(self.n_hidden, self.n_hidden))
                w.append(torch.zeros(self.n_hidden))
                w.append(torch.zeros(self.n_hidden))
        w = torch.concat([x.view(-1) for x in w])
        return w

    def get_weights_and_biases(self, W: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        '''Transform a vector representing the terms to the weights and biases of s and t
        Pretty ugly code but it gets the job done. 
        '''

        if W.shape[-1] != self.feature_size:
            raise ValueError(f'Weight vector W should have size{self.feature_size} not {W.shape}')

        if W.dim() == 1:
            W = W.unsqueeze(0) 

        batch_size = W.shape[0]
        s_w, s_b, t_w, t_b = [], [], [], []
        consumed = 0 
        for i in range(self.n_layers):
            if i == 0:
                w = W[:, consumed:consumed+(self.dim*self.n_hidden)].reshape(batch_size, self.dim, self.n_hidden)
                s_w.append(w)
                consumed += self.dim * self.n_hidden
                w = W[:, consumed:consumed+(self.dim*self.n_hidden)].reshape(batch_size, self.dim, self.n_hidden)
                t_w.append(w)
                consumed += self.dim * self.n_hidden

                w = W[:, consumed:consumed+self.n_hidden]
                s_b.append(w.unsqueeze(1))
                consumed += self.n_hidden
                w = W[:, consumed:consumed+self.n_hidden]
                t_b.append(w.unsqueeze(1))
                consumed += self.n_hidden
            elif i == self.n_layers - 1:
                w = W[:, consumed:consumed+(self.n_hidden*self.dim)].reshape(batch_size, self.n_hidden, self.dim)
                s_w.append(w)
                consumed += self.dim * self.n_hidden
                w = W[:, consumed:consumed+(self.n_hidden*self.dim)].reshape(batch_size, self.n_hidden, self.dim)
                t_w.append(w)
                consumed += self.dim * self.n_hidden

                w = W[:, consumed:consumed+self.dim]
                s_b.append(w.unsqueeze(1))
                consumed += self.dim
                w = W[:, consumed:consumed+self.dim]
                t_b.append(w.unsqueeze(1))
                consumed += self.dim
            else:
                w = W[:, consumed:consumed+(self.n_hidden*self.n_hidden)].reshape(batch_size, self.n_hidden, self.n_hidden)
                s_w.append(w)
                consumed += self.n_hidden * self.n_hidden
                w = W[:, consumed:consumed+(self.n_hidden*self.n_hidden)].reshape(batch_size, self.n_hidden, self.n_hidden)
                t_w.append(w)
                consumed += self.n_hidden * self.n_hidden

                w = W[:, consumed:consumed+self.n_hidden]
                s_b.append(w.unsqueeze(1))
                consumed += self.n_hidden
                w = W[:, consumed:consumed+self.n_hidden]
                t_b.append(w.unsqueeze(1))
                consumed += self.n_hidden
        return s_w, s_b, t_w, t_b
    
    def get_s(self, masked_x, s_w, s_b):
        s = self.net_forward(masked_x, s_w, s_b)
        return self.clip * torch.tanh(s / self.clip)

    def forward(self, x, W):
        s_w, s_b, t_w, t_b = self.get_weights_and_biases(W)
        masked_x = self.mask * x
        s = self.get_s(masked_x, s_w, s_b)
        t = self.net_forward(masked_x, t_w, t_b)

        y = masked_x + (1-self.mask) * (x * (torch.exp(s)+self.eps) + t) 
        log_abs_det_jacobian = torch.sum(torch.abs(s), dim=-1)

        return y, log_abs_det_jacobian

    def backward(self, x, W):
        s_w, s_b, t_w, t_b = self.get_weights_and_biases(W)
        masked_x = self.mask * x
        s = self.get_s(masked_x, s_w, s_b)
        t = self.net_forward(masked_x, t_w, t_b)

        x = masked_x + (1-self.mask) * (x - t) * (torch.exp(-s) + self.eps)
        log_abs_det_jacobian = -torch.sum(torch.abs(s), dim=-1)

        return x, log_abs_det_jacobian

class ConceptDistribution():

    def __init__(self, dim=2, k=5, radius=1.0, eps=1e-9):
        '''
            Args:
                dim (int): number of dimensions
                k (float): scaling factor for how aggressive the falloff is. It should be fairly aggressive to avoid points near the edge of the unit ball. 
                radius (float): radius of the ball from which samples are drawn.
                eps (float): Epsilon value
        '''
        self.dim = dim
        self.radius = radius
        self.k = k
        self.eps = eps

    def log_prob(self, x: Tensor, negative_example=False):
        #We don't account for the prob of the angle of the vector
        #Luckily, that's uniform so while this technically isn't the PDF, it's proportionate to the PDF
        #Alternatively, it is the PDF if we view it as a function of ||x|| rather than x.

        r = torch.linalg.vector_norm(x, dim=-1) / self.radius
        if negative_example: # These are the derivatives of their CDF, (i.e., their PDF)
            prob = math.sqrt(math.pi)*math.erf(self.k) * torch.exp(torch.erfinv((r * math.erf(self.k)) ** 2)) / (2*self.k)
        else:
            prob = 2*self.k*torch.exp(-self.k**2 * r ** 2) / (math.sqrt(math.pi) * math.erf(self.k))

        return -torch.log(prob+self.eps)

    def log_cdf(self, x: Tensor, negative_example=False):
        r = torch.linalg.vector_norm(x, dim=-1) / self.radius
        if negative_example: # These are the derivatives of their CDF, (i.e., their PDF)
            prob = torch.erfinv(math.erf(self.k) * r) / self.k
        else:
            prob = 1-torch.erf(self.k*r) / math.erf(self.k) #1 - is just to make the CDF go from right to left rather than left to right
        return -torch.log(prob+self.eps)

    def sample(self, n: int, negative_example=False):
        '''Adapted from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/'''
        u = torch.normal(mean=torch.zeros(n, self.dim))
        u = F.normalize(u)
        r = torch.rand(n, 1)
        if negative_example:
            r = torch.erf(self.k *  r) / math.erf(self.k)
        else:
            r = torch.erfinv(math.erf(self.k) * r) / self.k
        return self.radius*r*u

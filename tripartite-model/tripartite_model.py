from typing import Tuple, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import transforms 
from torch.distributions.multivariate_normal import MultivariateNormal

from utils import ConceptDistribution

class BallHomeomorphism():
    ''' Takes points in Rn and transforms them to the unit ball, or vice versa'''

    def __init__(self, dim, radius=1.):
        self.dim = dim
        self.radius = radius

    def to_ball(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        ladj = (self.radius ** self.dim) / ((norm+1) ** (self.dim+1))
        ladj = torch.log(torch.abs(ladj)).squeeze(dim=-1)
        return self.radius*x / (1 + norm), ladj

    def from_ball(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        ladj = self.radius / ((self.radius - norm) ** (self.dim+1))
        ladj = torch.log(torch.abs(ladj)).squeeze(dim=-1)
        return x / (self.radius - norm), ladj

class MaskedCouplingFlow(nn.Module):
    def __init__(self, dim, mask=None, n_hidden=64, n_layers=2, activation=F.selu, clip=1.0, eps=1e-9):
        super().__init__()
        self.dim = dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.register_buffer('mask', mask)
        self.clip = clip
        self.eps = eps

    def net_forward(self, z: Tensor, W: List[Tensor], B: List[Tensor]) -> Tensor:
        ''' Run through a neural network of self.n_layer dimensions 
        given the weights, biases and the input.'''

        for w, b in zip(W, B):
            z = self.activation(F.linear(z, w.T, b))
        return z

    @property
    def feature_size(self) -> int:
        '''Determine how big a feature vector should be to represent this coupling'''

        weight_size = (self.dim * 2 + (self.n_layers - 2) * self.n_hidden) * self.n_hidden
        bias_size = self.dim + (self.n_layers - 1) * self.n_hidden
        return 2*(weight_size + bias_size)

    def get_weights_and_biases(self, W: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        '''Transform a vector representing the terms to the weights and biases of s and t
        Pretty ugly code but it gets the job done. 
        '''

        if W.shape !=(self.feature_size,):
            raise ValueError(f'Weight vector W should have size{self.feature_size} not {W.shape}')
        s_w, s_b, t_w, t_b = [], [], [], []
        consumed = 0 
        for i in range(self.n_layers):
            if i == 0:
                w = W[consumed:consumed+(self.dim*self.n_hidden)].reshape(self.dim, self.n_hidden)
                s_w.append(w)
                consumed += torch.numel(w)
                w = W[consumed:consumed+(self.dim*self.n_hidden)].reshape(self.dim, self.n_hidden)
                t_w.append(w)
                consumed += torch.numel(w)

                w = W[consumed:consumed+self.n_hidden]
                s_b.append(w)
                consumed += torch.numel(w)
                w = W[consumed:consumed+self.n_hidden]
                t_b.append(w)
                consumed += torch.numel(w)
            elif i == self.n_layers - 1:
                w = W[consumed:consumed+(self.n_hidden*self.dim)].reshape(self.n_hidden, self.dim)
                s_w.append(w)
                consumed += torch.numel(w)
                w = W[consumed:consumed+(self.n_hidden*self.dim)].reshape(self.n_hidden, self.dim)
                t_w.append(w)
                consumed += torch.numel(w)

                w = W[consumed:consumed+self.dim]
                s_b.append(w)
                consumed += torch.numel(w)
                w = W[consumed:consumed+self.dim]
                t_b.append(w)
                consumed += torch.numel(w)
            else:
                w = W[consumed:consumed+(self.n_hidden*self.n_hidden)].reshape(self.n_hidden, self.n_hidden)
                s_w.append(w)
                consumed += torch.numel(w)
                w = W[consumed:consumed+(self.n_hidden*self.n_hidden)].reshape(self.n_hidden, self.n_hidden)
                t_w.append(w)
                consumed += torch.numel(w)

                w = W[consumed:consumed+self.n_hidden]
                s_b.append(w)
                consumed += torch.numel(w)
                w = W[consumed:consumed+self.n_hidden]
                t_b.append(w)
                consumed += torch.numel(w)
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
        log_abs_det_jacobian = torch.sum((1-self.mask)*torch.abs(s), dim=-1)

        return y, log_abs_det_jacobian

    def backward(self, x, W):
        s_w, s_b, t_w, t_b = self.get_weights_and_biases(W)
        masked_x = self.mask * x
        s = self.get_s(masked_x, s_w, s_b)
        t = self.net_forward(masked_x, t_w, t_b)

        log_abs_det_jacobian = -torch.sum((1-self.mask)*torch.abs(s), dim=-1)
        x = masked_x + (1-self.mask) * (x - t) * (torch.exp(-s) + self.eps)

        return x, log_abs_det_jacobian



class TripartiteModel(nn.Module):

    def __init__(self, dim:int =32, n_couplings:int =4, n_hidden=32, clip=1.0, radius=2.0):
        '''
        Args:
            dim: Dimensionality of e-space
            n_coupling: Number of layers of couplings to go through
            buffer: Buffer for sigmoid function (where sigmoid(x)=0.5)
        '''
        super().__init__()
        self.distribution = ConceptDistribution(dim, radius=radius)
        self.homeomorphism = BallHomeomorphism(dim, radius=radius)
        self.radius = radius
        self.couplings = nn.ModuleList()
        mask = torch.ones(dim)
        mask[::2] = 0
        for i in range(n_couplings):
            self.couplings.append(MaskedCouplingFlow(dim, mask, n_hidden=n_hidden, clip=clip))
            mask = 1-mask

    @property
    def feature_size(self):
        return sum(x.feature_size for x in self.couplings)

    def split_weights(self, W):
        if W.shape !=(self.feature_size,):
            raise ValueError(f'Weight vector W should have size{self.feature_size} not {W.shape}')
        return torch.split(W, self.feature_size // len(self.couplings), dim=-1)

    def transform(self, x: Tensor, W: Tensor, with_ladj=False):
        '''
        Takes x in E-space coordinates and converts them to the predicate 
        described by the weighting tensor W
        '''
        predicate_position, log_abs_det_jacobian = self.homeomorphism.from_ball(x)

        for coupling, weight in zip(self.couplings, self.split_weights(W)):
            predicate_position, ladj = coupling.backward(predicate_position, weight)
            log_abs_det_jacobian += ladj

        if with_ladj: 
            return predicate_position, log_abs_det_jacobian
        return predicate_position

    def inverse_transform(self, x: Tensor, W: Tensor):
        e_position = x
        log_abs_det_jacobian = torch.zeros(x.shape[0], device=x.device)

        for coupling, weight in zip(reversed(self.couplings), reversed(self.split_weights(W))):
            e_position, ladj = coupling.forward(e_position, weight)
            log_abs_det_jacobian += ladj

        e_position, ladj = self.homeomorphism.to_ball(e_position)
        log_abs_det_jacobian += ladj
        return e_position, log_abs_det_jacobian

    def sample(self, W: Tensor, n: int = 128, with_ladj=False, with_log_probs=False):
        samples = self.distribution.sample(n).to(W.device)
        log_probs = self.distribution.log_prob(samples)
        e_position, log_abs_det_jacobian = self.inverse_transform(samples, W)
        if with_ladj and with_log_probs:
            return e_position, log_abs_det_jacobian, log_probs
        elif with_ladj:
            return e_position, log_abs_det_jacobian
        elif with_log_probs:
            return e_position, log_probs
        return e_position

    def forward(self, x: Tensor, W: Tensor, positive_predication: bool = True):
        ''' 
        Determines whether point x (in E-space) is a member of the predicated defined by weight W

        Args:
            x: Tensor, Input values in E-space coordinates
            W: Tensor, Weighting tensor describing weights of coupling layers
            positive_predication: bool, whether this is positive or negative sampling

        Returns:
            probs: Returns the negative log probability that x is a member of predicate W or, if positive_probs=False,
            the negative log probability that it is not.
        '''
        x_in_W = self.transform(x, W)
        if positive_predication:
            return torch.log_cdf(x_in_W)
        return torch.log_cdf(x_in_W)


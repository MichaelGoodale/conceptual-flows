from typing import Tuple, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import transforms 
from torch.distributions.multivariate_normal import MultivariateNormal

class MaskedCouplingFlow():
    def __init__(self, dim, mask=None, n_hidden=64, n_layers=2, activation=F.relu):
        self.dim = dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.mask = mask.detach()

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

    def backward(self, z, W):
        s_w, s_b, t_w, t_b = self.get_weights_and_biases(W)
        z_k = (self.mask * z)
        zp_D = z * torch.exp(self.net_forward(z_k, s_w, s_b)) + self.net_forward(z_k, t_w, t_b)
        return z_k + (1 - self.mask) * zp_D

    def forward(self, z, W):
        s_w, s_b, t_w, t_b = self.get_weights_and_biases(W)
        zp_k = (self.mask * z)
        z_D = (((1 - self.mask) * z) - self.net_forward(zp_k, t_w, t_b)) / (self.net_forward(zp_k, s_w, s_b) + 1e-8)
        return zp_k + z_D

    def log_abs_det_jacobian(self, z, W):
        s_w, s_b, t_w, t_b = self.get_weights_and_biases(W)
        return -torch.sum(torch.abs(self.net_forward(z * self.mask, s_w, s_b)))



class TripartiteModel(nn.Module):

    def __init__(self, dim:int =32, n_couplings:int =4, buffer:float =3., n_hidden=32):
        '''
        Args:
            dim: Dimensionality of e-space
            n_coupling: Number of layers of couplings to go through
            buffer: Buffer for sigmoid function (where sigmoid(x)=0.5)
        '''
        super().__init__()
        self.distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim, dim))
        self.couplings = []
        mask = torch.ones(dim)
        mask[::2] = 0
        for i in range(n_couplings):
            self.couplings.append(MaskedCouplingFlow(dim, mask, n_hidden=n_hidden))
            mask = 1-mask

        self.buffer = buffer

    @property
    def feature_size(self):
        return sum(x.feature_size for x in self.couplings)

    def split_weights(self, W):
        if W.shape !=(self.feature_size,):
            raise ValueError(f'Weight vector W should have size{self.feature_size} not {W.shape}')
        return torch.split(W, self.feature_size // len(self.couplings), dim=-1)

    def log_abs_det_jacobian(self, z, W):
        return sum(x.log_abs_det_jacobian(z, w) for x, w in zip(self.couplings, self.split_weights(W)))

    def transform(self, x: Tensor, W: Tensor):
        '''
        Takes x in E-space coordinates and converts them to the predicate 
        described by the weighting tensor W
        '''

        predicate_position = x
        for coupling, weight in zip(self.couplings, self.split_weights(W)):
            predicate_position = coupling.forward(predicate_position, weight)
        return predicate_position

    def inverse_transform(self, x: Tensor, W: Tensor):
        e_position = x
        for coupling, weight in zip(reversed(self.couplings), reversed(self.split_weights(W))):
            e_position = coupling.backward(e_position, weight)
        return e_position

    def sample(self, W: Tensor, n: int = 128):
        samples = self.distribution.sample((n, ))
        return self.inverse_transform(samples, W), self.log_abs_det_jacobian(samples, W)

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
            return -torch.log_sigmoid(self.buffer-x_in_W)
        return -torch.log_sigmoid(self.buffer+x_in_W)


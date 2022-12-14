from typing import Tuple, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import transforms 
from torch.distributions.multivariate_normal import MultivariateNormal

from utils import ConceptDistribution, BallHomeomorphism, MaskedCouplingFlow, TranslateFlow

class TripartiteModel(nn.Module):


    # For all functions with W; to have all from a single W, just make sure the shape is (1, feature_size)
    # and broadcasting should take care of it. 

    def __init__(self, dim:int =32, n_couplings:int =4, n_hidden=32, clip=1.0,  k=5, k_neg=5, c=2/3.):
        '''
        Args:
            dim: Dimensionality of e-space
            n_coupling: Number of layers of couplings to go through
            buffer: Buffer for sigmoid function (where sigmoid(x)=0.5)
        '''
        super().__init__()
        self.distribution = ConceptDistribution(dim, k=k, k_neg=k_neg, c=c)
        self.couplings = nn.ModuleList(
                [TranslateFlow(dim)]
                )
        self.dim = dim
        mask = torch.ones(dim)
        mask[::2] = 0
        for i in range(n_couplings):
            self.couplings.append(MaskedCouplingFlow(dim, mask, n_hidden=n_hidden, clip=clip))
            mask = 1-mask

    @property
    def feature_size(self):
        return sum(x.feature_size for x in self.couplings)

    def split_weights(self, W):
        if W.shape[-1] != self.feature_size:
            raise ValueError(f'Weight vector W should have size{self.feature_size} not {W.shape}')
        return torch.split(W, [c.feature_size for c in self.couplings], dim=-1)

    def transform(self, x: Tensor, W: Tensor, with_log_probs=False, negative_example=False, with_both_prob=False):
        '''
        Takes x in E-space coordinates and converts them to the predicate 
        described by the weighting tensor W
        '''
        predicate_position = x 
        log_abs_det_jacobian = torch.zeros(*x.shape[:-1], device=x.device)

        for coupling, weight in zip(self.couplings, self.split_weights(W)):
            predicate_position, ladj = coupling.forward(predicate_position, weight)
            log_abs_det_jacobian += ladj

        if with_log_probs and not with_both_prob:
            log_probs = self.distribution.log_prob(predicate_position, negative_example=negative_example)
            return predicate_position, log_probs - log_abs_det_jacobian
        elif with_both_prob:
            log_probs_pos = self.distribution.log_prob(predicate_position, negative_example=False)
            log_probs_neg = self.distribution.log_prob(predicate_position, negative_example=True)
            return predicate_position, log_probs_pos - log_abs_det_jacobian, log_probs_neg - log_abs_det_jacobian
        return predicate_position

    def inverse_transform(self, x: Tensor, W: Tensor):
        e_position = x 
        log_abs_det_jacobian = torch.zeros(*x.shape[:-1], device=x.device)

        for coupling, weight in zip(reversed(self.couplings), reversed(self.split_weights(W))):
            e_position, ladj = coupling.backward(e_position, weight)
            log_abs_det_jacobian += ladj

        return e_position, log_abs_det_jacobian

    def sample(self, W: Tensor, n: int = 128, with_ladj=False, negative_example = False, on_boundary=False):
        if W.ndim == 1:
            W = W.unsqueeze(0)

        n_concepts = W.shape[0]
        
        if on_boundary:
            samples = self.distribution.generate_boundary((n_concepts, n)).to(W.device)
        else:
            samples = self.distribution.sample((n_concepts, n), negative_example=negative_example).to(W.device)

        e_position, log_abs_det_jacobian = self.inverse_transform(samples, W.unsqueeze(1).expand(-1, n,-1))
        if with_ladj:
            return e_position, log_abs_det_jacobian
        return e_position

    def forward(self, x: Tensor, W: Tensor, negative_example: bool = False):
        ''' 
        Determines whether point x (in E-space) is a member of the predicated defined by weight W

        Args:
            x: Tensor, Input values in E-space coordinates
            W: Tensor, Weighting tensor describing weights of coupling layers
            negative_example: bool, whether this is positive or negative sampling

        Returns:
            probs: Returns the log probability that x is a member of predicate W or, if negative_example=False,
            the negative log probability that it is not.

            Corresponds to probability it is from the positive distribution rather than the negative one, assuming that they are equally likely.
        '''
        x_in_W, pos_log_probs, neg_log_probs = self.transform(x, W, with_both_prob=True)
        marginal = torch.logsumexp(torch.stack((pos_log_probs, neg_log_probs), dim=-1), dim=-1) #Slower but should be much more stable
        if negative_example:
            return neg_log_probs - marginal
        return pos_log_probs - marginal


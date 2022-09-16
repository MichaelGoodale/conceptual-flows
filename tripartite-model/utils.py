import torch
import torch.distributions as D
import torch.nn.functional as F

from torch import Tensor
import numpy as np

class ConceptDistribution():

    def __init__(self, dim=2, k=2, neg_k=None, radius=1.0, eps=1e-9):
        self.dim = dim
        self.radius = radius
        #If k gets too high it'll get crazy large. 
        self.k = k

        #Neg_k will be uniform acroos sphere if neg_k == dim so 
        #should be neg_k should be higher than dim.
        self.neg_k = neg_k
        if self.neg_k is None: 
            self.neg_k = self.dim + 1
        self.eps = eps

    def log_prob(self, x: Tensor, negative_example=False):
        #We don't account for the prob of the angle of the vector
        #Luckily, that's uniform so while this technically isn't the PDF, it's proportionate to the PDF
        #Alternatively, it is the PDF if we view it as a function of ||x|| rather than x.

        r = torch.linalg.vector_norm(x, dim=-1) / self.radius
        if negative_example: # These are the derivatives of their CDF, (i.e., their PDF)
            prob = self.neg_k * r ** (self.neg_k - 1)
        else:
            prob = (r ** (1/self.k - 1)) / self.k

        return -torch.log(prob+self.eps)

    def log_cdf(self, x: Tensor, negative_example=False):
        r = torch.linalg.vector_norm(x, dim=-1) / self.radius
        if negative_example: # These are the derivatives of their CDF, (i.e., their PDF)
            prob = r ** self.neg_k
        else:
            prob = 1 - r ** (1/self.k) #1 - is just to make the CDF go from right to left rather than left to right
        return -torch.log(prob+self.eps)

    def sample(self, n: int, negative_example=False):
        '''Adapted from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/'''
        u = torch.normal(mean=torch.zeros(n, self.dim))
        u = F.normalize(u)
        if negative_example:
            r = (torch.rand(n,1)*0.8) ** (1/self.neg_k) #Inverse CDF
        else:
            r = (torch.rand(n,1)*0.8) ** self.k #Inverse CDF (which incidentally is the inverse of the previous)
        return self.radius*r*u

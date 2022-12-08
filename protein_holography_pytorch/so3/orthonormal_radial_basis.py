'''

Part of this code was adapted from https://github.com/muhrin/mrs-tutorial

'''


import math
import torch
import numpy as np
import scipy as sp
import scipy.special


class ZernickeRadialFunctions:
    
    def __init__(self, rcut, number_of_basis, lmax, complex_sph=False, record_zeros=False):
        self.rcut = rcut
        self.number_of_basis = number_of_basis
        self.lmax = lmax
        self.complex_sph = complex_sph
        self.record_zeros = record_zeros
        self.radius_depends_on_l = True
        if record_zeros:
            self.multiplicities = [number_of_basis] * (lmax + 1)
        else:
            rv = torch.arange(number_of_basis)
            self.multiplicities = [rv[torch.logical_and(rv >= l, (rv - l) % 2 == 0)].shape[0] for l in range(lmax + 1)]
    
    def __call__(self, r):
        try:
            r = r.numpy()
        except:
            r = r.detach().numpy()
        
        # cap radiuses at self.rcut
        r[r > self.rcut] = self.rcut
        
        return_val = []
        for l in range(self.lmax+1):
            for n in range(self.number_of_basis):
                if (n-l) % 2 == 1 or (n < l):
                    if self.record_zeros:
                        return_val.append(np.full(r.shape[0], 0.0))
                    continue

                # dimension of the Zernike polynomial
                D = 3.
                # constituent terms in the polynomial
                A = np.power(-1,(n-l)/2.) 
                B = np.sqrt(2.*n + D)
                C = sp.special.binom(int((n+l+D)/2. - 1),
                                     int((n-l)/2.))
                E = sp.special.hyp2f1(-(n-l)/2.,
                                       (n+l+D)/2.,
                                       l+D/2.,
                                       np.array(r)/self.rcut*np.array(r)/self.rcut)
                F = np.power(np.array(r)/self.rcut,l)
                
                coeff = A*B*C*E*F
                
                return_val.append(coeff)
        
        return torch.tensor(np.transpose(np.vstack(return_val))).type(torch.float)

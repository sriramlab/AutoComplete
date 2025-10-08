
import numpy
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class LinearNet(nn.Module):
    def __init__(self, indim=80, zdim=128):
        super().__init__()
        # N x N linear predictors
        self.net = nn.Linear(indim, indim)

    def forward(self, x):
        x = self.net(x)
        return x

parseFloat = lambda raw: float(raw[0] + '.'+raw[1:])
getconf = lambda tags, name: tags.split(name)[1].split('_')[0]

class DeepCoder(nn.Module):
    def __init__(self, indim=80,
                 width=1, n_depth=1, n_multiples=0,
                 tagstring=None, nonlin=lambda indim: torch.nn.LeakyReLU(),verbose=True):
        super().__init__()

        outdim = indim
        use_dropout = None
        use_batchnorm = False
        if tagstring is not None:

            if 'WIDTH' in tagstring:
                width = getconf(tagstring, 'WIDTH')
                width = parseFloat(width)
            if 'DEPTH' in tagstring:
                n_depth = int(getconf(tagstring, 'DEPTH'))
            if 'MULT' in tagstring:
                n_multiples = int(getconf(tagstring, 'MULT'))
            if 'NOPCS' in tagstring:
                outdim -= 20
            if 'DROPOUT' in tagstring:
                use_dropout = float(tagstring.split('DROPOUT')[1].split('_')[0])/100
            if 'BN' in tagstring:
                use_batchnorm = True

            nonlins = dict(
                PRELU=lambda dim: torch.nn.PReLU(num_parameters=dim),
                RELU=lambda dim: torch.nn.LeakyReLU(inplace=True)
            )
            for match, fn in nonlins.items():
                if match in tagstring:
                    nonlin = fn
                    break

        if verbose:
            print('WIDTH', width)
            print('DEPTH', n_depth)
            print('MULT', n_multiples)
            print('In D', indim)
            print('OutD', outdim)
            print('Dropout', use_dropout)

        spec = []
        zdim = int(indim/width)
        zlist = list(np.linspace(indim, zdim, n_depth+1).astype(int))
        if verbose: print('Zlist', zlist)

        for li in range(n_depth):
            dnow = zlist[li]
            dnext = zlist[li+1]
            spec += [(dnow, dnext)]
            if li != n_depth-1:
                for mm in range(n_multiples):
                    spec += [(dnext, dnext)]

        if verbose: print('Spec:', spec)

        layers = []
        for si, (d1, d2) in enumerate(spec):
            layers += [nn.Linear(d1, d2)]
            if use_batchnorm: layers += [nn.BatchNorm1d(d2)]
            layers += [nonlin(d2)]

        if use_dropout is not None:
            layers += [nn.Dropout(use_dropout, inplace=False)]

        for si, (d2, d1) in enumerate(spec[::-1]):
            d2 = outdim if si == len(spec)-1 else d2
            layers += [nn.Linear(d1, d2)]
            if si != len(spec)-1:
                if use_batchnorm: layers += [nn.BatchNorm1d(d2)]
                layers += [nonlin(d2)]

        self.net = nn.Sequential(*layers)

        if verbose: print('Zdim:', zlist[-1])

    def forward(self, x):
        x = self.net(x)
        return x

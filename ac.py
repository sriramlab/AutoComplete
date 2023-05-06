
import torch
import numpy as np
import torch.nn as nn

parseFloat = lambda raw: float(raw[0] + '.'+raw[1:])
getconf = lambda tags, name: tags.split(name)[1].split('_')[0]

class AutoComplete(nn.Module):
    def __init__(self,
			indim=80, # input data dimension
			width=10, # encoding dim ratio; 10=x1.0, 20=x0.5
			n_depth=4, # number of layers between input layer & encoding layer
			n_multiples=0, # repeated layers of same dim per layer
			nonlin=lambda dim: torch.nn.LeakyReLU(inplace=True), # the nonlinearity
			verbose=False
		):
        super().__init__()

        outdim = indim

        if verbose:
            print('WIDTH', width)
            print('DEPTH', n_depth)
            print('MULT', n_multiples)
            print('NONLIN', nonlin)
            print('In D', indim)
            print('OutD', outdim)

        spec = []
        zdim = int(indim/width)
        zlist = list(np.linspace(indim, zdim, n_depth+1).astype(int))
        if verbose: print('Encoding progression:', zlist)

        for li in range(n_depth):
            dnow = zlist[li]
            dnext = zlist[li+1]
            spec += [(dnow, dnext)]
            if li != n_depth-1:
                for mm in range(n_multiples):
                    spec += [(dnext, dnext)]

        if verbose: print('Fc layers spec:', spec)

        layers = []
        for si, (d1, d2) in enumerate(spec):
            layers += [nn.Linear(d1, d2)]
            layers += [nonlin(d2)]

        for si, (d2, d1) in enumerate(spec[::-1]):
            d2 = outdim if si == len(spec)-1 else d2
            layers += [nn.Linear(d1, d2)]
            if si != len(spec)-1:
                layers += [nonlin(d2)]

        self.net = nn.Sequential(*layers)

        if verbose: print('Zdim:', zlist[-1])

    def forward(self, x):
        x = self.net(x)
        return x
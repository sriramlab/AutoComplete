
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CopymaskDataset(Dataset):
    def __init__(self, data, split, copymask_amount=0.3):

        self.data = data
        self.missing = np.isnan(data)
        self.split = split
        self.copymask_amount = copymask_amount

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        datarow = self.data[idx,].copy()
        missing_inds = self.missing[idx,]

        # get an empty starting mask (no copy mask)
        mask_inds = np.zeros(len(datarow), dtype=bool)

        # grab one random from population
        if np.random.rand() < self.copymask_amount:
            rnd_ind = np.random.randint(self.data.shape[0])
            mask_inds = self.missing[rnd_ind,].copy()


        observed_inds = ~missing_inds
        if np.sum(mask_inds & observed_inds) == np.sum(observed_inds):
            # fix when everything was zerod, then flip back one
            mask_inds[np.where(mask_inds & observed_inds)] = False

        # set true existing nans to zero
        datarow[missing_inds] = 0

        return datarow, missing_inds, mask_inds

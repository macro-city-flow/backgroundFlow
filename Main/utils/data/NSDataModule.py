import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions

#Non-sequential
class NSDataModule(pl.LightningDataModule):
    def __init__(self, feat_path: str, adj_path: str,
                 seq_len: int = 4, pre_len: int = 1,
                 split_ratio: float = 0.8, normalize: bool = True, **kwargs):
        super(NSDataModule, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = utils.data.functions.load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--batch-size', type=int, default=1)
        parser.add_argument('--seq-len', type=int, default=1)
        parser.add_argument('--pre-len', type=int, default=1)
        parser.add_argument('--split-ratio', type=float, default=0.8)
        #parser.add_argument('--normalize', type=bool, default=True)
        normlize_parser = parser.add_mutually_exclusive_group(required=True)
        normlize_parser.add_argument('--normalize',dest='normalize',action='store_true')
        normlize_parser.add_argument('--no-normalize',dest='normalize',action='store_false')
        parser.set_defaults(normalize=True)
        return parser

    def setup(self, stage: str = None):
        self.train_dataset, self.val_dataset = \
            utils.data.functions.generate_torch_datasets(self._feat, self.seq_len, self.pre_len,
                                                         split_ratio=self.split_ratio, normalize=self.normalize,sequential =False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,num_workers=8)

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj

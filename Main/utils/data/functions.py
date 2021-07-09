import numpy as np
import pickle as pkl
import torch
import random


def load_features(feat_path, dtype=np.float32):

    with open(feat_path, 'rb') as data_file:
        feat = pkl.load(data_file)

    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):

    with open(adj_path, 'rb') as adj_file:
        adj = pkl.load(adj_file)

    return adj


def generate_dataset(data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True,sequential=False):

    raw = []

    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data/max_val

    size = len(data)-pre_len-seq_len
    for _ in range(size):
        raw.append(data[_:_+pre_len+seq_len])
    if not sequential:    
        random.shuffle(raw)

    train_size = int(time_len * split_ratio)
    train_set, val_set = raw[0:train_size], raw[train_size: time_len]
    train_X, train_Y, val_X, val_Y = [], [], [], []
    for _ in train_set:
        train_X.append(_[0:seq_len])
        train_Y.append(_[seq_len: pre_len + seq_len])
    for _ in val_set:
        val_X.append(_[0: seq_len])
        val_Y.append(_[seq_len:pre_len + seq_len])
    return train_X, train_Y, val_X, val_Y


def generate_torch_datasets(data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True,sequential=False):
    train_X, train_Y, test_X, test_Y = generate_dataset(data, seq_len, pre_len, time_len=time_len,
                                                        split_ratio=split_ratio, normalize=normalize,sequential=sequential)
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y))
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y))
    return train_dataset, test_dataset

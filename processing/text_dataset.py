import gc
import os
import pickle
import hickle
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import Sampler
import torchvision
import json
from utils import *
import os

class TextDataset(data.Dataset):
    def __init__(self, info_data_filename: str, batch_size, do_train=True, split_num=1, **kwargs):
        super(TextDataset, self).__init__()
        print("loading data")
        info_data = json.load(open(info_data_filename, "r"))
        embedding_file = info_data["embedding_file"]
        if not os.path.isabs(embedding_file):
            embedding_file = os.path.join(os.path.dirname(os.path.abspath(info_data_filename)), embedding_file)
        if not os.path.exists(embedding_file) and "embedding_file" in info_data:
            download_file(url=info_data["embedding_file_url"], filename=embedding_file)
        data = load_data(embedding_file)
        # assert len(data)==info_data["len"]
        print("loading data is over")
        self.shape = list(map(int, info_data["shape"]))
        self.column_name = info_data["column"]
        self.batch_size = batch_size
        self.do_train = do_train
        # self.data = self.data['data']
        # data_split_path = os.path.join(data_path, 'splits', 'split_{}.pkl'.format(split_num))
        # if os.path.exists(data_split_path):
        #     self.train_idx, self.valid_idx = pickle.load(open(data_split_path, 'rb'))
        #     self.train_data = self.create_dataset(text, self.train_idx)
        #     self.valid_data = self.create_dataset(text, self.valid_idx)
        #     self.train_idx = np.asarray(range(len(self.train_data)))
        #     self.valid_idx = len(self.train_idx) + np.asarray(range(len(self.valid_data)))
        # else:
        self.train_data = list(map(lambda x: (x, x.shape[0]), data))

    def create_dataset(self, text_type, keys):
        t = []
        for j, k in enumerate(keys):
            for i, v in enumerate(self.data[k]):
                l = v['text_{}_len'.format(text_type)] // 512 + (int(v['text_{}_len'.format(text_type)] % 512 > 0))
                if (l == 0):
                    continue
                tt = (v['text_embedding_{}'.format(text_type)], l)
                t.append(tt)
        return t

    def __getitem__(self, index):
        if (hasattr(self, 'train_idx') and index in self.train_idx) or (index < len(self.train_data)):
            d = self.train_data[index]
        else:
            d = self.valid_data[index - len(self.train_data)]
        x = self.process_data(d)
        return x

    def process_data(self, d):
        x = torch.tensor(d[0]).view(*self.shape)
        i = torch.tensor([d[1] - 1])
        return x, i

    def __len__(self):
        return len(self.train_data)


def collate_fn(data):
    x_text, i_s = zip(*data)
    x_text = torch.stack(x_text, dim=1)
    i_s = torch.stack(i_s, dim=0)
    b_is = torch.arange(i_s.shape[0]).reshape(tuple(i_s.shape))
    return x_text, i_s.squeeze(), b_is.squeeze()


class ImbalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        self.label_to_count = label_to_count

        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        elif hasattr(dataset, 'get_label'):
            return dataset.get_label(idx)
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate, seed=0):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        self.dataset = dataset

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0 and not hasattr(self.dataset, 'valid_idx'):
            return None, None
        idx_full = np.arange(self.n_samples)
        np.random.seed(self.seed)

        # shuffle indexes only if shuffle is true
        # if order matters don't shuffle
        # added for med2vec dataset where order matters
        len_valid = int(self.n_samples * split)
        if (self.shuffle):
            if (hasattr(self.dataset, 'valid_idx')):
                valid_idx = self.dataset.valid_idx
                train_idx = self.dataset.train_idx
            else:
                valid_idx = idx_full[0:len_valid]

            train_sampler = SubsetRandomSampler(train_idx)
            # use the balanced dataset sampler if balanced_data is set
            # this option can be passed to the dataset class
            if (hasattr(self.dataset, 'balanced_data') and self.dataset.balanced_data):
                train_sampler = ImbalancedSampler(self.dataset, train_idx)

            valid_sampler = SubsetRandomSampler(valid_idx)

        else:
            num_intervals = len(idx_full) // len_valid
            rand_i = np.random.randint(0, num_intervals)
            valid_idx = idx_full[rand_i * len_valid: (rand_i + 1) * len_valid]
            train_idx = np.delete(idx_full, np.arange(rand_i * len_valid, (rand_i + 1) * len_valid))

            if (hasattr(self.dataset, 'valid_idx')):
                valid_idx = self.dataset.valid_idx
                train_idx = self.dataset.train_idx

            train_sampler = MySequentialSampler(train_idx)
            valid_sampler = MySequentialSampler(valid_idx)
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class MySequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return (self.data_source[i] for i in range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class TextDataLoader(BaseDataLoader):
    """
    Mortality prediction task
    """

    def __init__(self, info_data_filename, text, batch_size, shuffle, validation_split, num_workers, do_train=True, **kwargs):
        # self.info_data_filename = os.path.expanduser(expand_vars(info_data_filename))
        self.info_data_filename = os.path.expanduser(info_data_filename)
        self.text = text
        self.batch_size = batch_size
        self.do_train = do_train
        self.dataset = TextDataset(self.info_data_filename, self.batch_size, self.do_train, **kwargs)
        super(TextDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                                             collate_fn=collate_fn)

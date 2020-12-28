import json
from functools import wraps
from typing import Union

import h5py
import pandas as pd
import numpy as np
from itertools import chain
import gc
import torch

try:
    import dill as pickle
except:
    import pickle
import hickle
import torch
import os
from tqdm import tqdm
import logging
import GPUtil
import random
import tables
from transformers import load_tf_weights_in_bert, BertForPreTraining, BertConfig

logger = logging.getLogger(__name__)


def exception(logger, reraise=False):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur

    @param logger: The logging object
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"There was an exception in {func.__name__}")
            if reraise:
                raise

        return wrapper

    return decorator


def isnan(o):
    return type(o) == type(np.nan) and np.isnan(o)


def flatten(x):
    return list(chain(*x))


def to_hickle(in_object, filename: str):
    path_save = os.path.dirname(filename)
    if not os.path.exists(path_save):
        os.makedirs(path_save, exist_ok=True)
    hickle.dump(in_object, filename, mode="w")


def np_to_h5(in_object: np.array, filename: str, name_dataset='dataset_1'):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset(name_dataset, data=in_object, compression="gzip", compression_opts=9)


def np_from_h5(filename: str, name_dataset='dataset_1') -> np.array:
    with h5py.File(filename, 'r') as hf:
        return np.array(hf.get(name_dataset))


def to_h5(in_object: Union[np.array, torch.Tensor], filename: str):
    with tables.open_file(filename, 'w') as h5_file:
        if isinstance(in_object, torch.Tensor):
            h5_file.create_array('/', 'data', in_object.detach().cpu().numpy())
        else:
            h5_file.create_array('/', 'data', in_object)


@exception(logger, reraise=False)
def from_h5(filename: str):
    with tables.open_file(filename, 'r') as h5_file:
        return h5_file.root.data.read()


@exception(logger, reraise=False)
def from_hickle(filename: str):
    return hickle.load(filename)


@exception(logger, reraise=False)
def to_pickle(in_object, filename: str, protocol=pickle.HIGHEST_PROTOCOL):
    with open(filename, 'wb') as f:
        pickle.dump(in_object, f, protocol)


@exception(logger, reraise=False)
def from_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def setup_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def prepare_device(no_cuda=False):
    return (
        torch.device('cuda')
        if not no_cuda and torch.cuda.is_available()
        else torch.device('cpu')
    )


def wget_file(url, filename):
    import requests
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)


def print_gpu():
    if not torch.cuda.is_available():
        print("There is no available GPU")
        return
    print(f"GPU(s) available {torch.cuda.device_count()}")
    print(f"use {torch.cuda.current_device()} {torch.cuda.get_device_name(device=None)}")
    GPUs = GPUtil.getGPUs()
    gpu = GPUs[0]
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree,
                                                                                                gpu.memoryUsed,
                                                                                                gpu.memoryUtil * 100,
                                                                                                gpu.memoryTotal))


def save_model(path, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = path
    torch.save(model_to_save.state_dict(), output_model_file)


def save_data(in_object, filename):
    outfile_ext = get_extension(filename)
    if outfile_ext == "h5":
        to_h5(in_object=in_object, filename=filename)
    elif outfile_ext == "hkl":
        to_hickle(in_object=in_object, filename=filename)
    elif outfile_ext == "json":
        json.dump(in_object, open(filename, 'w'))
    elif outfile_ext == "npy":
        if isinstance(in_object, torch.Tensor):
            np.save(filename, in_object.numpy())
        else:
            np.save(filename, in_object)
    elif outfile_ext == "pkl":
        to_pickle(in_object=in_object, filename=filename)
    elif outfile_ext == "pt":
        torch.save(in_object, filename)
    else:
        print("unrecognized file type")

@exception(logger, reraise=False)
def load_data(filename):
    file_ext = get_extension(filename)
    if file_ext == "h5":
        return from_h5(filename=filename)
    elif file_ext == "hkl":
        return from_hickle(filename=filename)
    elif file_ext == "json":
        return json.load(open(filename, 'r'))
    elif file_ext == "npy":
        return np.load(filename)
    elif file_ext == "pkl":
        return from_pickle(filename=filename)
    elif file_ext == "pt":
        return torch.load(filename)
    elif file_ext == "csv":
        return pd.read_csv(filename)
    else:
        raise Exception("unrecognized file type")


def expand_vars(input_str: str):
    import config
    return input_str.format(PROJECT_DIR=getattr(config, "PROJECT_DIR"),
                            PROJECT_DATA_DIR=getattr(config, "PROJECT_DATA_DIR"),
                            MIMIC_PATH=getattr(config, "MIMIC_PATH"))


def get_extension(filename: str) -> str:
    import pathlib
    return pathlib.Path(filename).suffix[1:] if pathlib.Path(filename).suffix.startswith(".") else pathlib.Path(
        filename).suffix

def download_file(url, filename):
    if "drive.google.com" in url:
        import gdown
        gdown.download(url, filename, quiet=False)
    else:
        import subprocess
        subprocess.run(["wget", url, "-O", filename])

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


def chunks(iterable, n):
    for i in range(len(iterable), n):
        yield iterable[i:i + n]


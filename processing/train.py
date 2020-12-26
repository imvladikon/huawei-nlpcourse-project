import numpy as np
import argparse
import json
import os
import nni
import torch
from torchsummary import summary
from utils import *
from processing.base_trainer import BaseTrainer
from processing.gru_ae import *
import logging
import processing.text_dataset as module_data
import processing.loss as module_loss
import processing.metric as module_metric
from processing.text_dataset import TextDataLoader
from processing.text_summarization_trainer import TextSummTrainer

logger = logging.getLogger(__name__)


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def import_module(name, config):
    return getattr(__import__("{}.{}".format(name, config[name]['module_name'])), config[name]['type'])


def mod_config(config, nni_params):
    if (nni_params == None):
        return config

    def recurse_dict(d, k, v):
        if (k in d):
            d[k] = v
            return d
        for kk, vv in d.items():
            if (type(vv) == dict):
                d[kk] = recurse_dict(vv, k, v)
        return d

    for k, v in nni_params.items():

        if k in config:
            config[k] = v
            continue
        for kk, vv in config.items():
            if (type(vv) == dict):
                config[kk] = recurse_dict(vv, k, v)
    return config


class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """

    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)


def main(config, resume, device, nni_params=None):
    if nni_params is None:
        nni_params = {}
    config = mod_config(config, nni_params)
    train_logger = Logger()
    model = GRUAE(**config['model']['args']).to(device).train()
    # print(model)
    summary(model,(1, 768))
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met, lambda: None) for met in config['metrics']]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    trainer = TextSummTrainer(model, loss, metrics, optimizer,
                              resume=resume,
                              config=config,
                              data_loader=data_loader,
                              valid_data_loader=valid_data_loader,
                              lr_scheduler=lr_scheduler,
                              train_logger=train_logger)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Structmed Trainer')
    parser.add_argument('-c', '--config', default='None', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument("--no-cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")

    args = parser.parse_args()
    print("setup seed: {}".format(args.seed))
    setup_seed(args.seed)
    print_gpu()
    device = prepare_device(args.no_cuda)
    print(device)

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    params = {}
    try:
        params = nni.get_next_parameter()
    except:
        pass
    # params = {"text": False}
    # params = {"text": True, "codes": False, "learning_rate": 0.0001, "demographics_size": 0, "batch_size": 16, "div_factor": 1, "step_size": 40, "class_weight_1": 4.616655939419362, "class_weight_0": 0.81750651640358}
    main(config, args.resume, device, params)
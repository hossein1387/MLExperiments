import argparse
import os
import random
import sys
import torchvision.datasets as dsets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.parallel
import yaml
from torch.autograd import Variable
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=False, default="config.yaml")
    parser.add_argument('-t', '--modelype', help='type of model to run', required=False, default="MLP")
    args = parser.parse_args()
    return vars(args)

def load_dataset(config):
    if config['dataset'] == 'mnist':
        # import ipdb as pdb; pdb.set_trace()
        # MNIST Dataset
        train_dataset = dsets.MNIST(root='./data/',
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    download=True)

        test_dataset = dsets.MNIST(root='./data/',
                                   train=False, 
                                   transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config["batchsize"], 
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=config["batchsize"], 
                                                  shuffle=False)
    else: 
        print ("Unsupported dataset type".format(config['dataset']))
        sys.exit()
    return train_loader, test_loader, train_dataset, test_dataset

import numpy as np, sys, unicodedata, requests, os, random, pdb, json, uuid, time, argparse, pickle
from random import randint
from pprint import pprint
import logging, logging.config, itertools, pathlib
import scipy.sparse as sp
from pymongo import MongoClient
from collections import defaultdict as ddict
from itertools import product

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import Parameter as Param
from torch import tensor

import networkx as nx, socket
from tqdm import tqdm, trange
from ordered_set import OrderedSet

np.set_printoptions(precision=4)

def makeDirectory(dirpath):
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus
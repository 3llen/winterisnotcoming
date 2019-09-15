import copy
import json
import os
import time
from collections import OrderedDict
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

gpu_available = torch.cuda.is_available()

if not gpu_available:
    print('CUDA is not available. Training will be done with CPU.')
else:
    print('CUDA is available. Training will be done with GPU.')
torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
print(torch_device)



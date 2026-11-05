import contextlib
import copy
import hashlib
import importlib.metadata
import inspect
import io
import itertools
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from IPython.display import display
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from tqdm import tqdm
import clip

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, '| device:', DEVICE)
if REQUIRE_GPU and DEVICE.type != 'cuda':
    raise RuntimeError('Select CSCI566 ViT in a GPU-backed OnDemand session before running.')
if DEVICE.type == 'cuda':
    x = torch.ones((16, 16), device=DEVICE)
    assert (x @ x).sum().item() == 4096
    torch.cuda.synchronize()
    print('GPU operation passed:', torch.cuda.get_device_name(0))

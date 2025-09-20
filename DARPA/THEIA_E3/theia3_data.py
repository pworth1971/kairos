import functools
import os
import json
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *
import networkx as nx

import hashlib
def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

filePath="./log/"

filelist = os.listdir(filePath)
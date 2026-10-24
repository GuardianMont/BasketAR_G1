import random
import numpy as np
import torch

def set_seed(seed: int):
    """Sets the random seed across python, numpy, and pytorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (optional, might slightly reduce training speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
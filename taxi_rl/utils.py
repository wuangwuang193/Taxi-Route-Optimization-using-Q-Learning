import random
import numpy as np
import os

def set_seed(seed: int = 42):
    """Cố định seed cho toàn bộ project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
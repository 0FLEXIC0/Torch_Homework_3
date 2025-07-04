import random
import numpy as np
import torch
from torch.utils.data import Dataset

def set_seed(seed=42):
    """
    Фиксирует seed для всех генераторов случайных чисел (Python, NumPy, PyTorch).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_classification_data(n=1000, num_features=2, num_classes=2, seed=42, source='random'):
    """
    Генерирует синтетические данные для задачи классификации.
    """
    np.random.seed(seed)
    if source == 'random':
        X = np.random.randn(n, num_features)
        w = np.random.randn(num_features, num_classes)
        logits = X @ w + np.random.randn(n, num_classes) * 0.5
        y = np.argmax(logits, axis=1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    else:
        raise ValueError("Unknown source")

class ClassificationDataset(Dataset):
    """
    PyTorch Dataset для задачи классификации.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

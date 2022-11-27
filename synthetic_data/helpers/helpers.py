import torch


def get_device():
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

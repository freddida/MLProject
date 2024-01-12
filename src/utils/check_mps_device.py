import torch
from torch import Tensor


def check_mps_device() -> None:
    """
    Check if the MPS (Multi-Process Service) device is available and print a tensor on that device if found.

    Returns:
        None
    """
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)  # Creating a tensor on MPS device
        print(x)
    else:
        print("MPS device not found.")

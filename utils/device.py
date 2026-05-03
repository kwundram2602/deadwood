import torch


def get_device() -> torch.device:
    """Return best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {name} ({mem_gb:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device

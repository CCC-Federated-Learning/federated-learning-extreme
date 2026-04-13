import torch


def check_gpu(keyword: str = "3070") -> None:
    """Print CUDA availability and list all GPU devices."""
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device found. Running on CPU.")
        return

    count = torch.cuda.device_count()
    print(f"GPU count: {count}")

    matched = None
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        tag = "  ← match" if keyword.lower() in name.lower() else ""
        print(f"  [{i}] {name}{tag}")
        if keyword.lower() in name.lower():
            matched = i

    if matched is None:
        print(f"No GPU matched keyword: '{keyword}'")
    else:
        print(f"Preferred device: cuda:{matched}")


if __name__ == "__main__":
    check_gpu()

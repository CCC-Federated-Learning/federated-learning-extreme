import torch

#very good test to check if the GPU is available and which one is being used. It also allows you to find a specific GPU by name and get the preferred device based on a keyword. This can be useful for ensuring that your code is running on the correct hardware, especially when working with multiple GPUs.
def main() -> None:
    keyword = "3070"
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device found. Current device: cpu")
        return

    print(f"CUDA device count: {torch.cuda.device_count()}")
    matched_index = None
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        marker = " <= matched" if keyword in name.lower() else ""
        print(f"[{i}] {name}{marker}")
        if keyword in name.lower():
            matched_index = i

    if matched_index is None:
        print(f"No GPU matched keyword: {keyword}")
    else:
        print(f"Matched GPU index: {matched_index}")
        preferred_device = f"cuda:{matched_index}"
        print(f"Preferred device: {preferred_device}")


if __name__ == "__main__":
    main()

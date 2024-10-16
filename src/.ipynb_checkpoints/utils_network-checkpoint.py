import torch 


def check_select_device()->torch.device:
    device  = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda is available")
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}:")
            print(torch.cuda.get_device_properties(i))
    else:
        print("cuda is not available")
    return device
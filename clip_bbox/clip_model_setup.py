import numpy as np
import torch
import os
from .model import build_model
from .clip import load

# os.path.abspath(os.path.dirname(__file__))

MODELS = {
    "RN50": (
        "https://openaipublic.azureedge.net/clip/models/"
        "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
        "/RN50.pt"
    ),
    "RN101": (
        "https://openaipublic.azureedge.net/clip/models/"
        "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599"
        "/RN101.pt"
    ),
    "RN50x4": (
        "https://openaipublic.azureedge.net/clip/models/"
        "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd"
        "/RN50x4.pt"
    ),
    "ViT-B/32": (
        "https://openaipublic.azureedge.net/clip/models/"
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
        "ViT-B-32.pt"
    ),
}


def get_clip_model(device, model_name="RN50", input_res=(720, 1280)):
    """Downloads pre-trained CLIP model and adjusts model to accept
    desired input resolution.

    Args:
        device (Torch.device): Device Torch should use to run CLIP. For example, the device can be
            `torch.device("cpu")` or `torch.device("cuda")`
        model_name (str): The name of the desired CLIP model to download. The options are "RN50", "RN101", "RN50x4", or
            "ViT-B/32".

        input_res (tuple[int]): Input resolution represented as (height, width).

    Returns:
        Modified Torch model accepting the desired input resolution.

    """
    print("Torch version:", torch.__version__)

    input_resolution = input_res

    if not os.path.exists("model.pt"):
        os.system("wget -q {} -O model.pt".format(MODELS[model_name]))

    clip_model, _ = load("model.pt", device, jit=False, download_root='./')
    clip_model = clip_model.eval()
    # clip_model = torch.load("model.pt", map_location=device).eval()
    context_length = clip_model.context_length
    # context_length = clip_model.context_length.item()
    vocab_size = clip_model.vocab_size
    # vocab_size = clip_model.vocab_size.item()
    # print("model location: ", clip_model.device)

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}",
    )
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    clip_model.state_dict()["input_resolution"] = torch.tensor(input_res)
    # print("Model's state_dict:")
    for param_tensor in clip_model.state_dict():
        # print(param_tensor + '            ' + str(clip_model.state_dict()[param_tensor].size()))
        clip_model.state_dict()[param_tensor] = clip_model.state_dict()[param_tensor].to(device)

    model_modded = build_model(clip_model.state_dict())

    model_modded.to(device)

    return model_modded

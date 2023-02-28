import numpy as np
import torch
import os

from ..clip.model import build_model

MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/"
    "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
    "/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/"
    "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599"
    "/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/"
    "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd"
    "/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
    "ViT-B-32.pt",
}


def get_clip_model(model_name="RN50", input_res=(720, 1280)):
    print("Torch version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    input_resolution = input_res
    os.system("wget {} -O model.pt".format(MODELS[model_name]))

    clip_model = torch.jit.load("model.pt").cuda().eval()
    context_length = clip_model.context_length.item()
    vocab_size = clip_model.vocab_size.item()

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
        clip_model.state_dict()[param_tensor] = clip_model.state_dict()[
            param_tensor
        ].cuda()

    model_modded = build_model(clip_model.state_dict())

    print("check model cuda: {}".format(next(model_modded.parameters()).is_cuda))
    model_modded.to(device)
    print("device count: {}".format(torch.cuda.device_count()))

    return model_modded

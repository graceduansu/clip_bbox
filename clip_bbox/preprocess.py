from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import torch
import numpy as np

from .simple_tokenizer import SimpleTokenizer


def preprocess_imgs(img_path_list, device, input_resolution=None):
    """Preprocess list of images for CLIP.

    Args:
        img_path_list (List[str]): List of image paths to preprocess
        input_resolution (tuple[int]): Input resolution represented as (height, width)

    Returns:
        List[Torch tensor]: List of images
        Torch tensor: Array of images preprocessed as a
            Torch tensor

    """
    if not input_resolution:
        input_resolution = Image.open(img_path_list[0]).size

    preprocess = Compose(
        [
            Resize(input_resolution, interpolation=Image.BICUBIC),
            ToTensor(),
        ]
    )

    # TODO: allow mean and std as optional arguments
    # statistics based on default image dataset from scikit-image
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)

    images = []

    for filename in [filename for filename in img_path_list if filename.endswith(".png") or filename.endswith(".jpg")]:
        image = preprocess(Image.open(filename).convert("RGB"))
        images.append(image)

    image_input = torch.tensor(np.stack(images)).to(device)
    image_input = image_input.detach().cpu().type(torch.FloatTensor)

    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    # torch.save(images, 'rocket.pt')
    # torch.save(image_input, 'rocket_preprocessed.pt')
    return images, image_input


def preprocess_texts(caption_list, context_length, device):
    """Preprocess list of texts for CLIP.

    Args:
        caption_list (List[str]): List of captions
        context_length (int): CLIP model parameter

    Returns:
        Torch tensor: Array of preprocessed texts

    """
    tokenizer = SimpleTokenizer()
    text_tokens = [tokenizer.encode("This is " + desc) for desc in caption_list]

    text_input = torch.zeros(len(text_tokens), context_length, dtype=torch.long)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    for i, tokens in enumerate(text_tokens):
        tokens = [sot_token] + tokens + [eot_token]
        text_input[i, : len(tokens)] = torch.tensor(tokens)

    text_input = text_input.to(device)
    text_input = text_input.detach().cpu().type(torch.FloatTensor)
    # torch.save(text_input, "rocket_text_preprocessed.pt")

    return text_input

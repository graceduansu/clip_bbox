from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import torch

import os
import skimage
import numpy as np

from clip.simple_tokenizer import SimpleTokenizer


# TODO: allow other img+text pair inputs
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse",
    "coffee": "a cup of coffee on a saucer",
}


def preprocess_imgs(input_resolution=(720, 1280)):
    preprocess = Compose(
        [
            Resize(input_resolution, interpolation=Image.BICUBIC),
            ToTensor(),
        ]
    )

    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

    images = []

    for filename in [
        filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")
    ]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue

        image = preprocess(Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB"))
        images.append(image)

    image_input = torch.tensor(np.stack(images)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    print("image stack size:")
    print(image_input.size())

    return images, image_input


def preprocess_texts(context_length):
    texts = []
    for filename in [
        filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")
    ]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue

        texts.append(descriptions[name])

    tokenizer = SimpleTokenizer()
    text_tokens = [tokenizer.encode("This is " + desc) for desc in texts]

    text_input = torch.zeros(len(text_tokens), context_length, dtype=torch.long)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    for i, tokens in enumerate(text_tokens):
        tokens = [sot_token] + tokens + [eot_token]
        text_input[i, : len(tokens)] = torch.tensor(tokens)

    text_input = text_input.cuda()

    return text_input


# TEXT PREPROCESSING UTILS

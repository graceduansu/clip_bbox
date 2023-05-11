#!/usr/bin/env python3

import torch

from . import clip_model_setup
from . import bbox_utils
from . import preprocess
from math import sqrt, ceil


def get_square_int_factors(n):
    val = ceil(sqrt(n))
    while True:
        if not n % val:
            val2 = n // val
            break
        val -= 1

    return val, val2


def run_clip_bbox(img_path, caption, out_path):
    """Draws bounding boxes on an input image.

    Args:
        img_path (str): path to input image.
        caption (str): caption of input image.
        out_path (str): path to output image displaying bounding boxes.

    Returns:
        None.

    """
    device = torch.device("cpu")

    print('torch device:', device)

    # input_resolution = (720, 1280)
    images, image_input = preprocess.preprocess_imgs([img_path], device)

    input_resolution = images[0].size()[1:]

    model_modded = clip_model_setup.get_clip_model(device, input_res=input_resolution)
    text_input = preprocess.preprocess_texts([caption], model_modded.context_length, device)

    with torch.no_grad():
        image_features = model_modded.encode_image(image_input).float()
        text_features = model_modded.encode_text(text_input).float()

    # get bbox results
    img_fts = image_features[1:]
    # print("fts size: ", img_fts.size())
    # torch.save(img_fts, "rocket_img_fts.pt")
    # torch.save(text_features, "rocket_txt_fts.pt")

    heatmap_list = img_fts_to_heatmap(img_fts, text_features)
    pred_bboxes = []
    for h in range(len(heatmap_list)):
        heat = heatmap_list[h]
        # np.savez("heat.npz", heat)
        bboxes = bbox_utils.heat2bbox(heat, input_resolution)
        pred_bboxes.append([torch.tensor(b["bbox_normalized"]).unsqueeze(0) for b in bboxes])
        bbox_utils.img_heat_bbox_disp(images[h].permute(1, 2, 0).cpu(), heat, out_path, bboxes=bboxes)

    print("CLIP_BBox run complete!")


def img_fts_to_heatmap(img_fts, txt_fts):
    """Computes the similarity heatmap between a pair of
    image and text embeddings from CLIP.

    Args:
        img_fts (numpy array): Image embedding from CLIP.
        txt_fts (numpy array): Text embedding from CLIP.

    Returns:
        numpy array: Similarity heatmap between the image and text embeddings.

    """

    # dot product with normalization
    img_norm = img_fts / img_fts.norm(dim=-1, keepdim=True)
    txt_norm = txt_fts / txt_fts.norm(dim=-1, keepdim=True)

    batch_size = len(txt_norm)
    # print(batch_size)

    # img_norm = F.interpolate(img_norm.permute(2,1,0), size=int(input_resolution[0]*input_resolution[1]/64),
    #      mode='nearest').permute(2,1,0)
    # resize = torch.flatten(img_norm).size()[0] / batch_size / img_fts.size()[0]
    # resize = int(math.sqrt(resize))
    dim1, dim2 = get_square_int_factors(img_fts.size(0))
    print(f"image feature dimensions: {dim1} x {dim2}")
    img_reshape = torch.reshape(img_norm, (dim1, dim2, batch_size, img_fts.size()[2]))
    # img_reshape = torch.reshape(img_norm, (7, 7, batch_size, 1024))
    # img_reshape = torch.reshape(img_norm, (resize, resize, batch_size, img_fts.size()[0]))
    # print(img_reshape.shape)
    # print(txt_norm.shape)

    img_reshape = img_reshape.cpu()

    heatmap_norm = img_reshape.cpu().numpy() @ txt_norm.cpu().numpy().T

    # heatmap_list = [-1.0 * heatmap_norm[:,:,h,h] + heatmap_norm[:,:,h,h].min() for h in range(batch_size)]

    # my loop
    heatmap_list = []
    for h in range(batch_size):
        # maxi = heatmap_norm[:, :, h, h].max()
        # print(maxi)
        hm = -1.0 * heatmap_norm[:, :, h, h] + heatmap_norm[:, :, h, h].max()
        # hm = hm / np.linalg.norm(hm)
        heatmap_list.append(hm)

    return heatmap_list

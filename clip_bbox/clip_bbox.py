#!/usr/bin/env python3

import torch

# TODO: why does from . not work when name = main
import clip_model_setup, bbox_utils, preprocess

# TODO: arg parser for bbox params


def run_clip_bbox(input_res):
    input_resolution = input_res
    model_modded = clip_model_setup.get_clip_model()

    images, image_input = preprocess.preprocess_imgs()
    text_input = preprocess.preprocess_texts(model_modded.context_length)

    with torch.no_grad():
        image_features = model_modded.encode_image(image_input).float()
        text_features = model_modded.encode_text(text_input).float()

    # get bbox results
    img_fts = image_features[1:]
    # print("fts size: ", img_fts.size())

    heatmap_list = img_fts_to_heatmap(img_fts, text_features)
    pred_bboxes = []
    for h in range(len(heatmap_list)):
        title = "Image " + str(h)
        heat = heatmap_list[h]
        bboxes = bbox_utils.heat2bbox(heat, input_resolution)
        pred_bboxes.append(
            [torch.tensor(b["bbox_normalized"]).unsqueeze(0) for b in bboxes]
        )
        save_path = "img_{}_bbox.png".format(h)
        bbox_utils.img_heat_bbox_disp(
            images[h].permute(1, 2, 0).cpu(),
            heat,
            save_path,
            title=title,
            bboxes=bboxes,
            order="xyxy",
        )


def img_fts_to_heatmap(img_fts, txt_fts):
    """
    img_fts: list of image features with no global avg pooling layer,
    shape=(img_dim x img_dim, batch_size, features)
    Return: list of heatmaps
    """

    # dot product with normalization
    img_norm = img_fts / img_fts.norm(dim=-1, keepdim=True)
    txt_norm = txt_fts / txt_fts.norm(dim=-1, keepdim=True)

    batch_size = len(txt_norm)
    print(batch_size)

    # img_norm = F.interpolate(img_norm.permute(2,1,0), size=int(input_resolution[0]*input_resolution[1]/64), mode='nearest').permute(2,1,0)
    # resize = torch.flatten(img_norm).size()[0] / batch_size / img_fts.size()[0]
    # resize = int(math.sqrt(resize))
    img_reshape = torch.reshape(img_norm, (22, 40, batch_size, img_fts.size()[2]))
    # img_reshape = torch.reshape(img_norm, (7, 7, batch_size, 1024))
    # img_reshape = torch.reshape(img_norm, (resize, resize, batch_size, img_fts.size()[0]))
    print(img_reshape.shape)
    print(txt_norm.shape)

    img_reshape = img_reshape.cpu()

    heatmap_norm = img_reshape.cpu().numpy() @ txt_norm.cpu().numpy().T

    # heatmap_list = [-1.0 * heatmap_norm[:,:,h,h] + heatmap_norm[:,:,h,h].min() for h in range(batch_size)]

    # my loop
    heatmap_list = []
    for h in range(batch_size):
        maxi = heatmap_norm[:, :, h, h].max()
        print(maxi)
        hm = -1.0 * heatmap_norm[:, :, h, h] + heatmap_norm[:, :, h, h].max()
        # hm = hm / np.linalg.norm(hm)
        heatmap_list.append(hm)

    return heatmap_list


if __name__ == "__main__":
    run_clip_bbox(input_res=(720, 1280))

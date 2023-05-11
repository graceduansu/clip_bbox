import clip_bbox.clipbbox as cb
import clip_bbox.bbox_utils as bu
import clip_bbox.preprocess as p
from clip_bbox.__main__ import main

import numpy as np
from math import isclose
from matplotlib.figure import Figure
from PIL import Image
from PIL import ImageChops
import torch
import os


def imgs_are_same(img1_path, img2_path):
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    equal_size = img1.height == img2.height and img1.width == img2.width
    diff = ImageChops.difference(img1, img2).getbbox()

    if diff:
        equal_content = len(set(diff)) < 5
    else:
        equal_content = True

    return equal_size and equal_content


# Ground truth outputs:
heat = np.load("clip_bbox/tests/assets/heat.npz")['arr_0']
gt_bboxes = [
    {
        'score': 0.7454576,
        'bbox': (362, 182, 428, 357),
        'bbox_normalized': np.array([0.2828125, 0.25277778, 0.334375, 0.49583333]),
    },
    {
        'score': 0.69931805,
        'bbox': (603, 212, 687, 365),
        'bbox_normalized': np.array([0.47109375, 0.29444444, 0.53671875, 0.50694444]),
    },
]
input_resolution = (720, 1280)

# UNIT TESTS


# bbox_utils
def test_heat2bbox():
    bboxes = bu.heat2bbox(heat, input_resolution)
    assert len(bboxes) == 2

    for i in range(len(gt_bboxes)):
        gt_score = float(gt_bboxes[i]['score'])
        score = float(bboxes[i]['score'])
        assert isclose(gt_score, score, rel_tol=1e-6)
        assert gt_bboxes[i]['bbox'] == bboxes[i]['bbox']
        assert np.allclose(gt_bboxes[i]['bbox_normalized'], bboxes[i]['bbox_normalized'], atol=1e-6)


def test_img_heat_bbox_disp():
    image = np.load("clip_bbox/tests/assets/rocket.npy")
    fig = bu.img_heat_bbox_disp(image, heat, "clip_bbox/tests/test_img_heat_bbox_disp.png", bboxes=gt_bboxes)
    assert isinstance(fig, Figure)
    assert imgs_are_same(
        "clip_bbox/tests/assets/test_img_heat_bbox_disp-gt_output.png", "clip_bbox/tests/test_img_heat_bbox_disp.png"
    )
    os.remove("clip_bbox/tests/test_img_heat_bbox_disp.png")


# preprocess
def test_preprocess_imgs():
    device = torch.device("cpu")
    images, input = p.preprocess_imgs(["clip_bbox/tests/assets/rocket.png"], device, input_resolution=input_resolution)
    gt_images = torch.load("clip_bbox/tests/assets/rocket.pt")
    gt_input = torch.load("clip_bbox/tests/assets/rocket_preprocessed.pt")
    assert len(gt_images) == len(images)
    assert torch.allclose(gt_images[0], images[0])
    assert torch.allclose(gt_input, input)


def test_preprocess_texts():
    device = torch.device("cpu")
    text_input = p.preprocess_texts(["a rocket standing on a launchpad"], 77, device)
    gt_input = torch.load("clip_bbox/tests/assets/rocket_text_preprocessed.pt")
    assert torch.allclose(gt_input, text_input)


# clipbbox
def test_img_fts_to_heatmap():
    img_fts = torch.load("clip_bbox/tests/assets/rocket_img_fts.pt")
    text_features = torch.load("clip_bbox/tests/assets/rocket_txt_fts.pt")
    heatmap_list = cb.img_fts_to_heatmap(img_fts, text_features)
    assert len(heatmap_list) == 1
    assert np.allclose(heatmap_list[0], heat, atol=1e-6)


# INTEGRATION TESTS


def test_run_clip_bbox():
    cb.run_clip_bbox(
        "clip_bbox/tests/assets/rocket.png", "a rocket standing on a launchpad", "clip_bbox/tests/test_all-rocket.png"
    )
    assert imgs_are_same("clip_bbox/tests/assets/test_all-gt_rocket.png", "clip_bbox/tests/test_all-rocket.png")
    os.remove("clip_bbox/tests/test_all-rocket.png")

    cb.run_clip_bbox("clip_bbox/tests/assets/camera.png", "a camera on a tripod", "clip_bbox/tests/test_all-camera.png")
    assert imgs_are_same("clip_bbox/tests/assets/test_all-gt_camera.png", "clip_bbox/tests/test_all-camera.png")
    os.remove("clip_bbox/tests/test_all-camera.png")


def test_main():
    cli_args = [
        "clip_bbox/tests/assets/rocket.png",
        "a rocket standing on a launchpad",
        "clip_bbox/tests/test_all-rocket.png",
    ]
    main(cli_args)
    assert imgs_are_same("clip_bbox/tests/assets/test_all-gt_rocket.png", "clip_bbox/tests/test_all-rocket.png")
    os.remove("clip_bbox/tests/test_all-rocket.png")

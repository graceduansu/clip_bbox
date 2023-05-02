import clip_bbox.clipbbox as cb

# from unittest.mock import patch

from PIL import Image
from PIL import ImageChops


def imgs_are_same(img1_path, img2_path):
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    equal_size = img1.height == img2.height and img1.width == img2.width
    equal_content = len(set(ImageChops.difference(img1, img2).getbbox())) < 100

    return equal_size and equal_content


# UNIT TESTS


# bbox_utils
def test_heat2bbox():
    pass


def test_img_heat_bbox_disp():
    pass


# preprocess
def test_preprocess_imgs():
    pass


def test_preprocess_txts():
    pass


# clipbbox
def test_run_img_fts_to_heatmap():
    pass


# INTEGRATION TESTS


def test_run_clip_bbox():
    cb.run_clip_bbox(
        "clip_bbox/tests/images/rocket.png", "a rocket standing on a launchpad", "clip_bbox/tests/test_1.png"
    )
    assert imgs_are_same("clip_bbox/tests/images/gt_output_1.png", "clip_bbox/tests/test_1.png")


# def test_main():
#     cli_args = ["1"]
#     main(cli_args)

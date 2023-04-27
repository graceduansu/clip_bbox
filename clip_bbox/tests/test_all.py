import clip_bbox.clipbbox as cb

# from unittest.mock import patch

from PIL import Image
from PIL import ImageChops


def imgs_are_same(img1_path, img2_path):
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    equal_size = img1.height == img2.height and img1.width == img2.width
    equal_content = not ImageChops.difference(img1, img2).getbbox()

    return equal_size and equal_content


# INTEGRATION TESTS


# UNIT TESTS


def test_run_clip_bbox():
    cb.run_clip_bbox(
        "clip_bbox/tests/images/rocket.png", "a rocket standing on a launchpad", "clip_bbox/tests/test_1.png"
    )
    assert imgs_are_same("clip_bbox/tests/images/gt_output_1.png", "clip_bbox/tests/test_1.png")


# @patch('builtins.print')
# def test_print_hello(mock_print):
#     print_hello()
#     assert mock_print.call_args.args == ("Hello, world!",)

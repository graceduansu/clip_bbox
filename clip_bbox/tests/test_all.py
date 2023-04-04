from clip_bbox import run_clip_bbox

# from unittest.mock import patch
import pathlib as pl

# INTEGRATION TESTS


# UNIT TESTS
def test_run_clip_bbox():
    # TODO: create "example inputs" folder with skimage and descriptions
    run_clip_bbox(input_res=(720, 1280))
    root = "../"
    for i in range(8):
        path = pl.Path(root + "img_{}_bbox.png".format(i))
        assert path.isfile()


# @patch('builtins.print')
# def test_print_hello(mock_print):
#     print_hello()
#     assert mock_print.call_args.args == ("Hello, world!",)

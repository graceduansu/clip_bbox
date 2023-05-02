from argparse import ArgumentParser
from clip_bbox.clipbbox import run_clip_bbox


def main(arg_strings=None):
    parser = ArgumentParser()
    parser.add_argument('imgpath', help="path to input image")
    parser.add_argument('caption', help="caption of input image")
    parser.add_argument('outpath', help="path to output image displaying bounding boxes")
    # parser.add_argument('--height', help="height of output image")
    # parser.add_argument('--width', help="width of output image")
    args = parser.parse_args(arg_strings)
    run_clip_bbox(args.imgpath, args.caption, args.outpath)


if __name__ == "__main__":
    main()

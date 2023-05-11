# Welcome to clip_bbox's documentation!

## Overview / About

[CLIP](https://github.com/openai/CLIP) is a neural network, pretrained on image-text pairs, that can predict the most relevant text snippet for a given image. 

Given an image and a natural language text label, `CLIP_BBox` will obtain the image's spatial embedding and text label's embedding from CLIP, compute the similarity heatmap between the embeddings, then draw bounding boxes around the image regions with the highest image-text correspondences. 

## Installation

Use pip to install clip_bbox as a Python package:

    $ pip install clip-bbox

### Use As a Command Line Script

```
usage: python -m clip_bbox [-h] imgpath caption outpath

positional arguments:
  imgpath     path to input image
  caption     caption of input image
  outpath     path to output image displaying bounding boxes

optional arguments:
  -h, --help  show this help message and exit
```

To draw bounding boxes on an image based on its caption, run

    $ python -m clip_bbox "path/to/img.png" "caption of your image" "path/to/output_path.png"

#### Usage Example
```eval_rst
.. image:: ../images/command_line_usage.gif
```

### Use As a Python Module

To draw bounding boxes on an image based on its caption, do the following:

```python
from clip_bbox import run_clip_bbox

run_clip_bbox("path/to/img.png", "caption of your image", "path/to/output_path.png")
```

Using the `run_clip_bbox()` will result in an output image displaying the original image, the similarity heatmap between then spatial and text embeddings from CLIP, and the image with bounding boxes & heatmap drawn on top of it. 

Here is an example output image for the caption `"a camera on a tripod"`:
```eval_rst
.. image:: ../images/example_output.png
```

#### Usage Example

```python
from clip_bbox import run_clip_bbox

run_clip_bbox("rocket.png", "a rocket standing on a launchpad", "result.png")
```

Result: Produces `result.png`, which displays the similarity heatmap and bounding boxes for an image of a rocket and its caption. 
```eval_rst
.. image:: ../images/rocket_result.png
```

```eval_rst
.. toctree::
   :maxdepth: 4
   :caption: Contents

   modules.rst
```

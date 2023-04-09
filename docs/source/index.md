# Welcome to clip_bbox's documentation!

## Installation

Use pip to install clip_bbox as a Python package:

    $ pip install clip-bbox

### Use As a Command Line Script

To draw bounding boxes on an image, run

    $ python clip_bbox.py --img "path/to/img.png" 

### Use As a Python Module

To draw bounding boxes on an image, do the following:

```python
from clip_bbox import run_clip_bbox

run_clip_bbox('path/to/img.png')
```

Using the `run_clip_bbox()` will result in an output image displaying the original image, the similarity heatmap between then spatial and text embeddings from CLIP, and the image with bounding boxes & heatmap drawn on top of it. 

Here is an example output image:
```eval_rst
.. image:: ../images/example_output.png
```

```eval_rst
.. toctree::
   :maxdepth: 4
   :caption: Contents

   modules.rst
```

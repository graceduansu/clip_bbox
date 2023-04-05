
# CLIP_BBox

`CLIP_BBox` is a Python library for detecting image objects with natural language text labels. 

[![Build Status](https://github.com/graceduansu/clip_bbox/workflows/Build%20Status/badge.svg?branch=main)](https://github.com/graceduansu/clip_bbox/actions?query=workflow%3A%22Build+Status%22)
[![codecov](https://codecov.io/gh/graceduansu/clip_bbox/branch/main/graph/badge.svg)](https://codecov.io/gh/graceduansu/clip_bbox)
[![GitHub](https://img.shields.io/github/license/graceduansu/clip_bbox)](./LICENSE)
![GitHub issues](https://img.shields.io/github/issues/graceduansu/clip_bbox)
[![PyPI](https://img.shields.io/pypi/v/clip-bbox)](https://pypi.org/project/clip-bbox/)


## Overview

[CLIP](https://github.com/openai/CLIP) is a neural network, pretrained on image-text pairs, that can predict the most relevant text snippet for a given image. 

Given an image and a natural language text label, `CLIP_BBox` will obtain the image's spatial embedding and text label's embedding from CLIP, compute the similarity heatmap between the embeddings, then draw a bounding box around the image region with the highest image-text correspondence. 

### Features

The library will provide functions for the following operations:
* Getting and appropriately reshaping an image's spatial embedding from the CLIP model before it performs attention-pooling
* Getting a text snippet's embedding from the CLIP model
* Computing the similarity heatmap between a pair of spatial and text embeddings from CLIP
* Drawing bounding boxes on an image, given a similarity heatmap and a similarity threshold

## Install

Use pip to install clip_bbox as a Python package:

    $ pip install clip_bbox

## Example Usage

### Use As a Command Line Script

To draw bounding boxes on an image, run

    $ python clip_bbox.py --img "path/to/img.png" 

### Use As a Python Module

To draw bounding boxes on an image, do the following:

```python
from clip_bbox import run_clip_bbox

run_clip_bbox('path/to/img.png')
```

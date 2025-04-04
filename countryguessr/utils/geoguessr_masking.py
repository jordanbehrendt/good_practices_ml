# -*- coding: utf-8 -*-
"""
utils.geoguessr_masking
utils------------------

Methods to allow geoguessr overlay masking.

:copyright: Cognitive Systems Lab, 2025
"""
# Imports
# Built-in
import PIL.Image
import importlib.resources as resources

# Local
import countryguessr.utils as utils

# 3r-party
import numpy as np


def get_overlay_mask_file():
    with resources.path(utils, "overlay_mask.png") as mask_finder:
        path = mask_finder

    return path


OVERLAY_MASK_FILE = get_overlay_mask_file()


def mask_overlay(original, mask=OVERLAY_MASK_FILE, mask_fill=False):
    """Mask the overlays of geoguessr images.

    Parameters
    ----------
    original : str
        Path to iamge file to apply mask to.
    mask : str | Path, optional
        path to mask file. By default uses
        the resource `overlay_mask` that already marks geoguessr
        overlays.
    mask_fill : bool
        whether to fill the masked overlays with uniform noise.
        By default is False.
    """

    image = PIL.Image.open(original)

    mask = PIL.Image.open(mask)
    mask = mask.convert("1", dither=None)
    mask.save("mask.png")

    shape = (image.size[1], image.size[0], 3)
    if mask_fill:
        fill = np.random.uniform(0, 255, shape).astype(np.uint8)
    else:
        fill = np.zeros(shape).astype(np.uint8)

    fill = PIL.Image.fromarray(fill, mode="RGB")
    fill.save("fill.png")

    masked = PIL.Image.composite(image, fill, mask)

    return masked

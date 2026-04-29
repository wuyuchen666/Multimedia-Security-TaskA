"""Compatibility layer for JPEG DCT access.

Backend order:
1. jpegio, if available.
2. jpeglib, if available.
3. Pure OpenCV/Numpy fallback for aarch64 servers where both C extensions fail.

The OpenCV fallback computes JPEG-like luminance DCT coefficients from decoded
pixels and writes by inverse DCT + cv2.imwrite. It is not a bit-exact JPEG
coefficient writer, but it keeps the experiment runnable on ARM machines.
"""

from dataclasses import dataclass

import cv2
import numpy as np


STD_LUMA_QTABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


try:
    import jpegio as _jpegio
except ImportError:
    _jpegio = None

try:
    import jpeglib as _jpeglib
except ImportError:
    _jpeglib = None

if _jpegio is not None:
    BACKEND = "jpegio"
elif _jpeglib is not None:
    BACKEND = "jpeglib"
else:
    BACKEND = "opencv_dct"


@dataclass
class OpenCVDCTJPEG:
    coef_arrays: list
    quant_tables: list
    quality: int = 95


def quality_to_luma_qtable(quality):
    quality = int(np.clip(quality, 1, 100))
    scale = 5000 / quality if quality < 50 else 200 - quality * 2
    table = np.floor((STD_LUMA_QTABLE * scale + 50) / 100)
    return np.clip(table, 1, 255).astype(np.int32)


def pad_to_block(image):
    h, w = image.shape
    hp = int(np.ceil(h / 8) * 8)
    wp = int(np.ceil(w / 8) * 8)
    if (h, w) == (hp, wp):
        return image
    return np.pad(image, ((0, hp - h), (0, wp - w)), mode="edge")


def spatial_to_coefficients(gray, qtable):
    gray = pad_to_block(gray).astype(np.float64)
    h, w = gray.shape
    coef = np.empty((h, w), dtype=np.int32)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = gray[y:y + 8, x:x + 8] - 128.0
            coef[y:y + 8, x:x + 8] = np.rint(cv2.dct(block) / qtable).astype(np.int32)
    return coef


def coefficients_to_spatial(coef, qtable):
    h, w = coef.shape
    image = np.empty((h, w), dtype=np.float64)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = coef[y:y + 8, x:x + 8].astype(np.float64) * qtable
            image[y:y + 8, x:x + 8] = cv2.idct(block) + 128.0
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def read(path, quality=95):
    if _jpegio is not None:
        return _jpegio.read(path)
    if _jpeglib is not None:
        return _jpeglib.to_jpegio(_jpeglib.read_dct(path))

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read image for OpenCV DCT backend: {path}")
    qtable = quality_to_luma_qtable(quality)
    return OpenCVDCTJPEG([spatial_to_coefficients(gray, qtable)], [qtable], int(quality))


def write(jpeg, path):
    if _jpegio is not None:
        return _jpegio.write(jpeg, path)
    if _jpeglib is not None:
        return jpeg.write(path)

    image = coefficients_to_spatial(jpeg.coef_arrays[0], jpeg.quant_tables[0])
    return cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg.quality)])

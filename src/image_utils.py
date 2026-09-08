"""
image_utils.py
Utilidades de conversión/normalización de imágenes numpy.
"""

import numpy as np


def normalize_if_needed(img):
    """Normaliza a rango [0,1] si la imagen viene en escala 0-255."""
    img = img.astype(np.float32)
    max_val = np.nanmax(img)
    if max_val > 1.5:
        img = img / 255.0
    return img


def to_uint8_vis(img):
    """Stretch simple (÷0.3) para visualización tipo 'realce de contraste'."""
    img = np.clip(img / 0.3, 0, 1)
    return (img * 255).astype(np.uint8)


def to_uint8_rgb(img):
    """Conversión directa a uint8 sin stretch (reflectancia real 0-1 -> 0-255)."""
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

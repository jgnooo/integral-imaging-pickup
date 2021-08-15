import os
import cv2
import numpy as np

from PIL import Image

""" Convert elemental image array to sub aperture image array """

import utils


def inpainting(sub_aperture):
    """

    Args:
    
    Returns:
    
    """
    color = sub_aperture
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(thresh)

    inpainted = cv2.inpaint(color, mask, 3, cv2.INPAINT_NS)
    return inpainted


def generate_sub_apertures(elem_plane, P_L, num_of_lenses):
    """

    Args:
    
    Returns:
    
    """
    sub_apertures = np.zeros((P_L * num_of_lenses, P_L * num_of_lenses, 3))
    sub_aperture = np.zeros((num_of_lenses, num_of_lenses, 3))

    for elem_i in range(P_L):
        for elem_j in range(P_L):
            y_start = elem_i * num_of_lenses
            y_end = elem_i * num_of_lenses + num_of_lenses
            x_start = elem_j * num_of_lenses
            x_end = elem_j * num_of_lenses + num_of_lenses

            for i in range(num_of_lenses):
                for j in range(num_of_lenses):
                    sub_aperture[i, j, :] = elem_plane[elem_i + (i * P_L), elem_j + (j * P_L), :]

            sub_apertures[y_start:y_end, x_start:x_end] = inpainting(sub_aperture.astype(np.uint8))

    return sub_apertures
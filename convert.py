"""Generate central depth, pixel size, valid depth range, converted depth"""

import utils


def central_depth(f, g):
    """Calculate central depth of the integral imaging system.

    Args:
        f : Focal length of elemental lens.
        g : Gap between lens and display.
    Returns:
        d : Central depth
    """
    d = (f * g) / (g - f)
    return d

def pixel_size_object_img(d, g, P_D):
    """Calculate a pixel size of the image.

    Args:
        d   : Central depth.
        g   : Gap between lens and display.
        P_D : Pixel pitch of LCD.
    Returns:
        P_I : Pixel size of the object image.
    """
    P_I = (d / g) * P_D
    return P_I


def convert_depth(depth, f, g, P_D, P_L, output_dir):
    """Convert the depth map.

    Args:
        depth      : Depth map corresponding the RGB image.
        f          : Focal length of elemental lens.
        g          : Gap between lens and display.
        P_D        : Pixel pitch of LCD.
        P_L        : Size of elemental lens.
        ourput_dir : Output directory for converted depth image.
    Returns:
        d          : Central depth.
        P_I        : Pixel size of the object image.
        delta_d    : Depth range of integral imaging.
        L          : Converted depth map.
    """
    d = central_depth(f, g)
    P_I = pixel_size_object_img(d, g, P_D)

    delta_d = ((2 * d) / P_L) * P_I
    converted_depth_min = d - delta_d / 2
    converted_depth_max = d + delta_d / 2

    L = (d * (depth.max() + depth.min())) / (depth * 2)
    
    L[L < converted_depth_min] = converted_depth_min
    L[L > converted_depth_max] = converted_depth_max

    return d, P_I, delta_d, L
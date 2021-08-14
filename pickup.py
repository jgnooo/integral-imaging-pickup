""" Integral Imaing Pickup System """

import os
import cv2
import numpy as np

from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import utils


def generate_object_coords(color, L):
    """Generate coordinates of object pixel.

    Args:
        color        : Color image.
        L            : Converted depth image.
    Returns:
        pixel_coords : Coordinates of object pixels. (x, y, depth)
    """
    height, width, _ = color.shape
    coords = utils.generate_coords(height, width, L, is_depth=True)
    return coords


def generate_virtual_lens(num_of_lenses, P_L):
    """Generate virtual lens array.

    Args:
        num_of_lens : Number of lenses of lens array.
        P_L         : Size of elemental lens.
    Returns:
        lenses_idx  : Indices of virtual lenses.
        lenses_loc  : Location of each lenses.
    """
    lenses_idx = []
    for i_L in range(num_of_lenses):
        for j_L in range(num_of_lenses):
            lenses_idx.append((i_L, j_L))

    lenses_loc = utils.generate_coords(num_of_lenses, num_of_lenses)
    return lenses_idx, lenses_loc


def generate_elemental_plane(num_of_lenses, P_L):
    """Generate elemental image plane.

    Args:
        num_of_lenses : Number of lenses of lens array.
        P_L           : Size of elemental lens.
    Returns:
        elem_plane    : Elemental image plane.
        elem_coords   : Coordinates of elemental image.
    """
    elem_plane_h = P_L * num_of_lenses
    elem_plane_w = P_L * num_of_lenses

    elem_plane = np.zeros((elem_plane_h, elem_plane_w, 3))
    elem_coords = utils.generate_coords(elem_plane_h, elem_plane_h)
    return elem_plane, elem_coords


def points_transfrom(i, j, i_L, j_L, P_L, P_I, g, L):
    """Transform points of object pixels to elemental image pixels.

    Args:
        i   : Location of pixel.
        j   : Location of pixel.
        i_L : Elemental lens index.
        j_L : Elemental lens index.
        P_L : Size of elemental lens.
        P_I : Pixel size of the object image.
        g   : Gap between lens and display.
        L   : Converted depth information.
    Returns:
        u   : Transformed coordinate corresponding 'x'.
        v   : Transformed coordinate corresponding 'y'.
    """
    u = P_L * i_L - ((i * P_I) - (P_L * i_L)) * (g / L)
    v = P_L * j_L - ((j * P_I) - (P_L * j_L)) * (g / L)
    return u, v


def generate_elemental_imgs_GPU(color, L, P_L, P_I, g, num_of_lenses):
    """Generate elemental images by paper's method.

    Args:
        color         : Color image.
        L             : Converted depth information.
        P_L           : Size of elemental lens.
        P_I           : Pixel size of the object image.
        g             : Gap between lens and display.
        num_of_lenses : Number of lenses of lens array.
    Returns:
        elem_plane    : Elemental image array.
    """
    height, width, _ = color.shape

    pixel_coords = generate_object_coords(color, L)
    lenses_idx, lenses_loc= generate_virtual_lens(num_of_lenses, P_L)
    elem_plane, elem_coords = generate_elemental_plane(num_of_lenses, P_L)
    elem_plane_h, elem_plane_w, _ = elem_plane.shape
    half_elem = elem_plane_h // 2
    half_h = height // 2
    half_w = width // 2

    elem_plane_R = elem_plane[:, :, 0].astype(np.float32)
    elem_plane_G = elem_plane[:, :, 1].astype(np.float32)
    elem_plane_B = elem_plane[:, :, 2].astype(np.float32)
    elem_plane_R_gpu = cuda.mem_alloc(elem_plane_R.nbytes)
    elem_plane_G_gpu = cuda.mem_alloc(elem_plane_G.nbytes)
    elem_plane_B_gpu = cuda.mem_alloc(elem_plane_B.nbytes)
    cuda.memcpy_htod(elem_plane_R_gpu, elem_plane_R)
    cuda.memcpy_htod(elem_plane_G_gpu, elem_plane_G)
    cuda.memcpy_htod(elem_plane_B_gpu, elem_plane_B)

    pixel_x = pixel_coords[0].astype(np.float32)
    pixel_y = pixel_coords[1].astype(np.float32)
    pixel_L = pixel_coords[2].astype(np.float32)
    pixel_x_gpu = cuda.mem_alloc(pixel_x.nbytes)
    pixel_y_gpu = cuda.mem_alloc(pixel_y.nbytes)
    pixel_L_gpu = cuda.mem_alloc(pixel_L.nbytes)
    cuda.memcpy_htod(pixel_x_gpu, pixel_x)
    cuda.memcpy_htod(pixel_y_gpu, pixel_y)
    cuda.memcpy_htod(pixel_L_gpu, pixel_L)

    lens_loc_x = lenses_loc[0].astype(np.float32)
    lens_loc_y = lenses_loc[1].astype(np.float32)
    lens_loc_x_gpu = cuda.mem_alloc(lens_loc_x.nbytes)
    lens_loc_y_gpu = cuda.mem_alloc(lens_loc_y.nbytes)
    cuda.memcpy_htod(lens_loc_x_gpu, lens_loc_x)
    cuda.memcpy_htod(lens_loc_y_gpu, lens_loc_y)

    R = color[:, :, 0].astype(np.float32)
    G = color[:, :, 1].astype(np.float32)
    B = color[:, :, 2].astype(np.float32)
    R_gpu = cuda.mem_alloc(R.nbytes)
    G_gpu = cuda.mem_alloc(G.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    cuda.memcpy_htod(R_gpu, R)
    cuda.memcpy_htod(G_gpu, G)
    cuda.memcpy_htod(B_gpu, B)

    elem_coords_x = elem_coords[0].astype(np.float32)
    elem_coords_y = elem_coords[1].astype(np.float32)
    elem_coords_x_gpu = cuda.mem_alloc(elem_coords_x.nbytes)
    elem_coords_y_gpu = cuda.mem_alloc(elem_coords_y.nbytes)
    cuda.memcpy_htod(elem_coords_x_gpu, elem_coords_x)
    cuda.memcpy_htod(elem_coords_y_gpu, elem_coords_y)

    mod = SourceModule("""
        __global__ void generate_EIA(float * R, float * G, float * B,
                                        float * elem_R, float * elem_G, float * elem_B,
                                        float * pixel_x, float * pixel_y, float * pixel_L,
                                        float * elem_coords_x, float * elem_coords_y,
                                        float * lens_loc_x, float * lens_loc_y,
                                        float P_L, float P_I, float g,
                                        int height, int width, int num_of_lenses) {

            int p_x = threadIdx.x + blockDim.x * blockIdx.x;
            int p_y = threadIdx.y + blockDim.y * blockIdx.y;

            int i, j;
            float u, v;
            float p_i, p_j;
            int shift_x, shift_y;

            int half_h = (int)(height / 2);
            int half_w = (int)(width / 2);

            int h_of_elem = num_of_lenses * (int)P_L;
            int w_of_elem = num_of_lenses * (int)P_L;
            
            int half_h_elem = (int)(h_of_elem / 2);
            int half_w_elem = (int)(w_of_elem / 2);

            float lens_min_x, lens_min_y;
            
            if (p_x < width && p_y < height) {
                for (i = 0; i < num_of_lenses; i++) {
                    for (j = 0; j < num_of_lenses; j++) {
                        shift_x = i * (int)P_L;
                        shift_y = j * (int)P_L;
                        
                        u = P_L * lens_loc_x[i + j * num_of_lenses] - ((pixel_x[p_x + p_y * width] * P_I) - (P_L * lens_loc_x[i + j * num_of_lenses])) * (g / pixel_L[p_x + p_y * width]);
                        v = P_L * lens_loc_y[i + j * num_of_lenses] - ((pixel_y[p_x + p_y * width] * P_I) - (P_L * lens_loc_y[i + j * num_of_lenses])) * (g / pixel_L[p_x + p_y * width]);

                        lens_min_x = elem_coords_x[shift_x + shift_y * w_of_elem];
                        lens_min_y = elem_coords_y[shift_x + shift_y * w_of_elem];

                        if ((lens_min_x <= u && u <= lens_min_x + P_L) && (lens_min_y <= v && v <= lens_min_y + P_L)) {
                            u += half_w_elem;
                            v += half_h_elem;

                            p_i = pixel_x[p_x + p_y * width] + half_w;
                            p_j = pixel_y[p_x + p_y * width] + half_h;

                            if ((0 <= (int)u && (int)u < w_of_elem) && (0 <= (int)v && (int)v < h_of_elem)) {
                                elem_R[(int)u + (int)v * w_of_elem] = R[(int)p_i + (int)p_j * width];
                                elem_G[(int)u + (int)v * w_of_elem] = G[(int)p_i + (int)p_j * width];
                                elem_B[(int)u + (int)v * w_of_elem] = B[(int)p_i + (int)p_j * width];
                            }
                        }
                    }
                }
            }   
        }
    """)
    
    gird_h = height // 20
    gird_w = width // 20
    func = mod.get_function("generate_EIA")
    func(R_gpu, G_gpu, B_gpu,
            elem_plane_R_gpu, elem_plane_G_gpu, elem_plane_B_gpu,
            pixel_x_gpu, pixel_y_gpu, pixel_L_gpu,
            elem_coords_x_gpu, elem_coords_y_gpu,
            lens_loc_x_gpu, lens_loc_y_gpu,
            np.float32(P_L), np.float32(P_I), np.float32(g),
            np.int32(height), np.int32(width), np.int32(num_of_lenses),
            block=(20, 20, 1),
            grid=(gird_w, gird_h))

    elem_R = np.empty_like(elem_plane_R)
    elem_G = np.empty_like(elem_plane_G)
    elem_B = np.empty_like(elem_plane_B)
    cuda.memcpy_dtoh(elem_R, elem_plane_R_gpu)
    cuda.memcpy_dtoh(elem_G, elem_plane_G_gpu)
    cuda.memcpy_dtoh(elem_B, elem_plane_B_gpu)
    
    EIA = np.stack([elem_R, elem_G, elem_B], axis=2)
    inpainted_EIA = inpainting(EIA.astype(np.uint8), num_of_lenses, P_L)
    return inpainted_EIA


def inpainting(EIA, num_of_lenses, P_L):
    inpainted_EIA = EIA.copy()
    for i in range(num_of_lenses):
        for j in range(num_of_lenses):
            elem = EIA[i * P_L:i * P_L + P_L, j * P_L:j * P_L + P_L]
            gray = cv2.cvtColor(elem, cv2.COLOR_RGB2GRAY)

            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_not(thresh)

            inpainted = cv2.inpaint(elem, mask, 5, cv2.INPAINT_NS)

            inpainted_EIA[i * P_L:i * P_L + P_L, j * P_L:j * P_L + P_L] = inpainted
    return inpainted_EIA
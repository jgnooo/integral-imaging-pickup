import os
import math
import time
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image

import monodepth.depth_estimator as estimator
import convert
import pickup
import subaperture

import utils 


parser = argparse.ArgumentParser(description='Generation Light Field using Depth Estimation.')
parser.add_argument('--color_path', type=str, default='./inputs/color.png', help='Path of input image.')
parser.add_argument('--depth_path', type=str, default='./inputs/depth.png', help='Path of depth image.')
parser.add_argument('--output_path', type=str, default='./results/', help='Output root directory.')

parser.add_argument('--model_path', type=str, default='./monodepth/model.h5', help='Model file for predicting a depth.')

parser.add_argument('--is_prediction', action='store_true', help='Whether or not need to predict a depth map.')
parser.add_argument('--is_gpu', action='store_true', help='Select GPU or Not.')

parser.add_argument('--num_of_lenses', type=int, default=200, help='Number of elemental lenses.')
parser.add_argument('--P_D', type=float, default=0.1245, help='Pixel pitch of LCD.')
parser.add_argument('--P_L', type=int, default=1.8675, help='Size of elemental lens.')
parser.add_argument('--f', type=float, default=10, help='Focal length of elemental lens.')
parser.add_argument('--g', type=float, default=12, help='Gap between lens and display.')

args = parser.parse_args()


def get_depth_map(color):
    """Predict a depth map from a single RGB image.

    Args:
        color : Input color image.
    Returns:
        depth : Predicted a depth image corresponding a input RGB image.
    """
    height, width, _ = color.shape

    # if color height != 480 and color width != 640, Resize the input image.
    if height != 480 and width != 640:
        color = estimator.resize_image(color)
    
    net_input = estimator.preprocess_image(color)
    depth = estimator.estimate_depth(net_input, height, width, args.model_path)
    return depth


def cvt_mm2pixel(inputs, pitch_of_pixel):
    """Convert mm unit to pixel unit.

    Args:
        inputs : Input dictionary including image information and lens parameters.
        pitch_of_pixel : Pixel pitch of LCD.
    Returns:
        cvt_inputs : Input dictionary converted pixel units.
    """
    cvt_inputs = {}
    cvt_inputs['depth'] = utils.cvt_mm2pixel(inputs['depth'], pitch_of_pixel)
    cvt_inputs['P_D'] = utils.cvt_mm2pixel(inputs['P_D'], pitch_of_pixel)
    cvt_inputs['P_L'] = utils.cvt_mm2pixel(inputs['P_L'], pitch_of_pixel)
    cvt_inputs['f'] = utils.cvt_mm2pixel(inputs['f'], pitch_of_pixel)
    cvt_inputs['g'] = utils.cvt_mm2pixel(inputs['g'], pitch_of_pixel)
    return cvt_inputs


def get_input_params():
    """Parameters
    
    Image information
        - color : Color image.
        - depth : Depth image corresponding a color image.
    
    Lens Parameters
        Information of Lens-array
            - P_L           : Size of elemental lens.
            - num_of_lenses : Number of elemental lens.
            - f             : Focal length of elemental lens.

        Information of Display
            - P_D           : Pixel pitch of LCD.
            - g             : Gap between lens and display.
    """
    name = args.color_path.split('/')[-1].split('.')[0]
    
    color = utils.load_image(args.color_path)

    # if you have a depth map corresponding a color image, Do not need to predict.
    if args.is_prediction:
        depth = get_depth_map(color)
    else:
        depth = np.load(args.depth_path)
    
    inputs = {}
    inputs['name'] = name
    inputs['color'] = color
    inputs['depth'] = depth
    inputs['num_of_lenses'] = args.num_of_lenses
    inputs['P_D'] = args.P_D
    inputs['P_L'] = args.P_L
    inputs['f'] = args.f
    inputs['g'] = args.g
    return inputs


def main():
    '''
        Input Stage : Generate converted depth,
                               central depth,
                               depth range,
                               pixel size of image from input parameters.
    '''
    print('\nGenerate parameters for pickup system...')

    # Setup the input parameters.
    inputs = get_input_params()

    # Make directory for saving result images
    output_dir = os.path.join(
        args.output_path, 'N{}F{}G{}_{}'.format(args.num_of_lenses, args.f, args.g, inputs['name'])
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert mm to pixel
    cvt_inputs = cvt_mm2pixel(inputs, pitch_of_pixel=inputs['P_D'])

    # Convert depth data
    d, P_I, delta_d, L = convert.convert_depth(cvt_inputs['depth'],
                                               cvt_inputs['f'],
                                               cvt_inputs['g'],
                                               cvt_inputs['P_D'],
                                               cvt_inputs['P_L'],
                                               output_dir)
    
    utils.save_image(utils.visualize_depth(inputs['depth']), os.path.join(output_dir, 'depth.jpg'))
    utils.save_image(utils.visualize_depth(L), os.path.join(output_dir, 'converted_depth.jpg'))
    print('Parameters generated.')

    # Print parameters
    # utils.print_params(inputs, cvt_inputs, d, P_I, delta_d, color, L)

    '''
        Generate Elemental image array.
    '''
    print('\nPickup Stage...')

    if args.is_gpu:
        EIA = pickup.generate_elemental_imgs_GPU(inputs['color'], 
                                                 L.astype(np.int32),
                                                 int(cvt_inputs['P_L']),
                                                 P_I,
                                                 cvt_inputs['g'],
                                                 inputs['num_of_lenses'])
    else:
        EIA = pickup.generate_elemental_imgs_CPU(inputs['color'], 
                                                 L,
                                                 int(cvt_inputs['P_L']),
                                                 P_I,
                                                 cvt_inputs['g'],
                                                 inputs['num_of_lenses'])
    
    utils.save_image(EIA, os.path.join(output_dir, 'elemental_image_array.jpg'))
    print('Elemental Image Array generated.')

    '''
        Generate Sub Aperture image array.
    '''
    print('\nGenerate sub aperture image array...')
    sub_apertures = subaperture.generate_sub_apertures(EIA,
                                                       int(cvt_inputs['P_L']),
                                                       inputs['num_of_lenses'])
    print('Sub-Aperture Images generated.')
    utils.save_image(sub_apertures, os.path.join(output_dir, 'sub_aperture_array.jpg'))
    
    print('\nDone.')


if __name__ == "__main__":
    main()

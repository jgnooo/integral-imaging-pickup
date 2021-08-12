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
import sub_aperture

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
    height, width, _ = color.shape

    # if color height != 480 and color width != 640, Resize the input image.
    if height != 480 and width != 640:
        color = estimator.resize_image(color)
    
    net_input = estimator.preprocess_image(color)
    depth = estimator.estimate_depth(net_input, height, width, args.model_path)
    return depth


def cvt_mm2pixel(inputs, pitch_of_pixel):
    cvt_inputs = {}
    cvt_inputs['depth'] = utils.cvt_mm2pixel(inputs['depth'], pitch_of_pixel)
    cvt_inputs['P_D'] = utils.cvt_mm2pixel(inputs['P_D'], pitch_of_pixel)
    cvt_inputs['P_L'] = utils.cvt_mm2pixel(inputs['P_L'], pitch_of_pixel)
    cvt_inputs['f'] = utils.cvt_mm2pixel(inputs['f'], pitch_of_pixel)
    cvt_inputs['g'] = utils.cvt_mm2pixel(inputs['g'], pitch_of_pixel)
    return cvt_inputs


def get_input_params():
    """Parameters
    
    Image
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
    print('\nInput Stage...')
    start = time.time()

    # Setup the input parameters.
    inputs = get_input_params()

    output_dir = os.path.join(
        args.output_path, 'N{}F{}G{}'.format(args.num_of_lenses, args.f, args.g)
    )

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Convert mm to pixel
    cvt_inputs = cvt_mm2pixel(inputs, pitch_of_pixel=inputs['P_D'])

    # Convert depth data
    inputstage = convet(output_dir)
    d, P_I, delta_d, color, L = inputstage.convert_depth(inputs['color'],
                                                         cvt_inputs['depth'],
                                                         cvt_inputs['f'],
                                                         cvt_inputs['g'],
                                                         cvt_inputs['P_D'],
                                                         cvt_inputs['P_L'],
                                                         inputs['roi_w'])

    print('Input Stage Done.')

    # Print parameters
    # utils.print_params(inputs, cvt_inputs, d, P_I, delta_d, color, L)

    '''
        Calculation Stage
    '''
    # Generate elemental images
    # print('\nCalculation Stage...')
    # start = time.time()
    # calculationstage = CalculationStage(output_dir)
    # if args.is_gpu:
    #     elem_plane = calculationstage.generate_elemental_imgs_GPU(color, 
    #                                                               L.astype(np.int32),
    #                                                               int(cvt_inputs['P_L']),
    #                                                               P_I,
    #                                                               cvt_inputs['g'],
    #                                                               inputs['num_of_lenses'])
    # else:
    #     elem_plane = calculationstage.generate_elemental_imgs_CPU(color, 
    #                                                               L,
    #                                                               int(cvt_inputs['P_L']),
    #                                                               P_I,
    #                                                               cvt_inputs['g'],
    #                                                               inputs['num_of_lenses'])

    # print('Elemental Image Array generated.')

    # '''
    #     Generate Sub Aperture
    # '''
    # print('\nGenerate sub aperture images...')
    # aperture = SubAperture(output_dir)
    # sub_apertures = aperture.generate_sub_apertures(elem_plane,
    #                                                 int(cvt_inputs['P_L']),
    #                                                 inputs['num_of_lenses'])
    # print('Sub-Aperture Images generated.')

    # print('\nElapsed time : {}s'.format(time.time() - start))
    # print('Done.')


if __name__ == "__main__":
    main()

# Integral Imaging Pickup System
Python Implementation of integral imaging pickup system.
> Li, Gang, et al. "Simplified integral imaging pickup method for real objects using a depth camera." Journal of the Optical Society of Korea 16.4 (2012): 381-385.

Integral imaging system consist of pickup system and display system.   
- In pickup system, the system generate **Elemental Image Array**.
- In display system, the observer can observe the 3D display using physical micro-lens array from generated **Elemental Image Array**.

This code is for integral imaging pickup system to generate **Elemental Image Array** and **Sub-aperture Image Array**.   
   
_In this system, use monocular depth estimaion network instead of a depth camera._

## Process

## Requirements
- PyCuda
- Tensorflow 2.2
- Numpy
- Pilow

## Pre-trained monocular depth estimation model weights
* [Trained by NYU RGB-D V2](https://drive.google.com/uc?export=download&id=1k8McRE2vOtrkHmG9ZU6Cd-IUDtr2Fbbv) (650 MB)

## Usage
- Download the model weights.
    - Go to the link above, and Download model weights.
    - Locate at `monodepth` directory.
- Prepare the input image.
    - Locate the input color image to `inputs` directory or `/your/own/dir/`.
    - If you have a depth image corresponding a input color image,   
      locate a depth map to `depths` directory or `/your/own/dir/`.
- Start integral imaging pickup system.
    ```Bash
    python main.py \
        --color_path ./inputs or /your/own/path/ \
        --depth_path ./depths or /your/own/path/ \
        --output_path ./results or /your/own/path/ \
        --model_path ./monodepth/model.h5 \
        --is_prediction \
        --is_gpu \
        --num_of_lenses 200 \
        --P_D 0.1245 \
        --P_L 1.8675 \ 
        --f 10 \
        --g 12
    ```
    - See `main.py` parser for more details.
    
## Results
- Depth image
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/55485826/127944218-2c72c094-2bc6-4b15-8241-f7e36e25dbde.png"></p> -->
- Elemental Image Array
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/55485826/127944218-2c72c094-2bc6-4b15-8241-f7e36e25dbde.png"></p> -->
- Sub-aperture Image Array
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/55485826/127944218-2c72c094-2bc6-4b15-8241-f7e36e25dbde.png"></p> -->

## To-Do List
1. Update the code generating sub-aperture images using PyCuda.
2. GUI.
3. Check these codes.
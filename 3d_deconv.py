import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,signal
import cv2
from imageio import imread,imwrite
from matplotlib.widgets import Slider
from util import *
from deconv import *
import glob
import config

root_path = config.root_path
target_tiff_path = config.target_tiff_path
psf_stack_path = config.psf_stack_path
psf_range_indices = config.psf_range_indices



def reading_psf_stack(psf_stack_path,psf_range_indices):
    x1,x2,y1,y2 = psf_range_indices
    stack_depth_name_list = sorted(glob.glob(psf_stack_path+'*'),key=extract_depth_from_name)
    depth_len = len(stack_depth_name_list)
    psf_stack = np.zeros((depth_len,x2-x1,y2-y1))
    for i in range(depth_len):
        psf_stack[i] = imread(stack_depth_name_list[i]+'/Averaged_Image.jpg')[x1:x2,y1:y2]
    norm_psf_stack = normalize_psf_stack(psf_stack)

    print('The PSF stack size is: ')
    print(norm_psf_stack.shape)
    return norm_psf_stack

def reading_target_tiff(target_tiff_path):
    tiff = tifffile.TiffFile(target_tiff_path)
    target_img = tiff.pages[0].asarray()
    target_shape = (len(tiff.pages),target_img.shape[0],target_img.shape[1])
    fish_stack = np.zeros(target_shape)
    for i in range(target_shape[0]):
        fish_stack[i] = tiff.pages[i].asarray()
        fish_stack[i] = fish_stack[i]/fish_stack[i].max()

    print('The target (fish) volume size is: ')
    print(target_shape)
    return fish_stack

def size_rematching(psf_stack, target_stack):
    # Z dimensions of PSF Stack and Target Volume should be the same.
    assert psf_stack.shape[0] >= target_stack.shape[0], 'PSF Stack z dim is less than Target z dim!'

    if psf_stack.shape[0] > target_stack.shape[0]:
        print('Drop the last {} slices of the PSF stack'.format(psf_stack.shape[0]-target_stack.shape[0]))

        psf_stack = psf_stack[:target_stack.shape[0]]

    # Matching the Target stack size to be the same as the final LFM image size after convolution.
    conv_image = signal.fftconvolve(target_stack[0],psf_stack[0],'full')
    padding_fish_stack = padding_stack(target_stack,[target_stack.shape[0],conv_image.shape[0],conv_image.shape[1]])
    print('Fish Stack after padding: ')
    print(padding_fish_stack.shape)

    return psf_stack, padding_fish_stack

def data_preprocess_pipeline(psf_stack_path,target_tiff_path,psf_range_indices):
    norm_psf_stack = reading_psf_stack(psf_stack_path,psf_range_indices)
    fish_stack = reading_target_tiff(target_tiff_path)
    psf_stack, target_stack = size_rematching(norm_psf_stack,fish_stack)
    return psf_stack, target_stack

if __name__ == '__main__':
    
    psf_stack, target_stack = data_preprocess_pipeline(psf_stack_path,target_tiff_path,psf_range_indices)
    lfm_image = stack_fftconv_3d(target_stack,psf_stack)

    # Modify RL iteration number here! 
    iteration_num = 1

    deconv_stack = richardson_lucy_3d_fft(lfm_image,psf_stack,iteration_num,gt_stack=target_stack)
    np.save(config.saved_deconv_path + '/{}_conv_vol.npy'.format(iteration_num),deconv_stack)

    plt.figure()
    plt.title('Simulated Light Field Image')
    plt.imshow(lfm_image)
    plt.show()
    
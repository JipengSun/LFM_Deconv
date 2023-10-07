import numpy as np
from scipy import ndimage,signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

def stackalize_img(img,zdim):
    res = np.zeros((zdim,img.shape[0],img.shape[1]))
    for i in range(zdim):
        res[i] = img
    return res

def normalize_psf_stack(psf_stack):
    psf_stack = psf_stack.astype(np.float64)
    for i in range(psf_stack.shape[0]):
        psf_stack[i] = psf_stack[i]/np.sum(psf_stack[i])
    return psf_stack

def transpose_psf_stack(psf_stack):
    PSF_STACK_HAT = np.ones_like(psf_stack)
    for i in range(psf_stack.shape[0]):
        PSF_STACK_HAT[i] = np.transpose(psf_stack[i])
    return PSF_STACK_HAT

def stack_conv_3d(img_stack,psf_stack):
    z_dim = img_stack.shape[0]
    res = np.zeros_like(img_stack[0])
    for i in range(z_dim):
        res += signal.convolve2d(img_stack[i],psf_stack[i],'same')
    return res

def stack_fftconv_3d(img_stack,psf_stack):
    z_dim = img_stack.shape[0]
    res = np.zeros_like(signal.fftconvolve(img_stack[0],psf_stack[0],'same'))
    #res = np.zeros_like(img_stack[0])
    for i in range(z_dim):
        res += signal.fftconvolve(img_stack[i],psf_stack[i],'same')
    return res

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def display_stack(img_stack):
    # Setting Plot and Axis variables as subplots()
    # function returns tuple(fig, ax)
    fig, ax = plt.subplots()
    
    # Adjust the bottom size according to the
    # requirement of the user
    plt.subplots_adjust(bottom=0.25)
    
    current_slice = ax.imshow(img_stack[0])
    
    # Choose the Slider color
    slider_color = 'White'

    # Set the initial slider position
    initial_z = 0
    
    # Set the axis and slider position in the plot
    axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],
                            facecolor = slider_color)
    slider_position = Slider(axis_position,
                            'Z', 0, img_stack.shape[0]-1, valinit=initial_z,valstep = 1)
    
    # update() function to change the graph when the
    # slider is in use
    def update(val):
        z = int(slider_position.val)
        current_slice.set_data(img_stack[z, :, :])
        fig.canvas.draw_idle()
    
    # update function called using on_changed() function
    slider_position.on_changed(update)
    
    # Display the plot
    plt.show()

def find_digit_from_name(filename):
    filename = os.path.basename(filename)
    if '_' in filename:
        str_list = filename.split('_')
    else:
        str_list = filename.split('.')
    return float(str_list[0])

def extract_depth_from_name(filename):
    filename = os.path.basename(filename)
    #print(filename)
    return float(filename)

def padding_stack(img_stack,padding_size):
    orig_size = img_stack.shape
    padding_shape_list = []
    for i in range(3):
        padding_left = int((padding_size[i]-orig_size[i])/2)
        padding_right = padding_size[i]-orig_size[i]-padding_left
        padding_shape_list.append([padding_left,padding_right])   
    padding_shape_array = np.array(padding_shape_list)    
    pad_stack = np.pad(img_stack,padding_shape_array)
    #print(pad_stack.shape)
    return pad_stack



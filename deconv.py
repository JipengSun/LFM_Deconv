import numpy as np
from scipy import ndimage,signal
from util import *
from imageio import imwrite

root_path = '/Users/Jipeng/VSCodeProjects/LFM_Simulation'

def up_dim_conv_stack(img_2d, psf_stack):
    z_dim = psf_stack.shape[0]
    res_stack = np.zeros((z_dim,img_2d.shape[0],img_2d.shape[1]))
    for i in range(z_dim):
        res_stack[i] = signal.convolve2d(img_2d,psf_stack[i],'same')
    return res_stack

def up_dim_fftconv_stack(img_2d, psf_stack):
    z_dim = psf_stack.shape[0]
    res_stack = np.zeros((z_dim,img_2d.shape[0],img_2d.shape[1]))
    for i in range(z_dim):
        res_stack[i] = signal.fftconvolve(img_2d,psf_stack[i],'same')
    return res_stack

def l1_loss(est_img_stack,gt_img_stack):
    z_dim = est_img_stack.shape[0]
    l1 = 0
    for i in range (z_dim):
        l1 += np.abs(est_img_stack[i]-gt_img_stack[i]).mean()
    return l1

def richardson_lucy_3d(img,psf_stack,iter,gt_stack=None):
    img = img.astype(np.float64)/np.max(img)
    latent_est_stack = stackalize_img(img,psf_stack.shape[0])
    PSF_STACK = normalize_psf_stack(psf_stack)
    
    PSF_STACK_HAT = transpose_psf_stack(psf_stack)
    for i in range(iter):
        est_conv = stack_conv_3d(latent_est_stack,PSF_STACK)
        est_conv = est_conv/np.max(est_conv)
        relative_blur = img/est_conv
        error_est = up_dim_conv_stack(relative_blur,PSF_STACK_HAT)
        error_est = error_est/np.max(error_est)
        latent_est_stack = latent_est_stack * error_est
        latent_est_stack = latent_est_stack/latent_est_stack.max()
        if gt_stack is not None:
            print(i,l1_loss(latent_est_stack,gt_stack))
    result = latent_est_stack
    return result

def richardson_lucy_3d_fft(img,psf_stack,iter,gt_stack=None):
    img = img.astype(np.float64)/np.max(img)
    latent_est_stack = stackalize_img(img,psf_stack.shape[0])
    PSF_STACK = normalize_psf_stack(psf_stack)
    
    PSF_STACK_HAT = transpose_psf_stack(psf_stack)
    for i in range(iter):
        print('Running RL iteration {}'.format(iter))
        est_conv = stack_fftconv_3d(latent_est_stack,PSF_STACK)
        est_conv = est_conv/np.max(est_conv)
        relative_blur = img/est_conv
        error_est = up_dim_fftconv_stack(relative_blur,PSF_STACK_HAT)
        error_est = error_est/np.max(error_est)
        latent_est_stack = latent_est_stack * error_est
        latent_est_stack = latent_est_stack/latent_est_stack.max()
        if gt_stack is not None:
            print(i,l1_loss(latent_est_stack,gt_stack))
            conv_res = stack_fftconv_3d(latent_est_stack,psf_stack)
            img_diff = np.abs(conv_res-img)
            #imwrite(root_path+'/3D_Deconv/Results/RL/res/{}_conv_res.png'.format(i),conv_res)
            #imwrite(root_path+'/3D_Deconv/Results/RL/diff/{}_conv_res.png'.format(i),img_diff)
        #if i%10 == 0:
    #np.save(root_path+'/3D_Deconv/Results/RL/vol/{}_conv_vol.npy'.format(1),latent_est_stack)
    result = latent_est_stack
    return result
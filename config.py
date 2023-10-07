# Project Root Path. Adopting abs path for the sake of cluster programming
root_path = '/Users/Jipeng/VSCodeProjects/LFM_Simulation'
# Zebrafish Tiff File Path
target_tiff_path = root_path+ '/Data/zebrafish_tiff/HuCH2B_GCaMP6s_f2_2p_300_2.tif'
# PSF Stack Folder Path
psf_stack_path = root_path + '/Data/-10_5_0.1/'

# Cropping indices of PSF
psf_range_indices = [300,1300,250,1250]

# Deconv Saved Result
saved_deconv_path = root_path + '/Output/'
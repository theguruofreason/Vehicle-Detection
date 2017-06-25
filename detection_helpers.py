import cv2
import numpy as np
from skimage.feature import hog

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'BGR2GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if conv == 'BGR2HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if conv == 'BGR2Luv':
        return cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    if conv == 'BGR2RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conv == 'BGR2YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if conv == 'RGB2BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2Luv':
        return cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True, channels = [0,1,2]):
    features = []
    for channel in channels:
        features.append(hog(img[:,:,channel], orientations=orient,
                            pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block),
                            transform_sqrt=False,
                            visualise=vis, feature_vector=feature_vec))
    return np.concatenate(features)


# returns a rescaled list of lists of color channel values
def bin_spatial(img, color_space = 'BGR', size=(32, 32), color_channels = [0,1,2]):
    # Convert image to new color space (if specified)
    if color_space != 'BGR':
        if color_space == 'HSV':
            img = convert_color(img, 'BGR2HSV')
        elif color_space == 'Luv':
            img = convert_color(img, 'BGR2Luv')
        elif color_space == 'HLS':
            img = convert_color(img, 'BGR2HLS')
        elif color_space == 'YUV':
            img = convert_color(img, 'BGR2YUV')
    features = []
    for channel in color_channels:
        features.append(cv2.resize(img[:, :, channel], size).ravel())
    return np.concatenate(features)


# returns an array of counts of pixels within the corresponding value ranges for each color channel
def color_hist(img, nbins=16):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(img, use_spatial = True, spatial_colorspace='BGR', spatial_size=(32, 32), color_channels = [0,1,2],
                     use_histogram = True, hist_bins=32, use_hog = True, hog_conversion = 'BGR2HSV', hog_channels = [0,1,2], orient = 9,
                     pix_per_cell = 8, cell_per_block = 2):
    features_used = []
    if spatial_colorspace != 'BGR':
        feature_img = convert_color(img, conv = 'BGR2' + spatial_colorspace)
    else:
        feature_img = np.copy(img)
    if use_spatial:
        # Apply bin_spatial() to get spatial color features
        color_features = bin_spatial(feature_img, size = spatial_size, color_channels = color_channels)
        features_used.append(color_features)
    if use_histogram:
        # Apply color_hist() also with a color space option now
        color_histogram_features = color_hist(feature_img, nbins = hist_bins)
        features_used.append(color_histogram_features)
    if use_hog:
        hog_feature_img = convert_color(img, conv = hog_conversion)
        hog_features = get_hog_features(hog_feature_img, orient, pix_per_cell, cell_per_block,
        vis=False, feature_vec=True, channels = hog_channels)
        features_used.append(hog_features)
    # Append the new feature vector to the features list
    features = np.concatenate(features_used)
    # Return list of feature vectors
    return features

# set up parameters
use_spatial = True
spatial_colorspace = 'HSV'
spatial_size = (32, 32)
color_channels = [1,2]
use_histogram = True
histogram_bins = 12
use_hog = True
hog_conversion = 'BGR2YUV'
hog_channels = [0,1,2]
orient = 12
pix_per_cell = 8
cell_per_block = 2

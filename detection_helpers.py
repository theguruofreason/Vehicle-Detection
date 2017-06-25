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
    if conv == 'BGR2RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    features = []
    for channel in range(img.shape[2]):
        features.append(hog(img[:,:,channel], orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block),
                              transform_sqrt=False,
                              visualise=vis, feature_vector=feature_vec))
    return np.concatenate(features)


# returns a rescaled list of lists of color channel values
def bin_spatial(img, color_space = 'RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            img = convert_color(img, 'RGB2HSV')
        elif color_space == 'LUV':
            img = convert_color(img, 'RGB2LUV')
        elif color_space == 'HLS':
            img = convert_color(img, 'RGB2HLS')
        elif color_space == 'YUV':
            img = convert_color(img, 'RGB2YUV')
    channel_1 = cv2.resize(img[:, :, 0], size).ravel()
    channel_2 = cv2.resize(img[:, :, 1], size).ravel()
    channel_3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((channel_1, channel_2, channel_3))


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

def extract_features(img, spatial_colorspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hog_conversion = 'BGR2HSV', channel = 0, orient = 9,
                     pix_per_cell = 8, cell_per_block = 2):
    features = []
    if spatial_colorspace != 'BGR':
        feature_img = convert_color(img, 'BGR2' + spatial_colorspace)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_img, size = spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_img, nbins = hist_bins)
        hog_feature_img = convert_color(feature_img, conv = hog_conversion)
        hog_features = get_hog_features(hog_feature_img, orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True)
    else:
        feature_img = np.copy(img)
        spatial_features = bin_spatial(feature_img, size = spatial_size)
        hist_features = color_hist(feature_img, nbins = hist_bins)
        # Apply get_hog_features()
        hog_feature_img = convert_color(feature_img, conv = hog_conversion)
        hog_features = get_hog_features(hog_feature_img, orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features = np.concatenate((spatial_features, hist_features, hog_features))
    # Return list of feature vectors
    return features

# set up parameters
spatial_colorspace = 'BGR'
spatial_size = (32, 32)
histogram_bins = 16
hog_conversion = 'BGR2HSV'
orient = 12
pix_per_cell = 8
cell_per_block = 2

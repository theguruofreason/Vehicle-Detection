import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import glob
import pickle
import time



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
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


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

def extract_features(filename, spatial_colorspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hog_conversion = 'BGR2HSV', channel = 0, orient = 9,
                     pix_per_cell = 8, cell_per_block = 2):
    img = cv2.imread(filename)
    if spatial_colorspace != 'BGR':
        converted_img = convert_color(img, 'BGR2' + spatial_colorspace)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(converted_img, size = spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(converted_img, nbins = hist_bins)
        hog_converted_img = convert_color(converted_img, conv = hog_conversion)
        hog_features = get_hog_features(hog_converted_img, orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True)
    else:
        spatial_features = bin_spatial(img, size = spatial_size)
        hist_features = color_hist(img, nbins = hist_bins)
        # Apply get_hog_features()
        hog_img = convert_color(img, conv = hog_conversion)
        hog_features = get_hog_features(hog_img, orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True)
    # Append the new feature vector to the features list
    features = []
    three_features = []
    three_features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features

car_image_filenames = glob.glob('./vehicles/*/*.png')
num_car_images = len(car_image_filenames)
non_car_image_filenames = glob.glob('./non-vehicles/*/*.png')
num_non_car_images = len(non_car_image_filenames)
print('# car images:', num_car_images, '\n# non-car images:', num_non_car_images)

data_subset_size = 500 # number of car and non-car images to use

# select random car and non-car examples
car_examples = np.random.choice(car_image_filenames, data_subset_size, replace = False)
non_car_examples = np.random.choice(non_car_image_filenames, data_subset_size, replace = False)

car_features = []
non_car_features = []

# set up parameters
spatial_colorspace = 'BGR'
spatial_size = (16, 16)
histogram_bins = 16
hog_conversion = 'BGR2GRAY'
orient = 9
pix_per_cell = 8
cell_per_block = 2

# extract the features for x
for file in car_examples:
    car_features.append(extract_features(file, spatial_colorspace = spatial_colorspace, spatial_size = spatial_size,
                                    hist_bins = histogram_bins, hog_conversion = hog_conversion, channel = 1,
                                    orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block))

for file in non_car_examples:
    non_car_features.append(extract_features(file, spatial_colorspace = spatial_colorspace, spatial_size = spatial_size,
                                    hist_bins = histogram_bins, hog_conversion = hog_conversion, channel = 1,
                                   orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block))


# put all examples together to be split later by train_test_split()
X = np.vstack((car_features, non_car_features)).astype(np.float64)
#print(car_features.shape)
#print(non_car_features.shape)
print(X.shape)
# fit per-column scaler
X_scaler = StandardScaler().fit(X)
# apply scaler
scaled_X = X_scaler.transform(X)

# labels vector
y = np.hstack((np.ones(data_subset_size), np.zeros(data_subset_size)))
print(len(scaled_X))
print(len(y))
# divide selected data into training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state = rand_state)

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
